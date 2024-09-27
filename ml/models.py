import math

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import MessagePassing


class RNNDecoder(nn.Module):
    def __init__(
        self,
        in_feat_size,
        out_feat_size,
        hidden_size,
        dropout,
        num_layers,
        device,
    ):
        super().__init__()
        assert isinstance(hidden_size, list)
        self.rnn_layer_module = nn.ModuleList()
        self.dense_layer_module = nn.ModuleList()
        assert len(hidden_size) == num_layers
        for i in range(num_layers):
            input_size = in_feat_size if i == 0 else hidden_size[i - 1]
            self.rnn_layer_module.append(
                nn.GRU(input_size=input_size, hidden_size=hidden_size[i], batch_first=True)
            )

        self.relu = torch.nn.ReLU()
        self.out1 = nn.Linear(hidden_size[-1], hidden_size[-1])
        self.out2 = nn.Linear(hidden_size[-1], out_feat_size)
        self.device = device
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, previous_hidden=None):
        if isinstance(previous_hidden, list):
            hidden = []
            for i, layer in enumerate(self.rnn_layer_module):
                y, hidden_state = layer(y, previous_hidden[i])
                y = self.dropout(y)
                hidden.append(hidden_state)
            output = self.out2(self.dropout(self.relu(self.out1(y))))
        elif previous_hidden is None: # For first input, there is no hidden state
            hidden = []
            for i, layer in enumerate(self.rnn_layer_module):
                y, hidden_state = layer(y)
                y = self.dropout(y)
                hidden.append(hidden_state)
            output = self.out2(self.dropout(self.relu(self.out1(y))))

        return output, hidden


class EncoderDecoder(nn.Module):
    """Autoencoder-like model, consisting of a RNNEncoder and a DecoderCell, see above."""

    def __init__(
        self,
        input_size,
        enc_hidden_size,
        encoded_feat_size,
        dec_hidden_size, 
        out_size,
        enc_dropout,
        dec_dropout,
        static_feat_size,
        enc_num_layers,
        dec_num_layers,
        embedding_size,
        device,
    ):
        super().__init__()
        self.encoder = RNNDecoder(
            in_feat_size=1,
            out_feat_size=1,
            hidden_size=dec_hidden_size,
            dropout=dec_dropout,
            num_layers=dec_num_layers,
            device=device,
        )

        self.device = device
        self.embedding_size = embedding_size
        self.out_size = out_size

    def forward(
        self,
        encoder_input,
        decoder_input,
        static,
        edge_attr,
        stat_edge_index,
        dyn_e_idx,
        dyn_e_attr,
        target,
        train_mask=None
    ):
        enc_output, prev_hidden = self.encoder(y=encoder_input)
        y_prev = enc_output[:, -1, :].unsqueeze(1)

        #Initialize
        decoder_output = torch.zeros(
            encoder_input.size(0), self.out_size, 1, device=self.device
        ).float()
        #Autoregressive
        if not self.training:
            #TODO teach forcing works in parallel!
            #input the last encoder input, i.e. last day covid
            #y_prev = encoder_input[:, -1,:].unsqueeze(2)
            for i in range(self.out_size):
                step_decoder_input = y_prev#.squeeze(1)#torch.cat with decoder_input[:, i, :]
                y_prev, prev_hidden = self.encoder(
                    y=step_decoder_input, previous_hidden=prev_hidden
                )
                decoder_output[:, i] = y_prev.squeeze(1)

        #in training teacher forcing is done!
        elif self.training:
            step_decoder_input = y_prev
            for i in range(self.out_size):
                #step_decoder_input = torch.cat((target[:,i-decoder_input.shape[1],:], decoder_input[:, i, :]), axis=1)
                y_prev, prev_hidden = self.encoder(
                    y=step_decoder_input, previous_hidden=prev_hidden
                )
                step_decoder_input = target[:,i,:].unsqueeze(1)
                decoder_output[:, i] = y_prev.squeeze(1)
        return decoder_output, enc_output



class GCN(torch.nn.Module):
    def __init__(self, features, dropout):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(-1, 63)
        #self.conv5 = torch_geometric.nn.GCNConv(-1, 32)
        
        #self.layer_norm_1 = torch.nn.LayerNorm([features])
        self.relu = torch.nn.ReLU()
        self.linear1 = nn.Linear(64,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,64)
        self.linear4 = nn.Linear(64,64)
        self.linear5 = nn.Linear(64,features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x_input = x
        x = self.conv1(x, edge_index, edge_weight=edge_weight)  # skip connection!
        x = torch.concat((x, x_input), 1)
        x = x.type(torch.float)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.linear3(x)
        x = self.dropout(x)
        x = self.relu(x)


        x = self.linear4(x)
        x = self.dropout(x)
        x = self.relu(x)

        #x = 0.5 * (self.conv5(x, edge_index, edge_weight=edge_weight)  + x)  # skip connection
        x = x.type(torch.float)
        x = self.linear5(x)
        return x


class RNN_GNN_Fusion(nn.Module):
    def __init__(
            self,
            rnn_hidden_size,
            forecast_len,
            output_len,
            dropout,
            device,
    ):
        super().__init__()
        self.rnn = RNNDecoder(
            in_feat_size=1,
            out_feat_size=1,
            hidden_size=rnn_hidden_size,
            dropout=dropout,
            num_layers=len(rnn_hidden_size),
            device=device,
        )
        self.forecast_len = forecast_len
        #self.GCN = GCN(features=1, dropout=dropout)
        self.GNN = CustomMessagePassing(dropout=dropout)
        #self.readout = nn.Linear(2,1, bias=False)
        self.out_param = torch.nn.Parameter(torch.tensor(0.5))
        '''
        self.params = nn.ModuleDict({
            'rnn': nn.ModuleList([self.rnn]),
            'gnn': nn.ModuleList([self.GNN])})
        '''
        self.device = device
        self.only_rnn = False

    def forward(
        self,
        encoder_input,
        decoder_input,
        static,
        edge_attr,
        stat_edge_index,
        dyn_e_idx,
        dyn_e_attr,
        target,
        train_mask=None,
    ):
        #encoder_input = torch.cat([gnn_output, encoder_input], dim = 2)
        enc_output, prev_hidden = self.rnn(y=encoder_input)
        y_prev = enc_output[:, -1, :]

        #Initialize
        decoder_output = torch.zeros(
            encoder_input.size(0), self.forecast_len, 1, device=self.device
        ).float()
        
        step_graph_input = encoder_input[:,-6:,:]

        #Autoregressive
        if not self.training:
            for i in range(self.forecast_len):
                step_graph_input = torch.cat((step_graph_input,y_prev.unsqueeze(2)),dim = 1)[:,-7:,:]

                graph_data =  torch_geometric.data.Data(x=step_graph_input.squeeze(2), edge_index=stat_edge_index, edge_weight=edge_attr).to(self.device)
                #graph_data =  torch_geometric.data.Data(x=y_prev, edge_index=stat_edge_index, edge_weight=edge_attr).to(self.device)
                diffusion = self.GNN(graph_data)
                y_prev, prev_hidden = self.rnn(
                    y=y_prev.unsqueeze(1), previous_hidden=prev_hidden
                )
                out = (torch.sigmoid(self.out_param) * y_prev[:,0,:]) + ((1-torch.sigmoid(self.out_param)) * diffusion)
                #out = y_prev[:,0,:] + diffusion
                y_prev = out
                #y_prev = self.readout(out)
                decoder_output[:, i] = y_prev

        #in training teacher forcing is done!
        elif self.training:
            step_decoder_input = y_prev.unsqueeze(1)
            for i in range(self.forecast_len):
                if self.only_rnn:
                    y_prev, prev_hidden = self.rnn(
                        y=step_decoder_input, previous_hidden=prev_hidden
                    )
                    step_decoder_input = target[:,i,:].unsqueeze(1) #Teacher forcing
                    decoder_output[:, i] = y_prev.squeeze(1)
                else:
                    step_graph_input = torch.cat((step_graph_input,step_decoder_input),dim = 1)[:,-7:,:]
                    graph_data =  torch_geometric.data.Data(x=step_graph_input[:,:,0], edge_index = stat_edge_index, edge_weight=edge_attr).to(self.device)

                    #input for x has dim [400,1]
                    #graph_data =  torch_geometric.data.Data(x=step_decoder_input[:,:,0], edge_index = stat_edge_index, edge_weight=edge_attr).to(self.device)
                    diffusion = self.GNN(graph_data).unsqueeze(2)
                    #with torch.no_grad():
                    y_prev, prev_hidden = self.rnn(
                        y=step_decoder_input, previous_hidden=prev_hidden
                    )
                    step_decoder_input = target[:,i,:].unsqueeze(1) #Teacher forcing
                    #out = torch.cat((y_prev.squeeze(1), diffusion.squeeze(1)),dim=1)
                    #out = self.readout(out)
                    out = (torch.sigmoid(self.out_param) * y_prev[:,0,:]) + ((1-torch.sigmoid(self.out_param)) * diffusion[:,0,:])
                    #out = y_prev[:,0,:] + diffusion[:,0,:]
                    decoder_output[:, i] = out#[400,1]
        return decoder_output, enc_output



class TSTEmbedding(nn.Module):
    """
    this should transform an input vector (token) of dim
    [400,7,3]
    to
    [400, 7, d_model] (i think?)
    """

    def __init__(self, input_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.projection_matrix = torch.nn.Linear(input_size, d_model)#TODO is this correct?
        # assert d_model >= input_size
        # self.repeat = -(-d_model // input_size) #rounding up vs down
    def forward(self, x):
        #x = x.repeat(1, 1, self.repeat)[:, :, : self.d_model]
        x = self.projection_matrix(x)
        # what should the output dimension be?[400, d_model * n_heads]
        # this is easily possible as 7*4 <<<< 64*8!
        return x


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        mse = nn.MSELoss()(y_true, y_pred)
        rmse = torch.sqrt(mse)
        return rmse

class CustomMessagePassing(MessagePassing):
    def __init__(self, dropout):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(1, 1, bias=False)
        self.normalize_vec = None
        self.lin_mess_1 = torch.nn.Linear(14, 64)
        self.lin_mess_2 = torch.nn.Linear(64, 32)
        self.lin_mess_out = torch.nn.Linear(32, 1)
        self.acti = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        #if self.normalize_vec == None:
        self.normalize_vec = torch.bincount(edge_index[1], weights=edge_weight)
        edge_index = edge_index.type(torch.int64)
        out = self.propagate(edge_index, x=x, ed_we=edge_weight, norm_vec=self.normalize_vec)
        return out

    def message(self, x_j, x_i, ed_we):
        #maybe use also sqrt(degree)
        #out = torch.cat((x_j, x_j - x_i, ed_we.view(-1,1).type(torch.float32)), dim = 1)
        out = torch.cat((x_j, x_j - x_i), dim = 1)

        out = self.lin_mess_1(out)
        out = self.acti(out)
        out = self.dropout(out)

        out = self.lin_mess_2(out)
        out = self.acti(out)
        out = self.dropout(out)

        out = self.lin_mess_out(out)
        out = out * ed_we.view(-1,1)
        #out = x_j * ed_we.view(-1,1) # inverse distance weighting
        return out #is of shape [num_edges, 1]

    def update(self, message, x, norm_vec):
        #out = torch.cat((x , (message / norm_vec.view(-1,1))), dim=1)
        out = message / norm_vec.view(-1,1)
        out = out.type(torch.float32)
        out = self.lin(out)
        return out #torch.ones(400,1)




# Brandstetter from here on
# taken from https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers/blob/master/experiments/models_gnn.py

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 n_variables: int):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super().__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + 1, hidden_features),
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          Swish()
                                          )
        # TODO read this
        self.norm = torch_geometric.nn.InstanceNorm(hidden_features)

    def forward(self, h, u, dist, edge_index, batch):#jump
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index.type(torch.int64),h=h, u=u, edge_weight=dist)
        # TODO not mentioned?
        #x = self.norm(x, batch)
        return x

    def message(self,h_i, h_j, u_i, u_j, edge_weight):
        """
        Message update following formula 8 of the paper
        """
        edge_weight = edge_weight.unsqueeze(1)
        message = self.message_net_1(torch.cat((h_i, h_j, u_i-u_j, edge_weight), dim=-1).type(torch.float))
        message = self.message_net_2(message)
        return message

    def update(self, message, h):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((h, message), dim=-1))
        update = self.update_net_2(update)
        #if self.in_features == self.out_features:
        return h + update
        # else:
        #    return update


class MP_PDE_Solver(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super().__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        # assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        # self.eq_variables = eq_variables

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=1
                                        )
                               )
        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window, self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==14):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=5),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        elif(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        elif (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        elif(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data, _a, _b, dist, edge_index, _c, _d, targets, train_mask) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        # data.shape: [400, 14, 1]

        # +++ begin encoder +++
        u = data.squeeze(2)
        # Encode and normalize coordinate information
        # pos = data.pos
        # pos_x = pos[:, 1][:, None] / self.pde.L
        # pos_t = pos[:, 0][:, None] / self.pde.tmax
        # edge_index = data.edge_index given already
        batch = data

        # Encode equation specific parameters
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        # variables = pos_t    # time is treated as equation variable
        # Encoder and processor (message passing)
        # node_input = torch.cat((u, pos_x, variables), -1)
        node_input = data
        h = self.embedding_mlp(node_input.squeeze())

        # +++ end encoder +++
        for i in range(self.hidden_layer):
            # gnn_layer take = node_data, 
            # h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)
            h = self.gnn_layers[i](h, u, dist, edge_index, batch)
        
        # Decoder (formula 10 in the paper)
        dt = torch.arange(1, self.time_window + 1, 1) .to(h.device)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        # test output_mlp with

        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff
        #out.shape: [400, 14]

        return out.unsqueeze(2), None



# taken from https://github.com/HySonLab/pandemic_tgnn/blob/main/code/models.py
class GraphEncoding(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout,forecast_length, device):
        super().__init__()
        self.forecast_length = forecast_length
        #window = 14
        self.window = window #?
        self.n_nodes = n_nodes 
        self.nhid = nhid
        self.device = device
        # self.nfeat = nfeat #context length
        #self.nfeat = 1 #amount features / timeseries
        #nfeat = 1
        self.conv1 = torch_geometric.nn.conv.GCNConv(1, nhid) #in_channels, out_channels
        self.conv2 = torch_geometric.nn.conv.GCNConv(nhid, nhid)
        
        #TODO not referenced, not used?
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, x, adj, edge_weights):
        out = torch.ones((x.shape[0],self.forecast_length), device=self.device)
        orig_x = x #shape 400, 14, 1
        for i in range(self.forecast_length):
            lst = []
            weight = edge_weights #adj.coalesce().values()
            skip = x.view(x.shape[0],-1)
            #adj = adj.coalesce().indices()
            # skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
            # skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes
            # x.shape= 400,14,1
            # x = self.conv1(x.transpose(0,1), adj, edge_weight=weight) # TODO check if 14 gradients
            x = self.relu(self.conv1(x.transpose(0,1), adj, edge_weight=weight)).float()
            # we tested if batching works:
            # out = x.mean()
            # out.backward
            # for name, param in self.conv1.named_parameters(): 
            #     print(f"Gradients for {name}: {param.grad.size()}")
            # didnt show the batch-dim, as its already averaged.
            
            x = self.bn1(x.transpose(1,2))#works, but is it correct?
            x = self.dropout(x)
            lst.append(x)
            
            x = self.relu(self.conv2(x.transpose(1,2), adj,edge_weight=weight)).float()
            x = self.bn2(x.transpose(1,2))
            x = self.dropout(x)
            lst.append(x)
            
            x = torch.cat(lst, dim=1) #14, 256, 400
            #--------------------------------------
            # x = x.view(-1, self.window, self.n_nodes, x.size(1))
            # x = torch.transpose(x, 0, 1)
            # x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes
            
            x, (hn1, cn1) = self.rnn1(x.transpose(1,2))
            
            out2, (hn2,  cn2) = self.rnn2(x)
            
            #x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)
            x = torch.cat([x[-1,:,:], out2[-1,:,:]], dim = 1)
                    
            x = torch.cat([x,skip], dim=1)
            #--------------------------------------
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))# [400,1]
            
            out[:,i] = x.squeeze()
            orig_x = torch.cat((orig_x, x.unsqueeze(2)), dim=1)[:, 1:, :]
            x = orig_x
        return out.unsqueeze(2), None


class TST(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super(TST, self).__init__()
        self.model_type = 'Transformer'

        # Encoder and Decoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)

        #self.encoder_embedding = nn.Embedding(ntoken, d_model)
        #self.decoder_embedding = nn.Embedding(ntoken, d_model)
        #from [batchsize, seq_len, input_dim] to [batchsize, seq_len, 
        self.encoder_embedding = TSTEmbedding(input_size=1, d_model=d_model)
        self.decoder_embedding = TSTEmbedding(input_size=1, d_model=d_model)

        # self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.positional_encoding = PositionalEncoding(feature_size=1, max_len=56)

        self.output_layer = nn.Linear(d_model, 1)#ntoken replaced with 1
        # TODO d_hid should be used, in a multi-layer setting
        self.output_layer = nn.Sequential(nn.Linear(d_model, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid,1),
                )
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        #self.decoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer[-1].bias.data.zero_()
        self.output_layer[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self,
            encoder_input,
            decoder_input,
            static,
            edge_attr,
            stat_edge_index,
            dyn_e_idx,
            dyn_e_attr,
            target,
            train_mask):
        # link function to defined
        src=encoder_input
        src_mask=None
        src_padding_mask=None
        tgt=target
        # tgt_mask=None
        #target_sequence_length = tgt.size(1) # 14?!
        #tgt_mask = self.generate_square_subsequent_mask(target_sequence_length).to(tgt.device)
        is_inference=True if tgt == None else False
        src_embed = self.positional_encoding(self.encoder_embedding(src))
        memory = self.transformer_encoder(src_embed.transpose(0,1))

        if is_inference:
            # Start with the last value of the source sequence as the first input to the target
            tgt = src[:, -1].unsqueeze(1)
            for i in range(14):
                # Prepare the target embedding at each step
                tgt_embed = self.positional_encoding(self.decoder_embedding(tgt), shift=14)
                # Decode the target embeddings with the previously computed memory
                output = self.transformer_decoder(tgt_embed.transpose(0, 1), memory)
                # Pass the decoder's output through the final linear layer
                next_output = self.output_layer(output).transpose(0, 1)

                # Get the last step's output for the forecast
                next_step = next_output[:, -1, :]

                # Append the predicted next step to the target sequence
                tgt = torch.cat((tgt, next_step.unsqueeze(1)), dim=1)
            return tgt[:, 1:, :], None
        else:
            tgt =  torch.cat((src[:,-1].unsqueeze(2), tgt), 1)[:,:-1,:]
            target_sequence_length = tgt.size(1) # 14?!
            tgt_mask = self.generate_square_subsequent_mask(target_sequence_length).to(tgt.device)
            tgt_embed = self.positional_encoding(self.decoder_embedding(tgt))
            output = self.transformer_decoder(tgt_embed.transpose(0,1), memory, tgt_mask) #target, memory target_mask, 
            output = self.output_layer(output).transpose(0,1)
            return output, None


class PositionalEncoding(torch.nn.Module):
    def __init__(self, feature_size, max_len=5000):#d_model, dropout
        super(PositionalEncoding, self).__init__()
        # Create a long enough `position_encoding`
        pe = torch.zeros(max_len, feature_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_size, 2).float() * (-math.log(10000.0) / feature_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x, shift=0):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, feature_size]
            shift: Integer, shifts the positional encodings forward (positive value) or backward (negative value)
        """
        # Adjust start and end positions based on shift
        start_pos = max(shift, 0)
        end_pos = x.size(1) + shift

        # Use expand to match the batch size without explicit repetition and apply shift
        pe = self.pe[:, start_pos:end_pos].expand(x.size(0), -1, -1)
        return x + pe
