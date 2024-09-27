import torch
import numpy as np
import matplotlib.pyplot as plt
import dataset
import os

#Plotting utilities
def branch_plot(targets, predictions, llv=0, stride = 1, name_string="", output_dir=""):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    fig.suptitle(f"Val loss: {llv}")
    # stuttgart, berlin, barnim, goslar
    indices = [0, 140, 145, 200] if targets.shape[1] == 400 else [0, 5, 10, 20] 
    for en_index, index in enumerate(indices):
        assert (targets.shape[2] > stride)
        branch_target = torch.cat((targets[:,index, :stride, :].flatten(), targets[-1, index, 1:, :].squeeze()), dim = 0)

        x_axis = torch.tensor(np.arange(branch_target.shape[0]))

        axs[en_index].plot(x_axis, branch_target.cpu())
        #axs[en_index].set_xlabel(str(index) + " over time")
        axs[en_index].set_ylabel(f"Infection on {index}")
        axs[en_index].tick_params(labelrotation=45)
        
        pred_len = predictions.shape[2]
        for i in range(predictions.shape[0]):
            #print(f"shape of x_axis: {x_axis[i: i + pred_len].shape}")
            #print(f"shape of branch: {predictions[i, en_index, :, 0].shape}")
            axs[en_index].plot(x_axis[i: i + pred_len], predictions[i, index, :, 0].cpu())
        if (en_index < 3):
            axs[en_index].tick_params(axis="x",which="both", bottom=False, top=False, labelbottom=False)
    filepath = os.path.join(output_dir, f"branch_plot_{name_string}.png")
    plt.savefig(filepath, dpi = 600)
    plt.close()

        

def box_plot():
    pass

def training_loss_plot(train, validation, name_string="", output_dir=""):
    x = range(len(train))
    fig, ax1 = plt.subplots()
    ax1.plot(x, train, color="#E2725B")
    ax1.set_xlabel("Epochs")
    ax1.set_yscale("log")
    ax1.set_ylabel("Train Loss")
    ax1.text(len(train)-1, train[-1], f'{train[-1]:.4f}', ha='left', va='center', color="#E2725B")

    ax2 = ax1.twinx()

    ax2.plot(x, validation, color="#7F00FF")
    ax2.set_yscale("log")
    ax2.set_ylabel("Validation Loss")
    ax2.text(len(validation)-1, validation[-1], f'{validation[-1]:.4f}', ha='left', va='center', color="#7F00FF")

    '''
    plt.plot(train, label = "Train Loss")
    plt.plot(validation, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    '''
    #plt.legend()
    filepath = os.path.join(output_dir, f"loss_plot_{name_string}.png")
    plt.savefig(filepath)
    plt.close()


def daily_loss_plot(daily_loss, name_string, output_dir):
    filepath = os.path.join(output_dir, f"{name_string}_daily_loss")
    plt.plot(daily_loss.cpu())
    plt.savefig(filepath)
    plt.close()

class add_noise_transform:
    def __init__(self, mean = 0, std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        out = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(out, min=0)
        #return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return f"add_noise_transform, mean={self.mean}, std={self.std}"     

class remove_locations_transform:
    def __init__(self, perc = 1.):
        self.perc = perc
    def __call__(self, tensor):
        total_locations = tensor.size(1)  # Dynamically get the size of the tensor along dimension 1
        abs_drop = int(self.perc * total_locations)  # Calculate the absolute number of locations to drop based on the tensor's size
        indices = torch.randperm(total_locations)[:abs_drop]  # Adjust to use the dynamic size
        tensor[:, indices] = 0
        return tensor
    def __repr__(self):
        return f"randomly remove some entries" 

def get_dataset(key, config, train):
    """
    Returns a dataset based on the given key, configuration, and train flag.

    Args:
        key (str): The key to identify the dataset type. Possible values are:
            - 'simulation': for the SIDiffusionEquationDataset
            - 'wave': for the WaveEquationDataset
            - 'advection': for the AdvectionDiffusionEquationDataset
        config (SimpleNamespace): The configuration object given by the ConfigLoader.
        train (bool): A flag indicating whether the dataset is for training or evaluation.

    Returns:
        dataset (Dataset): The dataset object based on the given key and configuration.
    """
    noise_config = config.noise.get('train', {}) if train else config.noise.get('eval', {})
    
    mean = noise_config.get('gaussian_mean', 0)
    std = noise_config.get('gaussian_std', 0.01)
    locations_perc = noise_config.get('locations_perc', 1.)

    remove_transform = remove_locations_transform(locations_perc)
    noise_transform = add_noise_transform(mean, std)
    combined_transform = lambda x: remove_transform(noise_transform(x))
    
    dataset_map = {
        'simulation': dataset.SIDiffusionEquationDataset,
        'wave': dataset.WaveEquationDataset,
        'advection': dataset.AdvectionDiffusionEquationDataset
    }
    return dataset_map[key](config.encoder_length, config.forecast_length, transform=combined_transform)

def get_adjacency(key, device):
    filepaths = {   
        'simulation': os.path.join('..','data', 'SI_diffusion_equation','nuts3_adjacent_distances.pt'),
        'wave': os.path.join('..','data', 'wave_equation','germany_coastline_adjacency.pt'),
        'advection': os.path.join('..','data', 'advection_diffusion_equation','nuts3_adjacent_distances.pt')
        }

    adjacent_distances = torch.load(filepaths[key]).T.to(device)
    edge_index = adjacent_distances[:2, :].int()
    dist = (1/adjacent_distances[2,:])
    return dist, edge_index
        