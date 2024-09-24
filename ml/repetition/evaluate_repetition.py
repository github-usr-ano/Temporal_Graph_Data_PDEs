import sys
import os
from torch.utils.data import DataLoader, ConcatDataset
import torch

# Set up the project root directory
ml_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ml_dir)
sys.path.append(ml_dir)

from config_loader import ConfigLoader 
import models
import utils
import numpy as np


config = ConfigLoader(os.path.abspath(os.path.join(ml_dir, "repetition", "repetition.yml"))).get_config()
device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")

eval_datasources = config.eval_datasources

loss_func = models.RMSELoss()

logged_loss_val = []
logged_loss_train = []
# Training loop
for eval_data_sources in eval_datasources:
    datasets = [utils.get_dataset(source, config, train=False) for source in eval_data_sources]
    validation_dataset = ConcatDataset([torch.utils.data.Subset(ds, range(int(0.8 * len(ds)), len(ds))) for ds in datasets])
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    eval_data_sources_str = '_'.join([ds[:4] for ds in eval_data_sources])
    model_dir = f"{config.model_name}_" + eval_data_sources_str
    model_path = os.path.join(model_dir, model_dir)
    
    for epoch in range(1):
        running_loss_val = 0.0
        daily_loss = []

        with torch.no_grad():
            targets_val = []
            pred_val = []
            logg_loss_val = 0
            for batch_idx, data in enumerate(validation_loader):
                inputs, targets = data  # Assuming your data loader returns inputs and targets
            
                inputs = torch.permute(inputs, (2, 1, 0)).to(device)
                targets = torch.permute(targets, (2, 1, 0)).to(device)
                
                # Forward pass
                outputs = inputs[:,-1,:].repeat(1,config.forecast_length).unsqueeze(2)
                # Calculate the loss

                daily_loss.append((outputs-targets).double().square().mean(axis=0).squeeze())
                val_loss = loss_func(outputs, targets)
                running_loss_val += val_loss.item()

                targets_val.append(targets)
                pred_val.append(outputs)
                
                if batch_idx % 100 == 99:  # Print every 100 batches
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Val-Loss: {running_loss_val / 100:.4f}")
                    logg_loss_val += running_loss_val
                    running_loss_val = 0.0
            logged_loss_val.append(logg_loss_val)
        daily_loss = torch.stack(daily_loss).sqrt().mean(axis=0)
        targets_val = torch.stack(targets_val)
        pred_val = torch.stack(pred_val)

    os.makedirs(model_dir, exist_ok=True)
    np.savetxt(os.path.join("", f"{model_dir}_daily_losses.csv"), np.array(daily_loss.unsqueeze(0).cpu()), delimiter=",")

    utils.daily_loss_plot(daily_loss,  name_string=model_dir, output_dir=model_dir)
    utils.branch_plot(targets_val, pred_val, llv=logged_loss_val[-1], name_string=eval_data_sources_str , output_dir=model_dir)


print("Evaluation finished")
