import torch
import numpy as np

import os


class SIDiffusionEquationDataset(torch.utils.data.Dataset):
    path = os.path.join('..', 'data', 'SI_diffusion_equation', 'SI_equation_dataset.npy')

    def __init__(self, context_len, forecast_len, stride=1, transform = None):
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.stride = stride
        self.data = torch.tensor(np.load(self.path)[:,:,1], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return (len(self.data) - 2*(self.context_len + self.forecast_len)) // self.stride
    def __getitem__(self, idx):
        context_data = self.data[idx*self.stride : idx*self.stride + self.context_len,:]
        target_data = self.data[idx*self.stride + self.context_len : idx*self.stride + self.context_len + self.forecast_len,:]
        if self.transform:
            context_data = self.transform(context_data)
        return context_data, target_data


class WaveEquationDataset(torch.utils.data.Dataset):
    path = os.path.join('..', 'data', 'wave_equation', 'wave_equation_dataset.npy')

    def __init__(self, context_len, forecast_len, stride=1, transform = None):
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.stride = stride
        self.data = torch.tensor(np.load(self.path)[:,:,0], dtype=torch.float32)
        self.transform = transform
        def normalize_data(data):
            min_val = data.min(dim=0, keepdim=True)[0]
            max_val = data.max(dim=0, keepdim=True)[0]
            normalized_data = (data - min_val) / (max_val - min_val)
            return normalized_data
        self.data = normalize_data(self.data)

    def __len__(self):
        return (len(self.data) - 2*(self.context_len + self.forecast_len)) // self.stride
    def __getitem__(self, idx):
        context_data = self.data[idx*self.stride : idx*self.stride + self.context_len,:]
        target_data = self.data[idx*self.stride + self.context_len : idx*self.stride + self.context_len + self.forecast_len,:]
        if self.transform:
            context_data = self.transform(context_data)
        return context_data, target_data


class AdvectionDiffusionEquationDataset(torch.utils.data.Dataset):
    path = os.path.join('..', 'data', 'advection_diffusion_equation', 'advection_diffusion_dataset.npy')

    def __init__(self, context_len, forecast_len, stride=1, transform = None):
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.stride = stride
        self.data = torch.tensor(np.load(self.path)[:,:,0], dtype=torch.float32)
        self.data[self.data < 1e-8] = 0	
        self.transform = transform

    def __len__(self):
        return (len(self.data) - 2*(self.context_len + self.forecast_len)) // self.stride
    def __getitem__(self, idx):
        context_data = self.data[idx*self.stride : idx*self.stride + self.context_len,:]
        target_data = self.data[idx*self.stride + self.context_len : idx*self.stride + self.context_len + self.forecast_len,:]
        if self.transform:
            context_data = self.transform(context_data)
        return context_data, target_data                                                  