import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, sdf_values):
        self.point_clouds = point_clouds
        self.sdf_values = sdf_values
        
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        point_cloud = torch.FloatTensor(self.point_clouds[idx])
        sdf_value = torch.FloatTensor(self.sdf_values[idx])
        return point_cloud, sdf_value

def create_data_loader(point_clouds, sdf_values, batch_size=32, shuffle=True):
    dataset = PointCloudDataset(point_clouds, sdf_values)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
