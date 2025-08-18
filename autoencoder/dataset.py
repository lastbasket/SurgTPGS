import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
        self.data_dic = {}
        for i in range(len(data_names)):
            features = np.load(data_names[i])
            name = data_names[i].split('/')[-1].split('.')[0]
            self.data_dic[name] = features.shape[0] 
            if i == 0:
                data = features
            else:
                data = np.concatenate([data, features], axis=0)
        self.data = data

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data

    def __len__(self):
        return self.data.shape[0] 
    
    
class Seg_dataset(Dataset):
    def __init__(self, data_dir):
        data_names = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        self.data_dic = {}
        self.data_path = []
        
        for i in range(len(data_names)):
            features = np.load(data_names[i])
            self.data_path.append(data_names[i])
            name = data_names[i].split('/')[-1].split('.')[0]
            self.data_dic[name] = features.shape[0]
            if i == 0:
                data = features[None]
            else:
                data = np.concatenate([data, features[None]], axis=0)
        self.data = data

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data, self.data_path[index]

    def __len__(self):
        return self.data.shape[0] 