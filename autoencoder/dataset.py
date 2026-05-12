import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir, model_type):
        if "agg" in data_dir and "wo_agg" not in data_dir:
            self.data_names = glob.glob(os.path.join(data_dir, '*agg.npy'))
        elif "surgB" in data_dir:
            self.data_names = glob.glob(os.path.join(data_dir, '*surgB.npy'))
        elif "surg" in data_dir:
            self.data_names = glob.glob(os.path.join(data_dir, '*surg.npy'))
        elif "wo_agg" in data_dir:
            self.data_names = glob.glob(os.path.join(data_dir, '*_f.npy'))
        else:
            self.data_names = glob.glob(os.path.join(data_dir, '*_f.npy'))
        self.data_dic = {}
        data_list = []
        for i in tqdm(range(len(self.data_names))):
            features = np.load(self.data_names[i])
            if ("agg" or "surg") in data_dir and ("wo_agg" not in data_dir):
                features = features.transpose(0, 2, 3, 1) #.reshape(1, -1, 512) # [1, 512, 192, 192] -> [1, 192, 192, 512] -> [1, 192*192, 512]
                if model_type == 'mlp':
                    features = features.reshape(1, -1, 512)
            name = self.data_names[i].split('/')[-1].split('.')[0]
            self.data_dic[name] = features.shape[0] 
            data_list.append(features)
        data = np.concatenate(data_list, axis=0)
        self.data = torch.from_numpy(data) # 只学习从[1, 512]->[1, 3]的映射

    def __getitem__(self, index):
        data = self.data[index]
        # features = np.load(self.data_names[index])
        # if "agg" in self.data_names[index]:
        #     data = features.transpose(0, 2, 3, 1)#.reshape(1, -1, 512) # [1, 512, 192, 192] -> [1, 192, 192, 512] -> [1, 192*192, 512]
        # else:
        #     data = features
        # data = torch.tensor(data).squeeze(0)
        return data

    def __len__(self):
        return self.data.shape[0] 
        # return len(self.data_names)
    
    
class Seg_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_names = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        self.data_dic = {}
        self.data_path = []
        
        for i in range(len(self.data_names)):
            features = np.load(self.data_names[i])
            self.data_path.append(self.data_names[i])
            name = self.data_names[i].split('/')[-1].split('.')[0]
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