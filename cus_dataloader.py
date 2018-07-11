import torch
import os
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils

class CustomDataset(Dataset):
        # These three functions are necessary function for Dataset Object
        # Declaring these functions enables Dataloader object to work well
        # Caution : CSV file amplifies data size about 9~20 times
        # It may explode RAM memory
        def __init__(self, data_dir, data_file, window_size, transform=None):
                self.data = os.path.join(data_dir, data_file)
                self.csv = list(csv.reader(open(self.data,'r')))[1:]
                self.window_size = window_size
                self.transform = transform
                self.data_num = len(self.csv) #specific function for data

        def __len__(self):
                return self.data_num-self.window_size

        def __getitem__(self, idx):
                #getting data
                target_data = []
                for i in range(idx, idx+self.window_size):
                        target_data.append(self.csv[i])
                # transform data if it exists
                if self.transform:
                        target_data = self.transform(target_data)
                # getting data's label if exists
                target_label = float(target_data[-1][1]) # getting Label
                target_data[-1][1]=0.0
                return target_data, target_label, idx+self.window_size-1, self.csv[idx+self.window_size-1][0]

