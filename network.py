import numpy as np
import datetime
import csv
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as d_utils
import torchvision
import torchvision.transforms as transforms

from cus_dataloader import CustomDataset

BASE_DIR = '/workspace/dataset'
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--window', type=int, default=64, help='input batch size')
parser.add_argument('--hidden', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--num_layer', type=int, default=1, help='number of layer')
parser.add_argument('--load', type=int, default=-1, help='Determines whether to load from previous model or not. -1 : Not to import')

opt = parser.parse_args()

model_dir = 'norm_%d_%d_%d_%d/' %(opt.epoch, opt.window, opt.hidden, opt.num_layer)

def str_to_datetime(str):
        return datetime.datetime.strptime(str, "%Y-%m-%d %H:%M:%S")

def normalize_timestamp(data):
        last_time = str_to_datetime(data[-1][0])
        first_time = str_to_datetime(data[0][0])
        time_window = (last_time - first_time).total_seconds()
        new_data = [[0.0, 0.0] for i in range(len(data))]
        for i in range(len(data)):
                new_data[i][0]=float((last_time - str_to_datetime(data[i][0])).total_seconds()) / time_window      # Normalize timestamp inside window
                new_data[i][1]=float(data[i][1])

        return new_data

transform = transforms.Compose([
                        transforms.Lambda(lambda x : normalize_timestamp(x)),
                        transforms.Lambda(lambda x : torch.FloatTensor(x)),
                        ])

train_set = CustomDataset(data_dir = BASE_DIR, data_file = 'train_art.csv', window_size = opt.window, transform = transform)

test_set = CustomDataset(data_dir = BASE_DIR, data_file = 'test_art.csv', window_size = opt.window, transform = transform)

train_loader = d_utils.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)

test_loader = d_utils.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=2)

class BASE_RNN(nn.Module):

        def __init__(self, hidden_size, num_layer, num_feature, rnn_fn='LSTM'):
                super(BASE_RNN, self).__init__()

                if rnn_fn == 'RNN':
                        self.rnn_fn = nn.RNN
                elif rnn_fn == 'LSTM':
                        self.rnn_fn = nn.LSTM
                else :  
                        self.rnn_fn = nn.GRU

                self.RNN = self.rnn_fn(
                        input_size = num_feature,
                        hidden_size = hidden_size,
                        num_layers = num_layer,
                        batch_first = True,
                )
                self.FC = nn.Linear(hidden_size, 1)

        def forward(self, input_data):
                output, hn = self.RNN(input_data)
                return self.FC(output[:, -1, :])

criterion = nn.MSELoss()
RNN = BASE_RNN(hidden_size = opt.hidden, num_layer = opt.num_layer, num_feature = 2)
optimizer = optim.Adam(RNN.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(opt.epoch):
        print("Epoch : %d" %(epoch+1))
        train_result = {}
        test_result = {}
        for i, _data in enumerate(train_loader, 0):
                RNN.zero_grad()
                data, label, idx, time = _data
                idx = idx.numpy().tolist()                              # For saving results
                lab = label.numpy().tolist()                            # For saving results
                data = Variable(data)
                label = Variable(label.float())
                output = RNN(data).squeeze()
                out = output.data.numpy().tolist()                      # For saving results
                error = criterion(output, label)
                error.backward()
                optimizer.step()
                for j in range(len(idx)):
                        train_result[idx[j]] = [time[j], out[j], lab[j]]
        for i, _data in enumerate(test_loader, 0):
                RNN.zero_grad()
                data, label, idx, time= _data
                idx = idx.numpy().tolist()                              # For saving results
                lab = label.numpy().tolist()                            # For saving results
                data = Variable(torch.FloatTensor(data))
                label = Variable(label.float())
                output = RNN(data).squeeze()
                out = output.data.numpy().tolist()                      # For saving results
                error = criterion(output, label)
                for j in range(len(idx)):
                        test_result[idx[j]] = [time[j], out[j], lab[j]]

        output_dst = '/workspace/results/'
        try:
                os.makedirs(os.path.join(output_dst, model_dir))
        except OSError:
                pass
        f=open(output_dst+model_dir+'train_%d.csv' %(epoch+1), 'w')
        writer = csv.writer(f)
        for key in sorted(train_result):
                writer.writerow([key]+train_result[key])

        f=open(output_dst+model_dir+'test_%d.csv' %(epoch+1), 'w')
        writer = csv.writer(f)
        for key in sorted(test_result):
                writer.writerow([key]+test_result[key])

        try:
                os.makedirs(os.path.join('/workspace/model/', model_dir))
        except OSError:
                pass
        model_name = '/workspace/model/'+model_dir+'model_%d.pth' %(epoch+1)
        torch.save(RNN.state_dict(), model_name)

