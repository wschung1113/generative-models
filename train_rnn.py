from textwrap import dedent
import pandas as pd
import numpy as np
import random
import pickle
import time
import argparse

from tqdm import tqdm
# from tqdm import tqdm_notebook, tnrange

import torch

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
from torch.utils.data.sampler import SubsetRandomSampler # Train-Val split 함수

from torch.nn.utils.rnn import pad_sequence # 자동패딩해주는 함수
import torch.nn as nn
import torch.nn.functional as F

# from model import Net # 이렇게 하면 torch.save()할 때 source code warning 나서 안 씀

# from pytorchtools import EarlyStopping
specs = torch.load('data_specs.pt')
char_to_index = specs[0]
index_to_char = specs[1]
max_len = specs[2]
lines = specs[3]
lines_encoded = specs[4]

# hyperparameters
device = torch.device('cuda:1')

vocab_size = len(char_to_index)
input_size = vocab_size # 각 시점 입력의 크기 = 원-핫 벡터의 길이
learning_rate = 0.001 # 0.01로 했을 때 convergence_loss = 0.52, 0.001은 0.046, 0.0005
recurrent_cell = 'lstm'
batch_size = 128 # Segler et al. 128
num_layers = 3 # Segler et al. 3
hidden_size = 1024 # Segler et al. 1024
drop_out = 0.2 # Segler et al. 0.2
# optimizer = 'optim.Adam()'
# max_len = 64 # Segler et al. 

parallel_train = False

# create data set; split into train and test set
class CustomDataset(Dataset): 
    def __init__(self):
        self.data = lines_encoded

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        t = torch.LongTensor(self.data[idx])
        return t, len(t) - 1
dataset = torch.load('dataset.pt')

valid_size = 0.2

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(valid_size*dataset_size))
# np.random.seed(1)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

def collate_fn(data):
    b, lengths = zip(*data)
    max_len = torch.tensor(np.max(lengths))
    lengths = torch.tensor(lengths)
    b = torch.nn.utils.rnn.pad_sequence(b, batch_first = True, padding_value = 0)
    padded_X = []
    padded_Y = []
    for i in range(len(b)):
        padded_X.append(b[i][:-1])
        padded_Y.append(b[i][1:])
    padded_X_onehot = [torch.eye(input_size)[x] for x in padded_X]
    tensor_X = torch.stack(padded_X_onehot)
    tensor_Y = torch.stack(padded_Y)
    return tensor_X, tensor_Y, lengths.long()

train_loader = DataLoader(
    dataset,
    batch_size=batch_size, collate_fn=collate_fn, sampler=train_sampler)

test_loader = DataLoader(
    dataset,
    batch_size=batch_size, collate_fn=collate_fn, sampler=valid_sampler)

# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) # 자동배치 ON

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, drop_out, cell):
        super(Net, self).__init__()
        if cell == 'rnn':
            self.recurrent = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, dropout=drop_out, batch_first=True)
        elif cell == 'lstm':
            self.recurrent = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, dropout=drop_out, batch_first=True)
        else: # cell == 'gru'
            self.recurrent = torch.nn.GRU(input_dim, hidden_dim, num_layers=layers, dropout=drop_out, batch_first=True)
        self.fc = torch.nn.Linear(in_features=hidden_dim, out_features=input_dim)

    def forward(self, x, lengths, h, c, cell, train_or_sample): # 'rnn' 'lstm' 'gru'
        total_length = x.shape[1]
        self.recurrent.flatten_parameters()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        if train_or_sample == 'train':
            x, _ = self.recurrent(x)
            x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=total_length)
            x = self.fc(x[0])
            return x
        else:
            if cell == 'lstm':
                x, (hn, cn) = self.recurrent(x, (h, c))
            else: # if cell == 'rnn' or 'gru'
                x, hn = self.recurrent(x, h) 
            x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=total_length)
            x = self.fc(x[0])
            if cell == 'lstm':
                return x, (hn, cn)
            else: # if cell == 'rnn' or 'gru'
                return x, hn
net = Net(input_size, hidden_size, num_layers, drop_out, recurrent_cell).to(device)
if parallel_train:
    net = nn.DataParallel(net, device_ids=[0, 1])
# DistributedDataParallel() is preferred for memory efficiency

# samples = next(iter(dataloader))
# X, Y, lengths = samples[0].to(device), samples[1].to(device), samples[2].to(device)
# lstm_cell = torch.nn.LSTM(29, 1024, num_layers=3, dropout=0.2, batch_first=True).to(device)
# X = lstm_cell(X)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), learning_rate)

train_loss = []
train_loss_per_epoch = []
valid_loss = []
valid_loss_per_epoch = []
valid_loss_min = 1000

nb_epochs = 20
es_patience = 2
idx_patience = 0

train_times = []
start_time = time.time()
train_times.append(start_time)

for epoch in range(nb_epochs + 1): # 0 ~ nb_epochs
    net.train()
    train_loss_tmp = []
    for batch_idx, samples in enumerate(train_loader):
        X, Y, lengths = samples[0].to(device), samples[1].to(device), samples[2].to(device)
        outputs = net(X, lengths, None, None, recurrent_cell, 'train')
        optimizer.zero_grad() # initialize gradients for recalculation
        loss = criterion(outputs.view(-1, vocab_size), Y.view(-1)) # compute loss
        loss.backward() # backprop gradient computation
        optimizer.step() # 업데이트
        
        train_loss.append(loss.item())
        train_loss_tmp.append(loss.item())

        if batch_idx % 1000 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Running Loss: {:.6f}'.format(
                epoch, nb_epochs, batch_idx + 1, len(train_loader),
                np.mean(train_loss)
                ))    
    train_loss_per_epoch.append(np.mean(train_loss_tmp))

    net.eval()
    valid_loss_tmp = []
    for batch_idx, samples in enumerate(test_loader):
        X, Y, lengths = samples[0].to(device), samples[1].to(device), samples[2].to(device)
        outputs = net(X, lengths, None, None, recurrent_cell, 'train')
        loss = criterion(outputs.view(-1, vocab_size), Y.view(-1)) # compute loss
        valid_loss.append(loss.item())
        valid_loss_tmp.append(loss.item())

    valid_loss_current = np.mean(valid_loss_tmp)
    valid_loss_per_epoch.append(valid_loss_current)

    # early stopping
    if epoch > 0:
        if valid_loss_min < valid_loss_current: # if val loss hasn't been improved
            idx_patience += 1
        else: # if val loss improved
            idx_patience = 0
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss_current:.6f})')
            valid_loss_min = valid_loss_per_epoch[epoch]

    if idx_patience == es_patience:
        print("Early Stopping")
        break

train_times.append(time.time() - start_time)

# check-point
torch.save({'model':net, 'num_hidden_layers':num_layers, 'hidden_layer_dim':hidden_size,
'train_loss':train_loss, 'valid_loss':valid_loss,
'train_loss_per_epoch':train_loss_per_epoch, 'valid_loss_per_epoch':valid_loss_per_epoch,
'training_time':train_times}, recurrent_cell + '.pt')
