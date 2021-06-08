import pandas as pd
import numpy as np
import random
import pickle
import time

from tqdm import tqdm
# from tqdm import tqdm_notebook, tnrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
from torch.nn.utils.rnn import pad_sequence # 자동패딩해주는 함수

from pytorchtools import EarlyStopping

dtype = torch.float
# device = torch.device('cpu')
device = torch.device('cuda:0')

char_to_index = torch.load('char_to_index.pt')
index_to_char = torch.load('index_to_char.pt')
max_len = torch.load('max_len.pt')
lines = torch.load('lines.pt')
lines_encoded = torch.load('lines_encoded.pt')
find_max_len = torch.load('find_max_len.pt')

dataset = torch.load('dataset.pt')
# tensor_dataset = torch.load('tensor_dataset.pt')

# dataloader
def collate_fn(data):
    b, lengths = zip(*data)
    max_len = torch.tensor(np.max(lengths))
    lengths = torch.tensor(lengths)
    # print(lengths)
    b = torch.nn.utils.rnn.pad_sequence(b, batch_first = True, padding_value = 0)
    padded_X = []
    padded_Y = []
    for i in range(len(b)):
        padded_X.append(b[i][:-1])
        padded_Y.append(b[i][1:])
    padded_X_onehot = [torch.eye(input_size)[x] for x in padded_X]
    tensor_X = torch.stack(padded_X_onehot)
    tensor_Y = torch.stack(padded_Y)
    return tensor_X, tensor_Y, lengths.long(), max_len.long()

# dataloader = DataLoader(dataset, batch_size = 512, shuffle = True, collate_fn = collate_fn) # 자동배치 ON 
# dataloader = DataLoader(dataset, batch_size = None) # 자동배치 OFF # batch_size 와 batch_sample가 모두 None일 때 자동배치가 해제됩니다
# batch = next(iter(dataloader))

# hyperparameters
vocab_size = len(char_to_index)
input_size = len(char_to_index) # 각 시점 입력의 크기 = 원-핫 벡터의 길이
hidden_size = len(char_to_index) # hidden_size 역시 원-핫 벡터의 길이
output_size = len(char_to_index) # 출력 사이즈 역시 vocab_size

# RNN
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers): # 현재 hidden_size는 dic_size와 같음.
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x, lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted = False)
        x, _status = self.rnn(x)
        x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first = True, total_length = torch.max(lengths))
        x = self.fc(x[0])
        return x
net_rnn = Net(input_dim = len(char_to_index), hidden_dim = hidden_size, layers = 2).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net_rnn.parameters(), learning_rate)
nb_epochs = 0
loss_rnn = []
dataloader = DataLoader(dataset, batch_size = 512, shuffle = True, collate_fn = collate_fn) # 자동배치 ON 
learning_rate = 0.001 # 0.01로 했을 때 convergence_loss = 0.52, 0.001은 0.046, 0.0005

start_time = time.time()
for epoch in tqdm(range(nb_epochs + 1), desc = 'epoch'):
    for batch_idx, samples in tqdm(enumerate(dataloader), desc = 'batch'):
        time.sleep(.01)
        # print(batch_idx)
        # print(samples)
        # print(samples)
        X, Y, lengths = samples[0].to(device), samples[1].to(device), samples[2].to(device)
        # X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first = True, enforce_sorted = False)
        outputs = net_lstm(X, lengths)
        optimizer.zero_grad() # initialize gradients for recalculation
        loss = criterion(outputs.view(-1, vocab_size), Y.view(-1)) # compute loss
        loss.backward() # backprop gradient computation
        optimizer.step() # 업데이트
        if batch_idx % 5000 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader),
                loss.item()
                ))
        loss_rnn.append(loss.item())
print("--- %s seconds ---" % (time.time() - start_time))

# LSTM
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x, lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted = False)
        x, _status = self.lstm(x)
        x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first = True, total_length = torch.max(lengths))
        x = self.fc(x[0])
        return x
net_lstm = Net(input_dim = len(char_to_index), hidden_dim = hidden_size, layers = 3).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net_lstm.parameters(), learning_rate)
nb_epochs = 10
loss_lstm = []
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, collate_fn = collate_fn) # 자동배치 ON
batch = next(iter(dataloader)) 
learning_rate = 0.0001 # 0.01로 했을 때 convergence_loss = 0.52, 0.001은 0.046, 0.0005

start_time = time.time()
# for epoch in tqdm(range(nb_epochs + 1), desc = 'epoch'):
    # for batch_idx, samples in tqdm(enumerate(dataloader), desc = 'batch'):
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        time.sleep(.01)
        # print(batch_idx)
        # print(samples)
        # print(samples)
        X, Y, lengths = samples[0].to(device), samples[1].to(device), samples[2].to(device)
        # X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first = True, enforce_sorted = False)
        outputs = net_lstm(X, lengths)
        optimizer.zero_grad() # initialize gradients for recalculation
        loss = criterion(outputs.view(-1, vocab_size), Y.view(-1)) # compute loss
        loss.backward() # backprop gradient computation
        optimizer.step() # 업데이트
        if batch_idx % 1000 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader),
                loss.item()
                ))
        loss_lstm.append(loss.item())
print("--- %s seconds ---" % (time.time() - start_time))

# check-point
# torch.save(net.state_dict(), 'parameters.pt')
torch.save(net_rnn, 'rnn_model.pt')
torch.save(loss_rnn, 'loss_rnn.pt')
# torch.save(net_gru, 'gru_model.pt')
torch.save(net_lstm, 'lstm_model.pt')
torch.save(loss_lstm, 'loss_lstm.pt')

# checking parameters compatibility
# m_state_dict = torch.load('parameters.pt')
# new_m = Net(input_dim = vocab_size, hidden_dim = hidden_size, layers = 1)
# new_m.load_state_dict(m_state_dict)