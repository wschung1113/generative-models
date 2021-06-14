from textwrap import dedent
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

# from pytorchtools import EarlyStopping

dtype = torch.float
device = torch.device('cuda:0')

char_to_index = torch.load('data_specs.pt')[0]
index_to_char = torch.load('data_specs.pt')[1]
max_len = torch.load('data_specs.pt')[2]
lines = torch.load('data_specs.pt')[3]
lines_encoded = torch.load('data_specs.pt')[4]

class CustomDataset(Dataset): 
    def __init__(self):
        # self.x_data = train_X
        # self.y_data = train_y
        self.data = lines_encoded

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        # x = torch.cuda.FloatTensor(self.x_data[idx])
        # y = torch.cuda.FloatTensor(self.y_data[idx])
        # x = self.x_data[idx]
        # y = self.y_data[idx]
        l = torch.LongTensor(self.data[idx])
        # l = self.data[idx]
        return l, len(l) - 1

dataset = torch.load('dataset.pt')

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
input_size = vocab_size # 각 시점 입력의 크기 = 원-핫 벡터의 길이
learning_rate = 0.0001 # 0.01로 했을 때 convergence_loss = 0.52, 0.001은 0.046, 0.0005
recurrent_cell = 'gru'
is_bidirectional = False
batch_size = 128 # Segler et al. 128
num_layers = 3 # Segler et al. 3
hidden_size = 1024 # Segler et al. 1024
drop_out = 0.2 # Segler et al. 0.2
# optimizer = 'optim.Adam()'
# max_len = 64 # Segler et al.

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, drop_out, is_bidirectional, cell):
        super(Net, self).__init__()
        if cell == 'rnn':
            self.recurrent = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, dropout = drop_out, bidirectional = is_bidirectional, batch_first=True)
        if cell == 'lstm':
            self.recurrent = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, dropout = drop_out, bidirectional = is_bidirectional, batch_first=True)
        if cell == 'gru':
            self.recurrent = torch.nn.GRU(input_dim, hidden_dim, num_layers=layers, dropout = drop_out, bidirectional = is_bidirectional, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim*2 if is_bidirectional else hidden_dim, input_dim)

    def forward(self, x, lengths, h, c):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted = False)

        if h is None:
            x, _status = self.recurrent(x)
        else:
            if c is None:
                x, hn = self.recurrent(x, h)
            else:
                x, (hn, cn) = self.recurrent(x, (h, c))
            
        
        x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first = True, total_length = torch.max(lengths))
        x = self.fc(x[0])
        if h is None:
            return x
        else:
            if c is None:
                return x, hn
            else:
                return x, (hn, cn)
            

net = Net(input_dim = input_size, hidden_dim = hidden_size, layers = num_layers, drop_out = drop_out, is_bidirectional = is_bidirectional, cell = recurrent_cell).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), learning_rate)

loss_per_iter = []
avg_loss_per_epoch = []
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn) # 자동배치 ON
# batch = next(iter(dataloader)) 

# 추가로 훈련시키고 싶으면 여기부터 추가 epoch 정해서 실행
nb_epochs = 100

start_time = time.time()
# for epoch in tqdm(range(nb_epochs + 1), desc = 'epoch'):
    # for batch_idx, samples in tqdm(enumerate(dataloader), desc = 'batch'):
for epoch in range(nb_epochs + 1):
    loss_tmp = []
    for batch_idx, samples in enumerate(dataloader):
        # time.sleep(.01)
        X, Y, lengths = samples[0].to(device), samples[1].to(device), samples[2].to(device)
        # print(X)
        outputs = net(X, h = None, c = None, lengths = lengths)
        # outputs = net(X, lengths)
        optimizer.zero_grad() # initialize gradients for recalculation
        loss = criterion(outputs.view(-1, vocab_size), Y.view(-1)) # compute loss
        loss.backward() # backprop gradient computation
        optimizer.step() # 업데이트

        loss_per_iter.append(loss.item())
        loss_tmp.append(loss.item())

        if batch_idx % 1000 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Loss: {:.6f} Running Loss: {:.6f}'.format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader),
                loss.item(), np.mean(loss_per_iter)
                ))
        
    avg_loss_per_epoch.append(np.mean(loss_tmp))

    # 원하면 여기에 validation data로 loss 구해서 early stopping 구현 가능
print("--- %s seconds ---" % (time.time() - start_time))

# check-point
# torch.save(net.state_dict(), 'parameters.pt')
# torch.save([net, num_layers, is_bidirectional, hidden_size, loss_per_iter, avg_loss_per_epoch], 'rec_mod.pt')
# torch.save([net, num_layers, is_bidirectional, hidden_size, loss_per_iter, avg_loss_per_epoch], 'rec_mod_2.pt')
torch.save([net, num_layers, is_bidirectional, hidden_size, loss_per_iter, avg_loss_per_epoch], 'gru.pt')

# checking parameters compatibility
# m_state_dict = torch.load('parameters.pt')
# new_m = Net(input_dim = vocab_size, hidden_dim = hidden_size, layers = 1)
# new_m.load_state_dict(m_state_dict)