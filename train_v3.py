import pandas as pd
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
from torch.nn.utils.rnn import pad_sequence # 자동패딩해주는 함수

dtype = torch.float
# device = torch.device('cpu')
device = torch.device('cuda:0')

df = pd.read_csv('train.csv')
n = df.shape[0]

lines = []
for i in range(n):
    line = str(df.iloc[i, ][0]) # SMILES as string
    if len(line) > 0:
        lines.append(line)

max_len = len(lines[0])
for i in range(1, len(lines)):
    max_len = max(max_len, len(lines[i]))
print(max_len) # 가장 긴 SMILES 길이 # 나중에 샘플할 때 stopping threshold로 쓰임 (이거 혹은 eos)

text = ' '.join(lines)
print('문자열의 길이 또는 총 글자의 개수: %d' % len(text))
tmp = list(set(text))
tmp.append('<') # sos token
tmp.append('>') # eos token
char_vocab = sorted(tmp)
vocab_size = len(char_vocab)
print ('글자 집합의 크기 : {}'.format(vocab_size))
print(char_vocab)

char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 글자에 고유한 정수 인덱스 부여
print(char_to_index)

index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value] = key

# SMILES 정수 인코딩 & 패딩
train_X = []
train_y = []
for i in range(n): 
    X_sample = lines[i][:-1]
    X_encoded = [char_to_index['<']] + [char_to_index[c] for c in X_sample] # 정수 인코딩 # SoS 토큰 부착
    X_encoded = X_encoded + [char_to_index[' ']]*(max_len - len(X_sample) - 1) + [char_to_index['>']] # EoS 토큰 부착 # max_len에 모자란만큼 ' ' 부착 # 모든 정수 인코딩 길이 맞추기 위해
    train_X.append(X_encoded)

    y_sample = lines[i][1:]
    y_encoded = [char_to_index['<']] + [char_to_index[c] for c in y_sample] # 정수 인코딩 # SoS 토큰 부착
    y_encoded = y_encoded + [char_to_index[' ']]*(max_len - len(y_sample) - 1) + [char_to_index['>']] # EoS 토큰 부착 # max_len에 모자란만큼 ' ' 부착 # 모든 정수 인코딩 길이 맞추기 위해
    train_y.append(y_encoded)

x_one_hot = [np.eye(vocab_size)[x] for x in train_X] # X 데이터만 원-핫 인코딩

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(train_y)

print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))

tensor_dataset = TensorDataset(X, Y)

torch.save(dataset, 'tensor_dataset.pt')






input_size = vocab_size # 각 시점 입력의 크기 = 원-핫 벡터의 길이
hidden_size = vocab_size # hidden_size 역시 원-핫 벡터의 길이
output_size = vocab_size # 출력 사이즈 역시 vocab_size
learning_rate = 0.01

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers): # 현재 hidden_size는 dic_size와 같음.
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

net = Net(input_dim = vocab_size, hidden_dim = hidden_size, layers = 1)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

nb_epochs = 1
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        X, Y = samples
        
        outputs = net(X)

        optimizer.zero_grad() # initialize gradients for recalculation
        loss = criterion(outputs.view(-1, vocab_size), Y.view(-1)) # compute loss
        loss.backward() # backprop gradient computation
        optimizer.step() # 업데이트
        if batch_idx % 1000 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader),
                loss.item()
                ))

# check-point
torch.save(net.state_dict(), 'parameters.pt')
torch.save(char_to_index, 'char_to_index.pt')
torch.save(index_to_char, 'index_to_char.pt')
torch.save(max_len, 'max_len.pt')
torch.save(lines, 'lines.pt')
torch.save(dataset, 'dataset.pt')

# checking parameters compatibility
# m_state_dict = torch.load('parameters.pt')
# new_m = Net(input_dim = vocab_size, hidden_dim = hidden_size, layers = 1)
# new_m.load_state_dict(m_state_dict)