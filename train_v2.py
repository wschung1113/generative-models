import pandas as pd
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

df_orig = pd.read_csv('train.csv')

data_size = df_orig.shape[0]
df = df_orig.iloc[:data_size, ]

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

# SMILES 정수 인코딩
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
print(x_one_hot[0].shape)

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(train_y)

print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))

dataset = TensorDataset(X, Y)









# 모델 구현
input_size = vocab_size # 각 시점 입력의 크기 = 원-핫 벡터의 길이
hidden_size = vocab_size # hidden_size 역시 원-핫 벡터의 길이
output_size = vocab_size # 출력 사이즈 역시 vocab_size
learning_rate = 0.1

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers): # 현재 hidden_size는 dic_size와 같음.
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        dict = []
        x, tmp = self.rnn(x)
        dict.append(tmp)
        x, tmp = self.fc(x)
        dict.append(tmp)
        return x, dict

net = Net(input_dim = vocab_size, hidden_dim = hidden_size, layers = 1)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

epoch = 1
for i in range(epoch):
    
    # data_size//batch_size + 1

    

    optimizer.zero_grad() # initialize gradients for recalculation
    outputs = net(X) # (data_size, max_len + 1, vocab_size) 크기를 가진 텐서를 매 에포크마다 모델의 입력으로 사용
    loss = criterion(outputs.view(-1, vocab_size), Y.view(-1)) # compute loss
    loss.backward() # backprop gradient computation
    optimizer.step() # 업데이트

    # results의 텐서 크기는 (data_size, max_len + 1)
    results = outputs.argmax(dim=2)
    predict_str = ""
    for j, result in enumerate(results):
        if j == 0: # 처음에는 예측 결과를 전부 가져오지만
            predict_str += ''.join([char_vocab[t] for t in result])
        else: # 그 다음에는 마지막 글자만 반복 추가
            predict_str += char_vocab[result[-1]]

    print(predict_str[:20])

