import pandas as pd
import numpy as np
import random
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset # 커스텀데이터셋
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
from torch.nn.utils.rnn import pad_sequence # 자동패딩해주는 함수

dtype = torch.float
# device = torch.device('cpu')
device = torch.device('cuda')

df = pd.read_csv('train.csv')
# df = df[:10000]
n = df.shape[0]

lines = []
for i in range(n):
    line = str(df.iloc[i, ][0]) # SMILES as string
    if len(line) > 0:
        line = '<' + line + '>' # cat sos and eos token
        lines.append(line)

def find_max_len(l):
    max_len = len(l[0])
    for i in range(1, len(l)):
        max_len = max(max_len, len(l[i]))
    return max_len

max_len = find_max_len(lines) # 가장 긴 SMILES 길이 # 나중에 샘플할 때 stopping threshold로 쓰임 (이거 혹은 eos)

text = ' '.join(lines)
print('문자열의 길이 또는 총 글자의 개수: %d' % len(text))

char_vocab = sorted(list(set(text)))
vocab_size = len(char_vocab)
print ('글자 집합의 크기 : {}'.format(vocab_size))
print(char_vocab)

char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 글자에 고유한 정수 인덱스 부여
print(char_to_index)

index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value] = key

# SMILES 정수 인코딩
lines_encoded = []
# train_y = []
for i in range(n): 
    # X_sample = lines[i][:-1]
    # X_encoded = [char_to_index['<']] + [char_to_index[c] for c in X_sample]
    line = [char_to_index[c] for c in lines[i]]
    lines_encoded.append(line)

    # y_sample = lines[i][1:]
    # y_encoded = [char_to_index[c] for c in y_sample] + [char_to_index['>']]
    # train_y.append(y_encoded)

# # TensorDataset
# # SMILES 정수 인코딩 & 패딩
# train_X = []
# train_y = []
# for i in range(n):
#     X_sample = lines[i]
#     X_encoded = [char_to_index['<']] + [char_to_index[c] for c in X_sample] + [char_to_index['>']]
#     X_encoded = X_encoded + [char_to_index[' ']]*(max_len - len(X_sample))

#     train_X.append(X_encoded[:-1])
#     train_y.append(X_encoded[1:])
# x_one_hot = [np.eye(vocab_size)[x] for x in train_X] # X 데이터만 원-핫 인코딩
# X = torch.FloatTensor(x_one_hot)
# Y = torch.LongTensor(train_y)
# tensor_dataset = TensorDataset(X, Y)

# CustomDataset
# Dataset 상속
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

dataset = CustomDataset()

# check-point
torch.save([char_to_index, index_to_char, max_len, lines, lines_encoded], 'data_specs.pt')

# torch.save(dataset, 'tensor_dataset.pt')
torch.save(dataset, 'dataset.pt')

# x_one_hot = [np.eye(vocab_size)[x] for x in train_X] # X 데이터만 원-핫 인코딩
# # print(x_one_hot[0].shape)

# start_time = time.time()
# X = torch.FloatTensor(x_one_hot)
# print("--- %s seconds ---" % (time.time() - start_time))

# # start_time = time.time()
# # X = torch.cuda.FloatTensor(x_one_hot)
# # print("--- %s seconds ---" % (time.time() - start_time))

# Y = torch.LongTensor(train_y)

# print('훈련 데이터의 크기 : {}'.format(X.shape))
# print('레이블의 크기 : {}'.format(Y.shape))

# dataset = TensorDataset(X, Y)