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
device = torch.device('cuda')

df = pd.read_csv('train.csv')
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

max_len = find_max_len(lines) # 가장 긴 SMILES 길이 # 나중에 샘플할 때 stopping threshold로 쓰임 (이거 혹은 eos 토큰 생성 시)

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
for i in range(n):
    line = [char_to_index[c] for c in lines[i]]
    lines_encoded.append(line)

# CustomDataset
# Dataset 상속
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

dataset = CustomDataset()

# check-point
torch.save([char_to_index, index_to_char, max_len, lines, lines_encoded], 'data_specs.pt')
torch.save(dataset, 'dataset.pt')