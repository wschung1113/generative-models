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
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence # 자동패딩해주는 함수

device = torch.device('cuda:0')

# rnn_model = torch.load('rnn_model.pt')
lstm_model = torch.load('lstm_model.pt')

char_to_index = torch.load('char_to_index.pt')
index_to_char = torch.load('index_to_char.pt')
max_len = torch.load('max_len.pt')

def sample(model, char_to_ix, ix_to_char):
    vocab_size = len(char_to_ix)
    # iter 0
    x = torch.zeros((vocab_size, 1))
    x[char_to_index['<']] = 1
    x = x.reshape((-1, 1, vocab_size)).to(device)
    indices = []
    counter = 1
    eos_character = char_to_ix['>']
    outputs = lstm_model(x, torch.as_tensor([counter], dtype = torch.int64, device = 'cpu'))
    idx = outputs.cpu().data.numpy().argmax()
    indices.append(idx)
    while (idx != eos_character and counter != max_len):
        counter += 1
        x = [torch.eye(np.max(indices) + 1)[x] for x in indices]
        x = torch.stack(x).to(device)
        x = x.unsqueeze(0)
        outputs = model(x, torch.as_tensor([counter], dtype = torch.int64, device = 'cpu'))
        idx = outputs.cpu().data.numpy().argmax(axis=2)
        print(idx)
        indices.append(np.squeeze(idx)[len(idx)])
    return indices

sampl = sample(lstm_model, char_to_index, index_to_char)
result_str = ''.join([index_to_char[c] for c in sampl])
print(result_str)


















# def sample(model, char_to_ix):
#     vocab_size = len(char_to_ix)
#     x = torch.zeros((vocab_size, 1))
#     x[char_to_index['<']] = 1
#     x = x.reshape((-1, 1, vocab_size)).to(device)
#     indices = []
#     idx = -1 
#     counter = 0
#     eos_character = char_to_ix['>']
#     while (idx != eos_character and counter != max_len):
#         outputs = model(x, torch.as_tensor([len(x)], dtype = torch.int64, device = 'cpu'))
#         # print(outputs)
#         idx = outputs.cpu().data.numpy().argmax()
#         indices.append(idx)
#         x = torch.zeros((vocab_size, 1))
#         x[idx] = 1
#         x = x.reshape((-1, 1, vocab_size)).to(device)
#         counter += 1
#     # if (counter == max_len):
#     #     indices.append(char_to_ix['>'])
#     # indices = [char_to_index['<']] + indices
#     return indices