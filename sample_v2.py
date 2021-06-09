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

mod_pt = 'rec_mod.pt'

model = torch.load(mod_pt)[0]
num_layers = torch.load(mod_pt)[1]
is_bidirectional = torch.load(mod_pt)[2]
hidden_size = torch.load(mod_pt)[3]

char_to_index = torch.load('data_specs.pt')[0]
index_to_char = torch.load('data_specs.pt')[1]
max_len = torch.load('data_specs.pt')[2]

vocab_size = len(char_to_index)

# sample function
def sample(model, char_to_ix):
    # create start-token one-hot vector
    x = torch.zeros((vocab_size, 1))
    x[char_to_index['<']] = 1
    x = x.reshape(1, vocab_size)
    x = x.unsqueeze(0).to(device)

    eos_idx = char_to_ix['>']
    indices = []
    counter = 1

    # create initial hidden and cell state
    h0 = torch.zeros(num_layers*(2 if is_bidirectional else 1), 1, hidden_size).to(device)
    c0 = torch.zeros(num_layers*(2 if is_bidirectional else 1), 1, hidden_size).to(device)

    output, (hn, cn) = model(x = x, lengths = torch.as_tensor([1], dtype = torch.int64, device = 'cpu'), h = h0, c = c0)
    idx = output.cpu().data.numpy().argmax()
    indices.append(idx)

    x = torch.zeros((vocab_size, 1))
    x[idx] = 1
    x = x.reshape(1, vocab_size)
    x = x.unsqueeze(0).to(device)

    while (idx != eos_idx and counter != max_len):
        output, (hn, cn) = model(x, lengths = torch.as_tensor([1], dtype = torch.int64, device = 'cpu'), h = hn, c = cn)
        idx = output.cpu().data.numpy().argmax()
        indices.append(idx)

        x = torch.zeros((vocab_size, 1))
        x[idx] = 1
        x = x.reshape(1, vocab_size)
        x = x.unsqueeze(0).to(device)

        counter += 1
    return indices

# sample sequence and print
sampl = sample(model, char_to_index)
# sampl = sample(model, char_to_index)
result_str = ''.join([index_to_char[c] for c in sampl])
print(result_str)


















# def sample(model, char_to_ix, ix_to_char):
#     vocab_size = len(char_to_ix)
#     # iter 0
#     x = torch.zeros((vocab_size, 1))
#     x[char_to_index['<']] = 1
#     x = x.reshape((-1, 1, vocab_size)).to(device)
#     indices = []
#     counter = 1
#     eos_character = char_to_ix['>']
#     outputs = lstm_model(x, torch.as_tensor([counter], dtype = torch.int64, device = 'cpu'))
#     idx = outputs.cpu().data.numpy().argmax()
#     indices.append(idx)
#     while (idx != eos_character and counter != max_len):
#         counter += 1
#         x = [torch.eye(np.max(indices) + 1)[x] for x in indices]
#         x = torch.stack(x).to(device)
#         x = x.unsqueeze(0)
#         outputs = model(x, torch.as_tensor([counter], dtype = torch.int64, device = 'cpu'))
#         idx = outputs.cpu().data.numpy().argmax(axis=2)
#         print(idx)
#         indices.append(np.squeeze(idx)[len(idx)])
#     return indices








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