import pandas as pd
import numpy as np
import random
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence # 자동패딩해주는 함수

from model import Net

device = torch.device('cuda:0')

# 'rnn' 'lstm' 'gru'
mod = torch.load('rnn.pt')
recurrent_cell = 'rnn'

model = mod['model']
num_layers = mod['num_hidden_layers']
hidden_size = mod['hidden_layer_dim']
avg_loss_per_epoch = mod['avg_loss_per_epoch']

char_to_index = torch.load('data_specs.pt')[0]
index_to_char = torch.load('data_specs.pt')[1]
max_len = torch.load('data_specs.pt')[2]
lines = max_len = torch.load('data_specs.pt')[3]

vocab_size = len(char_to_index)

# sample function
def sample(model, cell, char_to_ix, start_token, n):
    out = []
    for i in range(n):
        # create start-token one-hot vector
        x = torch.zeros((vocab_size, 1))
        x[char_to_index[start_token]] = 1
        x = x.reshape(1, 1, vocab_size).to(device)

        eos_idx = char_to_ix['>']
        indices = []
        counter = 1

        if cell == 'lstm':
            h0 = torch.zeros(num_layers, 1, hidden_size).to(device) # create initial hidden and cell state
            c0 = torch.zeros(num_layers, 1, hidden_size).to(device)
            net.eval()
            output, (hn, cn) = model(x = x, lengths = torch.as_tensor([1], dtype = torch.int64, device = 'cpu'), h = h0, c = c0, cell = cell)
            output = F.softmax(output.view(-1), dim = 0)
            idx = torch.multinomial(output, 1).item()
            indices.append(idx)

            x = torch.zeros((vocab_size, 1))
            x[idx] = 1 # output as next input
            x = x.reshape(1, 1, vocab_size).to(device)

            while (idx != eos_idx and counter != max_len):
                output, (hn, cn) = model(x, lengths = torch.as_tensor([1], dtype = torch.int64, device = 'cpu'), h = hn, c = cn, cell = cell)
                output = F.softmax(output.view(-1), dim = 0)
                idx = torch.multinomial(output, 1).item()
                indices.append(idx)

                x = torch.zeros((vocab_size, 1))
                x[idx] = 1
                x = x.reshape(1, 1, vocab_size).to(device)
                
                counter += 1

        else: # cell == 'rnn' or 'gru'
            h0 = torch.zeros(num_layers, 1, hidden_size).to(device) # create initial hidden and cell state
            net.eval()
            output, hn = model(x = x, lengths = torch.as_tensor([1], dtype = torch.int64, device = 'cpu'), h = h0, c = None, cell = cell)
            output = F.softmax(output.view(-1), dim = 0)
            idx = torch.multinomial(output, 1).item()
            indices.append(idx)

            x = torch.zeros((vocab_size, 1))
            x[idx] = 1 # output as next input
            x = x.reshape(1, 1, vocab_size).to(device)

            while (idx != eos_idx and counter != max_len):
                output, hn = model(x, lengths = torch.as_tensor([1], dtype = torch.int64, device = 'cpu'), h = hn, c = None, cell = cell)
                output = F.softmax(output.view(-1), dim = 0)
                idx = torch.multinomial(output, 1).item()
                indices.append(idx)

                x = torch.zeros((vocab_size, 1))
                x[idx] = 1
                x = x.reshape(1, 1, vocab_size).to(device)
                
                counter += 1
        
        indices = [char_to_index[start_token]] + indices
        result_str = ''.join([index_to_char[c] for c in indices])
        out.append(result_str)
    
    return out

sampl = sample(model = model, cell = recurrent_cell, char_to_ix = char_to_index, start_token = '<', n = 1)
for s in sampl:
    print(s)


for i in range(100):
    print(lines[random.randint(0, len(lines))])
    


















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