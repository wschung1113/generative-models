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

    def forward(self, x, lengths, h, c, cell): # 'rnn' 'lstm' 'gru'
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted = False)

        if cell == 'lstm':
            x, (hn, cn) = self.recurrent(x, (h, c))
        else: # if cell == 'rnn' or 'gru'
            x, hn = self.recurrent(x, h)            
        
        x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first = True, total_length = torch.max(lengths))
        x = self.fc(x[0])

        if cell == 'lstm':
            return x, (hn, cn)
        else: # if cell == 'rnn' or 'gru'
            return x, hn