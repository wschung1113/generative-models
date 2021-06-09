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

nb_epochs = 5

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
        if batch_idx % 1000 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader),
                loss.item()
                ))
        loss_per_iter.append(loss.item())
        loss_tmp.append(loss.item())
    avg_loss_per_epoch.append(np.mean(loss_tmp))
print("--- %s seconds ---" % (time.time() - start_time))

# check-point
# torch.save(net.state_dict(), 'parameters.pt')
torch.save([net, num_layers, is_bidirectional, hidden_size, loss_per_iter, avg_loss_per_epoch], 'rec_mod.pt')