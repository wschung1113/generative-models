from functools import total_ordering
from textwrap import dedent
import pandas as pd
import numpy as np
import random
import pickle
import time
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence # 자동패딩해주는 함수
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset

from __future__ import print_function
from torchvision import datasets, transforms
from torchvision.utils import save_image

from sklearn.model_selection import train_test_split

###############################################################

specs = torch.load('data_specs.pt')
char_to_index = specs[0]
index_to_char = specs[1]
max_len = specs[2]
lines = specs[3]
lines_encoded = specs[4]

vocab_size = len(char_to_index)

parser = argparse.ArgumentParser(description='VAE SMILES Example')
parser.add_argument('--parallel-training', type=bool, default=False, metavar='N',
                    help='Boolean of whether to train with multiple GPUs or not')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--learning-rate', type=float, default=.001, metavar='N',
                    help='how much increment per iteration')
parser.add_argument('--input-dim', type=int, default=vocab_size, metavar='N',
                    help='Depends on what embedding to use')
parser.add_argument('--validation-split', type=float, default=0.2, metavar='N',
                    help='Ratio of validation set')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)

device = torch.device("cuda:0" if args.cuda else "cpu")

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

dataset = torch.load('dataset.pt')

# split into train and test set
# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(args.validation_split*dataset_size))
# np.random.seed(args.seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def collate_fn(data):
    b, lengths = zip(*data)
    max_len = torch.tensor(np.max(lengths))
    lengths = torch.tensor(lengths)
    b = torch.nn.utils.rnn.pad_sequence(b, batch_first = True, padding_value = 0)
    padded_X = []
    padded_Y = []
    for i in range(len(b)):
        padded_X.append(b[i][:-1])
        padded_Y.append(b[i][1:])
    padded_X_onehot = [torch.eye(input_size)[x] for x in padded_X]
    tensor_X = torch.stack(padded_X_onehot)
    tensor_Y = torch.stack(padded_Y)
    return tensor_X, tensor_Y, lengths.long()

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size, **kwargs, collate_fn=collate_fn, sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size, **kwargs, collate_fn=collate_fn, sampler=valid_sampler)

sample_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1, **kwargs, collate_fn=collate_fn, shuffle=True)

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, input_dim):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE(input_dim=args.input_dim).to(device)
if args.parallel_training:
    model = nn.DataParallel(model, device_ids=[0, 1])
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, input_dim):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, args.input_dim)
        loss = loss_function(recon_batch, data, mu, logvar, args.input_dim)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data, args.input_dim)
            test_loss += loss_function(recon_batch, data, mu, logvar, args.input_dim).item()

            # # 뭐하는 부분인지 모르겠음
            # if i == 0: # 첫번째 dataloader iteration이면
            #     n = min(data.size(0), 8) # batch_size와 8중에 작은 숫자가 n
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              '../charrnn/data/results/sample_' + str(epoch) + '.png', nrow=n)
    
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # Early Stopping 적용

# SMILES generation with VAE
sample_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1, **kwargs, collate_fn=collate_fn, shuffle=True)
    
with torch.no_grad():
    for i in range(5):
        # sample = torch.randn(64, 20).to(device)
        s, _, _ = next(iter(sample_loader))
        s = s.to(device)
        sample, _, _ = model(s, args.input_dim)
        sample = sample.cpu()
        # sample = model.decode(sample).cpu()
        indices = torch.argmax(sample, 1).cpu().tolist()
        result_str = ''.join([index_to_char[c] for c in indices]) + '>'

        # retrieve original SMILES
        s_indices = torch.argmax(s[0], 1).cpu().tolist()
        orig_str = ''.join([index_to_char[c] for c in s_indices]) + '>'
        print('Original: ' + orig_str + '\nGenerated: ' + result_str + '\n')