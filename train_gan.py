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

# from pytorch GAN tutorial
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# from pytorchtools import EarlyStopping

device = torch.device('cuda')