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
from torch.nn.utils.rnn import pad_sequence # 자동패딩해주는 함수

params= torch.load('parameters.pt')
char_to_ix = torch.load('char_to_index.pt')
ix_to_char = torch.load('index_to_char.pt')
max_len = torch.load('max_len.pt')

params.keys() # keys of the parameter dictionary

Waa = np.array(params["rnn.weight_hh_l0"])
Wax = np.array(params['rnn.weight_ih_l0'])
Wya = np.array(params['fc.weight'])
by = np.array(params['fc.bias'])
b = np.array(params['rnn.bias_ih_l0']) + np.array(parameters['rnn.bias_hh_l0'])

# softmax function
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def sample(parameters, char_to_ix):

    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa = np.array(parameters["rnn.weight_hh_l0"])
    Wax = np.array(parameters['rnn.weight_ih_l0'])
    Wya = np.array(parameters['fc.weight'])
    by = np.array(parameters['fc.bias'])
    by = np.reshape(by, (29, 1))
    b = np.array(parameters['rnn.bias_ih_l0']) + np.array(parameters['rnn.bias_hh_l0'])
    b = np.reshape(b, (29, 1))
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    # Step 1: Create the one-hot vector x for the first character '<'
    x = np.zeros((vocab_size, 1))
    x[char_to_ix['<']] = 1

    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros((n_a, 1))
    
    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []
    
    # Idx is a flag to detect eos, we initialize it to -1
    idx = -1 
    
    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append 
    # its index to "indices". We'll stop if we reach max_len characters (which should be very unlikely with a well 
    # trained model), which helps debugging and prevents entering an infinite loop. 
    counter = 0
    eos_character = char_to_ix['>']
    
    while (idx != eos_character and counter != max_len):
        
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax,x)+ np.dot(Waa,a_prev) + b)
        z = np.dot(Wya,a)+by
        y = softmax(z)
        
        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        
        idx = np.random.choice(range(vocab_size), p = y.ravel()) 
        # <Gema> first parameter: If an ndarray, a random sample is generated from its elements. 
        # <Gema> range(vocab_size) will give a normal python list - [ 0 1 2 3 ..... (vocab_size-1)]
        # <Gema> second parameter: The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.
        # <Gema> numpy.ndarray.ravel() Return a flattened array. If x = np.array([[1, 2, 3], [4, 5, 6]]), np.ravel(x)=[1 2 3 4 5 6]
        # <Gema> y.ravel() just readjust m by n matrix to mn by 1 array.

        # Append the index to "indices"
        indices.append(idx)
        
        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        # <Gema> Look at https://gist.github.com/karpathy/d4dee566867f8291f086
        
        # Update "a_prev" to be "a"
        a_prev = a
        
        counter +=1

    if (counter == max_len):
        indices.append(char_to_ix['>'])
    
    return indices

sampl = sample(params, char_to_ix)
result_str = ''.join([index_to_char[c] for c in sampl])
print(result_str)