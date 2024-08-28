import json
from math import e
import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from torch import tensor
from torch.distributions import Bernoulli, RelaxedBernoulli
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
from tonic import MemoryCachedDataset, DiskCachedDataset
import snntorch as snn
import pickle


dtype = torch.float

'''
HyperParameters For Net:
    General:
        - Device: Device to run the model on
    RelBerLeaky:
        - Beta: Decay rate of membrane potential
        - Threshold: Threshold for the membrane potential
        - Temp: Temperature for the RelaxedBernoulli distribution
    Net:
        - num_steps: Number of time steps to run the model for
        - lr: Learning rate for the optimizer
        - loss: Loss function to use
        - num_epochs: Number of epochs to train for
        - train_loader: Data loader for the training data
        - test_loader: Data loader for the test data

'''

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
    torch.backends.cudnn.benchmark = False     # disables the inbuilt cudnn auto-tuner which can introduce randomness

set_seed(42)

import json
from math import e
import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from torch import tensor
from torch.distributions import Bernoulli, RelaxedBernoulli
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
from tonic import MemoryCachedDataset, DiskCachedDataset
import snntorch as snn
import pickle

dtype = torch.float

'''
HyperParameters For Net:
    General:
        - Device: Device to run the model on
    RelBerLeaky:
        - Beta: Decay rate of membrane potential
        - Threshold: Threshold for the membrane potential
        - Temp: Temperature for the RelaxedBernoulli distribution
    Net:
        - num_steps: Number of time steps to run the model for
        - lr: Learning rate for the optimizer
        - loss: Loss function to use
        - num_epochs: Number of epochs to train for
        - train_loader: Data loader for the training data
        - test_loader: Data loader for the test data

'''

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
    torch.backends.cudnn.benchmark = False     # disables the inbuilt cudnn auto-tuner which can introduce randomness

set_seed(42)

surrogate_value = 1

from snntorch import surrogate

def custom_surrogate_rectangle(input_, grad_input, spikes):
    ## The hyperparameter slope is defined inside the function.
    a = torch.tensor(surrogate_value)
    grad = grad_input*(1/torch.sqrt(2*torch.pi*a))*torch.exp(-torch.square(input_)/(2*a))
    return grad

spike_grad = surrogate.custom_surrogate(custom_surrogate_rectangle)
        

class Net(nn.Module):
    ### Convnet with 64C7 - P2 - 128C7 - P2 - 128C7 - P2 - (dense to 11 classes)
    def __init__(self, beta, device, temp, threshold):
        super().__init__()

        self.beta = beta
        self.temp = temp
        self.device = device

        ## Initialize layers
        ## Input like (2, 32, 32)
        self.Conv_1 = nn.Conv2d(2, 64, 7, bias=False, padding = 2, device=device)
        self.pool_1 = nn.AvgPool2d(2)
        self.LiF_1 = snn.Leaky(self.beta, threshold=threshold, spike_grad = spike_grad)

        self.Conv_2 = nn.Conv2d(64, 128, 7, bias=False, padding = 2, device=device)
        self.pool_2 = nn.AvgPool2d(2)
        self.LiF_2 = snn.Leaky(self.beta, threshold=threshold, spike_grad = spike_grad)

        self.Conv_3 = nn.Conv2d(128, 128, 7, bias=False, padding = 2, device=device)
        self.pool_3 = nn.AvgPool2d(2)
        self.LiF_3 = snn.Leaky(self.beta, threshold=threshold, spike_grad = spike_grad)

        self.Flatten = nn.Flatten()
        self.FC = nn.Linear(5*5*128, 11, bias = False, device = device)
        self.LiF_out = snn.Leaky(self.beta, threshold=threshold, spike_grad = spike_grad)

    def forward(self, input): ## Note here input is [Time, Batch, Shape]
        num_steps = input.shape[0]
        batch_size = input.shape[1]

        ## initialize hidden states (membranes)
        mem1 = self.LiF_1.init_leaky()
        mem2 = self.LiF_2.init_leaky()
        mem3 = self.LiF_3.init_leaky()
        mem4 = self.LiF_out.init_leaky()

        out_sum_spike = torch.zeros(batch_size, 11, device=self.device) ## 10 classes
        
        # Record the final layer
        spk4_rec = []
        mem4_rec = []
        LiF_1_list = []
        LiF_2_list = []
        LiF_3_list = []


        for step in range(num_steps):
            x = input[step]
            cur_1 = self.Conv_1(x)
            cur_1 = self.pool_1(cur_1)
            spk1, mem1 = self.LiF_1.forward(cur_1, mem1)
            LiF_1_list.append(spk1)

            cur_2 = self.Conv_2(spk1)
            # cur_2 = self.pool_2(cur_2)
            spk2, mem2 = self.LiF_2.forward(cur_2, mem2)
            LiF_2_list.append(spk2)

            cur_3 = self.Conv_3(spk2)
            cur_3 = self.pool_3(cur_3)
            spk3, mem3 = self.LiF_3.forward(cur_3, mem3)
            LiF_3_list.append(spk3)

            cur_4 = self.FC(self.Flatten(spk3))
            spk4, mem4 = self.LiF_out.forward(cur_4, mem4)
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)
            out_sum_spike += spk4

        out_sum_spike = out_sum_spike/num_steps
        
        return out_sum_spike, torch.stack(spk4_rec, dim=0), torch.stack(mem4_rec, dim=0), torch.stack(LiF_1_list, dim=0), torch.stack(LiF_2_list, dim=0), torch.stack(LiF_3_list, dim=0)