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
# import snntorch as snn
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

class RelBerLeaky(nn.Module):
    def __init__(self, beta, device, threshold):
        self.device = device
        self.beta = tensor(beta).to(device)
        self.threshold = tensor(threshold).to(device)

        self._is_spiking = None
        super(RelBerLeaky, self).__init__()


    def forward(self, input_, mem=None, temp=0.):
        batch_size = input_.shape[0]

        if mem is None:
            mem = torch.zeros_like(input_, device=self.device)

        if self._is_spiking is None or self._is_spiking.shape != (batch_size,):
            self._is_spiking = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        mem = self.beta * mem + input_
        mem[self._is_spiking] -= self.threshold

        spk = self.fire(mem - self.threshold, temp)
        self._is_spiking = (spk > 0)

        return spk, mem
    
    def fire(self, mem, temp):
        if temp != 0:
            dist = RelaxedBernoulli(temp, logits=mem)
            spk = dist.rsample().to(self.device)
        else:
            dist = Bernoulli(logits = mem)
            spk = dist.sample().to(self.device)
        return spk
        


class Net(nn.Module):
    ### Conv net with 12C5 - P2 - 64C5 - P2
    def __init__(self, beta, device, temp, threshold):
        super().__init__()

        self.beta = beta
        self.temp = temp
        self.device = device

        ## Initialize layers
        self.Conv_1 = nn.Conv2d(2, 12, 5, bias=False, device=device)
        self.pool_1 = nn.AvgPool2d(2)
        self.LiF_1 = RelBerLeaky(self.beta, device = device, threshold=threshold)


        self.Conv_2 = nn.Conv2d(12, 64, 5, bias=False, device=device)
        self.pool_2 = nn.AvgPool2d(2)
        self.LiF_2 = RelBerLeaky(self.beta, device=device, threshold=threshold)

        self.Flatten = nn.Flatten()
        self.FC = nn.Linear(1600, 10, bias = False, device = device)
        self.LiF_3 = RelBerLeaky(self.beta, device=device, threshold=threshold)

    def forward(self, input): ## Note here input is [Time, Batch, Shape]
        num_steps = input.shape[0]
        batch_size = input.shape[1]

        ## initialize hidden states (membranes)
        mem1 = None
        mem2 = None
        mem3 = None

        out_sum_spike = torch.zeros(batch_size, 10, device=self.device) ## 10 classes
        
        # Record the final layer
        spk3_rec = []
        mem3_rec = []


        for step in range(num_steps):
            x = input[step]
            cur_1 = self.Conv_1(x)
            cur_1 = self.pool_1(cur_1)
            spk1, mem1 = self.LiF_1.forward(cur_1, mem1, self.temp)

            cur_2 = self.Conv_2(spk1)
            cur_2 = self.pool_2(cur_2)
            spk2, mem2 = self.LiF_2.forward(cur_2, mem2, self.temp)

            cur_3 = self.FC(self.Flatten(spk2))
            spk3, mem3 = self.LiF_3.forward(cur_3, mem3, self.temp)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
            out_sum_spike += spk3

        out_sum_spike = out_sum_spike/num_steps
        
        return out_sum_spike, torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0) ## return spike train and final membrane history
