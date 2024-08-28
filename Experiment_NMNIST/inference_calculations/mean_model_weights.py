from calendar import c
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
import Det_SNN_with_sparsity


os.chdir('Experiment_NMNIST')

device = torch.device('cuda:7') ## Manual Change

## Manual change
model_path = 'deterministic_model/Models/Det_beta0.9_num_epochs200_num_steps30_batch_size128_lr0.001_thr_0.5.pt' 
beta = 0.9
num_steps = 30
batch_size = 128
v_thr_train = 0.5

net = Det_SNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)

net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)

model2_path = 'deterministic_model/Models/Det_beta0.9_num_epochs200_num_steps30_batch_size128_lr0.001_thr_1.5.pt' 
v_thr_train = 1.5
net2 = Det_SNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net2.load_state_dict(torch.load(model2_path, map_location=device))
net2.to(device)

model3_path = 'deterministic_model/Models/Det_beta0.9_num_epochs200_num_steps30_batch_size128_lr0.001_thr_4.5.pt' 
v_thr_train = 4.5
net3 = Det_SNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net3.load_state_dict(torch.load(model3_path, map_location=device))
net3.to(device)

print('Det models')

# Initialize variables to accumulate the sum of L2 norms and count the layers
l2_sum = 0.0
layer_count = 0

# Calculate the L2 norm for each layer's weights and accumulate
for name, param in net.named_parameters():
    if 'weight' in name:
        l2_norm = torch.norm(param, p=2)
        l2_sum += l2_norm.item()
        layer_count += 1

# Calculate the average L2 norm across all layers
average_l2 = l2_sum / layer_count if layer_count > 0 else 0

print(f'Average L2 norm of weights across all layers: {average_l2}')
        
# Initialize variables to accumulate the sum of L2 norms and count the layers
l2_sum = 0.0
layer_count = 0

# Calculate the L2 norm for each layer's weights and accumulate
for name, param in net2.named_parameters():
    if 'weight' in name:
        l2_norm = torch.norm(param, p=2)
        l2_sum += l2_norm.item()
        layer_count += 1

# Calculate the average L2 norm across all layers
average_l2 = l2_sum / layer_count if layer_count > 0 else 0

print(f'Average L2 norm of weights across all layers: {average_l2}')

# Initialize variables to accumulate the sum of L2 norms and count the layers
l2_sum = 0.0
layer_count = 0

# Calculate the L2 norm for each layer's weights and accumulate
for name, param in net3.named_parameters():
    if 'weight' in name:
        l2_norm = torch.norm(param, p=2)
        l2_sum += l2_norm.item()
        layer_count += 1

# Calculate the average L2 norm across all layers
average_l2 = l2_sum / layer_count if layer_count > 0 else 0

print(f'Average L2 norm of weights across all layers: {average_l2}')



### Now for ASSnet
import StochasticSNN_with_sparsity

## Manual change
model_path = 'ASSnet_tests/Results/ASSnet_beta0.9_tempmulti_num_epochs200_num_steps30_batch_size128_lr0.001_thr_0.5.pt' 
beta = 0.9
num_steps = 30
batch_size = 128
v_thr_train = 0.5

net = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)

net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)

model2_path = 'ASSnet_tests/Results/ASSnet_beta0.9_tempmulti_num_epochs200_num_steps30_batch_size128_lr0.001_thr_1.5.pt' 
v_thr_train = 1.5
net2 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net2.load_state_dict(torch.load(model2_path, map_location=device))
net2.to(device)

model3_path = 'ASSnet_tests/Results/ASSnet_beta0.9_tempmulti_num_epochs200_num_steps30_batch_size128_lr0.001_thr_4.5.pt' 
v_thr_train = 4.5
net3 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net3.load_state_dict(torch.load(model3_path, map_location=device))
net3.to(device)

print('For ASSnet')

# Initialize variables to accumulate the sum of L2 norms and count the layers
l2_sum = 0.0
layer_count = 0

# Calculate the L2 norm for each layer's weights and accumulate
for name, param in net.named_parameters():
    if 'weight' in name:
        l2_norm = torch.norm(param, p=2)
        l2_sum += l2_norm.item()
        layer_count += 1

# Calculate the average L2 norm across all layers
average_l2 = l2_sum / layer_count if layer_count > 0 else 0

print(f'Average L2 norm of weights across all layers: {average_l2}')
        
# Initialize variables to accumulate the sum of L2 norms and count the layers
l2_sum = 0.0
layer_count = 0

# Calculate the L2 norm for each layer's weights and accumulate
for name, param in net2.named_parameters():
    if 'weight' in name:
        l2_norm = torch.norm(param, p=2)
        l2_sum += l2_norm.item()
        layer_count += 1

# Calculate the average L2 norm across all layers
average_l2 = l2_sum / layer_count if layer_count > 0 else 0

print(f'Average L2 norm of weights across all layers: {average_l2}')

# Initialize variables to accumulate the sum of L2 norms and count the layers
l2_sum = 0.0
layer_count = 0

# Calculate the L2 norm for each layer's weights and accumulate
for name, param in net3.named_parameters():
    if 'weight' in name:
        l2_norm = torch.norm(param, p=2)
        l2_sum += l2_norm.item()
        layer_count += 1

# Calculate the average L2 norm across all layers
average_l2 = l2_sum / layer_count if layer_count > 0 else 0

print(f'Average L2 norm of weights across all layers: {average_l2}')



## Manual change
model_path = 'fixed_temp/beta0.9_temp0.1_num_epochs200_num_steps30_batch_size128_lr0.00025_thr_0.5.pt' 
beta = 0.9
num_steps = 30
batch_size = 128
v_thr_train = 0.5

net = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)

net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)

model2_path = 'fixed_temp/beta0.9_temp0.1_num_epochs200_num_steps30_batch_size128_lr0.00025_thr_1.5.pt' 
v_thr_train = 1.5
net2 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net2.load_state_dict(torch.load(model2_path, map_location=device))
net2.to(device)

model3_path = 'fixed_temp/beta0.9_temp0.1_num_epochs200_num_steps30_batch_size128_lr0.00025_thr_4.5.pt' 
v_thr_train = 4.5
net3 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net3.load_state_dict(torch.load(model3_path, map_location=device))
net3.to(device)

print('For FT')

# Initialize variables to accumulate the sum of L2 norms and count the layers
l2_sum = 0.0
layer_count = 0

# Calculate the L2 norm for each layer's weights and accumulate
for name, param in net.named_parameters():
    if 'weight' in name:
        l2_norm = torch.norm(param, p=2)
        l2_sum += l2_norm.item()
        layer_count += 1

# Calculate the average L2 norm across all layers
average_l2 = l2_sum / layer_count if layer_count > 0 else 0

print(f'Average L2 norm of weights across all layers: {average_l2}')
        
# Initialize variables to accumulate the sum of L2 norms and count the layers
l2_sum = 0.0
layer_count = 0

# Calculate the L2 norm for each layer's weights and accumulate
for name, param in net2.named_parameters():
    if 'weight' in name:
        l2_norm = torch.norm(param, p=2)
        l2_sum += l2_norm.item()
        layer_count += 1

# Calculate the average L2 norm across all layers
average_l2 = l2_sum / layer_count if layer_count > 0 else 0

print(f'Average L2 norm of weights across all layers: {average_l2}')

# Initialize variables to accumulate the sum of L2 norms and count the layers
l2_sum = 0.0
layer_count = 0

# Calculate the L2 norm for each layer's weights and accumulate
for name, param in net3.named_parameters():
    if 'weight' in name:
        l2_norm = torch.norm(param, p=2)
        l2_sum += l2_norm.item()
        layer_count += 1

# Calculate the average L2 norm across all layers
average_l2 = l2_sum / layer_count if layer_count > 0 else 0

print(f'Average L2 norm of weights across all layers: {average_l2}')
