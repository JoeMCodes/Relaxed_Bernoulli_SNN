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
import StochasticSNN_with_sparsity

os.chdir('Experiment_NMNIST')

### Need to check ALL file paths, consider using/adding brier score for robustness check

device = torch.device('cuda:2') ## Manual Change


# ## Manual change
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


### Load the Data

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.Compose([
                                    transforms.ToFrame(sensor_size=sensor_size,
                                                        n_time_bins=num_steps) 
                                    ])

testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)


class BinarizeTransform:
    def __call__(self, tensor):
        return (tensor > 0).float()

transform = tonic.transforms.Compose([torch.from_numpy,
                                    BinarizeTransform()])

cached_testset = DiskCachedDataset(testset, transform=transform, cache_path='./cache/nmnist/test')

testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), drop_last=True)

def corrupt_image(img, prob):
    # Generate a random tensor with the same shape as the image
    random_tensor = torch.rand_like(img, dtype=torch.float32)

    # Create a mask for the bits to flip
    flip_mask = random_tensor < prob

    # Flip the bits where the mask is True
    corrupted_img = img.clone()  # Clone the original image tensor
    corrupted_img[flip_mask] = 1 - corrupted_img[flip_mask]

    return corrupted_img



def get_probs_normalise(x):
    # Calculate the sum of entries across the num_classes dimension
    row_sums = torch.sum(x, dim=1, keepdim=True)
    
    # Check if any row sums are zero
    zero_row_mask = (row_sums == 0)
    
    # If a row sum is zero, replace the corresponding row in x with ones
    x[zero_row_mask.expand_as(x)] = 1.0
    
    # Recalculate the row sums after replacement
    row_sums = torch.sum(x, dim=1, keepdim=True)
    
    # Normalize by dividing each entry by the sum of its row
    probs = x / row_sums
    
    return probs



no_repeats = 5

prob_list = np.linspace(0, 0.5, 51)

accs = []
accs2 = []
accs3 = []
briers = []
briers2 = []
briers3 = []
briers_sm = []
briers2_sm = []
briers3_sm = []

net.eval()
for p in prob_list:
    t = time.time()
    with torch.no_grad():
        acc_sum = 0
        acc_sum2 = 0
        acc_sum3 = 0
        brier_sum = 0
        brier_sum2 = 0
        brier_sum3 = 0
        brier_sum_sm = 0
        brier_sum2_sm = 0
        brier_sum3_sm = 0
        for _ in range(no_repeats):
            total = 0
            correct = 0
            correct2 = 0
            correct3 = 0
            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device, dtype=torch.float32)
                targets_oh = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=10).to(device, dtype=torch.float32)
                # test_spike_data = data.reshape(30, -1, 34*34*2)
                test_spike_data = corrupt_image(data, p)

                # forward pass
                test_output, _, _, _, _ = net(test_spike_data)
                test_output2, _, _, _, _ = net2(test_spike_data)
                test_output3, _, _, _, _ = net3(test_spike_data)

                # calculate total accuracy
                _, predicted = test_output.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                output_acc = get_probs_normalise(test_output)
                brier_sum += torch.sum((output_acc - targets_oh)**2).item()/10
                brier_sum_sm += torch.sum((nn.functional.softmax(test_output, dim=1) - targets_oh)**2).item()/10


                _, predicted2 = test_output2.max(1)
                correct2 += (predicted2 == targets).sum().item()

                output_acc2 = get_probs_normalise(test_output2)
                brier_sum2 += torch.sum((output_acc2 - targets_oh)**2).item()/10
                brier_sum2_sm += torch.sum((nn.functional.softmax(test_output2, dim=1) - targets_oh)**2).item()/10

                _, predicted3 = test_output3.max(1)
                correct3 += (predicted3 == targets).sum().item()

                output_acc3 = get_probs_normalise(test_output3)
                brier_sum3 += torch.sum((output_acc3 - targets_oh)**2).item()/10
                brier_sum3_sm += torch.sum((nn.functional.softmax(test_output3, dim=1) - targets_oh)**2).item()/10

            test_acc = 100 * correct / total
            acc_sum+=test_acc

            test_acc2 = 100 * correct2 / total
            acc_sum2+=test_acc2

            test_acc3 = 100 * correct3 / total
            acc_sum3+=test_acc3

    accs.append(acc_sum/no_repeats)
    accs2.append(acc_sum2/no_repeats)
    accs3.append(acc_sum3/no_repeats)
    briers.append(brier_sum/(no_repeats*total))
    briers2.append(brier_sum2/(no_repeats*total))
    briers3.append(brier_sum3/(no_repeats*total))
    briers_sm.append(brier_sum_sm/(no_repeats*total))
    briers2_sm.append(brier_sum2_sm/(no_repeats*total))
    briers3_sm.append(brier_sum3_sm/(no_repeats*total))
    print(f'Finished prob {p} in time {time.time()-t} s', flush = True)


result = {'probs': prob_list,
            'accs' : accs,
            'accs2' : accs2,
            'accs3' : accs3,
            'briers' : briers,
            'briers2' : briers2,
            'briers3' : briers3,
            'briers_sm': briers_sm,
            'briers2_sm': briers2_sm,
            'briers3_sm': briers3_sm}

# Save all results to a file
with open(f'Plots/Plot_data/ASSnet_prob_Robustness_beta{beta}_num_steps{num_steps}_trainedthr_compare.pkl', 'wb') as f:
    pickle.dump(result, f)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for different thresholds
plt.plot(prob_list, accs, label='Threshold = 0.5')
plt.plot(prob_list, accs2, label='Threshold = 1.5')
plt.plot(prob_list, accs3, label='Threshold = 4.5')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Robustness For Different Training Thresholds)')
plt.xlabel('Prob')
plt.ylabel('Accuracy')

# Save the combined plot to the specified file path
plt.savefig(f'Plots/ASSnet_prob_Robustness_beta{beta}_num_steps{num_steps}_thr_train{v_thr_train}.png')

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for different thresholds
plt.plot(prob_list, briers, label='Threshold = 0.5')
plt.plot(prob_list, briers2, label='Threshold = 1.5')
plt.plot(prob_list, briers3, label='Threshold = 4.5')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Robustness For Different Training Thresholds')
plt.xlabel('Prob')
plt.ylabel('Brier Score')

# Save the combined plot to the specified file path
plt.savefig(f'Plots/ASSnet_prob_brier_Robustness_beta{beta}_num_steps{num_steps}_thr_train{v_thr_train}.png')


### NOW FOR FT

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


no_repeats = 5

prob_list = np.linspace(0, 0.5, 51)

accs = []
accs2 = []
accs3 = []
briers = []
briers2 = []
briers3 = []
briers_sm = []
briers2_sm = []
briers3_sm = []

net.eval()
for p in prob_list:
    t = time.time()
    with torch.no_grad():
        acc_sum = 0
        acc_sum2 = 0
        acc_sum3 = 0
        brier_sum = 0
        brier_sum2 = 0
        brier_sum3 = 0
        brier_sum_sm = 0
        brier_sum2_sm = 0
        brier_sum3_sm = 0
        for _ in range(no_repeats):
            total = 0
            correct = 0
            correct2 = 0
            correct3 = 0
            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device, dtype=torch.float32)
                targets_oh = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=10).to(device, dtype=torch.float32)
                # test_spike_data = data.reshape(30, -1, 34*34*2)
                test_spike_data = corrupt_image(data, p)

                # forward pass
                test_output, _, _, _, _ = net(test_spike_data)
                test_output2, _, _, _, _ = net2(test_spike_data)
                test_output3, _, _, _, _ = net3(test_spike_data)

                # calculate total accuracy
                _, predicted = test_output.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                output_acc = get_probs_normalise(test_output)
                brier_sum += torch.sum((output_acc - targets_oh)**2).item()/10
                brier_sum_sm += torch.sum((nn.functional.softmax(test_output, dim=1) - targets_oh)**2).item()/10


                _, predicted2 = test_output2.max(1)
                correct2 += (predicted2 == targets).sum().item()

                output_acc2 = get_probs_normalise(test_output2)
                brier_sum2 += torch.sum((output_acc2 - targets_oh)**2).item()/10
                brier_sum2_sm += torch.sum((nn.functional.softmax(test_output2, dim=1) - targets_oh)**2).item()/10

                _, predicted3 = test_output3.max(1)
                correct3 += (predicted3 == targets).sum().item()

                output_acc3 = get_probs_normalise(test_output3)
                brier_sum3 += torch.sum((output_acc3 - targets_oh)**2).item()/10
                brier_sum3_sm += torch.sum((nn.functional.softmax(test_output3, dim=1) - targets_oh)**2).item()/10

            test_acc = 100 * correct / total
            acc_sum+=test_acc

            test_acc2 = 100 * correct2 / total
            acc_sum2+=test_acc2

            test_acc3 = 100 * correct3 / total
            acc_sum3+=test_acc3

    accs.append(acc_sum/no_repeats)
    accs2.append(acc_sum2/no_repeats)
    accs3.append(acc_sum3/no_repeats)
    briers.append(brier_sum/(no_repeats*total))
    briers2.append(brier_sum2/(no_repeats*total))
    briers3.append(brier_sum3/(no_repeats*total))
    briers_sm.append(brier_sum_sm/(no_repeats*total))
    briers2_sm.append(brier_sum2_sm/(no_repeats*total))
    briers3_sm.append(brier_sum3_sm/(no_repeats*total))
    print(f'Finished prob {p} in time {time.time()-t} s', flush = True)


result = {'probs': prob_list,
            'accs' : accs,
            'accs2' : accs2,
            'accs3' : accs3,
            'briers' : briers,
            'briers2' : briers2,
            'briers3' : briers3,
            'briers_sm': briers_sm,
            'briers2_sm': briers2_sm,
            'briers3_sm': briers3_sm}

# Save all results to a file
with open(f'Plots/Plot_data/FT_prob_Robustness_beta{beta}_num_steps{num_steps}_trainedthr_compare.pkl', 'wb') as f:
    pickle.dump(result, f)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for different thresholds
plt.plot(prob_list, accs, label='Threshold = 0.5')
plt.plot(prob_list, accs2, label='Threshold = 1.5')
plt.plot(prob_list, accs3, label='Threshold = 4.5')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Robustness For Different Training Thresholds)')
plt.xlabel('Prob')
plt.ylabel('Accuracy')

# Save the combined plot to the specified file path
plt.savefig(f'Plots/FT_prob_Robustness_beta{beta}_num_steps{num_steps}_thr_train{v_thr_train}.png')

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for different thresholds
plt.plot(prob_list, briers, label='Threshold = 0.5')
plt.plot(prob_list, briers2, label='Threshold = 1.5')
plt.plot(prob_list, briers3, label='Threshold = 4.5')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Robustness For Different Training Thresholds')
plt.xlabel('Prob')
plt.ylabel('Brier Score')

# Save the combined plot to the specified file path
plt.savefig(f'Plots/FT_prob_brier_Robustness_beta{beta}_num_steps{num_steps}_thr_train{v_thr_train}.png')


### Now for Det

import Det_SNN_with_sparsity


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

print('did model loads')

no_repeats = 5

prob_list = np.linspace(0, 0.5, 51)

accs = []
accs2 = []
accs3 = []
briers = []
briers2 = []
briers3 = []
briers_sm = []
briers2_sm = []
briers3_sm = []

net.eval()
for p in prob_list:
    t = time.time()
    with torch.no_grad():
        acc_sum = 0
        acc_sum2 = 0
        acc_sum3 = 0
        brier_sum = 0
        brier_sum2 = 0
        brier_sum3 = 0
        brier_sum_sm = 0
        brier_sum2_sm = 0
        brier_sum3_sm = 0
        for _ in range(no_repeats):
            total = 0
            correct = 0
            correct2 = 0
            correct3 = 0
            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device, dtype=torch.float32)
                targets_oh = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=10).to(device, dtype=torch.float32)
                # test_spike_data = data.reshape(30, -1, 34*34*2)
                test_spike_data = corrupt_image(data, p)

                # forward pass
                test_output, _, _, _, _ = net(test_spike_data)
                test_output2, _, _, _, _ = net2(test_spike_data)
                test_output3, _, _, _, _ = net3(test_spike_data)
                # calculate total accuracy
                _, predicted = test_output.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                output_acc = get_probs_normalise(test_output)
                brier_sum += torch.sum((output_acc - targets_oh)**2).item()/10
                brier_sum_sm += torch.sum((nn.functional.softmax(test_output, dim=1) - targets_oh)**2).item()/10


                _, predicted2 = test_output2.max(1)
                correct2 += (predicted2 == targets).sum().item()

                output_acc2 = get_probs_normalise(test_output2)
                brier_sum2 += torch.sum((output_acc2 - targets_oh)**2).item()/10
                brier_sum2_sm += torch.sum((nn.functional.softmax(test_output2, dim=1) - targets_oh)**2).item()/10

                _, predicted3 = test_output3.max(1)
                correct3 += (predicted3 == targets).sum().item()

                output_acc3 = get_probs_normalise(test_output3)
                brier_sum3 += torch.sum((output_acc3 - targets_oh)**2).item()/10
                brier_sum3_sm += torch.sum((nn.functional.softmax(test_output3, dim=1) - targets_oh)**2).item()/10

            test_acc = 100 * correct / total
            acc_sum+=test_acc

            test_acc2 = 100 * correct2 / total
            acc_sum2+=test_acc2

            test_acc3 = 100 * correct3 / total
            acc_sum3+=test_acc3

    accs.append(acc_sum/no_repeats)
    accs2.append(acc_sum2/no_repeats)
    accs3.append(acc_sum3/no_repeats)
    briers.append(brier_sum/(no_repeats*total))
    briers2.append(brier_sum2/(no_repeats*total))
    briers3.append(brier_sum3/(no_repeats*total))
    briers_sm.append(brier_sum_sm/(no_repeats*total))
    briers2_sm.append(brier_sum2_sm/(no_repeats*total))
    briers3_sm.append(brier_sum3_sm/(no_repeats*total))
    print(f'Finished prob {p} in time {time.time()-t} s', flush = True)


result = {'probs': prob_list,
            'accs' : accs,
            'accs2' : accs2,
            'accs3' : accs3,
            'briers' : briers,
            'briers2' : briers2,
            'briers3' : briers3,
            'briers_sm': briers_sm,
            'briers2_sm': briers2_sm,
            'briers3_sm': briers3_sm}

# Save all results to a file
with open(f'Plots/Plot_data/Det_prob_Robustness_beta{beta}_num_steps{num_steps}_trainedthr_compare.pkl', 'wb') as f:
    pickle.dump(result, f)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for different thresholds
plt.plot(prob_list, accs, label='Threshold = 0.5')
plt.plot(prob_list, accs2, label='Threshold = 1.5')
plt.plot(prob_list, accs3, label='Threshold = 4.5')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Robustness For Different Training Thresholds)')
plt.xlabel('Prob')
plt.ylabel('Accuracy')

# Save the combined plot to the specified file path
plt.savefig(f'Plots/Det_prob_Robustness_beta{beta}_num_steps{num_steps}_thr_train{v_thr_train}.png')

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for different thresholds
plt.plot(prob_list, briers, label='Threshold = 0.5')
plt.plot(prob_list, briers2, label='Threshold = 1.5')
plt.plot(prob_list, briers3, label='Threshold = 4.5')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Robustness For Different Training Thresholds')
plt.xlabel('Prob')
plt.ylabel('Brier Score')

# Save the combined plot to the specified file path
plt.savefig(f'Plots/Det_prob_brier_Robustness_beta{beta}_num_steps{num_steps}_thr_train{v_thr_train}.png')
