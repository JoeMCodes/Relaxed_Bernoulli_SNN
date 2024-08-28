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

os.chdir('Experiment_DVSGesture')
### Need to check ALL file paths, consider using/adding brier score for robustness check

device = torch.device('cuda:6') ## Manual Change
print('Start', flush = True)

## Manual change
model_path = 'ASSnet_tests/Models/MSE_count_beta0.7_num_epochs500_num_steps60_batch_size64_lr0.0005_thr_0.5.pt' 
beta = 0.7
num_steps = 60
batch_size = 64
v_thr_train = 0.5

net = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)

net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)

model2_path = 'ASSnet_tests/Models/MSE_count_beta0.7_num_epochs500_num_steps60_batch_size64_lr0.0005_thr_1.5.pt' 
v_thr_train = 1.5
net2 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net2.load_state_dict(torch.load(model2_path, map_location=device))
net2.to(device)

model3_path = 'ASSnet_tests/Models/MSE_count_beta0.7_num_epochs500_num_steps60_batch_size64_lr0.0005_thr_4.5.pt' 
v_thr_train = 4.5
net3 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net3.load_state_dict(torch.load(model3_path, map_location=device))
net3.to(device)

## Load the Data

sensor_size = tonic.datasets.DVSGesture.sensor_size

class BinarizeTransform:
    def __call__(self, tensor):
        return (tensor > 0.)
    
w,h=32,32
n_frames=num_steps #100
debug = False

transform = tonic.transforms.Compose([
    tonic.transforms.Denoise(filter_time=10000), # removes outlier events with inactive surrounding pixels for 10ms
    tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(w,h)), # downsampling image
    tonic.transforms.ToFrame(sensor_size=(w,h,2), n_time_bins=n_frames) # n_frames frames per trail
])

testset = tonic.datasets.DVSGesture(save_to='./data3', transform=transform, train=False)

cache_transform = tonic.transforms.Compose([BinarizeTransform()])

cached_testset = DiskCachedDataset(testset, transform=cache_transform, cache_path='./cache3/dvsgesture/test')

testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))
#####

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



no_repeats = 10

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
        correct = 0
        correct2 = 0
        correct3 = 0
        brier_sum = 0
        brier_sum2 = 0
        brier_sum3 = 0
        brier_sum_sm = 0
        brier_sum2_sm = 0
        brier_sum3_sm = 0
        for _ in range(no_repeats):
            total = 0

            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device, dtype=torch.float32)
                targets_oh = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=11).to(device, dtype=torch.float32)
                # test_spike_data = data.reshape(30, -1, 34*34*2)
                test_spike_data = corrupt_image(data, p)

                # forward pass
                test_output, _, _, _, _, _ = net(test_spike_data)
                test_output2, _, _, _, _, _ = net2(test_spike_data)
                test_output3, _, _, _, _, _ = net3(test_spike_data)

                # calculate total accuracy
                _, predicted = test_output.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                output_acc = get_probs_normalise(test_output)
                brier_sum += torch.sum((output_acc - targets_oh)**2).item()/11
                brier_sum_sm += torch.sum((nn.functional.softmax(test_output, dim=1) - targets_oh)**2).item()/11

                _, predicted2 = test_output2.max(1)
                correct2 += (predicted2 == targets).sum().item()

                output_acc2 = get_probs_normalise(test_output2)
                brier_sum2 += torch.sum((output_acc2 - targets_oh)**2).item()/11
                brier_sum2_sm += torch.sum((nn.functional.softmax(test_output2, dim=1) - targets_oh)**2).item()/11


                _, predicted3 = test_output3.max(1)
                correct3 += (predicted3 == targets).sum().item()

                output_acc3 = get_probs_normalise(test_output3)
                brier_sum3 += torch.sum((output_acc3 - targets_oh)**2).item()/11
                brier_sum3_sm += torch.sum((nn.functional.softmax(test_output3, dim=1) - targets_oh)**2).item()/11


    accs.append(100*correct/(no_repeats*total))
    accs2.append(100*correct2/(no_repeats*total))
    accs3.append(100*correct3/(no_repeats*total))
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
model_path = 'fixed_temp/Models/MSE_count_beta0.7_temp0.05_num_epochs500_num_steps60_batch_size64_lr0.0005_thr_0.5.pt' 
beta = 0.7
num_steps = 60
batch_size = 64
v_thr_train = 0.5

net = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)

net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)

model2_path = 'fixed_temp/Models/MSE_count_beta0.7_temp0.05_num_epochs500_num_steps60_batch_size64_lr0.0005_thr_1.5.pt' 
v_thr_train = 1.5
net2 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net2.load_state_dict(torch.load(model2_path, map_location=device))
net2.to(device)

model3_path = 'fixed_temp/Models/MSE_count_beta0.7_temp0.05_num_epochs500_num_steps60_batch_size64_lr0.0005_thr_4.5.pt' 
v_thr_train = 4.5
net3 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net3.load_state_dict(torch.load(model3_path, map_location=device))
net3.to(device)


no_repeats = 10

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
        correct = 0
        correct2 = 0
        correct3 = 0
        brier_sum = 0
        brier_sum2 = 0
        brier_sum3 = 0
        brier_sum_sm = 0
        brier_sum2_sm = 0
        brier_sum3_sm = 0
        for _ in range(no_repeats):
            total = 0

            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device, dtype=torch.float32)
                targets_oh = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=11).to(device, dtype=torch.float32)
                # test_spike_data = data.reshape(30, -1, 34*34*2)
                test_spike_data = corrupt_image(data, p)

                # forward pass
                test_output, _, _, _, _, _ = net(test_spike_data)
                test_output2, _, _, _, _, _ = net2(test_spike_data)
                test_output3, _, _, _, _, _ = net3(test_spike_data)

                # calculate total accuracy
                _, predicted = test_output.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                output_acc = get_probs_normalise(test_output)
                brier_sum += torch.sum((output_acc - targets_oh)**2).item()/11
                brier_sum_sm += torch.sum((nn.functional.softmax(test_output, dim=1) - targets_oh)**2).item()/11

                _, predicted2 = test_output2.max(1)
                correct2 += (predicted2 == targets).sum().item()

                output_acc2 = get_probs_normalise(test_output2)
                brier_sum2 += torch.sum((output_acc2 - targets_oh)**2).item()/11
                brier_sum2_sm += torch.sum((nn.functional.softmax(test_output2, dim=1) - targets_oh)**2).item()/11


                _, predicted3 = test_output3.max(1)
                correct3 += (predicted3 == targets).sum().item()

                output_acc3 = get_probs_normalise(test_output3)
                brier_sum3 += torch.sum((output_acc3 - targets_oh)**2).item()/11
                brier_sum3_sm += torch.sum((nn.functional.softmax(test_output3, dim=1) - targets_oh)**2).item()/11


    accs.append(100*correct/(no_repeats*total))
    accs2.append(100*correct2/(no_repeats*total))
    accs3.append(100*correct3/(no_repeats*total))
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



## Now for DET


import Det_SNN_with_sparsity


## Manual change
model_path = 'deterministic_models/Models/MSE_count_beta0.7_temp0.05_num_epochs500_num_steps60_batch_size64_lr0.0005_thr_0.5.pt' 
beta = 0.7
num_steps = 60
batch_size = 64
v_thr_train = 0.5

net = Det_SNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)

net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)

model2_path = 'deterministic_models/Models/MSE_count_beta0.7_temp0.05_num_epochs500_num_steps60_batch_size64_lr0.0005_thr_1.5.pt' 
v_thr_train = 1.5
net2 = Det_SNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net2.load_state_dict(torch.load(model2_path, map_location=device))
net2.to(device)

model3_path = 'deterministic_models/Models/MSE_count_beta0.7_temp0.05_num_epochs500_num_steps60_batch_size64_lr0.0005_thr_4.5.pt' 
v_thr_train = 4.5
net3 = Det_SNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net3.load_state_dict(torch.load(model3_path, map_location=device))
net3.to(device)



no_repeats = 10

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
        correct = 0
        correct2 = 0
        correct3 = 0
        brier_sum = 0
        brier_sum2 = 0
        brier_sum3 = 0
        brier_sum_sm = 0
        brier_sum2_sm = 0
        brier_sum3_sm = 0
        for _ in range(no_repeats):
            total = 0

            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device, dtype=torch.float32)
                targets_oh = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=11).to(device, dtype=torch.float32)
                # test_spike_data = data.reshape(30, -1, 34*34*2)
                test_spike_data = corrupt_image(data, p)

                # forward pass
                test_output, _, _, _, _, _ = net(test_spike_data)
                test_output2, _, _, _, _, _ = net2(test_spike_data)
                test_output3, _, _, _, _, _ = net3(test_spike_data)

                # calculate total accuracy
                _, predicted = test_output.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                output_acc = get_probs_normalise(test_output)
                brier_sum += torch.sum((output_acc - targets_oh)**2).item()/11
                brier_sum_sm += torch.sum((nn.functional.softmax(test_output, dim=1) - targets_oh)**2).item()/11

                _, predicted2 = test_output2.max(1)
                correct2 += (predicted2 == targets).sum().item()

                output_acc2 = get_probs_normalise(test_output2)
                brier_sum2 += torch.sum((output_acc2 - targets_oh)**2).item()/11
                brier_sum2_sm += torch.sum((nn.functional.softmax(test_output2, dim=1) - targets_oh)**2).item()/11


                _, predicted3 = test_output3.max(1)
                correct3 += (predicted3 == targets).sum().item()

                output_acc3 = get_probs_normalise(test_output3)
                brier_sum3 += torch.sum((output_acc3 - targets_oh)**2).item()/11
                brier_sum3_sm += torch.sum((nn.functional.softmax(test_output3, dim=1) - targets_oh)**2).item()/11


    accs.append(100*correct/(no_repeats*total))
    accs2.append(100*correct2/(no_repeats*total))
    accs3.append(100*correct3/(no_repeats*total))
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