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



#####
## Now vary and plot thr against acc

no_repeats = 5
v_thrs = np.linspace(0, 11, 111)
net.eval()
net2.eval()
net3.eval()

L1_1_sparss = []
L1_2_sparss = []
L1_3_sparss = []
L2_1_sparss = []
L2_2_sparss = []
L2_3_sparss = []
L3_1_sparss = []
L3_2_sparss = []
L3_3_sparss = []

accs = []
accs2 = []
accs3 = []
briers = []
briers2 = []
briers3 = []
briers_sm = []
briers2_sm = []
briers3_sm = []


for v_thr in v_thrs:
    t = time.time()

    v_thr = torch.tensor(v_thr)
    net.LiF_1.threshold = v_thr
    net.LiF_2.threshold = v_thr
    net.LiF_3.threshold = v_thr 

    net2.LiF_1.threshold = v_thr
    net2.LiF_2.threshold = v_thr
    net2.LiF_3.threshold = v_thr 

    net3.LiF_1.threshold = v_thr
    net3.LiF_2.threshold = v_thr
    net3.LiF_3.threshold = v_thr 

    
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

        L1_1_running_mean_sparsity = 0
        L1_2_running_mean_sparsity = 0
        L1_3_running_mean_sparsity = 0
        L2_1_running_mean_sparsity = 0
        L2_2_running_mean_sparsity = 0
        L2_3_running_mean_sparsity = 0
        L3_1_running_mean_sparsity = 0
        L3_2_running_mean_sparsity = 0
        L3_3_running_mean_sparsity = 0
        for _ in range(no_repeats):
            L1_1_running_sparsity = 0
            L1_2_running_sparsity = 0
            L1_3_running_sparsity = 0
            L2_1_running_sparsity = 0
            L2_2_running_sparsity = 0
            L2_3_running_sparsity = 0
            L3_1_running_sparsity = 0
            L3_2_running_sparsity = 0
            L3_3_running_sparsity = 0
            N = 0

            total = 0
            correct = 0
            correct2 = 0
            correct3 = 0
            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device, dtype=torch.float32)
                targets_oh = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=10).to(device, dtype=torch.float32)
                test_spike_data = data.to(device)

                # forward pass
                test_output, LiF3_1, _, LiF1_1, LiF2_1 = net(test_spike_data)
                test_output2, LiF3_2, _, LiF1_2, LiF2_2 = net2(test_spike_data)
                test_output3, LiF3_3, _, LiF1_3, LiF2_3 = net3(test_spike_data)


                # calculate layer sparsity 
                L1_1_running_sparsity += torch.mean(LiF1_1).item()
                L2_1_running_sparsity += torch.mean(LiF2_1).item()
                L3_1_running_sparsity += torch.mean(LiF3_1).item()

                L1_2_running_sparsity += torch.mean(LiF1_2).item()
                L2_2_running_sparsity += torch.mean(LiF2_2).item()
                L3_2_running_sparsity += torch.mean(LiF3_2).item()

                L1_3_running_sparsity += torch.mean(LiF1_3).item()
                L2_3_running_sparsity += torch.mean(LiF2_3).item()
                L3_3_running_sparsity += torch.mean(LiF3_3).item()

                N += 1

                
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

            
            L1_1_running_mean_sparsity += L1_1_running_sparsity/N
            L2_1_running_mean_sparsity += L2_1_running_sparsity/N
            L3_1_running_mean_sparsity += L3_1_running_sparsity/N
            
            L1_2_running_mean_sparsity += L1_2_running_sparsity/N
            L2_2_running_mean_sparsity += L2_2_running_sparsity/N
            L3_2_running_mean_sparsity += L3_2_running_sparsity/N

            L1_3_running_mean_sparsity += L1_3_running_sparsity/N
            L2_3_running_mean_sparsity += L2_3_running_sparsity/N
            L3_3_running_mean_sparsity += L3_3_running_sparsity/N

            test_acc = 100 * correct / total
            acc_sum+=test_acc

            test_acc2 = 100 * correct2 / total
            acc_sum2+=test_acc2

            test_acc3 = 100 * correct3 / total
            acc_sum3+=test_acc3

    L1_1_sparss.append(L1_1_running_mean_sparsity/no_repeats)
    L2_1_sparss.append(L2_1_running_mean_sparsity/no_repeats)
    L3_1_sparss.append(L3_1_running_mean_sparsity/no_repeats)

    L1_2_sparss.append(L1_2_running_mean_sparsity/no_repeats)
    L2_2_sparss.append(L2_2_running_mean_sparsity/no_repeats)
    L3_2_sparss.append(L3_2_running_mean_sparsity/no_repeats)

    L1_3_sparss.append(L1_3_running_mean_sparsity/no_repeats)
    L2_3_sparss.append(L2_3_running_mean_sparsity/no_repeats)
    L3_3_sparss.append(L3_3_running_mean_sparsity/no_repeats)

    accs.append(acc_sum/no_repeats)
    accs2.append(acc_sum2/no_repeats)
    accs3.append(acc_sum3/no_repeats)
    briers.append(brier_sum/(no_repeats*total))
    briers2.append(brier_sum2/(no_repeats*total))
    briers3.append(brier_sum3/(no_repeats*total))
    briers_sm.append(brier_sum_sm/(no_repeats*total))
    briers2_sm.append(brier_sum2_sm/(no_repeats*total))
    briers3_sm.append(brier_sum3_sm/(no_repeats*total))

    print(f'Finished Threshrold = {v_thr} : Took {time.time() - t}s', flush = True)

result = {'thresholds': v_thrs,
          'L1_1_sparss': L1_1_sparss,
          'L1_2_sparss': L1_2_sparss,
          'L1_3_sparss': L1_3_sparss,
          'L2_1_sparss': L2_1_sparss,
          'L2_2_sparss': L2_2_sparss,
          'L2_3_sparss': L2_3_sparss,
          'L3_1_sparss': L3_1_sparss,
          'L3_2_sparss': L3_2_sparss,
          'L3_3_sparss': L3_3_sparss}

# Save all results to a file
with open(f'Plots/Plot_data/Det_spars_vs_thr_beta{beta}_num_steps{num_steps}_trainedthr_{v_thr}_compare_extra.pkl', 'wb') as f:
    pickle.dump(result, f)

result = {'thresholds': v_thrs,
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
with open(f'Plots/Plot_data/Det_acc_vs_thr_beta{beta}_num_steps{num_steps}_trainedthr_{v_thr}_compare_extra.pkl', 'wb') as f:
    pickle.dump(result, f)


plt.figure()

# Plot the data
plt.plot(v_thrs, L1_1_sparss, label='Layer 1, Threshold = 0.5', color='red')
plt.plot(v_thrs, L1_2_sparss, label='Layer 1, Threshold = 1.5', color='blue')
plt.plot(v_thrs, L1_3_sparss, label='Layer 1, Threshold = 4.5', color='green')

# Plot the data
plt.plot(v_thrs, L2_1_sparss, label='Layer 2, Threshold = 0.5', color='red', linestyle='dashed')
plt.plot(v_thrs, L2_2_sparss, label='Layer 2, Threshold = 1.5', color='blue', linestyle='dashed')
plt.plot(v_thrs, L2_3_sparss, label='Layer 2, Threshold = 4.5', color='green', linestyle='dashed')

# Plot the data
plt.plot(v_thrs, L3_1_sparss, label='Layer 3, Threshold = 0.5', color='red', linestyle='dotted')
plt.plot(v_thrs, L3_2_sparss, label='Layer 3, Threshold = 1.5', color='blue', linestyle='dotted')
plt.plot(v_thrs, L3_3_sparss, label='Layer 3, Threshold = 4.5', color='green', linestyle='dotted')

# Create legend handles
color_legend_handles = [
    plt.Line2D([0], [0], color='red', lw=2),
    plt.Line2D([0], [0], color='blue', lw=2),
    plt.Line2D([0], [0], color='green', lw=2)
]
line_type_legend_handles = [
    plt.Line2D([0], [0], color='black', linestyle='-', lw=2),
    plt.Line2D([0], [0], color='black', linestyle='--', lw=2),
    plt.Line2D([0], [0], color='black', linestyle=':', lw=2)
]

# Create the legends
legend1 = plt.legend(color_legend_handles, ['Threshold = 0.5', 'Threshold = 1.5', 'Threshold = 4.5'], title='Threshold Legend', loc='upper left')
legend2 = plt.legend(line_type_legend_handles, ['Layer 1', 'Layer 2', 'Layer 3'], title='Layer Legend', loc='upper right')

# Add the legends to the plot
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

# Add title and labels
plt.title(f'Sparsity vs Threshold (beta={beta}, num_steps={num_steps})')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')

# Save the plot to the specified file path
plt.savefig(f'Plots/Det_Sparsity_vs_Thr_beta{beta}_num_steps{num_steps}.png')

plt.figure()

# Plot the data
plt.plot(v_thrs, torch.mean(torch.tensor([L1_1_sparss, L2_1_sparss, L3_1_sparss]), dim = 0), label='Threshold = 0.5', color='red')
plt.plot(v_thrs, torch.mean(torch.tensor([L1_2_sparss, L2_2_sparss, L3_2_sparss]), dim = 0), label='Threshold = 1.5', color='blue')
plt.plot(v_thrs, torch.mean(torch.tensor([L1_3_sparss, L2_3_sparss, L3_3_sparss]), dim = 0), label='Threshold = 4.5', color='green')

# Add title and labels
plt.title(f'Model Sparsity vs Threshold (beta={beta}, num_steps={num_steps})')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')

# Save the plot to the specified file path
plt.savefig(f'Plots/Det_Sparsity_vs_Thr_beta{beta}_num_steps{num_steps}_model.png')


## ACC vs THR
# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for different thresholds
plt.plot(v_thrs, accs, label='Threshold = 0.5')
plt.plot(v_thrs, accs2, label='Threshold = 1.5')
plt.plot(v_thrs, accs3, label='Threshold = 4.5')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Threshold vs Accuracy (beta={beta}, num_steps={num_steps})')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')

# Save the combined plot to the specified file path
plt.savefig(f'Plots/Det_Sparsity_vs_Acc_beta{beta}_num_steps{num_steps}.png')


### Now for ASSnet
import StochasticSNN_with_sparsity

## Manual change
model_path = 'ASSnet_tests/Models/ASSnet_beta0.9_tempmulti_num_epochs200_num_steps30_batch_size128_lr0.001_thr_0.5.pt' 
beta = 0.9
num_steps = 30
batch_size = 128
v_thr_train = 0.5

net = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)

net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)

model2_path = 'ASSnet_tests/Models/ASSnet_beta0.9_tempmulti_num_epochs200_num_steps30_batch_size128_lr0.001_thr_1.5.pt' 
v_thr_train = 1.5
net2 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net2.load_state_dict(torch.load(model2_path, map_location=device))
net2.to(device)

model3_path = 'ASSnet_tests/Models/ASSnet_beta0.9_tempmulti_num_epochs200_num_steps30_batch_size128_lr0.001_thr_4.5.pt' 
v_thr_train = 4.5
net3 = StochasticSNN_with_sparsity.Net(beta = beta, device = device, temp = 0., threshold = v_thr_train)
net3.load_state_dict(torch.load(model3_path, map_location=device))
net3.to(device)


#####
## Now vary and plot thr against acc

no_repeats = 5
v_thrs = np.linspace(0, 11, 111)
net.eval()
net2.eval()
net3.eval()

L1_1_sparss = []
L1_2_sparss = []
L1_3_sparss = []
L2_1_sparss = []
L2_2_sparss = []
L2_3_sparss = []
L3_1_sparss = []
L3_2_sparss = []
L3_3_sparss = []

accs = []
accs2 = []
accs3 = []
briers = []
briers2 = []
briers3 = []
briers_sm = []
briers2_sm = []
briers3_sm = []


for v_thr in v_thrs:
    t = time.time()

    v_thr = torch.tensor(v_thr)
    net.LiF_1.threshold = v_thr
    net.LiF_2.threshold = v_thr
    net.LiF_3.threshold = v_thr 

    net2.LiF_1.threshold = v_thr
    net2.LiF_2.threshold = v_thr
    net2.LiF_3.threshold = v_thr 

    net3.LiF_1.threshold = v_thr
    net3.LiF_2.threshold = v_thr
    net3.LiF_3.threshold = v_thr 

    
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

        L1_1_running_mean_sparsity = 0
        L1_2_running_mean_sparsity = 0
        L1_3_running_mean_sparsity = 0
        L2_1_running_mean_sparsity = 0
        L2_2_running_mean_sparsity = 0
        L2_3_running_mean_sparsity = 0
        L3_1_running_mean_sparsity = 0
        L3_2_running_mean_sparsity = 0
        L3_3_running_mean_sparsity = 0
        for _ in range(no_repeats):
            L1_1_running_sparsity = 0
            L1_2_running_sparsity = 0
            L1_3_running_sparsity = 0
            L2_1_running_sparsity = 0
            L2_2_running_sparsity = 0
            L2_3_running_sparsity = 0
            L3_1_running_sparsity = 0
            L3_2_running_sparsity = 0
            L3_3_running_sparsity = 0
            N = 0

            total = 0
            correct = 0
            correct2 = 0
            correct3 = 0
            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device, dtype=torch.float32)
                targets_oh = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=10).to(device, dtype=torch.float32)
                test_spike_data = data.to(device)

                # forward pass
                test_output, LiF3_1, _, LiF1_1, LiF2_1 = net(test_spike_data)
                test_output2, LiF3_2, _, LiF1_2, LiF2_2 = net2(test_spike_data)
                test_output3, LiF3_3, _, LiF1_3, LiF2_3 = net3(test_spike_data)


                # calculate layer sparsity 
                L1_1_running_sparsity += torch.mean(LiF1_1).item()
                L2_1_running_sparsity += torch.mean(LiF2_1).item()
                L3_1_running_sparsity += torch.mean(LiF3_1).item()

                L1_2_running_sparsity += torch.mean(LiF1_2).item()
                L2_2_running_sparsity += torch.mean(LiF2_2).item()
                L3_2_running_sparsity += torch.mean(LiF3_2).item()

                L1_3_running_sparsity += torch.mean(LiF1_3).item()
                L2_3_running_sparsity += torch.mean(LiF2_3).item()
                L3_3_running_sparsity += torch.mean(LiF3_3).item()

                N += 1

                
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
            
            L1_1_running_mean_sparsity += L1_1_running_sparsity/N
            L2_1_running_mean_sparsity += L2_1_running_sparsity/N
            L3_1_running_mean_sparsity += L3_1_running_sparsity/N
            
            L1_2_running_mean_sparsity += L1_2_running_sparsity/N
            L2_2_running_mean_sparsity += L2_2_running_sparsity/N
            L3_2_running_mean_sparsity += L3_2_running_sparsity/N

            L1_3_running_mean_sparsity += L1_3_running_sparsity/N
            L2_3_running_mean_sparsity += L2_3_running_sparsity/N
            L3_3_running_mean_sparsity += L3_3_running_sparsity/N

            test_acc = 100 * correct / total
            acc_sum+=test_acc

            test_acc2 = 100 * correct2 / total
            acc_sum2+=test_acc2

            test_acc3 = 100 * correct3 / total
            acc_sum3+=test_acc3

    L1_1_sparss.append(L1_1_running_mean_sparsity/no_repeats)
    L2_1_sparss.append(L2_1_running_mean_sparsity/no_repeats)
    L3_1_sparss.append(L3_1_running_mean_sparsity/no_repeats)

    L1_2_sparss.append(L1_2_running_mean_sparsity/no_repeats)
    L2_2_sparss.append(L2_2_running_mean_sparsity/no_repeats)
    L3_2_sparss.append(L3_2_running_mean_sparsity/no_repeats)

    L1_3_sparss.append(L1_3_running_mean_sparsity/no_repeats)
    L2_3_sparss.append(L2_3_running_mean_sparsity/no_repeats)
    L3_3_sparss.append(L3_3_running_mean_sparsity/no_repeats)

    accs.append(acc_sum/no_repeats)
    accs2.append(acc_sum2/no_repeats)
    accs3.append(acc_sum3/no_repeats)
    briers.append(brier_sum/(no_repeats*total))
    briers2.append(brier_sum2/(no_repeats*total))
    briers3.append(brier_sum3/(no_repeats*total))
    briers_sm.append(brier_sum_sm/(no_repeats*total))
    briers2_sm.append(brier_sum2_sm/(no_repeats*total))
    briers3_sm.append(brier_sum3_sm/(no_repeats*total))

    print(f'Finished Threshrold = {v_thr} : Took {time.time() - t}s', flush = True)

result = {'thresholds': v_thrs,
          'L1_1_sparss': L1_1_sparss,
          'L1_2_sparss': L1_2_sparss,
          'L1_3_sparss': L1_3_sparss,
          'L2_1_sparss': L2_1_sparss,
          'L2_2_sparss': L2_2_sparss,
          'L2_3_sparss': L2_3_sparss,
          'L3_1_sparss': L3_1_sparss,
          'L3_2_sparss': L3_2_sparss,
          'L3_3_sparss': L3_3_sparss}

# Save all results to a file
with open(f'Plots/Plot_data/ASSnet_spars_vs_thr_beta{beta}_num_steps{num_steps}_trainedthr_{v_thr}_compare_extra.pkl', 'wb') as f:
    pickle.dump(result, f)

result = {'thresholds': v_thrs,
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
with open(f'Plots/Plot_data/ASSnet_acc_vs_thr_beta{beta}_num_steps{num_steps}_trainedthr_{v_thr}_compare_extra.pkl', 'wb') as f:
    pickle.dump(result, f)


plt.figure()

# Plot the data
plt.plot(v_thrs, L1_1_sparss, label='Layer 1, Threshold = 0.5', color='red')
plt.plot(v_thrs, L1_2_sparss, label='Layer 1, Threshold = 1.5', color='blue')
plt.plot(v_thrs, L1_3_sparss, label='Layer 1, Threshold = 4.5', color='green')

# Plot the data
plt.plot(v_thrs, L2_1_sparss, label='Layer 2, Threshold = 0.5', color='red', linestyle='dashed')
plt.plot(v_thrs, L2_2_sparss, label='Layer 2, Threshold = 1.5', color='blue', linestyle='dashed')
plt.plot(v_thrs, L2_3_sparss, label='Layer 2, Threshold = 4.5', color='green', linestyle='dashed')

# Plot the data
plt.plot(v_thrs, L3_1_sparss, label='Layer 3, Threshold = 0.5', color='red', linestyle='dotted')
plt.plot(v_thrs, L3_2_sparss, label='Layer 3, Threshold = 1.5', color='blue', linestyle='dotted')
plt.plot(v_thrs, L3_3_sparss, label='Layer 3, Threshold = 4.5', color='green', linestyle='dotted')

# Create legend handles
color_legend_handles = [
    plt.Line2D([0], [0], color='red', lw=2),
    plt.Line2D([0], [0], color='blue', lw=2),
    plt.Line2D([0], [0], color='green', lw=2)
]
line_type_legend_handles = [
    plt.Line2D([0], [0], color='black', linestyle='-', lw=2),
    plt.Line2D([0], [0], color='black', linestyle='--', lw=2),
    plt.Line2D([0], [0], color='black', linestyle=':', lw=2)
]

# Create the legends
legend1 = plt.legend(color_legend_handles, ['Threshold = 0.5', 'Threshold = 1.5', 'Threshold = 4.5'], title='Threshold Legend', loc='upper left')
legend2 = plt.legend(line_type_legend_handles, ['Layer 1', 'Layer 2', 'Layer 3'], title='Layer Legend', loc='upper right')

# Add the legends to the plot
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

# Add title and labels
plt.title(f'Sparsity vs Threshold (beta={beta}, num_steps={num_steps})')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')

# Save the plot to the specified file path
plt.savefig(f'Plots/ASSnet_Sparsity_vs_Thr_beta{beta}_num_steps{num_steps}.png')

plt.figure()

# Plot the data
plt.plot(v_thrs, torch.mean(torch.tensor([L1_1_sparss, L2_1_sparss, L3_1_sparss]), dim = 0), label='Threshold = 0.5', color='red')
plt.plot(v_thrs, torch.mean(torch.tensor([L1_2_sparss, L2_2_sparss, L3_2_sparss]), dim = 0), label='Threshold = 1.5', color='blue')
plt.plot(v_thrs, torch.mean(torch.tensor([L1_3_sparss, L2_3_sparss, L3_3_sparss]), dim = 0), label='Threshold = 4.5', color='green')

# Add title and labels
plt.title(f'Model Sparsity vs Threshold (beta={beta}, num_steps={num_steps})')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')

# Save the plot to the specified file path
plt.savefig(f'Plots/ASSnet_Sparsity_vs_Thr_beta{beta}_num_steps{num_steps}_model.png')


## ACC vs THR
# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for different thresholds
plt.plot(v_thrs, accs, label='Threshold = 0.5')
plt.plot(v_thrs, accs2, label='Threshold = 1.5')
plt.plot(v_thrs, accs3, label='Threshold = 4.5')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Threshold vs Accuracy (beta={beta}, num_steps={num_steps})')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')

# Save the combined plot to the specified file path
plt.savefig(f'Plots/ASSnet_Sparsity_vs_Acc_beta{beta}_num_steps{num_steps}.png')


### Now for FT


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


#####
## Now vary and plot thr against acc

no_repeats = 5
v_thrs = np.linspace(0, 11, 111)
net.eval()
net2.eval()
net3.eval()

L1_1_sparss = []
L1_2_sparss = []
L1_3_sparss = []
L2_1_sparss = []
L2_2_sparss = []
L2_3_sparss = []
L3_1_sparss = []
L3_2_sparss = []
L3_3_sparss = []

accs = []
accs2 = []
accs3 = []
briers = []
briers2 = []
briers3 = []
briers_sm = []
briers2_sm = []
briers3_sm = []


for v_thr in v_thrs:
    t = time.time()

    v_thr = torch.tensor(v_thr)
    net.LiF_1.threshold = v_thr
    net.LiF_2.threshold = v_thr
    net.LiF_3.threshold = v_thr 

    net2.LiF_1.threshold = v_thr
    net2.LiF_2.threshold = v_thr
    net2.LiF_3.threshold = v_thr 

    net3.LiF_1.threshold = v_thr
    net3.LiF_2.threshold = v_thr
    net3.LiF_3.threshold = v_thr 

    
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

        L1_1_running_mean_sparsity = 0
        L1_2_running_mean_sparsity = 0
        L1_3_running_mean_sparsity = 0
        L2_1_running_mean_sparsity = 0
        L2_2_running_mean_sparsity = 0
        L2_3_running_mean_sparsity = 0
        L3_1_running_mean_sparsity = 0
        L3_2_running_mean_sparsity = 0
        L3_3_running_mean_sparsity = 0
        for _ in range(no_repeats):
            L1_1_running_sparsity = 0
            L1_2_running_sparsity = 0
            L1_3_running_sparsity = 0
            L2_1_running_sparsity = 0
            L2_2_running_sparsity = 0
            L2_3_running_sparsity = 0
            L3_1_running_sparsity = 0
            L3_2_running_sparsity = 0
            L3_3_running_sparsity = 0
            N = 0

            total = 0
            correct = 0
            correct2 = 0
            correct3 = 0
            for data, targets in testloader:
                data = data.to(device)
                targets = targets.to(device, dtype=torch.float32)
                targets_oh = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=10).to(device, dtype=torch.float32)
                test_spike_data = data.to(device)

                # forward pass
                test_output, LiF3_1, _, LiF1_1, LiF2_1 = net(test_spike_data)
                test_output2, LiF3_2, _, LiF1_2, LiF2_2 = net2(test_spike_data)
                test_output3, LiF3_3, _, LiF1_3, LiF2_3 = net3(test_spike_data)


                # calculate layer sparsity 
                L1_1_running_sparsity += torch.mean(LiF1_1).item()
                L2_1_running_sparsity += torch.mean(LiF2_1).item()
                L3_1_running_sparsity += torch.mean(LiF3_1).item()

                L1_2_running_sparsity += torch.mean(LiF1_2).item()
                L2_2_running_sparsity += torch.mean(LiF2_2).item()
                L3_2_running_sparsity += torch.mean(LiF3_2).item()

                L1_3_running_sparsity += torch.mean(LiF1_3).item()
                L2_3_running_sparsity += torch.mean(LiF2_3).item()
                L3_3_running_sparsity += torch.mean(LiF3_3).item()

                N += 1

                
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
            
            L1_1_running_mean_sparsity += L1_1_running_sparsity/N
            L2_1_running_mean_sparsity += L2_1_running_sparsity/N
            L3_1_running_mean_sparsity += L3_1_running_sparsity/N
            
            L1_2_running_mean_sparsity += L1_2_running_sparsity/N
            L2_2_running_mean_sparsity += L2_2_running_sparsity/N
            L3_2_running_mean_sparsity += L3_2_running_sparsity/N

            L1_3_running_mean_sparsity += L1_3_running_sparsity/N
            L2_3_running_mean_sparsity += L2_3_running_sparsity/N
            L3_3_running_mean_sparsity += L3_3_running_sparsity/N

            test_acc = 100 * correct / total
            acc_sum+=test_acc

            test_acc2 = 100 * correct2 / total
            acc_sum2+=test_acc2

            test_acc3 = 100 * correct3 / total
            acc_sum3+=test_acc3

    L1_1_sparss.append(L1_1_running_mean_sparsity/no_repeats)
    L2_1_sparss.append(L2_1_running_mean_sparsity/no_repeats)
    L3_1_sparss.append(L3_1_running_mean_sparsity/no_repeats)

    L1_2_sparss.append(L1_2_running_mean_sparsity/no_repeats)
    L2_2_sparss.append(L2_2_running_mean_sparsity/no_repeats)
    L3_2_sparss.append(L3_2_running_mean_sparsity/no_repeats)

    L1_3_sparss.append(L1_3_running_mean_sparsity/no_repeats)
    L2_3_sparss.append(L2_3_running_mean_sparsity/no_repeats)
    L3_3_sparss.append(L3_3_running_mean_sparsity/no_repeats)

    accs.append(acc_sum/no_repeats)
    accs2.append(acc_sum2/no_repeats)
    accs3.append(acc_sum3/no_repeats)
    briers.append(brier_sum/(no_repeats*total))
    briers2.append(brier_sum2/(no_repeats*total))
    briers3.append(brier_sum3/(no_repeats*total))
    briers_sm.append(brier_sum_sm/(no_repeats*total))
    briers2_sm.append(brier_sum2_sm/(no_repeats*total))
    briers3_sm.append(brier_sum3_sm/(no_repeats*total))

    print(f'Finished Threshrold = {v_thr} : Took {time.time() - t}s', flush = True)

result = {'thresholds': v_thrs,
          'L1_1_sparss': L1_1_sparss,
          'L1_2_sparss': L1_2_sparss,
          'L1_3_sparss': L1_3_sparss,
          'L2_1_sparss': L2_1_sparss,
          'L2_2_sparss': L2_2_sparss,
          'L2_3_sparss': L2_3_sparss,
          'L3_1_sparss': L3_1_sparss,
          'L3_2_sparss': L3_2_sparss,
          'L3_3_sparss': L3_3_sparss}

# Save all results to a file
with open(f'Plots/Plot_data/FT_spars_vs_thr_beta{beta}_num_steps{num_steps}_trainedthr_{v_thr}_compare_extra.pkl', 'wb') as f:
    pickle.dump(result, f)

result = {'thresholds': v_thrs,
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
with open(f'Plots/Plot_data/FT_acc_vs_thr_beta{beta}_num_steps{num_steps}_trainedthr_{v_thr}_compare_extra.pkl', 'wb') as f:
    pickle.dump(result, f)


plt.figure()

# Plot the data
plt.plot(v_thrs, L1_1_sparss, label='Layer 1, Threshold = 0.5', color='red')
plt.plot(v_thrs, L1_2_sparss, label='Layer 1, Threshold = 1.5', color='blue')
plt.plot(v_thrs, L1_3_sparss, label='Layer 1, Threshold = 4.5', color='green')

# Plot the data
plt.plot(v_thrs, L2_1_sparss, label='Layer 2, Threshold = 0.5', color='red', linestyle='dashed')
plt.plot(v_thrs, L2_2_sparss, label='Layer 2, Threshold = 1.5', color='blue', linestyle='dashed')
plt.plot(v_thrs, L2_3_sparss, label='Layer 2, Threshold = 4.5', color='green', linestyle='dashed')

# Plot the data
plt.plot(v_thrs, L3_1_sparss, label='Layer 3, Threshold = 0.5', color='red', linestyle='dotted')
plt.plot(v_thrs, L3_2_sparss, label='Layer 3, Threshold = 1.5', color='blue', linestyle='dotted')
plt.plot(v_thrs, L3_3_sparss, label='Layer 3, Threshold = 4.5', color='green', linestyle='dotted')

# Create legend handles
color_legend_handles = [
    plt.Line2D([0], [0], color='red', lw=2),
    plt.Line2D([0], [0], color='blue', lw=2),
    plt.Line2D([0], [0], color='green', lw=2)
]
line_type_legend_handles = [
    plt.Line2D([0], [0], color='black', linestyle='-', lw=2),
    plt.Line2D([0], [0], color='black', linestyle='--', lw=2),
    plt.Line2D([0], [0], color='black', linestyle=':', lw=2)
]

# Create the legends
legend1 = plt.legend(color_legend_handles, ['Threshold = 0.5', 'Threshold = 1.5', 'Threshold = 4.5'], title='Threshold Legend', loc='upper left')
legend2 = plt.legend(line_type_legend_handles, ['Layer 1', 'Layer 2', 'Layer 3'], title='Layer Legend', loc='upper right')

# Add the legends to the plot
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

# Add title and labels
plt.title(f'Sparsity vs Threshold (beta={beta}, num_steps={num_steps})')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')

# Save the plot to the specified file path
plt.savefig(f'Plots/FT_Sparsity_vs_Thr_beta{beta}_num_steps{num_steps}.png')

plt.figure()

# Plot the data
plt.plot(v_thrs, torch.mean(torch.tensor([L1_1_sparss, L2_1_sparss, L3_1_sparss]), dim = 0), label='Threshold = 0.5', color='red')
plt.plot(v_thrs, torch.mean(torch.tensor([L1_2_sparss, L2_2_sparss, L3_2_sparss]), dim = 0), label='Threshold = 1.5', color='blue')
plt.plot(v_thrs, torch.mean(torch.tensor([L1_3_sparss, L2_3_sparss, L3_3_sparss]), dim = 0), label='Threshold = 4.5', color='green')

# Add title and labels
plt.title(f'Model Sparsity vs Threshold (beta={beta}, num_steps={num_steps})')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')

# Save the plot to the specified file path
plt.savefig(f'Plots/FT_Sparsity_vs_Thr_beta{beta}_num_steps{num_steps}_model.png')


## ACC vs THR
# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for different thresholds
plt.plot(v_thrs, accs, label='Threshold = 0.5')
plt.plot(v_thrs, accs2, label='Threshold = 1.5')
plt.plot(v_thrs, accs3, label='Threshold = 4.5')

# Add legend
plt.legend()

# Add title and labels
plt.title(f'Threshold vs Accuracy (beta={beta}, num_steps={num_steps})')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')

# Save the combined plot to the specified file path
plt.savefig(f'Plots/Ft_Sparsity_vs_Acc_beta{beta}_num_steps{num_steps}.png')

