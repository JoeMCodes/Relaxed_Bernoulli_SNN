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
import ASSnet_model

os.chdir('Experiment_NMNIST')

print(os.getcwd())


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
    torch.backends.cudnn.benchmark = False     # disables the inbuilt cudnn auto-tuner which can introduce randomness

set_seed(42)

def trainer(net, train_loader, test_loader, device, num_epochs, lr, batch_size, temp_schedule):### numsteps now defined in train/ testloader
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = lr, betas = (0.9, 0.999), weight_decay=1e-8)
    loss_hist = []
    test_loss_hist = []
    test_acc_hist = []
    loss_val = 0
    counter = 0
    scaler = torch.cuda.amp.GradScaler()
    temps = temp_schedule

    if len(temps) != num_epochs:
        raise ValueError('Temp schedule must have num_epoch entries')
    
    net.train()
    
    for data, _ in iter(train_loader):
        num_steps = data.shape[0]
        break

    for epoch in range(num_epochs):
        net.temp = temps[epoch]
        iter_counter = 0
        epoch_loss_hist = []
        epoch_test_loss_hist = []
        t = time.time()

        for data, label in iter(train_loader):

            data = data.to(device)
            targets = torch.nn.functional.one_hot(label, num_classes=10).float().to(device)
            spike_data = data.to(device)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output, _, _ = net(spike_data)
                loss_val = temps[epoch] * loss(output, targets)
            scaler.scale(loss_val).backward()
            scaler.step(optimizer)
            scaler.update()

            # Store loss history for future plotting
            epoch_loss_hist.append(loss_val.item())

        with torch.no_grad():
            net.eval()
            total = 0.
            correct = 0.
            for test_data, test_targets in iter(test_loader):
                test_data = test_data.to(device)
                test_targets_oh = torch.nn.functional.one_hot(test_targets, num_classes=10).float().to(device)
                test_targets = test_targets.to(device)
                test_spike_data = test_data.to(device)
            

                # Test set forward pass
                test_output, _, _ = net(test_spike_data)

                # Test set loss
                test_loss = loss(test_output, test_targets_oh)
                epoch_test_loss_hist.append(test_loss.item())

                # calculate total accuracy
                _, predicted = test_output.max(1)
                total += test_targets.size(0)
                correct += (predicted == test_targets).sum().item()

        test_acc = 100 * correct / total
        
        loss_hist.append(np.mean(epoch_loss_hist))
        test_loss_hist.append(np.mean(epoch_test_loss_hist))
        test_acc_hist.append(test_acc)
        print(f'Train Loss for Epoch {epoch+1} is {np.mean(epoch_loss_hist)}: Test Loss is {np.mean(epoch_test_loss_hist)}: Test ACC is {test_acc}: Epoch Time: {time.time() - t}', flush=True)
        if (epoch+1)==20:
            if test_acc < 15.:
                return loss_hist, test_loss_hist, test_acc_hist
    return loss_hist, test_loss_hist, test_acc_hist



def train_and_test(device, train_loader, test_loader, beta, num_epochs, num_steps, batch_size, lr, threshold, temp_schedule): ### numsteps now defined in train/ testloader
    print(f'Started training for beta {beta} temp multi time {num_steps} lr {lr} thr {threshold}', flush=True)
    set_seed(123)

    net = ASSnet_model.Net(beta = beta, temp = 0., threshold=threshold, device = device).to(device) ## Create net

    train_hist, test_hist, test_acc_hist = trainer(net, train_loader, test_loader, num_epochs = num_epochs, device = device, batch_size=batch_size, lr = lr, temp_schedule = temp_schedule) # train net

    torch.save(net.state_dict(), f'ASSnet_tests/Models/ASSnet_beta{beta}_tempmulti_num_epochs{num_epochs}_num_steps{num_steps}_batch_size{batch_size}_lr{lr}_thr_{threshold}.pt') ## Save model

    ## Calc test Accuracies
    total = 0
    correct = 0


    with torch.no_grad():
        net.eval()
        net.temp = tensor(0.)
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device, dtype=torch.float32)
            test_spike_data = data.to(device)

            # forward pass
            test_output, _, _ = net(test_spike_data)

            # calculate total accuracy
            _, predicted = test_output.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        test_acc = 100 * correct / total

        result = {
            'beta': beta,
            'timestep': num_steps,
            'test_acc': test_acc,
            'train_hist': train_hist,
            'test_hist': test_hist,
            'test_acc_hist': test_acc_hist
        }

        print(f'Test Accuracy: {test_acc}', flush=True)

        # Save plot
        plt.figure()
        plt.plot(train_hist, label='Train Hist')
        plt.plot(test_hist, label='Test Hist')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train Hist vs Test Hist (beta={beta}, temp=multi, timestep={num_steps})')
        plt.savefig(f'ASSnet_tests/Results/ASSnet_beta{beta}_tempmulti_num_epochs{num_epochs}_num_steps{num_steps}_batch_size{batch_size}_lr{lr}_thr_{threshold}.png')
        plt.close()

        # Save all results to a file
        with open(f'ASSnet_tests/Results/ASSnet_beta{beta}_tempmulti_num_epochs{num_epochs}_num_steps{num_steps}_batch_size{batch_size}_lr{lr}_thr_{threshold}.pkl', 'wb') as f:
            pickle.dump(result, f)

    return result

device = torch.device('cuda:7')


### Load the Data

batch_size = 128
timestep = 30

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.Compose([
                                    transforms.ToFrame(sensor_size=sensor_size,
                                                        n_time_bins=timestep) 
                                    ])
### n_time_bins is How many timesteps

trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)


class BinarizeTransform:
    def __call__(self, tensor):
        return (tensor > 0).float()

transform = tonic.transforms.Compose([torch.from_numpy,
                                    BinarizeTransform()])



cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cacheAN/nmnist/train')
cached_testset = DiskCachedDataset(testset, transform=transform, cache_path='./cacheAN/nmnist/test')

trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, drop_last=True)
testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

num_epochs = 100

# ## Temp schedule must have num_epoch entries
temp_sched = np.linspace(1., 0.1, num_epochs, endpoint=False)

for threshold in [0.5, 1.5, 4.5]:
    train_and_test(device=device, train_loader=trainloader, test_loader=testloader, beta=0.9, num_epochs=num_epochs, num_steps=timestep, batch_size=batch_size, lr=0.001, threshold = threshold, temp_schedule = temp_sched)
