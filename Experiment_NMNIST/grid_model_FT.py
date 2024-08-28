import json
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
from tonic import DiskCachedDataset
# import snntorch as snn
import pickle

dtype = torch.float

os.chdir('Experiment_NMNIST')

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
# Load configuration file
with open('config_FT.json', 'r') as f:
    config = json.load(f)

# Parameters from the configuration file
betas = config['betas']
thresholds = config['thresholds']
temps = config['temps']
timesteps = config['timesteps']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
lrs = config['learning_rates']

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

        out_sum_spike = torch.zeros(batch_size, 10, device=device) ## 10 classes
        
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



def trainer(net, train_loader, test_loader, device, num_epochs, lr, batch_size):### numsteps now defined in train/ testloader
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = lr, betas = (0.9, 0.999), weight_decay=1e-8)
    loss_hist = []
    test_loss_hist = []
    test_acc_hist = []
    loss_val = 0
    counter = 0
    scaler = torch.cuda.amp.GradScaler()
    
    net.train()
    
    for data, _ in iter(train_loader):
        num_steps = data.shape[0]
        break

    for epoch in range(num_epochs):
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
                loss_val = loss(output, targets)
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
    return loss_hist, test_loss_hist, test_acc_hist



def train_and_test(device, train_loader, test_loader, beta, temp, num_epochs, num_steps, batch_size, lr): ### numsteps now defined in train/ testloader
    print(f'Started training for beta {beta} temp {temp} time {num_steps}', flush=True)
    set_seed(42)

    net = Net(beta = beta, temp = temp, threshold=threshold, device = device).to(device) ## Create net

    train_hist, test_hist, test_acc_hist = trainer(net, train_loader, test_loader, num_epochs = num_epochs, device = device, batch_size=batch_size, lr = lr) # train net

    torch.save(net.state_dict(), f'fixed_temp/Models/beta{beta}_temp{temp}_num_epochs{num_epochs}_num_steps{num_steps}_batch_size{batch_size}_lr{lr}.pt') ## Save model

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
            'temp': temp,
            'timestep': num_steps,
            'test_acc': test_acc,
            'train_hist': train_hist,
            'test_hist': test_hist,
            'test_acc_hist': test_acc_hist
        }

        # Save plot
        plt.figure()
        plt.plot(train_hist, label='Train Hist')
        plt.plot(test_hist, label='Test Hist')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train Hist vs Test Hist (beta={beta}, temp={temp}, timestep={num_steps})')
        plt.savefig(f'fixed_temp/Results/beta{beta}_temp{temp}_num_epochs{num_epochs}_num_steps{num_steps}_batch_size{batch_size}_lr{lr}.png')
        plt.close()

        # Save all results to a file
        with open(f'fixed_temp/Results/beta{beta}_temp{temp}_num_epochs{num_epochs}_num_steps{num_steps}_batch_size{batch_size}_lr{lr}.pkl', 'wb') as f:
            pickle.dump(result, f)

    return result



### Train Models
device = torch.device('cuda:7')


for timestep in timesteps:

    ### Load the Data

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

    cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')
    cached_testset = DiskCachedDataset(testset, transform=transform, cache_path='./cache/nmnist/test')

    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, drop_last=True)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

    for temp in temps:
        for beta in betas:
            for learning_rate in lrs:
                for threshold in thresholds:
                    big_t = time.time()
                    train_and_test(device=device, train_loader=trainloader, test_loader=testloader, beta=beta, temp=temp, num_epochs=num_epochs, num_steps=timestep, batch_size=batch_size, lr=learning_rate)
                    print(f'Total Time To Train was {time.time()-big_t}')

