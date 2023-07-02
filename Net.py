import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F 
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np

csv = 'DATA.csv'
data = pd.read_csv(csv)
label = data[:,0]
data = data[:,1:]

dataT = torch.tensor(data).float()
labelsT = torch.tensor(label).long()

train_data, test_data, train_labels, test_labels = train_test_split(dataT, labelsT, test_size=0.8)
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

batchsize = 128
train_loader =  DataLoader(train_data, batch_size=batchsize, shuffle=True)
test_loader =  DataLoader(test_data, batch_size=batchsize, shuffle=True)

def createNN():
    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(1422, 711)
            self.fc1 = nn.Linear(711, 237)
            self.fc2 = nn.Linear(237,40)
            self.output = nn.Linear(40, 3)

        def forward(self, x):
            x = self.relu(self.input(x))
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return torch.log_softmax(self.output(x), axis = 1)
    net = FFN()
    lossfun = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.01)

    return net, lossfun, optimizer

def trainModel():
    numepochs = 100
    net, lossfun, optimizer = createNN()

    losses = torch.zeros(numepochs)


    for epochi in range(numepochs):
        batchLoss  = []
        for X, y in train_loader: 
            yHat = net(X)
            loss = lossfun(yHat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batchLoss.append(loss.item())
        losses[epochi] = np.mean(batchLoss)

