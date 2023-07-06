import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as f
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

csv = 'landmarks.csv'
Data = pd.read_csv(csv)
label = Data['label']
data = Data.iloc[:,2:]

dataT = torch.tensor(data.values).float()
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
            self.input = nn.Linear(24, 120)
            self.fc1 = nn.Linear(120, 240)
            self.fc2 = nn.Linear(240,40)
            self.output = nn.Linear(40, 2)

        def forward(self, x):
            x = f.relu(self.input(x))
            x = f.relu(self.fc1(x))
            x = f.relu(self.fc2(x))
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
    return losses, net

losses, Net = trainModel()
print("final loss = ", str(losses[-1]))
plt.plot(losses, '.-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()



