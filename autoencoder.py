import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn 
import torch.nn.functional as f 

import matplotlib.pyplot as plt


data = np.loadtxt(open('DATA.csv', 'rb'), delimiter=',')
#remove labels
data = data[:, 1:]


dataNorm = data / np.max(data)
dataT = torch.tensor(dataNorm).float()

#create autoencoder networks
def createAE():
    
    class AEnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(1422, 400)
            self.encode = nn.Linear(400,150)
            self.bottleneck = nn.Linear(150, 400)
            self.decode = nn.Linear(400,1422) 
        def forward(self, x):
            x = f.relu(self.input(x))
            x = f.relu(self.encode(x))
            x = f.relu(self.bottleneck(x))
            y = torch.sigmoid(self.decode(x))
            return y 
    net = AEnet()
    lossfun = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = .001)
    return net, lossfun, optimizer

def trainModel():
    numofepoch = 100
    net, lossfun, optimizer = createAE() 
    losses = torch.zeros(numofepoch)

    for epochi in range(numofepoch):
        randomidx = np.random.choice(dataT.shape[0], size = 512)
        X = dataT[randomidx, :]

        yHat = net(X)
        loss = lossfun(yHat, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses[epochi] = loss.item()

    return losses, net

losses, Net = trainModel()
print("final loss = ", str(losses[-1]))
plt.plot(losses, '.-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()