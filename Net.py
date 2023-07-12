import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as f
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0')



csv = 'landmarks.csv'
Data = pd.read_csv(csv)
label = Data['label']
data = Data.iloc[:,2:]

dataT = torch.tensor(data.values).float()
labelsT = torch.tensor(label).long()

train_data, test_data, train_labels, test_labels = train_test_split(dataT, labelsT, test_size=0.2)
Test_data = test_data
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

batchsize = 256
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
    optimizer = optim.SGD(net.parameters(), lr = 0.001)

    return net, lossfun, optimizer

def trainModel():
    numepochs = 40000
    net, lossfun, optimizer = createNN()
    net.to(device)

    losses = torch.zeros(numepochs)


    for epochi in range(numepochs):
        batchLoss  = []
        for X, y in train_loader: 
            X = X.to(device)
            y = y.to(device)
            yHat = net(X)
            loss = lossfun(yHat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchLoss.append(loss.item())
        losses[epochi] = np.mean(batchLoss)
    net.cpu()
    return losses, net

losses, Net = trainModel()
torch.save(Net.state_dict(), 'model1.pt')
pred = Net(Test_data)
predictions = torch.max(pred,1)[1]
print(predictions)
print(test_labels)
misclassified = np.where(predictions != test_labels)[0]

# total accuracy
totalacc = 1 - len(misclassified)/len(test_labels)
print(misclassified)
print('Final accuracy: %g' %totalacc)

print("final loss = ", str(losses[-1]))
plt.plot(losses, '.-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


