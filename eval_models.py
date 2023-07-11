import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as f
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

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


csv = 'landmarks.csv'
Data = pd.read_csv(csv)
label = Data['label']
data = Data.iloc[:,2:]

dataT = torch.tensor(data.values).float()
labelsT = torch.tensor(label).long()

train_data, test_data, train_labels, test_labels = train_test_split(dataT, labelsT, test_size=0.2, random_state= True)
Test_data = test_data
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

Net, lossfun, optimizer = createNN()
Net.load_state_dict(torch.load('model1.pt'))
pred = Net(Test_data)
predictions = torch.max(pred,1)[1]
print(predictions)
print(test_labels)
misclassified = np.where(predictions != test_labels)[0]

# total accuracy
totalacc = 1 - len(misclassified)/len(test_labels)
print(misclassified)
print('Final accuracy: %g' %totalacc)