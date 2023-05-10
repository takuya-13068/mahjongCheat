import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from PIL import Image
import sys


class MyCNN(torch.nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        
        #self.fc1 = nn.LazyLinear(out_features=128)
        self.fc1 = nn.Linear(in_features=2*33, out_features=128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(in_features=64, out_features=3)
    
    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

path = 'data/val.csv'
df = pd.read_csv(path)
data_paths = df['file']
inputs = df['data'].tolist()
for i in range(len(inputs)):
    inputs[i] = [float(x.strip()) for x in inputs[i].split(' ')]
inputs = torch.tensor(inputs)
labels = df['label'].tolist()
#print(labels)

args = sys.argv
# Load model
loaded_model = torch.load(args[1], map_location=torch.device('cpu')) # for cpu


for i in range(len(labels)):
    val_data = inputs[i]
    val_label= labels[i]
    val_path = data_paths[i]

    # Predict
    softmax = torch.nn.Softmax(dim=-1)
    output = softmax(loaded_model(val_data))

    if torch.argmax(output)== val_label:
        print('Correct!')
    else :
        print('False! num: '+str(i) + ', label:'+str(val_label)+ ', path:'+val_path)
        

'''
print('stand:', '{:.2f}'.format(output[0].item()*100), '%')
print('hand:', '{:.2f}'.format(output[1].item()*100), '%')
print('down:', '{:.2f}'.format(output[2].item()*100), '%')

if torch.argmax(output)==0:
  print('\nThis is a stand.')
elif torch.argmax(output)==1:
  print('\nThis is a hands-up.')
elif torch.argmax(output)==2:
  print('\nThis is a down.')
  '''