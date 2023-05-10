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


batch_size = 12
vali_split = 0.8
learning_rate = 0.00002
epochs = 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = sys.argv

class MyCNN(torch.nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.fc1 = nn.Linear(in_features=2*33, out_features=2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(in_features=64, out_features=4)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x



class my_dataset(Dataset):
    def __init__(self, csv_path,transform=None):
        #csvファイル読み込み。 data: path, data, label
        df = pd.read_csv(csv_path)
        data_paths = df['file']
        inputs = df['data'].tolist()
        for i in range(len(inputs)):
            inputs[i] = [int(x.strip()) for x in inputs[i].split(' ')]
        inputs = torch.tensor(inputs)
        labels = df['label']

        self.data_paths = data_paths
        self.labels = labels
        self.inputs = inputs
        #self.transform = transform

    def __getitem__(self, index):
        '''#transform事前処理実施
        if self.transform is not None:
            img = self.transform(img)'''

        inputs = self.inputs[index]
        labels = self.labels[index]
        return inputs,labels

    def __len__(self):
        return len(self.data_paths)
    
dataset = my_dataset("data/"+args[1],transform =None)

train_size = (int) (len(dataset) * vali_split)
val_size = len(dataset) - train_size
dataset_sizes = {'train': train_size, 'val': val_size}
dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(dataset_val,   batch_size=batch_size, shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}





# Training algorithm
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #print(dataloaders[phase]) 
            # label: 正解ラベルの配列, data: 4つの66のパラメータがある

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)
                

                # Forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('best model saved')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = MyCNN().to(device)
# Set loss function
criterion = torch.nn.CrossEntropyLoss()
# Set optimizer 
optimizer_ft = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)
best_model = train_model(model, criterion, optimizer_ft, num_epochs=epochs)

# Save the model
torch.save(model, args[2])

#python model.py 'csvname' 'modelname'

'''
print('-' *10)
print('Validation')

path = 'data/val.csv'
df = pd.read_csv(path)
data_paths = df['file']
inputs = df['data'].tolist()
for i in range(len(inputs)):
    inputs[i] = [float(x.strip()) for x in inputs[i].split(' ')]
inputs = torch.tensor(inputs)
print(inputs.shape)
val_data = inputs[10]
print(val_data)
print(val_data.shape)

# Load model
loaded_model = torch.load('hockey_model.h5', map_location=torch.device('cpu')) # for cpu

# Predict
softmax = torch.nn.Softmax(dim=-1)
output = softmax(loaded_model(val_data))

print('stand:', '{:.2f}'.format(output[0].item()*100), '%')
print('hand:', '{:.2f}'.format(output[1].item()*100), '%')
print('down:', '{:.2f}'.format(output[2].item()*100), '%')

if torch.argmax(output)==0:
  print('\nThis is a stand.')
elif torch.argmax(output)==1:
  print('\nThis is a hands-up.')
elif torch.argmax(output)==2:
  print('\nThis is a down.')'''