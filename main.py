import cv2
import mediapipe as mp
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
from PIL import Image 

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

class MyCNN(torch.nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        '''
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.dropout1 = nn.Dropout(0.5)'''
        
        #self.fc1 = nn.LazyLinear(out_features=128)
        self.fc1 = nn.Linear(in_features=2*33, out_features=128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(in_features=64, out_features=3)
    
    def forward(self, x):
        '''
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.pool1(x)
        x = nn.ReLU()(self.conv3(x))
        x = self.dropout1(x)
        
        x = x.view(x.size(0), -1)
        '''
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

p1Position = {'xmin':0, 'xmax':630, 'ymin':0, 'ymax':1120}
p2Position = {'xmin':1290, 'xmax':1920, 'ymin':0, 'ymax':1120}
 

#cap = cv2.VideoCapture('usa.mp4')
cap = cv2.VideoCapture(0)

pTime = 0

# Load model
loaded_model = torch.load('best_model.h5', map_location=torch.device('cpu')) # for cpu

while True:
    flg, img = cap.read()
    p1 = img[p1Position['ymin']:p1Position['ymax'],p1Position['xmin']:p1Position['xmax']]
    p2 = img[p2Position['ymin']:p2Position['ymax'],p2Position['xmin']:p2Position['xmax']]
    p1RGB = cv2.cvtColor(p1, cv2.COLOR_BGR2RGB)
    p2RGB = cv2.cvtColor(p2, cv2.COLOR_BGR2RGB)
    p1results = pose.process(p1RGB)
    p2results = pose.process(p2RGB)
    p1data, p2data = [], []


    # p1, p2に対して骨格推定を行う
    if p1results.pose_landmarks:
        mpDraw.draw_landmarks(p1, p1results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(p1results.pose_landmarks.landmark):
            h, w, c = p1.shape
            #print(id, lm)
            #print(lm.visibility)
            cx, cy = float(int(lm.x * w)), float(int(lm.y * h))
            if lm.visibility > 0.1:
                cv2.circle(img, (int(cx), int(cy)), 5, (255, 0, 0), cv2.FILLED)
            p1data+=[cx,cy]
    cv2.rectangle(img,(p1Position['xmin'],p1Position['ymin']),(p1Position['xmax'],p1Position['ymax']),(255,0,255),2)
    
    if p2results.pose_landmarks:
        mpDraw.draw_landmarks(p2, p2results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(p2results.pose_landmarks.landmark):
            h, w, c = p2.shape
            cx, cy = float(int(lm.x * w)), float(int(lm.y * h))
            if lm.visibility > 0.1:
                cv2.circle(img, (int(cx) + p2Position['xmin'], int(cy)), 5, (255, 0, 0), cv2.FILLED)
            p2data+=[cx,cy]
    cv2.rectangle(img,(p2Position['xmin'],p2Position['ymin']),(p2Position['xmax'],p2Position['ymax']),(0,255,0),2)
    
    # p1, p2に対してポーズ判定を行う
    p1text = 'p1: '
    if len(p1data) >= 66:
        p1data = torch.tensor(p1data)
        # Predict
        softmax = torch.nn.Softmax(dim=-1)
        output = softmax(loaded_model(p1data))

        if torch.argmax(output)== 0:
            p1text += 'stand'
        elif torch.argmax(output)== 1:
            p1text += 'hand'
        elif torch.argmax(output)== 2:
            p1text += 'down'
        else :
            p1text += 'ERROR!!!!!!!!!!!'
    else:
        p1text += 'not found properly'

    cv2.putText(img, p1text, (p1Position['xmin']+20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (255, 0, 0), 5)

    p2text = 'p2: '
    if len(p2data) >= 66:
        p2data = torch.tensor(p2data)
        # Predict
        softmax = torch.nn.Softmax(dim=-1)
        output = softmax(loaded_model(p2data))

        if torch.argmax(output)== 0:
            p2text += 'stand'
        elif torch.argmax(output)== 1:
            p2text += 'hand'
        elif torch.argmax(output)== 2:
            p2text += 'down'
        else :
            p2text += 'ERROR!!!!!!!!!!!'
    else:
        p2text += 'not found properly'

    cv2.putText(img, p2text, (p2Position['xmin']+20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (255, 0, 0), 5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
    
    cv2.imshow("capture", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


'''
data format
16:9 -> 540:960
'''
