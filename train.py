import time
import cv2
import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import matplotlib.image as mpimg
from collections import OrderedDict
from skimage import io, transform
from math import *
import xml.etree.ElementTree as ET 

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transforms import Transforms
from dataset import FaceLandmarksDataset
# from model import Network
from model_mobile import Network


dataset = FaceLandmarksDataset(Transforms())

# Visualize Train Transforms
# image, landmarks = dataset[0]
# landmarks = (landmarks+0.5)*224
# plt.figure(figsize=(10, 10))
# plt.imshow(image.numpy().squeeze(), cmap='gray');

# plt.scatter(landmarks[:,0], landmarks[:,1], s=8);
# image, landmarks = dataset[0]
# landmarks = (landmarks+0.5)*224
# plt.figure(figsize=(10, 10))

# plt.imshow(image.numpy().squeeze().transpose(1,2,0));

# plt.scatter(landmarks[:,0], landmarks[:,1], s=8);
# plt.show()


#implement cross valiation
# split the dataset into validation and test sets
len_valid_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_valid_set

# print("The length of Train set is {}".format(len_train_set))
# print("The length of Valid set is {}".format(len_valid_set))

train_dataset , valid_dataset,  = torch.utils.data.random_split(dataset , [len_train_set, len_valid_set])

# shuffle and batch the datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)

images, landmarks = next(iter(train_loader))

# print(images.shape)
# print(landmarks.shape)

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.5f " % (step, total_step, loss))   
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.5f " % (step, total_step, loss))
        
    sys.stdout.flush()

torch.autograd.set_detect_anomaly(True)
network = Network()
network.cuda()
network.load_state_dict(torch.load('./content/weights/face_landmarks_best2.pth'))

criterion = nn.MSELoss()
# criterion = nn.GaussianNLLLoss()
optimizer = optim.Adam(network.parameters(), lr=0.0005)

loss_min = np.inf
num_epochs = 300

start_time = time.time()
losslog=[]
for epoch in range(1,num_epochs+1):
    
    loss_train = 0
    loss_valid = 0
    running_loss = 0
    
    network.train()
    for step in range(1,len(train_loader)+1):
    
        images, landmarks = next(iter(train_loader))
        
        images = images.cuda()
        landmarks = landmarks.view(landmarks.size(0),-1).cuda() 
        
        predictions = network(images)
        
        # clear all the gradients before calculating them
        optimizer.zero_grad()
        
        # find the loss for the current step
        
        loss_train_step = criterion(predictions, landmarks)
        
        # calculate the gradients
        loss_train_step.backward()
        
        # update the parameters
        optimizer.step()
        
        loss_train += loss_train_step.item()
        running_loss = loss_train/step
        
        print_overwrite(step, len(train_loader), running_loss, 'train')
        
    network.eval() 
    with torch.no_grad():
        
        for step in range(1,len(valid_loader)+1):
            
            images, landmarks = next(iter(valid_loader))
        
            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0),-1).cuda()
        
            predictions = network(images)

            # find the loss for the current step
            loss_valid_step = criterion(predictions, landmarks)

            loss_valid += loss_valid_step.item()
            running_loss = loss_valid/step

            print_overwrite(step, len(valid_loader), running_loss, 'valid')
    
    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)
    
    print('\n--------------------------------------------------')
    print('Epoch: {}  Train Loss: {:.5f}  Valid Loss: {:.5f}'.format(epoch, loss_train, loss_valid))
    print('--------------------------------------------------')
    losslog.append(loss_train)
    if loss_valid <= loss_min:
        loss_min = loss_valid
        torch.save(network.state_dict(), './content/weights/face_landmarks{}.pth'.format(epoch)) 
        print("\nMinimum Validation Loss of {:.5f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
        print('Model Saved\n')
     
print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))