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

# image, landmarks = dataset[0]
# landmarks = (landmarks+0.5)*224
# plt.figure(figsize=(10, 10))
# plt.imshow(image.numpy().squeeze().transpose(1,2,0));
# plt.scatter(landmarks[:,0], landmarks[:,1], s=8);
# plt.show()

len_valid_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_valid_set
train_dataset , valid_dataset,  = torch.utils.data.random_split(dataset , [len_train_set, len_valid_set])
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=20, shuffle=True, num_workers=4)
start_time = time.time()

with torch.no_grad():

    best_network = Network()
    best_network.cuda()
    best_network.load_state_dict(torch.load('./content/weights/face_landmarks26.pth')) 
    best_network.eval()
    
    images, landmarks = next(iter(valid_loader))
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0].numpy().squeeze().transpose(1,2,0));
    # plt.scatter(landmarks[:,0], landmarks[:,1], s=8);
    # plt.show()

    print (images.size())
    print (type(images[0]))
    
    
    images = images.cuda()
    landmarks = (landmarks +0.5) * 224
    # plt.figure(figsize=(10, 40))
    # plt.imshow(images[0].cpu().numpy().squeeze().transpose(1,2,0));
    # # plt.scatter(landmarks[:,0], landmarks[:,1], s=8);
    # plt.show()

    print (images[0].cpu().size())
    predictions = (best_network(images).cpu()+0.5 ) * 224
    predictions = predictions.view(-1,148,2)
    
    plt.figure(figsize=(10,40))
    
    for img_num in range(20):
        # plt.subplot(8,1,img_num+1)
        plt.imshow(images[img_num].cpu().numpy().transpose(1,2,0).squeeze())
        plt.scatter(predictions[img_num,:,0], predictions[img_num,:,1], c = 'r', s = 5)
        # plt.scatter(landmarks[img_num,:,0], landmarks[img_num,:,1], c = 'g', s = 5)
        plt.show()

print('Total number of test images: {}'.format(len(valid_dataset)))

end_time = time.time()
print("Elapsed Time : {}".format(end_time - start_time)) 