import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import matplotlib.image as mpimg
from collections import OrderedDict
from math import *
import xml.etree.ElementTree as ET 

import torch
import torchvision
from torch.utils.data import Dataset

from transforms import Transforms

from os import walk
import imghdr
class FaceLandmarksDataset(Dataset):

    def __init__(self, transform=None):

        self.image_filenames = []
        self.landmarks = []
        self.crops=[]
        self.transform = transform
        self.size=[]
        self.root_dir = './content/'
        with open('./content/wflw/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt') as file:
            rawdata= file.readlines()
            for i in range(7499):
                datafile=rawdata[i]
                points=datafile.split(' ')
                landmark = []
                xlist=[]
                ylist=[]
                for i in range(98):
                    x=points[i*2]
                    y=points[i*2+1]
                    landmark.append([floor(float(x)), floor(float(y[:-1]))])
                    xlist.append(float(x))
                    ylist.append(float(y))
                    
                    if i>92: # for the strengthened learning
                        for j in range(10):
                            landmark.append([floor(float(x)), floor(float(y[:-1]))])
        
                cropattr={}
                tempwidth=max(xlist)-min(xlist)
                tempheight=max(ylist)-min(ylist)
                cropattr['top']=min(ylist)-tempheight*1
                cropattr['left']=min(xlist)-tempwidth*1
                cropattr['height']=tempheight*3
                cropattr['width']=tempwidth*3
                filepath=os.path.join('./content/wflw/images/', points[-1].strip())
                if not os.path.isfile(filepath):
                    continue
                self.landmarks.append(landmark)
                self.crops.append(cropattr)
                self.image_filenames.append(filepath)


        with open('./content/wflw/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt') as file:
            rawdata= file.readlines()
            for i in range(2499):
                datafile=rawdata[i]
                points=datafile.split(' ')
                landmark = []
                xlist=[]
                ylist=[]
                for i in range(98):
                    x=points[i*2]
                    y=points[i*2+1]
                    landmark.append([floor(float(x)), floor(float(y[:-1]))])
                    xlist.append(float(x))
                    ylist.append(float(y))
                    if i>92:
                        for j in range(10):
                            landmark.append([floor(float(x)), floor(float(y[:-1]))])
                cropattr={}
                tempwidth=max(xlist)-min(xlist)
                tempheight=max(ylist)-min(ylist)
                cropattr['top']=min(ylist)-tempheight*1
                cropattr['left']=min(xlist)-tempwidth*1
                cropattr['height']=tempheight*3
                cropattr['width']=tempwidth*3
                filepath=os.path.join('./content/wflw/images/', points[-1].strip())
                if not os.path.isfile(filepath):
                    continue
                self.landmarks.append(landmark)
                self.crops.append(cropattr)
                self.image_filenames.append(filepath)

        self.landmarks = np.array(self.landmarks).astype('float32')     

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # image = cv2.imread(self.image_filenames[index], 0)
        image = cv2.imread(self.image_filenames[index],1)
        print(self.image_filenames[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = self.landmarks[index]
        cropcur=self.crops[index]
        if self.transform:
            image, landmarks = self.transform(image, landmarks, cropcur)

        landmarks = landmarks-0.5

        return image, landmarks



dataset = FaceLandmarksDataset(Transforms())

for i in range(100):
    image, landmarks = dataset[i]
    landmarks = (landmarks+0.5)*224
    plt.figure(figsize=(10, 10))

    plt.imshow(image.numpy().squeeze().transpose(1,2,0));

    plt.scatter(landmarks[:,0], landmarks[:,1], s=8);
    plt.show()