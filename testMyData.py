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

image = cv2.imread('./content/mydata/myphoto8.jpg', 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
image = TF.to_tensor(image)
image = TF.normalize(image, [0.5], [0.5])


# plt.imshow(image.numpy().squeeze().transpose(1,2,0));
# plt.show()

input_batch = image.unsqueeze(0)
# arr = []
# arr.append(input_batch)
# result = np.array(arr)


with torch.no_grad():

    best_network = Network()
    best_network.cuda()
    best_network.load_state_dict(torch.load('./content/weights/face_landmarks3.pth')) 
    best_network.eval()
    
    
    # image = image.cuda()

    input_batch = input_batch.cuda()
    predictions = (best_network(input_batch).cpu()+0.5 ) * 224
    predictions = predictions.view(-1,148,2)
    
    plt.figure(figsize=(10,40))

    plt.imshow(input_batch[0].cpu().numpy().transpose(1,2,0).squeeze())
    plt.scatter(predictions[0,:,0], predictions[0,:,1], c = 'r', s = 5)
    plt.show()
    


print('Total number of test images: {}'.format(len(valid_dataset)))

end_time = time.time()
print("Elapsed Time : {}".format(end_time - start_time)) 