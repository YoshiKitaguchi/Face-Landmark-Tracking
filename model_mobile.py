import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms

class Network(nn.Module):
    def __init__(self,num_classes=2*148):
        super().__init__()
        self.model_name='mobilenetv2'
        self.model=models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
        # self.model.in_features=nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x