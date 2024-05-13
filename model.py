import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import cv2
from pytube import YouTube
import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

class Original_Model(nn.Module):
    def __init__(self, verbose=True, input_dim = 9):
        super(Original_Model, self).__init__()
        self.verbose = verbose
        self.f = input_dim
        self.dim = self.f * 5 - 2
        self.dim1, self.dim2 = (self.dim-10)*2, (self.dim-20)*6
        print(self.dim, self.dim1, self.dim2)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,7,7), stride=1)
        self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,6), stride=1)
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.dim1, 3, bias=False)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=128, kernel_size=(7,4), stride=1)
        self.fc1 = nn.Linear(128, 6, bias=False)

    def forward(self, x, input_dim):
        print(input_dim)
        if self.verbose: print("연산 전:\t", x.size())
        assert x.size()[1] == 1
        (x1, x2, x3, x4, x5) = torch.split(x, [self.f-1,self.f-1,self.f,self.f,self.f], dim=2)
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.conv1(x2))
        x3 = F.relu(self.conv1(x3))
        x4 = F.relu(self.conv1(x4))
        x5 = F.relu(self.conv1(x5))
        x = torch.cat([x1, x2, x3, x4, x5], dim=2)
        if self.verbose: print("conv1 연산 후:\t", x.shape)
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.pool1(x)
        x = x.view(-1, 2, self.dim1//2, x.shape[2], x.shape[3])
        if self.verbose: print("pool1 연산 후:\t", x.shape)
        (x1, x2, x3, x4, x5) = torch.split(x, [self.f-3,self.f-3,self.f-2,self.f-2,self.f-2], dim=2)
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(self.conv2(x2))
        x3 = F.relu(self.conv2(x3))
        x4 = F.relu(self.conv2(x4))
        x5 = F.relu(self.conv2(x5))
        x = torch.cat([x1, x2, x3, x4, x5], dim=2)
        if self.verbose: print("conv2 연산 후:\t",x.shape)
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.pool2(x)
        x = x.view(-1, self.dim2, x.shape[2], x.shape[3])
        if self.verbose: print("pool2 연산 후:\t", x.shape)
        x = F.relu(self.conv3(x))
        if self.verbose: print("conv3 연산 후:\t", x.shape)
        x = x.view(-1, 128)
        if self.verbose: print("veiw 연산 후:\t", x.shape)
        x = F.relu(self.fc1(x))
        if self.verbose: print("fc1 연산 후:\t", x.shape)
        return x
  

#cnn = Original_Model(verbose=True).to(device)
#summary(cnn, (1,33,60,40))  # Input Size: (N, C_in=1, Dimension=33, Height=60, Width=40)