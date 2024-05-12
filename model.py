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


def Download(link, filename):
    if os.path.exists(filename):
        print("Already downloaded:", filename)
        return
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_lowest_resolution()
    try:
        youtubeObject.download(filename=filename)
    except:
        print("An error has occurred")
    print("Download is completed successfully")

# example video download
#link = "https://youtu.be/GUPu3ilbfbE?feature=shared"
#filename = 'video_example.mp4'
#Download(link, filename)

def read_dataset(directory, dataset="train"):
    if dataset == "train":
            filepath = os.path.join(directory, "train.p")
    elif dataset == "dev":
        filepath = os.path.join(directory, "dev.p")
    else:
        filepath = os.path.join(directory, "test.p")

    videos = pickle.load(open(filepath, "rb"))
 
def hardwire(filename):
    w, h, frames = 40, 60, 7
    input = np.zeros((frames, h, w, 3), dtype='float32')  # 7 input 'rgb' frames

    cap = cv2.VideoCapture(filename)
    for f in range(frames):
        _, frame = cap.read()
        input[f,:,:,:] = frame[100:100+h, 200:200+w, :]
    print(input.shape)

    gray = np.zeros((frames, h, w), dtype='uint8')
    hardwired = np.zeros((33, h,w)) # 7 for gray,gradient-x,y (7x3=21)  +   6 for optflow-x,y (6x2=12)
    for f in range(frames):
        # gray
        gray[f,:,:] = cv2.cvtColor(input[f,:,:,:], cv2.COLOR_BGR2GRAY)
        hardwired[0+f,:,:] = gray[f,:,:]
        # gradient-x, gradient-y
        hardwired[7+f,:,:], hardwired[14+f,:,:] = np.gradient(gray[f,:,:], axis=1), np.gradient(gray[f,:,:], axis=0)

    # optflow-x,y
    for f in range(frames-1):
        mask = np.zeros_like(gray[f,:,:])
        flow = cv2.calcOpticalFlowFarneback(gray[f,:,:],gray[f+1,:,:],None,0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        hardwired[21+f,:,:], hardwired[27+f,:,:] = flow[:,:,0], flow[:,:,1]

    hardwired = torch.from_numpy(hardwired).to(device)  # torch.randn(1, 1, 7, 60, 40)
    return hardwired

class Ensemble(nn.Module):
    def __init__(self, model, output):
        super(Ensemble, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(6, output)
    
    def forward(self, x):
        sum = 0
        for i in x:
            sum = sum + self.model(i)
        x = torch.softmax(x, dim = 1)
        return x

class Original_Model(nn.Module):
  def __init__(self, verbose=False):
    super(Original_Model, self).__init__()
    self.verbose = verbose
    self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,7,7), stride=1)
    self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,6), stride=1)
    self.pool1 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(23*2, 3, bias=False)
    self.pool2 = nn.MaxPool2d(3)
    self.conv3 = nn.Conv2d(in_channels=13*6, out_channels=128, kernel_size=(7,4), stride=1)
    self.fc1 = nn.Linear(128, 6, bias=False)
    
  def forward(self, x):
    if self.verbose: print("연산 전:\t\t", x.size())
#    assert x.size()[2] == 33

    (x1, x2, x3, x4, x5) = torch.split(x, [6,6,7,7,7], dim=2)
#    print(x1.shape)
#    print(x2.shape)
#    print(x3.shape)
#    print(x4.shape)
#    print(x5.shape)

    x1 = F.relu(self.conv1(x1))
    x2 = F.relu(self.conv1(x2))
    x3 = F.relu(self.conv1(x3))
    x4 = F.relu(self.conv1(x4))
    x5 = F.relu(self.conv1(x5))
    x = torch.cat([x1, x2, x3, x4, x5], dim=2)
    if self.verbose: print("conv1 연산 후:\t", x.shape)

    x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
    x = self.pool1(x)
    x = x.view(-1, 2, 23, x.shape[2], x.shape[3])
    if self.verbose: print("pool1 연산 후:\t", x.shape)

    (x1, x2, x3, x4, x5) = torch.split(x, [4,4,5,5,5], dim=2)
    x1 = F.relu(self.conv2(x1))
    x2 = F.relu(self.conv2(x2))
    x3 = F.relu(self.conv2(x3))
    x4 = F.relu(self.conv2(x4))
    x5 = F.relu(self.conv2(x5))
    x = torch.cat([x1, x2, x3, x4, x5], dim=2)
    if self.verbose: print("conv2 연산 후:\t",x.shape)

    x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
    x = self.pool2(x)
    x = x.view(-1, 78, x.shape[2], x.shape[3])
    if self.verbose: print("pool2 연산 후:\t", x.shape)

    x = F.relu(self.conv3(x))
    if self.verbose: print("conv3 연산 후:\t", x.shape)

    x = x.view(-1, 128)
    x = F.relu(self.fc1(x))
    if self.verbose: print("fc1 연산 후:\t", x.shape)

    return x
  

#cnn = Original_Model(verbose=True).to(device)
#summary(cnn, (1,33,60,40))  # Input Size: (N, C_in=1, Dimension=33, Height=60, Width=40)