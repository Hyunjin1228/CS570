import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import cv2
from pytube import YouTube
from torchvision import transforms

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

    
class RawDataset(Dataset):
    def __init__(self, directory, dataset="train", transform= None) :
        self.instances, self.labels, self.filepath = self.read_dataset(directory, dataset, transform)

        # print(self.instances)
        self.hardwireds = torch.tensor(self.hardwires(self.instances))
        # print(self.hardwireds.shape, self.hardwire_labels.shape)
        self.imgtransform = transform

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        return self.hardwireds[idx], self.labels[idx]

    def read_dataset(self, directory, dataset="train", imgtransform=None):
        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        for video in videos:
            frames = []
            # print(len(video['frames']))
            for frame in video["frames"]:
                if imgtransform:
                    frames.append(imgtransform(frame))
                else:
                    frames.append(frame)
            labels.append(CATEGORY_INDEX[video["category"]])
            instances.append(frames)

#        instances = np.array(instances, dtype=np.float32)
#        print(instances)
#        labels = np.array(labels, dtype=np.uint8)

        return instances, labels, filepath

    def hardwires(self, instances):
        hws = []
        for j in instances:
#            print(i, label)
            hw = self.hardwire(j)
#            print(i, j, hw)
            hws.append(hw)
        hws = np.array(hws, dtype=np.float32)
        return hws


    def hardwire(self, instance):
        w, h, frames = 40, 60, 7
        input = np.zeros((frames, h, w, 3), dtype='float32')  # 7 input 'rgb' frames

#        cap = cv2.VideoCapture(filename)
        hardwires = []
        # print(np.shape(instance))
        for i, frame in enumerate(instance[:-7]):
            for f in range(frames):
#            _, frame = cap.read()
#            print(instance.shape)
               input[f,:,:,:] = instance[i+f]
#        print(input.shape)

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
            #hardwired = torch.from_numpy(hardwired).to(device)  # torch.randn(1, 1, 7, 60, 40)
            hardwires.append(torch.from_numpy(hardwired).to(device))
        return hardwires
    

class BlockFrameDataset(Dataset):
    def __init__(self, directory, dataset="train", transform= None):
        self.instances, self.labels = self.read_dataset(directory, dataset, transform)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)
        self.img_trans = transform

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx], 
            "label": self.labels[idx] 
        }
        # print(sample)
        return self.instances[idx], self.labels[idx] 

    def read_dataset(self, directory, dataset="train", imgtransform=None):
        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        current_block = []
        for video in videos:
            for i, frame in enumerate(video["frames"]):
                if imgtransform:
                    frame = imgtransform(frame)
                current_block.append(frame)
                if len(current_block) % 7 == 0:
                    current_block = np.array(current_block)
                    instances.append(current_block)
                    labels.append(CATEGORY_INDEX[video["category"]])
                    current_block = []

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        return instances, labels