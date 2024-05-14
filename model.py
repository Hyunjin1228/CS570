# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Original_Model(nn.Module):
#     def __init__(self, verbose=False, input_dim = 9):
#         super(Original_Model, self).__init__()
#         self.verbose = verbose
#         self.f = input_dim
#         self.dim = self.f * 5 - 2
#         self.dim1, self.dim2 = (self.dim-10)*2, (self.dim-20)*6
#         # print(self.dim, self.dim1, self.dim2)

#         self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,9,7), stride=1)
#         self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,7), stride=1)
#         self.pool1 = nn.MaxPool2d(3)
        
#         self.pool2 = nn.MaxPool2d(3)
#         self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=128, kernel_size=(6,4), stride=1)
#         self.fc2 = nn.Linear(128, 6, bias=False)


#     def forward(self, x):
#         if self.verbose: print("연산 전:\t", x.size())
#         assert x.size()[1] == 1
#         (x1, x2, x3, x4, x5) = torch.split(x, [self.f-1,self.f-1,self.f,self.f,self.f], dim=2)
#         x1 = F.relu(self.conv1(x1))
#         x2 = F.relu(self.conv1(x2))
#         x3 = F.relu(self.conv1(x3))
#         x4 = F.relu(self.conv1(x4))
#         x5 = F.relu(self.conv1(x5))
#         x = torch.cat([x1, x2, x3, x4, x5], dim=2)
#         if self.verbose: print("conv1 연산 후:\t", x.shape)
#         x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
#         x = self.pool1(x)
#         x = x.view(-1, 2, self.dim1//2, x.shape[2], x.shape[3])
#         if self.verbose: print("pool1 연산 후:\t", x.shape)
#         (x1, x2, x3, x4, x5) = torch.split(x, [self.f-3,self.f-3,self.f-2,self.f-2,self.f-2], dim=2)
#         x1 = F.relu(self.conv2(x1))
#         x2 = F.relu(self.conv2(x2))
#         x3 = F.relu(self.conv2(x3))
#         x4 = F.relu(self.conv2(x4))
#         x5 = F.relu(self.conv2(x5))
#         x = torch.cat([x1, x2, x3, x4, x5], dim=2)
#         if self.verbose: print("conv2 연산 후:\t",x.shape)
#         x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
#         x = self.pool2(x)
#         x = x.view(-1, self.dim2, x.shape[2], x.shape[3])
#         if self.verbose: print("pool2 연산 후:\t", x.shape)
#         x = F.relu(self.conv3(x))
#         if self.verbose: print("conv3 연산 후:\t", x.shape)
#         x = x.view(-1, 128)
#         if self.verbose: print("veiw 연산 후:\t", x.shape)

#         x = F.relu(self.fc2(x))
#         if self.verbose: print("fc1 연산 후:\t", x.shape)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

def hardwire_layer(input, device, verbose=False):
    """
    Proprocess the given consecutive input (grayscaled) frames into 5 different styles. 
    input : array of shape (N, frames, height, width)
    ex) TRECVID dataset : (N, 7, 60, 40),  KTH dataset : (N, 9, 80, 60)
    
    ##################################### hardwired frames #####################################
    # content: [[[gray frames], [grad-x frames], [grad-y frames], [opt-x frames], [opt-y frames]], ...]
    # shape:   [[[---- f ----], [----- f -----], [----- f -----], [---- f-1 ---], [---- f-1 ---]], ...]
    #           => total: 5f-2 frames
    ############################################################################################
    """
    assert len(input.shape) == 4 
    if verbose: print("Before hardwired layer:\t", input.shape)
    N, f, h, w = input.shape
    
    hardwired = torch.zeros((N, 5*f-2, h, w)).to(device) 
    input = input.to(device)

    for j in range(f):
        # gray
        hardwired[:,0+j,:,:] = input[:,j,:,:]
        # gradient-x, gradient-y
        hardwired[:,f+j,:,:], hardwired[:,2*f+j,:,:] = torch.gradient(input[:,j,:,:], dim=[2,1]) 
        
    input = input.cpu()
    for i in tqdm(range(N), desc="hardwiring"):
        for j in range(f):
            # optflow-x,y
            if j == f-1: 
                continue
            flow = cv2.calcOpticalFlowFarneback(input[i,j,:,:].numpy(), input[i,j+1,:,:].numpy(),None,0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flow = torch.Tensor(flow).to(device)
            hardwired[i,3*f+j,:,:], hardwired[i,4*f-1+j,:,:] = flow[:,:,0], flow[:,:,1]#torch.Tensor(flow[:,:,0]).to(device), torch.Tensor(flow[:,:,1]).to(device)
    
    hardwired = hardwired.reshape(N, 1, 5*f-2, h, w)
    if verbose: print("After hardwired layer :\t", hardwired.shape)
    return hardwired


#* 3D-CNN Model
class Original_Model(nn.Module):
    """
    3D-CNN model designed by the '3D-CNN for HAR' paper. 
    Input Shape: (N, C_in=1, Dimension=5f-2, Height=h, Width=w)
    """
    def __init__(self, verbose=False, mode='KTH'):
        """You need to give the dimension (*, *, f, h, w) as a parameter."""
        super(Original_Model, self).__init__()
        self.verbose = verbose
        self.mode = mode
        if self.mode == 'KTH':
            self.f = 9
        elif self.mode == 'TRECVID':
            self.f = 7
        else:
            print("This mode is not available. Choose one of KTH or TRECVID.")
            return 
        self.dim = self.f * 5 - 2
        self.dim1, self.dim2 = (self.dim-10)*2, (self.dim-20)*6

        if self.mode == 'KTH':
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,9,7), stride=1)
            self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,7), stride=1)
            self.pool1 = nn.MaxPool2d(3)
            self.pool2 = nn.MaxPool2d(3)
            self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=128, kernel_size=(6,4), stride=1)
            self.fc1 = nn.Linear(128, 6, bias=False)

        elif self.mode == 'TRECVID':
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,7,7), stride=1)
            self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,6), stride=1)
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.MaxPool2d(3)
            self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=128, kernel_size=(7,4), stride=1)
            self.fc1 = nn.Linear(128, 3, bias=False)
        

    def forward(self, x):
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
        x = F.relu(self.fc1(x))
        if self.verbose: print("fc1 연산 후:\t", x.shape)

        return x
