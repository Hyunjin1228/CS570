import argparse
import os
from torchvision import transforms
from dataset import *
from model import *
#import dataset
# from train_helper import *
import torch.optim as optim

def hardwires(instances, labels):
    hws = []
    
    for j, label in zip(instances, labels):
        hw, _ = hardwire(j, label)
        hws = hws + hw
    hws = np.array(hws, dtype=np.float32)
    return hws

def hardwire_labels(instances, labels):
    ls = []
    for j, label in zip(instances, labels):
        _, l = hardwire(i, j, label)
        ls = ls + l
    ls = np.array(ls, dtype=np.uint8)
    return ls

def hardwire(instance, label):
    # print(filename)
    w, h, frames = 40, 60, 7
    input = np.zeros((frames, h, w, 3), dtype='float32')  # 7 input 'rgb' frames

#        cap = cv2.VideoCapture(filename)
    hardwires = []
    labels = []
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
        hardwires.append(hardwired)
        labels.append(label)
    return hardwires, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Frame ConvNet")
    parser.add_argument("--dataset_dir", type=str, default="data",
                        help="directory to dataset")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size for training (default: 16)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="number of epochs to train (default: 3)")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="start index of epoch (default: 1)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for training (default: 0.001)")
    parser.add_argument("--log", type=int, default=10,
                        help="log frequency (default: 10 iterations)")
    parser.add_argument("--cuda", type=int, default=0,
                        help="whether to use cuda (default: 0)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    start_epoch = args.start_epoch
    lr = args.lr
    log_interval = args.log
    
    if args.cuda == 1:
        cuda = True
    else:
        cuda = False

    print("Loading dataset")
    
    torchvision_transform = transforms.Compose([
        transforms.Resize((60, 40))
    ])
    print("training data")
    train_set = RawDataset(dataset_dir, "train", transform = torchvision_transform)
    print("training dataloader")
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    # dev_set = RawDataset(dataset_dir, "dev", transform = torchvision_transform)
    print("test data")
    test_set = TestDataset(dataset_dir, "test", transform = torchvision_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # Create model and optimizer.
    model = Original_Model()
#    ensemble_model = Ensemble()
    if start_epoch > 1:
        resume = True
    else:
        resume = False

    # Create directory for storing checkpoints.
    os.makedirs(os.path.join(dataset_dir, "original_model"), exist_ok=True)

    print("Start training")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()  # 학습을 위함
    for epoch in range(10):
        print("EPOCH: ", epoch)
        for index, (data, target) in enumerate(train_loader):
            # print(data)
            # print(data.shape)
            data = torch.reshape(data, (data.shape[0],1,33,60,40))
            optimizer.zero_grad()  # 기울기 초기화
            output = model(data)
            #print(output)
            loss = criterion(output, target)
            loss.backward()  # 역전파
            optimizer.step()

            if index % 100 == 0:
                print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))

    model.eval()  # test case 학습 방지를 위함
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = torch.reshape(data, (data.shape[0],1,33,60,40))
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))