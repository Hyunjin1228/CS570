import argparse
import os
from torchvision import transforms
from dataset import *
from model import *
#import dataset
# from train_helper import *
import torch.optim as optim


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
    parser.add_argument("--width", type=int, default = 60)
    parser.add_argument("--height", type=int, default = 80)
    parser.add_argument("--frames", type=int, default = 9)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    start_epoch = args.start_epoch
    lr = args.lr
    log_interval = args.log
    w = args.width
    h = args.height
    frames = args.frames
    
    if args.cuda == 1:
        cuda = True
    else:
        cuda = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')

    print("Loading dataset")
    
    torchvision_transform = transforms.Compose([
        transforms.Resize((h, w))
    ])
    print("training data")
    train_set = RawDataset(dataset_dir, "train", transform = torchvision_transform, height = h, width = w, frames = frames)
    print("training dataloader")
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    # dev_set = RawDataset(dataset_dir, "dev", transform = torchvision_transform)
    print("test data")
    test_set = TestDataset(dataset_dir, "test", transform = torchvision_transform, height = h, width = w, frames = frames)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # Create model and optimizer.
    model = Original_Model().to(device)
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
    hardwire_size = frames * 5 - 2
    model.train()  # 학습을 위함
    for epoch in range(10):
        print("EPOCH ", epoch)
        for index, (data, target) in enumerate(train_loader):
            # print(data)
            print(data.shape)
            data = torch.reshape(data, (data.shape[0],1,hardwire_size,h,w)).to(device)
            target = target.to(device)
            print(data.shape)
            optimizer.zero_grad()  # 기울기 초기화
            output = model(data, input_dim = frames)
            #print(output)
            print(output.shape)
            print(target)
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
            data = torch.reshape(data, (data.shape[0],1,hardwire_size,h,w)).to(device)
            target = target.to(device)
            output = model(data, frames)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))