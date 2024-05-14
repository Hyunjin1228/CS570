import argparse
import os
from torchvision import transforms
from dataset import *
from model import *
#import dataset
# from train_helper import *
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def evaluation(dataloader, frames, h, w):
    hardwire_size = frames * 5 - 2 
    test_loss = 0
    correct = 0
    val_loss = []
    with torch.no_grad():
        model.eval()
        for data, target in test_loader:
            data = torch.reshape(data, (data.shape[0],1,hardwire_size,h,w)).to(device)
            target = target.to(device)
            output = model(data, frames)
            test_loss += criterion(output, target).item() # sum up batch loss
            valid_loss = criterion(output,target).detach().cpu().numpy()
            val_loss.append(valid_loss)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                np.mean(val_loss), correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
                
    model.train()
    val_loss = np.mean(val_loss)
    return val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Frame ConvNet")
    parser.add_argument("--dataset_dir", type=str, default="data",
                        help="directory to dataset")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size for training (default: 16)")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="number of epochs to train (default: 10)")
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

    import wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="CS570",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "3D CNN",
        "dataset": "KTH",
        "epochs": args.num_epochs,
        }
    )

    print("Loading dataset")
    
    torchvision_transform = transforms.Compose([
        transforms.Resize((h, w))
    ])

    print("training data")
    train_set = RawDataset(dataset_dir, "train", transform = torchvision_transform, height = h, width = w, frames = frames)
    print(train_set)
    print("training dataloader")
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    #print("val data")
    #val_set = RawDataset(dataset_dir, "dev", transform = torchvision_transform, height = h, width = w, frames = frames)
    #print("val dataloader")
    #val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    print("test data")
    test_set = TestDataset(dataset_dir, "test", transform = torchvision_transform, height = h, width = w, frames = frames)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # k-fold
    validation_loss = []
    hardwire_size = frames * 5 - 2
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_set)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx) # index 생성
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_subsampler, num_workers=4) # 해당하는 index 추출
        valloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_subsampler, num_workers=4)

        model = Original_Model().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for index, (data, target) in enumerate(train_loader):
                inputs = torch.reshape(data, (data.shape[0],1,hardwire_size,h,w)).to(device)
                target = target.to(device)

                #print(data.shape)
                optimizer.zero_grad()  # 기울기 초기화
                output = model(inputs, input_dim = frames)
                #print(output)
                #print(output.shape)
                #print(target)
                loss = criterion(output, target)
                loss.backward()  # 역전파
                optimizer.step()

                if index % 250 == 0:
                    print("loss of {:2d} epoch, {:3d} index : {:.7f}".format(epoch, index, loss.item()))
                    # log metrics to wandb
                    wandb.log({"fold": fold, "epoch": epoch, "train loss": loss.item()})
        
        train_loss = evaluation(trainloader, frames, h, w)
        val_loss = evaluation(valloader, frames, h, w)
        print("k-fold", fold," Train Loss: %.4f, Validation Loss: %.4f" %(train_loss, val_loss)) 
        # log metrics to wandb
        wandb.log({"fold": fold, "train loss": train_loss, "valid loss": val_loss})
        validation_loss.append(val_loss)
    validation_loss = np.array(validation_loss)
    mean = np.mean(validation_loss)
    std = np.std(validation_loss)
    print("Validation Score: %.4f, ± %.4f" %(mean, std))
    
    '''
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
    validation_loss = []
    
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()  # 학습을 위함
    for epoch in range(num_epochs):
        print("EPOCH ", epoch)
        for index, (data, target) in enumerate(train_loader):
            # print(data)
            #print(data.shape)
            data = torch.reshape(data, (data.shape[0],1,hardwire_size,h,w)).to(device)
            target = target.to(device)
            #print(data.shape)
            optimizer.zero_grad()  # 기울기 초기화
            output = model(data, input_dim = frames)
            #print(output)
            #print(output.shape)
            #print(target)
            loss = criterion(output, target)
            loss.backward()  # 역전파
            optimizer.step()

            if index % 100 == 0:
                print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))
    '''
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