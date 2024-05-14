import argparse
from torchvision import transforms
from dataset import *
from model import *
import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold

# def evaluation(model, dataloader, frames, h, w):
#     hardwire_size = frames * 5 - 2 
#     test_loss = 0
#     correct = 0
#     val_loss = []
#     with torch.no_grad():
#         model.eval()
#         for data, target in dataloader:
#             data = torch.reshape(data, (data.shape[0],1,hardwire_size,h,w)).to(device)
#             target = target.to(device)
#             output = model(data, frames)
#             test_loss += criterion(output, target).item() # sum up batch loss
#             valid_loss = criterion(output,target).detach().cpu().numpy()
#             val_loss.append(valid_loss)
#             pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#         print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
#                 test_loss, correct, len(dataloader.dataset),
#                 100. * correct / len(dataloader.dataset)))
                
#     model.train()
#     val_loss = np.mean(val_loss)
#     print("Total loss: ", val_loss)
#     return val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Frame ConvNet")
    parser.add_argument("--dataset_dir", type=str, default="data",
                        help="directory to dataset")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size for training (default: 16)")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="number of epochs to train (default: 5)")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="start index of epoch (default: 1)")
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="learning rate for training (default: 0.00001)")
    parser.add_argument("--log", type=int, default=10,
                        help="log frequency (default: 10 iterations)")
    parser.add_argument("--cpu_only", action='store_true',
                        help="add the argument to use only cpu, not cuda")
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
    
    print(lr, num_epochs)

    device = torch.device('cuda:1' if torch.cuda.is_available() and not args.cpu_only else 'cpu')
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
    transforms.ToTensor(),
    transforms.Resize((80, 60)),
    transforms.Grayscale(num_output_channels=1)
])


    validation_loss = []
    # hardwire_size = frames * 5 - 2
    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # for fold, (train_idx, val_idx) in enumerate(kfold.split(train_set)):
    for i in range(5):
        seed = random.randint(0, 100)
        train_set = KTHDataset(dataset_dir, type="train", transform= torchvision_transform, frames = 9, seed=seed, device=device)
        test_set = KTHDataset(dataset_dir, type="test", transform= torchvision_transform, frames = 9, seed=seed, device=device)

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True) # 해당하는 index 추출
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=True)

        model = Original_Model(mode='KTH').to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            trainloss = 0
            data_num = 0.0
            for index, (data, target) in enumerate(train_loader):
                inputs = data.to(device)
                # inputs = torch.reshape(data, (data.shape[0],1,hardwire_size,h,w)).to(device)
                target = target.to(device)

                optimizer.zero_grad()  # 기울기 초기화
                output = model(inputs)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                loss = criterion(output, target)
                trainloss += loss.item() * data.shape[0]
                data_num += data.shape[0]
                loss.backward()  # 역전파
                optimizer.step()
                

                if index % 250 == 0:
                    print("loss of {} epoch, {:3d} index : {}".format(epoch, index, loss.item()))
                    
                if index % 10 == 0:
                    # log metrics to wandb
                    name = str(i)+"-th train losss"
                    wandb.log({name: loss.item()})

            print("Epoch {} Loss = {}".format(epoch, trainloss/data_num))
            
    #         hardwire_size = frames * 5 - 2 
    #         test_loss = 0
    #         correct = 0
    #         val_loss = []
    #         with torch.no_grad():
    #             model.eval()
    #             for data, target in test_loader:
    #                 data = torch.reshape(data, (data.shape[0],1,hardwire_size,h,w)).to(device)
    #                 target = target.to(device)
    #                 output = model(data, frames)
    #                 test_loss += criterion(output, target).item() # sum up batch loss
    #                 valid_loss = criterion(output,target).detach().cpu().numpy()
    #                 val_loss.append(valid_loss)
    #                 pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    #                 correct += pred.eq(target.view_as(pred)).sum().item()
    #             print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #                     test_loss, correct, len(test_loader.dataset),
    #                     100. * correct / len(test_loader.dataset)))
                    
    #     model.train()
    #     val_loss = np.mean(val_loss)
    #     print("Validation loss: ", val_loss)
    #     print("k-fold", fold," Validation Loss: %.4f" %(val_loss)) 
    #     # log metrics to wandb
    #     wandb.log({"fold": fold,  "valid loss": val_loss})
    #     validation_loss.append(val_loss)
    # validation_loss = np.array(validation_loss)
    # mean = np.mean(validation_loss)
    # std = np.std(validation_loss)
    # print("Validation Score: %.4f, ± %.4f" %(mean, std))
    
        model.eval()  # test case 학습 방지를 위함
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data, frames)
                test_loss += criterion(output, target).item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset))) 