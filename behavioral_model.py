import torch
import torch.nn as nn
import csv
import numpy as np
import argparse
import os
import cv2
from data_augmentation import augment_data
from training_mode import get_data,preprocess
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--log_file_dir',default=os.path.dirname(os.path.realpath(__file__)),type=dir_path,help='a directory where the log file is located')
    parser.add_argument('--epochs',default=80,type=int)
    parser.add_argument('--batch_size',default=60,type=int)
    
    args= parser.parse_args()
    return args

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs=nn.ModuleList(
            # input_channels, output_channels, kernel_size, stride, padding
            [nn.Conv2d(1,24,9,stride=2),
            nn.Conv2d(24,48,5,stride=2),
            nn.Conv2d(48,64,5,stride=2),
            nn.Conv2d(64,128,3,stride=2),
            ])
        flattened_dim=128*3*17
        self.mlp=nn.Sequential(
            nn.Linear(flattened_dim,1000),nn.ReLU(),nn.BatchNorm1d(1000),
            nn.Linear(1000,10),nn.ReLU(),
            nn.Linear(10,4),nn.ReLU(),
            nn.Linear(4,2))
        self.drop=nn.Dropout2d(p=0.5)
    def forward(self,obs):
        obs=obs/255.-0.5
        conv1=torch.relu(self.convs[0](obs))
        conv2=torch.relu(self.convs[1](conv1))
        conv3=torch.relu(self.convs[2](conv2))
        conv4=torch.relu(self.convs[3](conv3))
        drop1=self.drop(conv4)
        h=drop1.flatten(start_dim=1) #Flatten
        action = self.mlp(h)
        return action

def main():
    args=parse_args()
    model=Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-5)
    loss_func = torch.nn.MSELoss()  # loss function for regression, mean squared loss

    if torch.cuda.is_available():
        model=model.cuda()
        loss_func=loss_func.cuda()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(model)
    
    log_file_dir=args.log_file_dir
    log_file_augmented_path=augment_data(log_file_dir)
    samples=get_data(log_file_augmented_path)
    filtered_samples=preprocess(samples)
    train_samples, validation_samples=train_test_split(filtered_samples,test_size=0.2)


    def data_loader(samples,batch_size=60):
        num_samples=len(samples)
        for counter in range(0,num_samples,batch_size):
            batch_samples=samples[counter:counter+batch_size]
            images=[]
            steer_acc_pair=[]
            for batch_sample in batch_samples:
                filename=batch_sample[0]
                # print(filename)
                image=cv2.imread(filename) # read saved training image
                gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert RGB to grayscale
                steering_angle=batch_sample[1]
                acceleration=batch_sample[2]
                images.append(gray_image)
                steer_acc_pair.append([steering_angle,acceleration])
            x_train=np.array(images)[:,:,:,np.newaxis].astype('float') # (N,H,W,C)
            y_train=np.array(steer_acc_pair).astype('float')
            x_train,y_train=shuffle(x_train,y_train)
            yield x_train,y_train


    epochs=args.epochs
    batch_size=args.batch_size
    losses=[]
    # training loop
    for epoch in range(epochs):
        model.train()
        for i,data in enumerate(data_loader(train_samples,batch_size)):
            
            print(f'i={i}')
            batch_x,batch_y=data  
            batch_x=torch.tensor(batch_x,requires_grad=True)
            b_x=batch_x.permute(0,3,1,2).float().to(device) # pytorch accepts (N,C,H,W)
            batch_y=np.reshape(batch_y,(-1,2)) # first dim: minibatch size, second dim: steering_angle and acceleration
            b_y=torch.from_numpy(batch_y).float().to(device)
            print(f'b_y:{b_y}')
            prediction=model(b_x)
            ste,acc=prediction[:,0],prediction[:,1]
            print(f'prediction:{prediction}')
            loss1=loss_func(ste,b_y[:,0]) # steering_angle loss
            loss2=loss_func(acc,b_y[:,1]) # acceleration loss
            w1=1.75
            w2=1.
            loss=loss1*w1+loss2*w2 # weighted loss
            print(f'loss:{loss}')
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('epoch {}, loss {}'.format(epoch, loss.item()))

            losses.append(loss.item())

    # print(losses)
    np.save('loss.npy',losses)
    model_filename='model.pt'
    torch.save(model.state_dict(),model_filename)

    # validation loop
    val_losses=[]
    for epoch in range(epochs):
        model.eval()
        with torch.no_grad():
            for j,data in enumerate(data_loader(validation_samples,batch_size)):
                batch_x,batch_y=data

                batch_x=torch.tensor(batch_x,requires_grad=True)
                b_x=batch_x.permute(0,3,1,2).float().to(device)
                # b_x=batch_x.requires_grad()
                batch_y=np.reshape(batch_y,(-1,2))
                b_y=torch.from_numpy(batch_y).float().to(device)
                # print(b_y)
                model.eval()

                y_hat=model(b_x)
                
                ste,acc=y_hat[:,0],y_hat[:,1]
                val_loss1=loss_func(ste,b_y[:,0])
                val_loss2=loss_func(acc,b_y[:,1])
                val_loss=val_loss1+val_loss2
                val_losses.append(val_loss.item())

    # print(val_losses)
    np.save('val_loss.npy',val_losses)
    torch.cuda.empty_cache()

if __name__=='__main__':
    main()
