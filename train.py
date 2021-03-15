import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import argparse
import os

from model import *

parser = {
    'data_dir': './data/',
    'nb_epoch': 50,
    'test_size': 0.1,
    'learning_rate': 0.0001,
    'samples_per_epoch': 64,
    'batch_size': 36,
    'cuda': True,
    'seed': 7
}
args = argparse.Namespace(**parser)
args.cuda = args.cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df[['center', 'left', 'right']].values
    #and our steering commands as our output data
    y = data_df['steering'].values

    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0, shuffle=True)

    return X_train, X_valid, y_train, y_valid

X_train, X_valid, y_train, y_valid = load_data(args)

transformations = transforms.Compose([transforms.Lambda(lambda x: x/127.5 - 1)                                    
                                     ])
print('\n\n', args.data_dir)
#train_set = CarDataset(X_train, y_train, args.data_dir, False,transformations)
train_set = CarDataset3Img(X_train, y_train, args.data_dir,transformations)
valid_set = CarDataset(X_valid, y_valid, args.data_dir, False, transformations)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

def toVariable(data, use_cuda):
    input, target = data
    input, target = Variable(input.float()), Variable(target.float())
    if use_cuda:
        input, target = input.cuda(), target.cuda()
    
    return input, target




# Training
def train(epoch, net, dataloader, optimizer, criterion, use_cuda):
    net.train()
    train_loss = 0
    
    for batch_idx, (centers, lefts, rights) in enumerate(dataloader):

        optimizer.zero_grad()
        centers, lefts, rights = toVariable(centers, use_cuda), \
                                 toVariable(lefts, use_cuda), \
                                 toVariable(rights, use_cuda)
        datas = [lefts, rights, centers]        
        for data in datas:
            imgs, targets = data
            outputs = net(imgs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        if batch_idx % 100 == 0:
            print('Loss: %.3f '
                % (train_loss/((batch_idx+1)*3)))

def valid(epoch, net, validloader, criterion, use_cuda):
    global best_loss
    net.eval()
    valid_loss = 0
    for batch_idx, (inputs, targets) in enumerate(validloader):
        inputs, targets = Variable(inputs.float()), Variable(targets.float())
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs.squeeze(), targets)

        valid_loss += loss.item()
        
        avg_valid_loss = valid_loss/(batch_idx+1)
        if batch_idx % 100 == 0:
            print('Valid Loss: %.3f '
                % (valid_loss/(batch_idx+1)))
        if avg_valid_loss <= best_loss:
            best_loss = avg_valid_loss
            print('Best epoch: ' + str(epoch))
            state = {
                'net': net.module if args.cuda else net,
            }
            torch.save(state, './models/current.h5')

from model_architectures.registry import create_model

net = create_model('car_simple_model')
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

if args.cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.MSELoss()
best_loss = 0.999



for epoch in range(0,15):
    #optimizer = lr_scheduler(optimizer, epoch, lr_decay_epoch=args.lr_decay_epoch)	
    print('\nEpoch: %d' % epoch)
    train(epoch, net, train_loader, optimizer, criterion, args.cuda)
    valid(epoch, net, valid_loader, criterion, args.cuda)

state = {
        'net': net.module if args.cuda else net,
        }

torch.save(state, './models/last.h5')