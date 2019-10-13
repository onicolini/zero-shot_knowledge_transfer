#!/usr/bin/env python

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary

from wideresnet import Wide_ResNet

'''
Function that loads the dataset and returns the data-loaders
'''
def getData(batch_size,test_batch_size,val_percentage):
    # Normalize the training set with data augmentation
    transform_train = transforms.Compose([ 
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(20),
        torchvision.transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download/Load data
    full_training_data = torchvision.datasets.CIFAR10('./data',train = True,transform=transform_train,download=True)  
    test_data = torchvision.datasets.CIFAR10('./data',train = False,transform=transform_test,download=True)  

    # Create train and validation splits
    num_samples = len(full_training_data)
    training_samples = int((1-val_percentage)*num_samples+1)
    validation_samples = num_samples - training_samples
    training_data, validation_data = torch.utils.data.random_split(full_training_data, [training_samples, validation_samples])

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(training_data,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data,batch_size=batch_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size,shuffle=False)

    return train_loader, val_loader, test_loader

'''
Function to test that returns the loss per sample and the total accuracy
'''
def test(data_loader,net,cost_fun,device):
  
    net.eval()
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    for batch_idx, (inputs,targets) in enumerate(data_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net(inputs)[0]
        loss = cost_fun(outputs,targets)

        # Metrics computation
        samples+=inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss/samples, cumulative_accuracy/samples*100

'''
Function to train the nework on the data for one epoch that returns the loss per sample and the total accuracy
'''
def train(data_loader,net,cost_fun,device,optimizer):
    
    net.train()
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    for batch_idx, (inputs,targets) in enumerate(data_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net(inputs)[0]
        loss = cost_fun(outputs,targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Metrics computation
        samples+=inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss/samples, cumulative_accuracy/samples*100

def main(epochs, batch_size, test_batch_size,val_percentage,lr,test_freq,device,save_filename, res_depth, res_width):
    print('Architecture: WRN-' + str(res_depth) + '-' + str(res_width))
    print('Epochs: ' + str(epochs) + ' batch_size: ' + str(batch_size) + ' test_batch_size: ' + str(test_batch_size))
    print('Save and test frequency ' + str(test_freq) + ' model filename: ' + str(save_filename))
    print('LR: ' + str(lr) + ' momentum: ' + str(0.9) + ' weight decay: ' + str(5e-4))
    print('LR Scheduler: gamma= ' + str(0.2) + ' steps: [' + str(int(epochs*0.3)) + ',' + str(int(epochs*0.6)) + str(int(epochs*0.8)) )
    print('data augmentation: random crop 32, padding 4, random horizontal flip, random rotation 20,ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.03')
    
    
    # Define cost function
    cost_function = torch.nn.CrossEntropyLoss()

    # Create the network: Wide_ResNet(depth, width, dropout, num_classes)
    net = Wide_ResNet(res_depth,res_width,0,10)
    net = net.to(device)
    summary(net,input_size=(3,32,32))

    # Create the optimizer anche the learning rate scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=[int(epochs*0.3),int(epochs*0.6),int(epochs*0.8)], gamma=0.20)

    # Get the data
    train_loader, val_loader, test_loader = getData(batch_size,test_batch_size,val_percentage)

    for e in range(epochs):
        net.train() 

        train_loss, train_accuracy = train(train_loader,net,cost_function,device,optimizer)

        val_loss, val_accuracy = test(val_loader,net,cost_function,device)
        
        scheduler.step()

        print('Epoch: {:d}:'.format(e+1))
        print('\t Training loss: \t {:.6f}, \t Training accuracy \t {:.2f}'.format(train_loss, train_accuracy))
        print('\t Validation loss: \t {:.6f},\t Validation accuracy \t {:.2f}'.format(val_loss, val_accuracy))
        
        if((e) % test_freq) == 0:
            test_loss, test_accuracy = test(test_loader,net,cost_function,device)
            print('Test loss: \t {:.6f}, \t \t Test accuracy \t {:.2f}'.format(test_loss, test_accuracy))
            torch.save(net.state_dict(), save_filename)

    print('After training:')
    train_loss, train_accuracy = test(train_loader,net,cost_function,device)
    val_loss, val_accuracy = test(val_loader,net,cost_function,device)
    test_loss, test_accuracy = test(test_loader,net,cost_function,device)

    print('\t Training loss: \t {:.6f}, \t Training accuracy \t {:.2f}'.format(train_loss, train_accuracy))
    print('\t Validation loss: \t {:.6f},\t Validation accuracy \t {:.2f}'.format(val_loss, val_accuracy))
    print('Test loss: \t {:.6f}, \t \t Test accuracy \t {:.2f}'.format(test_loss, test_accuracy))
    
    torch.save(net.state_dict(), save_filename)

    net2 = Wide_ResNet(16,2,0,10)
    net2 = net.to(device)
    net2.load_state_dict(torch.load(save_filename))
    
    print('loaded net test:')
    test_loss, test_accuracy = test(test_loader,net2,cost_function,device)
    print('\t Test loss: \t {:.6f}, \t Test accuracy \t {:.2f}'.format(test_loss, test_accuracy))


# Parameters
epochs = 352
batch_size = 128
test_batch_size = 128
val_percentage = 0.01
lr = 0.1
test_freq = 20
device = 'cuda:0'
save_filename = './pretrained_models/teacher-16-2.pth'
res_depth = 16
res_width = 2

# Call the main
main(epochs, batch_size, test_batch_size,val_percentage,lr,test_freq,device,save_filename, res_depth, res_width)
