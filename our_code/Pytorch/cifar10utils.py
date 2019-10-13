import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import torch.nn.functional as F

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
    full_training_data = torchvision.datasets.CIFAR10('/home/test/data',train = True,transform=transform_train,download=True)  
    test_data = torchvision.datasets.CIFAR10('/home/test/data',train = False,transform=transform_test,download=True)  

    # Create train and validation splits
    num_samples = len(full_training_data)
    training_samples = int((1-val_percentage)*num_samples+1)
    validation_samples = num_samples - training_samples
    training_data, validation_data = torch.utils.data.random_split(full_training_data, [training_samples, validation_samples])

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(training_data,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data,batch_size=batch_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size,shuffle=False,drop_last=True,num_workers=0)

    return train_loader, val_loader, test_loader

'''
Function to test that returns the loss per sample and the total accuracy
'''
def test(data_loader,net,device):
    net.eval()
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.
    
    loss_funct = torch.nn.CrossEntropyLoss()

    for batch_idx, (inputs,targets) in enumerate(data_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net(inputs)[0]
        
        loss = loss_funct(outputs,targets)

        # Metrics computation
        samples+=inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        cumulative_accuracy += predicted.eq(targets).sum().item()
    
    net.train()

    return cumulative_loss/samples, cumulative_accuracy/samples*100