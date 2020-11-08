"""
The training and testing code is mostly adapted from the following PyTorch tutorial:
"https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

Sonat Baltacı sonat.baltaci@gmail.com
May 2020
"""

from cocob import COCOB
from models.mnist import CNN, FCN, init_weights

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

import argparse
import os

# Create arguments to set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "--bs", "--batch", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epoch",type=int, default=50)
parser.add_argument("--model", type=str, default="fcn")
parser.add_argument("--optim", type=str, default="cocob")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--save", type=bool, default=False)
args = parser.parse_args()

# Normalize the data
tr = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.1307], std=[0.3081])
     ])

# Seed the experiment
torch.manual_seed(args.seed)

# Load train and test dataset
mnist_trainset = datasets.MNIST(root='./data', train=True,
                                        download=True, transform=tr)
trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

mnist_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=tr)
testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

# Load the model
if(args.model=="cnn"):
    net = CNN()
elif(args.model=="fcn"):
    net = FCN()
    net.apply(init_weights)
else:
    print("Invalid model for MNIST experiments.")
    exit(-1)

# Set the criterion 
criterion = nn.CrossEntropyLoss()

# Set the optimizer
if(args.optim=="cocob"):
    optimizer = COCOB(net.parameters())
elif(args.optim=="adam"):
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif(args.optim=="adagrad"):
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr)
elif(args.optim=="adadelta"):
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
elif(args.optim=="rmsprop"):
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
else:
    print("Invalid optimizer for MNIST experiments.")
    exit(-1)

# Create files to log
if (os.path.isdir(os.getcwd()+"/logs") == False):
    os.mkdir("logs")
if (args.optim == "cocob"):
    f = open("logs/loss-error_rate_"+args.model+"_mnist_"+args.optim+".txt", 'w+')
else:
    f = open("logs/loss-error_rate_"+args.model+"_mnist_"+args.optim+"_lr_"+str(args.lr)+".txt", 'w+')

# Create folder to save the model
if (args.save == True and os.path.isdir(os.getcwd()+"/experiments") == False):
    os.mkdir("experiments")

# Loop over the dataset for "epoch" times
for epoch in range(args.epoch):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Feed the input forward to the network 
        outputs = net(inputs)

        # Compute the loss and back-propagate
        loss = criterion(outputs, labels)
        loss.backward()

        # Run the optimizer
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i%args.log_freq == args.log_freq-1: 
            print('[%d, %5d] loss: %.5f' %
                (epoch+1, i+1, running_loss/args.log_freq))
            f.write('[%d, %5d] loss: %.5f\n' %
                (epoch+1, i+1, running_loss/args.log_freq))
        running_loss = 0.0

    correct = 0
    total = 0

    # Test the model on test data
    with torch.no_grad():
        for j, data in enumerate(testloader, 0):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Error on the 10000 test images: %.5f' % (
        1-(correct/total)))
    f.write('Error on the 10000 test images: %.5f\n' % (
        1-(correct/total)))
f.close()

# Save the model
print('Finished Training')
if (args.save):
    if(args.optim == "cocob"):
        torch.save(net, "experiments/"+args.model+"_mnist_"+args.optim+".pkl")
    else:
        torch.save(net, "experiments/"+args.model+"_mnist_"+args.optim+"_lr_"+str(args.lr)+".pkl")
