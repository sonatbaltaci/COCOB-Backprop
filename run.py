"""
The training and testing code is mostly adapted from the following PyTorch tutorial:
"https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

Sonat Baltacı sonat.baltaci@gmail.com
May 2020
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from cocob import COCOB
from models.mnist import CNN, FCN, init_weights

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

import os
def plot(results, lr, batch_size, log_freq, epoch, model, train=True):
    """ Plot the experiments done with CNN model.
        Args: 
            path_list (list): Log paths of trained models, with following order: Adagrad, RMSprop, AdaDelta, Adam, COCOB.
            lr_list (list): Learning rate list of optimizers with the same order with path_list.
            batch_size (int): Batch size.
            model (str): Model type, CNN or FCN.
            train (bool): Plot type, train or test.
    """
    adagrad = results[0]
    rmsprop = results[1]
    adadelta = results[2]
    adam = results[3]
    cocob = results[4]

    if(train):
        log = (60000/batch_size)//log_freq
        x_tr = np.arange(0,log*epoch,1)/log
        
        plt.plot(x_tr, adagrad, color="lime")
        plt.plot(x_tr, rmsprop, color="magenta")
        plt.plot(x_tr, adadelta, color="blue")
        plt.plot(x_tr, adam, color="black")
        plt.plot(x_tr, cocob, color="red")
        plt.title("MNIST "+model+" Training Cost")
        plt.ylabel("Cross-Entropy Loss")

    else:
        plt.plot(adagrad, color="lime")
        plt.plot(rmsprop, color="magenta")
        plt.plot(adadelta, color="blue")
        plt.plot(adam, color="black")
        plt.plot(cocob, color="red")
        plt.title("MNIST "+model+" Test Error")
        plt.ylabel("Error Rate")
    
    plt.xticks(np.arange(0, 31, 10))
    plt.grid()
    plt.xlabel("Epochs")
    plt.legend(["AdaGrad "+str(lr[0]),"RMSprop "+str(lr[1]),"AdaDelta "+str(lr[2]), "Adam "+str(lr[0]), "COCOB"])

    plt.show()

# Create arguments to set hyperparameters
models = ["fcn","cnn"]
seed = 0
epochs = 1
optims = ["adagrad","rmsprop","adadelta","adam", "cocob"]
log_freq = 100

# Normalize the data
tr = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.1307], std=[0.3081])
     ])

# Seed the experiment
torch.manual_seed(seed)

for model in models:
    # Store data to plot the results
    train_res_all = []
    test_res_all = []
    lr_list = []

    for optim_ in optims:
        if(model=="cnn"):
            print("Started training CNN with "+optim_+".")
            lrs = [0.0001,0.00025,0.0001,0.075]

            # Set learning rates for CNN experiment
            net = CNN()
            
            # Load train and test dataset
            mnist_trainset = datasets.MNIST(root='./data', train=True,
                                                    download=True, transform=tr)
            trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=128,
                                                      shuffle=True, num_workers=2)

            mnist_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                   download=True, transform=tr)
            testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=128,
                                                     shuffle=False, num_workers=2)
            batch_size = 128

        elif(model=="fcn"):
            print("Started training FCN with "+optim_+".")
            
            # Set learning rates for FCN experiment
            lrs = [0.00075,0.0005,0.0001,0.01]
            net = FCN()
            net.apply(init_weights)
            
            # Load train and test dataset
            mnist_trainset = datasets.MNIST(root='./data', train=True,
                                                    download=True, transform=tr)
            trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=100,
                                                      shuffle=True, num_workers=2)

            mnist_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                   download=True, transform=tr)
            testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=100,
                                                     shuffle=False, num_workers=2)
            batch_size = 100
        
        else:
            print("Invalid model for MNIST experiments.")
            exit(-1)

        # Set the criterion 
        criterion = nn.CrossEntropyLoss()

        train_res = []
        test_res = []
        
        # Set the optimizer
        if(optim_=="cocob"):
            optimizer = COCOB(net.parameters())
        elif(optim_=="adam"):
            lr = lrs[3]
            optimizer = optim.Adam(net.parameters(), lr=lr)
            lr_list.append(lr)
        elif(optim_=="adagrad"):
            lr = lrs[0]
            optimizer = optim.Adagrad(net.parameters(), lr=lr)
            lr_list.append(lr)
        elif(optim_=="adadelta"):
            lr = lrs[2]
            optimizer = optim.Adadelta(net.parameters(), lr=lr)
            lr_list.append(lr)
        elif(optim_=="rmsprop"):
            lr = lrs[1]
            optimizer = optim.RMSprop(net.parameters(), lr=lr)
            lr_list.append(lr)
        else:
            print("Invalid optimizer for MNIST experiments.")
            exit(-1)

        # Create files to log
        if (os.path.isdir(os.getcwd()+"/logs") == False):
            os.mkdir("logs")
        if (optim_ == "cocob"):
            f = open("logs/loss-error_rate_"+model+"_mnist_"+optim_+".txt", 'w+')
        else:
            f = open("logs/loss-error_rate_"+model+"_mnist_"+optim_+"_lr_"+str(lr)+".txt", 'w+')

        # Create folder to save the model
        if (os.path.isdir(os.getcwd()+"/experiments") == False):
            os.mkdir("experiments")

        # Loop over the dataset for "epoch" times
        for epoch in range(epochs):

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
                if i%log_freq == log_freq-1: 
                    print('[%d, %5d] loss: %.5f' %
                        (epoch+1, i+1, running_loss/log_freq))
                    f.write('[%d, %5d] loss: %.5f\n' %
                        (epoch+1, i+1, running_loss/log_freq))
                    train_res.append(running_loss/log_freq)
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
            test_res.append(1-correct/total)
        f.close()
        train_res_all.append(train_res)
        test_res_all.append(test_res)
        # Save the model
        print('Finished training of '+model.upper()+' with '+optim_+'.')
        if (optim_ == "cocob"):
            torch.save(net, "experiments/"+model+"_mnist_"+optim_+".pkl")
        else:
            torch.save(net, "experiments/"+model+"_mnist_"+optim_+"_lr_"+str(lr)+".pkl")

    # Plot the experiments
    plot(train_res_all, lr_list, batch_size, log_freq, epochs, model.upper(), train=True)
    plot(test_res_all, lr_list, batch_size, log_freq, epochs, model.upper(), train=False)
