# COCOB

PyTorch implementation of NIPS 2017 paper, "Training Deep Networks without Learning Rates Through Coin Betting". <br />
Authors: F. Orabona, T. Tommasi <br />
https://arxiv.org/pdf/1705.07795.pdf

Optimizer and the experiment implementations have only cpu support, for now.

## MNIST Experiment
### Requirements
Experiments are conducted in Anaconda (version 4.8.2) environment with,

- Python 3.8.3
- PyTorch 1.5.0
- Torchvision 0.6.0
- NumPy 1.18.1
- Matplotlib 3.1.3

### Running the Experiments
The implementation of the optimizer is supported with the experiment setup of digit recognition task using MNIST dataset, which will be loaded automatically within the implementation.

To train the models with custom learning rates and optimizers, arguments to pass to ```train.py``` to run the experiments are as follows:
- model (string): Model type to experiment. Possible inputs are "fcn" and "cnn".
- batch_size (int): Batch size. For FCN, it is 100 and for CNN, 128.
- optim (string): Type of the optimizer. Possible inputs are "cocob", "adam", "adadelta", "adagrad" and "rmsprop".
- lr (float): Learning rate, for required optimizers.
- log_freq (int): Log frequency, in terms of iterations.
- epoch (int): Number of epochs.
- save (bool): Boolean value to save the model.

Here are two example commands:
```
python train.py --model cnn --batch_size 128 --optim cocob --log_freq 100 --epoch 30 --save True
python train.py --model cnn --batch_size 128 --optim adam --lr 0.075 --log_freq 100 --epoch 30 --save False
```
To reproduce the plots in the paper with given learning rates, an example script ```run.py``` is provided and can be executed as follows:
```
python run.py
```
