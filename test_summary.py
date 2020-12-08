# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle

# Custom Libraries
import utils
from torchsummaryX import summary
from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fc1.fc1().to(device)
x = torch.randn(1,3,32,32).to(device)

summary(model,x)
print(model.state_dict())

model_file = "/data/Lottery-Ticket-Hypothesis-in-Pytorch/saves/fc1/mnist/1_model_lt.pth.tar"
model_parameters = torch.load(model_file)
print(model_parameters)

model.load_state_dict(model_parameters)
x = torch.randn(1,3,32,32).to(device)
summary(model,x)