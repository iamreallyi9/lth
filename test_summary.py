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

def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()

batch_size =10
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
train_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss() # Default was F.nll_loss



model_file = "/data/Lottery-Ticket-Hypothesis-in-Pytorch/saves/fc1/mnist/1_model_lt.pth.tar"
model = torch.load(model_file)
optimizer = torch.optim.Adam(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
loss = train(model, train_loader, optimizer, criterion)

x = torch.randn(1,3,32,32).to(device)
summary(model,x)
