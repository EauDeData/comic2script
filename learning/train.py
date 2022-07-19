import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import matplotlib.pyplot as plt
import sys
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from loss_utils import *
from data_utils import *
from model_utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import random

cuda = True if torch.cuda.is_available() else False
device= 'cuda' if cuda else 'cpu'
criterion = torch.nn.BCELoss().to(device)

generator = GeneratorUNet(out_channels=1).to(device)

train, test = split_train_test_IMCDB()

dataloader, val_loader = DataLoader(train, batch_size=3, shuffle=True), DataLoader(test, batch_size=2, shuffle=True)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr = 1e-5)

def train_step(model, dataset, optimizer, criterion):

    loss_value = []

    for m, (image, gt) in enumerate(dataset):

        image, gt = image.to(device), gt.to(device)
        optimizer.zero_grad()
        h = model(image)
        loss = criterion(F.sigmoid(h), gt)
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())

    return sum(loss_value) / (m+1)

def test_step(model, dataset, epoch, criterion):

    loss_value = []

    for m, (image, gt) in enumerate(dataset):

        with torch.no_grad():
            
            image, gt = image.to(device), gt.to(device)
            h = model(image)
            loss = criterion(F.sigmoid(h), gt)
            loss_value.append(loss.item())

            numpy_image = h[random.randint(0, gt.shape[0] - 1)].squeeze().to('cpu').numpy() * 255
            cv2.imwrite(f'examples/{epoch} - {m}.png', numpy_image.astype(np.uint8))

    return sum(loss_value) / (m+1)

EPOCHES = 30

curve_train, curve_test = list(), list()
for e in range(EPOCHES):
    print('epoch', e)
    curve_train.append(train_step(generator, dataloader, optimizer_G, criterion))
    curve_test.append(test_step(generator, val_loader, e, criterion))
    print(f"Test Loss {curve_test[-1]}, train: {curve_train[-1]}")
    plt.plot(curve_train, label='train')
    plt.plot(curve_test, label = 'test')
    plt.legend()
    plt.savefig('state.png')
    plt.clf()
    torch.save(generator, f"model_checkpoints/{e}.pkl")
