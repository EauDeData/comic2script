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

cuda = True if torch.cuda.is_available() else False
criterion = PredictionLoss()

generator = GeneratorUNet(out_channels=1)

train, test = split_train_test()

dataloader, val_loader = DataLoader(train, batch_size=10), DataLoader(test, batch_size=10)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer_G = torch.optim.Adam(generator.parameters(), lr = 1e-3)

def train_step(model, dataset, optimizer, criterion):

    loss_value = []

    for m, image in enumerate(dataset):

        optimizer.zero_grad()
        h = F.sigmoid(model(image))
        loss = criterion(h, image)
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())

    return sum(loss_value) / (m+1)

def test_step(model, dataset, epoch, criterion):

    loss_value = []

    for m, image in enumerate(dataset):

        with torch.no_grad():
            h = F.sigmoid(model(image))
            loss = criterion(h, image)
            loss_value.append(loss.item())

            numpy_image = h[0].squeeze().to('cpu').numpy() * 255
            cv2.imwrite(f'examples/{epoch} - {m}.png', numpy_image.astype(np.uint8))

    return sum(loss_value) / (m+1)

EPOCHES = 30

curve_train, curve_test = list(), list()
for e in range(EPOCHES):
    print('epoch', e)
    curve_train.append(train_step(generator, dataloader, optimizer_G, criterion))
    curve_test.append(test_step(generator, val_loader, e, criterion))
    print(f"Test Loss {curve_test[-1]}, train: {curve_train[-1]}")
    plt.plot(curve_test)
    plt.plot(curve_test)
    plt.savefig('state.png')
    plt.clf()
