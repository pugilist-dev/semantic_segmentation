#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 07:37:39 2020

@author: rajiv
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
from utils.load_data import load_data
from model.model import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from utils.lr_scheduler import LRScheduler
import time
import os
from utils.get_dataset import dataset
from utils.data_augmenter import img_aug_transform

base_size = 1024
crop_size = 1024
resize = (1024,1024)
batch_size = 10
device = "cuda"
aux = False
momentum = 0.9
lr = 0.001
weight_decay = 1e-4
epochs = 100
num_clases = 19

transform = torchvision.transforms.Compose([
    img_aug_transform(),
    lambda x: torch.from_numpy(x),
    torchvision.transforms.RandomVerticalFlip()
])

train_names, val_names, train_mask, val_mask = load_data()
train_data = dataset(train_names, train_mask, transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
model = get_fast_scnn()
model.to(device)
criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=aux, aux_weight=0.4,
                                                        ignore_index=-1).to(device)
optimizer = torch.optim.SGD(model.parameters(),
                                         lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
lr_scheduler = LRScheduler(mode='poly', base_lr=lr, nepochs=epochs,
                                        iters_per_epoch=len(train_loader), power=0.9)
        
def checkpoint(model, epoch):
    filename = 'fscnn_{}.pth'.format(epoch)
    directory = './'
    save_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), save_path)

iterations = 0
start_time = time.time()
for epoch in range(epochs):
    model.train()
        
    for image, targets in train_loader:
        cur_lr = lr_scheduler(iterations)
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
        images = image.to(device)
        targets = targets.to(device)
        output = model(images)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        iterations += 1
        if iterations % 10 == 0:
            print('Epoch: %2d, Iteration: %4d/%4d, Time spent training: %4.2f sec,  learning rate: %.6f, Loss: %.3f' 
                  % (epoch, iterations, len(train_loader),
                time.time() - start_time, cur_lr, loss.item()))
    checkpoint(model, epoch)