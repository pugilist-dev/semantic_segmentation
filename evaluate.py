#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:50:23 2020

@author: rajiv
"""
import torch
from torch.utils.data import DataLoader

from torchvision import transforms
from utils.load_data import load_data
from model.model import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete
from utils.get_dataset import dataset
from utils.data_augmenter import img_aug_transform
import PIL.Image as Image
import numpy as np

base_size = 1024
crop_size = 1024
resize = (1024,1024)
batch_size = 6
device = "cuda"
aux = False
momentum = 0.9
lr = 0.001
weight_decay = 1e-4
epochs = 100
num_clases = 19

out_dir = "./eval_result/"
def mask_transform(img):
    img = img.resize(resize)
    img = np.array(img)
    img = torch.from_numpy(img)
    return img


transform = transforms.Compose([
    img_aug_transform(),
    lambda x: torch.from_numpy(x),
    transforms.RandomVerticalFlip()
])

    
train_names, val_names, train_mask, val_mask = load_data()
val_dataset = dataset(val_names, val_mask, transform)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False)
model = get_fast_scnn()

########### mention the right model name below ###################
model.load_state_dict(torch.load("./fscnn_ 0.pth", map_location="cuda:0"))

########## mention the right model name above ####################
model.to(device)
metric = SegmentationMetric(num_clases)
model.eval()
i = 0
for image, label in val_loader:
    image = image.to(device)
    label = label.to(device)
    prediction = model(image)
    pred = torch.argmax(prediction[0], 1)
    pred = pred.cpu().data.numpy()
    target = label.cpu().data.numpy()
    metric.update(pred, target)
    pixel_accuracy, mean_IoU = metric.get()
    print("Pixel Accuracy: %.2f%%, Mean IoU: %.3f%%" %(pixel_accuracy, mean_IoU))
    prediction = pred.squeeze(0)
    mask = get_color_pallete(prediction)
    output = out_dir + 'image_{}.png'.format(i)
    mask.save(output)
    i += 1