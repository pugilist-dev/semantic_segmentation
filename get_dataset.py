#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:19:11 2020

@author: rajiv
"""

import torch
import torchvision
from PIL import Image, ImageOps
import numpy as np
import random


resize = (1024,1024)
base_size = 1024
crop_size = 1024


class dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, labels, transform):
        self.labels = labels
        self.filenames = image_path
        self.transform = transform
        self.base_size = base_size
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
        
    def __len__(self):
        return len(self.filenames) 
    
    def class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)
    
    def image_transform(self, image):
        image = self.transform(image)
        return image
    
    def mask_transform(self, mask):
        target = self.class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))
    
    def _transform(self, image, mask):
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = image.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = image.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        image = self.image_transform(image)
        image = self.transform(image)
        mask = self.mask_transform(mask)
        return image, mask
    
    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert('RGB')
        label = Image.open(self.labels[index])
        image, mask = self._transform(image, label)
        return image.transpose(2,0).float(), mask