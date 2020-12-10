#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 23:52:14 2020

@author: rajiv
"""

import os
from itertools import chain


def load_data():

    ########## Change to the right data path here ########################
    data_path = ".../data/citys/leftImg8bit/"
    train_dir = ".../data/citys/leftImg8bit/train/"
    train_mask = ".../data/citys/gtFine/train/"
    val_dir =  ".../data/citys/leftImg8bit/val/"
    val_mask = ".../data/citys/gtFine/val/"
    ######### Change to the right data path here #########################

    # training directory of images
    list_train_dir = os.listdir(train_dir)
    train_path = [train_dir + s for s in list_train_dir]
    # Validation directory of images
    list_val_dir = os.listdir(val_dir)
    val_path = [val_dir + s for s in list_val_dir]
    # training directory of mask
    list_train_mask = os.listdir(train_mask)
    train_mask_path = [train_mask + s for s in list_train_mask]
    # validation directory for mask
    list_val_mask = os.listdir(val_mask)
    val_mask_path = [val_mask + s for s in list_val_mask]

    train_img_names = []
    for folder in train_path:
        train_img_names.append(os.listdir(folder))
    val_img_names = []
    for folder in val_path:
        val_img_names.append(os.listdir(folder))
    train_mask_names = []
    for folder in train_mask_path:
        train_mask_names.append(os.listdir(folder))
    val_mask_names = []
    for folder in val_mask_path:
        val_mask_names.append(os.listdir(folder))
    
    train_names = []
    for i, folder in enumerate(train_path):
        temp = []
        temp = [s for s in train_img_names[i]]
        temp = [folder + '/' + s for s in temp]
        train_names.append(temp)
    val_names = []
    for i, folder in enumerate(val_path):
        temp = []
        temp = [s for s in val_img_names[i]]
        temp = [folder + '/' + s for s in temp]
        val_names.append(temp)
    train_gt_mask = []
    for i, folder in enumerate(train_mask_path):
        temp = []
        temp = [s for s in train_mask_names[i]]
        temp = [folder + '/' + s for s in temp]
        train_gt_mask.append(temp)
    val_gt_mask = []
    for i, folder in enumerate(val_mask_path):
        temp = []
        temp = [s for s in val_mask_names[i]]
        temp = [folder + '/' + s for s in temp]
        val_gt_mask.append(temp)
    train_names = list(chain.from_iterable(train_names))
    val_names = list(chain.from_iterable(val_names))
    train_mask_names = list(chain.from_iterable(train_gt_mask))
    val_mask_names = list(chain.from_iterable(val_gt_mask))
    train_names.sort()
    train_mask_names.sort()
    val_names.sort()
    val_mask_names.sort()
    train_mask_names = train_mask_names[2::4]
    val_mask_names =val_mask_names[2::4]
    
    return train_names, val_names, train_mask_names, val_mask_names