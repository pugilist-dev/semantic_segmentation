#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:51:39 2020

@author: rajiv
"""

import torch

from torchvision import transforms
from model.model import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete




def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    ###### Paste the path along with the image name and extension .png below #######
    image = Image.open(".png")
    ###### Paste the path along with the image name and extension .png above #######
    
    image = transform(image).unsqueeze(0).to(device)
    model = get_fast_scnn()
    
    ##### Change the model name below #########
    model.load_state_dict(torch.load("./fscnn_100.pth", map_location="cuda:0"))
    ##### Change the model name above #########
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred)
    outname = "./eval_result/berlin_000000_000019_leftImg8bit.png"
    output_dir = "./"+ outname
    mask.save(output_dir)


if __name__ == '__main__':
    test()