#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 07:37:39 2020

@author: rajiv
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F

class conv_batch_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0):
        super(conv_batch_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
            )
    def forward(self, x):
        return self.conv(x)

class dep_separable_conv(nn.Module):
    def __init__(self, depth_channel, out_channel, stride = 1):
        super(dep_separable_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(depth_channel, depth_channel, 3, stride, 1, groups=depth_channel, bias=False),
            nn.BatchNorm2d(depth_channel),
            nn.ReLU(True),
            nn.Conv2d(depth_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
    def forward(self, x):
        return self.conv(x)
      
class depth_conv(nn.Module):
    def __init__(self, depth_channel, out_channel, stride = 1):
        super(depth_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(depth_channel, out_channel, 3, stride, 1, groups=depth_channel, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
    def forward(self, x):
        return self.conv(x)

class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(bottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            conv_batch_relu(in_channels, in_channels * t, 1),
            depth_conv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out
    
class pyramid_pooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pyramid_pooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = conv_batch_relu(in_channels, inter_channels, 1)
        self.conv2 = conv_batch_relu(in_channels, inter_channels, 1)
        self.conv3 = conv_batch_relu(in_channels, inter_channels, 1)
        self.conv4 = conv_batch_relu(in_channels, inter_channels, 1)
        self.out = conv_batch_relu(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x
        
class learning_to_downsample(nn.Module):
    def __init__(self, layer1_channel = 32, layer2_channel = 48, out_channel = 64):
        super(learning_to_downsample, self).__init__()
        self.conv1 = conv_batch_relu(3, out_channels = layer1_channel, kernel_size=3, stride=2)
        self.conv2 = dep_separable_conv(layer1_channel, layer2_channel, 2)
        self.conv3 = dep_separable_conv(layer2_channel, out_channel, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


    
class feature_fusion(nn.Module):
    def __init__(self, high_in_channels, lower_in_channels, out_channels, scale_factor=4):
        super(feature_fusion, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = depth_conv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(high_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)

class classifier(nn.Module):
    def __init__(self, depth_channel, num_classes, stride=1, **kwargs):
        super(classifier, self).__init__()
        self.dsconv1 = dep_separable_conv(depth_channel, depth_channel, stride)
        self.dsconv2 = dep_separable_conv(depth_channel, depth_channel, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(depth_channel, num_classes, 1)
        )        # It has same result with np.nanmean() when all class exist

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x
    
class global_feature_extractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3)):
        super(global_feature_extractor, self).__init__()
        self.bottleneck1 = self.generate_layer(bottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self.generate_layer(bottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self.generate_layer(bottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.pyramid_pool = pyramid_pooling(block_channels[2], out_channels)
    def generate_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.pyramid_pool(x)
        return x


class fast_scnn(nn.Module):
    def __init__(self, num_classes, aux = False):
        super(fast_scnn, self).__init__()
        self.aux = aux 
        self.layer1 = learning_to_downsample(32, 48, 64)
        self.layer2 = global_feature_extractor(64, [64, 96, 128], 128, 6, [3,3,3])
        self.layer3 = feature_fusion(64, 128, 128)
        self.layer4 = classifier(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )
    def forward(self, x):
        size = x.size()[2:]
        high_res = self.layer1(x)
        x = self.layer2(high_res)
        x = self.layer3(high_res, x)
        x = self.layer4(x)
        output = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        output.append(x)
        if self.aux:
            auxout = self.auxlayer(high_res)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            output.append(auxout)
        return tuple(output)

def get_fast_scnn():
    model = fast_scnn(num_classes=19)
    return model