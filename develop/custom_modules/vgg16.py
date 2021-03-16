import custom_modules.custom_modules as cm
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.quantization import QuantStub, DeQuantStub
from typing import Union, List
from copy import deepcopy
import math

class VGG16 (nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = cm.ConvReLU(in_planes=3, out_planes=64, kernel_size=3, stride=1, bias=True)
        self.conv1_2 = cm.ConvReLU(in_planes=64, out_planes=64, kernel_size=3, stride=1, bias=True)
        self.maxpool1 = cm.MaxPool2dRelu(kernel_size=2, stride=2, relu=False)
        self.conv2_1 = cm.ConvReLU(in_planes=64, out_planes=128, kernel_size=3, stride=1, bias=True)
        self.conv2_2 = cm.ConvReLU(in_planes=128, out_planes=128, kernel_size=3, stride=1, bias=True)
        self.maxpool2 = cm.MaxPool2dRelu(kernel_size=2, stride=2, relu=False)
        self.conv3_1 = cm.ConvReLU(in_planes=128, out_planes=256, kernel_size=3, stride=1, bias=True)
        self.conv3_2 = cm.ConvReLU(in_planes=256, out_planes=256, kernel_size=3, stride=1, bias=True)
        self.conv3_3 = cm.ConvReLU(in_planes=256, out_planes=256, kernel_size=3, stride=1, bias=True)
        self.maxpool3 = cm.MaxPool2dRelu(kernel_size=2, stride=2, relu=False)
        self.conv4_1 = cm.ConvReLU(in_planes=256, out_planes=512, kernel_size=3, stride=1, bias=True)
        self.conv4_2 = cm.ConvReLU(in_planes=512, out_planes=512, kernel_size=3, stride=1, bias=True)
        self.conv4_3 = cm.ConvReLU(in_planes=512, out_planes=512, kernel_size=3, stride=1, bias=True)
        self.maxpool4 = cm.MaxPool2dRelu(kernel_size=2, stride=2, relu=False)
        self.conv5_1 = cm.ConvReLU(in_planes=512, out_planes=512, kernel_size=3, stride=1, bias=True)
        self.conv5_2 = cm.ConvReLU(in_planes=512, out_planes=512, kernel_size=3, stride=1, bias=True)
        self.conv5_3 = cm.ConvReLU(in_planes=512, out_planes=512, kernel_size=3, stride=1, bias=True)
        self.maxpool5 = cm.MaxPool2dRelu(kernel_size=2, stride=2, relu=False)
        '''
        Remove average pooling. When IW/IH = 224/224, the average pooling essentially disappears
        The original model uses adaptive 2d average pooling with target OW/OH = 7/7
        To see this, take a look at the formulae for adaptive 2d avg pooling: 
        https://stackoverflow.com/a/58694174
        
        '''
        #self.avgpool = cm.AvgPool2dRelu(kernel_size=7, divisor_override=64, relu=False)
        self.fc1 = cm.LinearReLU(in_features=512*7*7, out_features=4096, bias=True)
        self.fc2 = cm.LinearReLU(in_features=4096, out_features=4096, bias=True)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000, bias=True)

        # Utils
        self.flatten = cm.Flatten()
        self.quant = QuantStub()
        self.deQuant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.maxpool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.maxpool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.deQuant(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == cm.ConvReLU or type(m) == cm.LinearReLU:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)

