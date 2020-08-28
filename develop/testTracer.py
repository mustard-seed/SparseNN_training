"""
Test file for the NN tracer.
Declare a mini ResNet-like model, prune and quantize it, and then feed it through the tracer to see what comes out of it.
"""
import custom_modules.custom_modules as cm
import custom_modules.resnet as resnet
import pruning.pruning as custom_pruning
import quantization.quantization as custom_quant
from    tracer.tracer import TraceDNN as Tracer

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.quantization import QuantStub, DeQuantStub
from typing import Union, List

import os
import yaml
import argparse

def toy_resnet() -> resnet.ResNet:
    """
    Generate and initialize a toy resent with only 5 layers.
    Input size is compatiable with CIFAR-10
    :return: The ResNet-56 network
    """
    block = resnet.BasicBlock
    # Number of blocks per stage. Each stage has two parametrized layers
    num_blocks = [1, 1, 1]
    network = resnet.ResNet(block, num_blocks, 'cifar10')
    return network

class TinyNet(nn.Module):
    """
    Tiny-Net just for generating a trace purpose
    """
    def __init__(self):
        super().__init__()
        self.block = resnet.BasicBlock(in_planes=4, planes=8, stride=1)
        self.averagePool = nn.AvgPool2d(kernel_size=8, divisor_override=64)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.block(x)
        x = self.averagePool(x)
        output = self.dequant(x)

        return output

        # Fuse layers prior to quantization

    def fuse_model(self):
        for m in self.modules():
            if type(m) == cm.ConvBNReLU:
                # Fuse the layers in ConvBNReLU module, which is derived from nn.Sequential
                # Use the default fuser function
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            elif type(m) == cm.ConvBN:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)

class Conv(nn.Module):
    """
    Tiny-Net just for generating a trace purpose
    """
    def __init__(self):
        super().__init__()
        self.block = resnet.conv3x3BN(in_planes=4, out_planes=8, stride=1, groups=1, dilation=1, require_relu=True)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.block(x)
        output = self.dequant(x)

        return output

        # Fuse layers prior to quantization

    def fuse_model(self):
        for m in self.modules():
            if type(m) == cm.ConvBNReLU:
                # Fuse the layers in ConvBNReLU module, which is derived from nn.Sequential
                # Use the default fuser function
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            elif type(m) == cm.ConvBN:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)

class MaxPool(nn.Module):
    """
    Tiny-Net just for generating a trace purpose
    """
    def __init__(self):
        super().__init__()
        self.block = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.block(x)
        output = self.dequant(x)

        return output


class TracerTest():
    def __init__(self, mode: str='resnet'):
        self.mode = mode
        if self.mode == 'resnet':
            self.model = toy_resnet()
        elif self.mode == 'tiny':
            self.model = TinyNet()
        elif self.mode == 'conv':
            self.model = Conv()
        elif self.mode == 'maxpool':
            self.model = MaxPool()
        else:
            print(self.mode)
            raise ValueError('Unsupported mode')
        self.pruneClusterSize=1
        self.pruneThreshold=1e-4
        qatRoundedConfig = torch.quantization.FakeQuantize.with_args(
            observer=custom_quant.RoundedMovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            averaging_constant=0.01
        )
        self.qatConfig = torch.quantization.QConfig(
            activation=qatRoundedConfig
            , weight=qatRoundedConfig
        )
        # Quantize the model
        if hasattr(self.model, 'fuse_model'):
            self.model.fuse_model()
        self.model.qconfig = self.qatConfig
        torch.quantization.prepare_qat(self.model, inplace=True)

        # Prune the model
        if self.mode == 'resnet':
            custom_pruning.applyClusterPruning(
                self.model.inputConvBNReLU[0],
                'weight',
                clusterSize=self.pruneClusterSize,
                threshold=self.pruneThreshold
            )
        if self.mode == 'resnet' or self.mode == 'tiny':
            for m in self.model.modules():
                if isinstance(m, resnet.BasicBlock):
                    custom_pruning.applyClusterPruning(m.convBN1[0],
                                                       "weight",
                                                       clusterSize=self.pruneClusterSize,
                                                       threshold=self.pruneThreshold)
                    custom_pruning.applyClusterPruning(m.convBN2[0],
                                                       "weight",
                                                       clusterSize=self.pruneClusterSize,
                                                       threshold=self.pruneThreshold)
                    if isinstance(m.shortcut, cm.ConvBN):
                        custom_pruning.applyClusterPruning(m.shortcut[0],
                                                           "weight",
                                                           clusterSize=self.pruneClusterSize,
                                                           threshold=self.pruneThreshold)
        elif self.mode == 'conv':
            custom_pruning.applyClusterPruning(self.model.block[0],
                                               "weight",
                                               clusterSize=self.pruneClusterSize,
                                               threshold=self.pruneThreshold)

    def trace(self, dirname: str, fileBaseName: str):
        tracer = Tracer(self.model)
        if self.mode == 'resnet':
            dummyInput = torch.randn(size=[1, 3, 32, 32])
        else:
            dummyInput = torch.randn(size=[1, 4, 8, 8])

        dummyOutput = tracer.traceModel(dummyInput)
        tracer.annotate(numMemRegions=3)
        tracer.dump(dirname, fileBaseName)

        """
        Saves the input and output
        """
        print("Saves the dummy input and output")
        inputArray = dummyInput.view(dummyInput.numel()).tolist()
        outputArray = dummyOutput.view(dummyOutput.numel()).tolist()
        blobPath = os.path.join(dirname, fileBaseName + '_inout.yaml')
        blobFile = open(blobPath, 'w')
        blobDict: dict = {}

        blobDict['input'] = inputArray
        blobDict['output'] = outputArray

        # We want list to be dumped as in-line format, hence the choice of the default_flow_style
        # See https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
        yaml.dump(blobDict, blobFile, default_flow_style=None)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="testTracer")
    parser.add_argument('--mode', type=str, choices=['resnet', 'tiny', 'conv', 'maxpool'], default='conv',
                        help='Mode. Valid choices are resnet, tiny, conv, and maxpool')
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.mode == 'resnet':
        fileBaseName = 'testTrace'
    elif args.mode == 'tiny':
        fileBaseName = 'tinyTrace'
    elif args.mode == 'conv':
        fileBaseName = 'convTrace'
    elif args.mode == 'maxpool':
        fileBaseName = 'mp'
    else:
        print(args.mode)
        raise ValueError("Unsupported mode")
    test = TracerTest(mode=args.mode)
    test.trace(dirname='.', fileBaseName=fileBaseName)

