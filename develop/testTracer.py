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
import torch.quantization
import torch.nn.init as init
from torch.quantization import QuantStub, DeQuantStub
from typing import Union, List

from torch.nn.intrinsic import qat as nniqat
import torch.nn.qat as nnqat

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
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            #nn.init.kaiming_normal_(m.weight, mode="fan_out")
            nn.init.normal_(m.weight, 0, 0.4)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            #HACK: need to centre the weights around a non-zero value for generating test trace,
            #Otherwise too many frac bits
            nn.init.normal_(m.weight, 0, 0.4)
            #nn.init.kaiming_normal_(m.weight, mode="fan_out")
            nn.init.zeros_(m.bias)
    return network

def test_cifar10_resnet() -> resnet.ResNet:
    """
    Generate and initialize a toy resent with only 5 layers.
    Input size is compatiable with CIFAR-10
    :return: The ResNet-56 network
    """
    block = resnet.BasicBlock
    # Number of blocks per stage. Each stage has two parametrized layers
    num_blocks = [9, 9, 9]
    stage_planes_override = [16, 32, 64]
    stage_strides_override = [1, 2, 2]
    avgpool2d_kernelsize = 8
    network = resnet.ResNet(block, num_blocks, 'cifar10', _flagTest=True,
                            _stage_planes_override=stage_planes_override,
                            _stage_strides_override=stage_strides_override,
                            _avgpool_2d_size_override= avgpool2d_kernelsize)
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            # torch.nn.init.normal_(m.weight, mean=0.0, std=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            #HACK: need to centre the weights around a non-zero value for generating test trace,
            #Otherwise too many frac bits
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            # torch.nn.init.normal_(m.weight, mean=0.0, std=0.5)
            nn.init.zeros_(m.bias)
    return network

def test_imagenet_resnet() -> resnet.ResNet:
    """
    Input size is compatiable with IMAGENET-1K
    :return: The ResNet-50 like network
    """
    block = resnet.BottleneckBlock
    # Number of blocks per stage. Each stage has two parametrized layers
    num_blocks = [1]
    stage_planes_override = [512]
    stage_strides_override = [2]
    avgpool2d_kernelsize = 8
    network = resnet.ResNet(block, num_blocks, 'imagenet1k', _flagTest=True,
                            _stage_planes_override=stage_planes_override,
                            _stage_strides_override=stage_strides_override,
                            _avgpool_2d_size_override= avgpool2d_kernelsize)
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            # torch.nn.init.normal_(m.weight, mean=0.0, std=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            #HACK: need to centre the weights around a non-zero value for generating test trace,
            #Otherwise too many frac bits
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            # torch.nn.init.normal_(m.weight, mean=0.0, std=0.5)
            nn.init.zeros_(m.bias)
    return network

class TinyNet(nn.Module):
    """
    Tiny-Net just for generating a trace purpose
    """
    def __init__(self):
        super().__init__()
        self.block = resnet.BasicBlock(in_planes=4, planes=8, stride=1)
        self.averagePool = cm.AvgPool2dRelu(kernel_size=8, divisor_override=64)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
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

class ResNet50_conv12(nn.Module):
    """
    Tiny-Net just for generating a trace purpose
    """
    def __init__(self):
        super().__init__()
        self.block = resnet.conv1x1BN(in_planes=256, out_planes=64, stride=1, groups=1, dilation=1, require_relu=True)
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

class Seq(nn.Module):
    """
    Two convolution layer back to back
    """
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.convBN1 = resnet.conv3x3BN(in_planes=in_planes,
                                 out_planes=planes,
                                 stride=stride,
                                 require_relu=True)
        self.convBN2 = resnet.conv3x3BN(in_planes=planes,
                                 out_planes=planes,
                                 stride=1)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.convBN1(x)
        x = self.convBN2(x)
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
        self.block = cm.MaxPool2dRelu(kernel_size=3, stride=1, padding=1, relu=True)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.block(x)
        output = self.dequant(x)

        return output

class Add(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = cm.EltwiseAdd(relu=True)
        self.quant0 = QuantStub()
        self.quant1 = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x, y):
        x = self.quant0(x)
        y = self.quant1(y)
        output = self.block(x, y)
        output = self.dequant(output)

        return output

class AvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = cm.AvgPool2dRelu(kernel_size=8, stride=8, divisor_override=64, relu=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        output = self.block(x)
        output = self.dequant(output)

        return output

class AvgPoolLinear(nn.Module):
    def __init__(self, _inFeatures: int, _outFeatures: int):
        super().__init__()
        self.inFeatures = _inFeatures
        self.outFeatures = _outFeatures
        self.pool = cm.AvgPool2dRelu(kernel_size=8, stride=8, divisor_override=64, relu=False)
        self.fc = cm.LinearReLU(in_features=_inFeatures, out_features=_outFeatures, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        output = self.pool(x)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        output = self.dequant(output)

        return output

    def fuse_model(self):
        for m in self.modules():
            if type(m) == cm.LinearReLU:
                # Fuse the layers in ConvBNReLU module, which is derived from nn.Sequential
                # Use the default fuser function
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)

class TracerTest():
    def __init__(self, mode: str='test'):
        self.mode = mode
        if self.mode == 'test':
            self.model = toy_resnet()
        elif self.mode == 'tiny':
            self.model = TinyNet()
        elif self.mode == 'conv':
            self.model = Conv()
        elif self.mode == 'maxpool':
            self.model = MaxPool()
        elif self.mode == 'add':
            self.model = Add()
        elif self.mode == "avg":
            self.model = AvgPool2d()
        elif self.mode == 'seq':
            self.model = Seq(in_planes=4, planes=8, stride=2)
        elif self.mode == 'avglinear':
            self.model = AvgPoolLinear(_inFeatures=4, _outFeatures=8)
        elif self.mode == 'restest_cifar10':
            self.model = test_cifar10_resnet()
        elif self.mode == 'restest_imagenet':
            self.model = test_imagenet_resnet()
        elif self.mode == 'resnet50_conv12':
            self.model = ResNet50_conv12()
        else:
            print(self.mode)
            raise ValueError('Unsupported mode')
        self.pruneClusterSize = 2
        self.pruneRangeInCluster = 4
        self.targetSparsity = 0.5
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

        # # Prune the model
        # if self.mode == 'resnet' or self.mode == 'tiny' or self.mode == 'test'
        #     for m in self.model.modules():
        #         if isinstance(m, resnet.BasicBlock):
        #             custom_pruning.applyBalancedPruning(m.convBN1[0],
        #                                                "weight",
        #                                                clusterSize=self.pruneClusterSize,
        #                                                 pruneRangeInCluster=self.pruneRangeInCluster,
        #                                                 sparsity=self.targetSparsity
        #                                                 )
        #             custom_pruning.applyBalancedPruning(m.convBN2[0],
        #                                                "weight",
        #                                                clusterSize=self.pruneClusterSize,
        #                                                pruneRangeInCluster=self.pruneRangeInCluster,
        #                                                 sparsity=self.targetSparsity
        #                                                 )
        #             if isinstance(m.shortcut, cm.ConvBN):
        #                 custom_pruning.applyBalancedPruning(m.shortcut[0],
        #                                                    "weight",
        #                                                    clusterSize=self.pruneClusterSize,
        #                                                     pruneRangeInCluster=self.pruneRangeInCluster,
        #                                                     sparsity=self.targetSparsity
        #                                                     )
        # if self.mode == 'resnet' or self.mode == 'tiny' or self.mode == 'test'
        #     for m in self.model.modules():
        #         if isinstance(m, resnet.BasicBlock):
        #             custom_pruning.applyBalancedPruning(m.convBN1[0],
        #                                                "weight",
        #                                                clusterSize=self.pruneClusterSize,
        #                                                 pruneRangeInCluster=self.pruneRangeInCluster,
        #                                                 sparsity=self.targetSparsity
        #                                                 )
        #             custom_pruning.applyBalancedPruning(m.convBN2[0],
        #                                                "weight",
        #                                                clusterSize=self.pruneClusterSize,
        #                                                pruneRangeInCluster=self.pruneRangeInCluster,
        #                                                 sparsity=self.targetSparsity
        #                                                 )
        #             if isinstance(m.shortcut, cm.ConvBN):
        #                 custom_pruning.applyBalancedPruning(m.shortcut[0],
        #                                                    "weight",
        #                                                    clusterSize=self.pruneClusterSize,
        #                                                     pruneRangeInCluster=self.pruneRangeInCluster,
        #                                                     sparsity=self.targetSparsity
        #                                                     )
        # elif self.mode == 'conv':
        #     custom_pruning.applyBalancedPruning(self.model.block[0],
        #                                        "weight",
        #                                        clusterSize=self.pruneClusterSize,
        #                                         pruneRangeInCluster=self.pruneRangeInCluster,
        #                                         sparsity=self.targetSparsity
        #                                         )
        # elif self.mode == 'seq':
        #     custom_pruning.applyBalancedPruning(self.model.convBN1[0],
        #                                        "weight",
        #                                        clusterSize=self.pruneClusterSize,
        #                                         pruneRangeInCluster=self.pruneRangeInCluster,
        #                                         sparsity=self.targetSparsity
        #                                         )
        #     custom_pruning.applyBalancedPruning(self.model.convBN2[0],
        #                                        "weight",
        #                                         clusterSize=self.pruneClusterSize,
        #                                         pruneRangeInCluster=self.pruneRangeInCluster,
        #                                         sparsity=self.targetSparsity
        #                                         )
        # elif self.mode == 'avglinear':
        #     custom_pruning.applyBalancedPruning(self.model.fc[0],
        #                                        "weight",
        #                                         clusterSize=self.pruneClusterSize,
        #                                         pruneRangeInCluster=self.pruneRangeInCluster,
        #                                         sparsity=self.targetSparsity
        #                                         )

        for module in self.model.modules():
            if isinstance(module, (nnqat.Linear, nnqat.Conv2d, nniqat.ConvBn2d, nniqat.ConvBnReLU2d, nniqat.ConvReLU2d, nniqat.LinearReLU)):
                custom_pruning.applyBalancedPruning(module,
                                                    "weight",
                                                    clusterSize=self.pruneClusterSize,
                                                    pruneRangeInCluster=self.pruneRangeInCluster,
                                                    sparsity=self.targetSparsity
                                                    )


    def trace(self, dirname: str, fileBaseName: str):
        if self.mode == 'test':
            dummyInput = torch.rand(size=[1, 3, 32, 32]) * (-1.0) + 4.0
        elif self.mode == 'add':
            # dummyInput = (torch.rand(size=[1, 4, 8, 8]) * (-1.0) + 4.0,
            #               torch.rand(size=[1, 4, 8, 8]) * (-1.0) + 4.0)
            dummyInput = (torch.rand(size=[1, 16, 8, 8]) * (4.0) - 2.0,
                          torch.rand(size=[1, 16, 8, 8]) * (2.0) - 1.0)
        elif self.mode == 'restest_cifar10':
            dummyInput = torch.rand(size=[1, 3, 32, 32]) * 4.0 - 2.0
        elif self.mode == 'restest_imagenet':
            dummyInput = torch.rand(size=[1, 3, 56, 56]) * 4.0 - 2.0
        elif self.mode == 'resnet50_conv12':
            dummyInput = torch.rand(size=[1,256,56, 56]) * 4.0 - 2.0
        else:
            dummyInput = torch.rand(size=[1, 4, 8, 8]) * (-1.0) + 4.0

        """
        Run inference twice. only save the input and output on the second try
        The first run is to calibrate the quantization parameters
        """
        self.model.eval()
        if isinstance(dummyInput, tuple):
            self.model(*dummyInput)
        else:
            self.model(dummyInput)

        self.model.apply(torch.quantization.disable_observer)

        # quantized_model = torch.quantization.convert(self.model.eval(), inplace=False)
        if isinstance(dummyInput, tuple):
            dummyOutput = self.model(*dummyInput)
        else:
            dummyOutput = self.model(dummyInput)
        print("Saves the dummy input and output")
        blobPath = os.path.join(dirname, fileBaseName + '_inout.yaml')
        blobFile = open(blobPath, 'w')
        blobDict: dict = {}

        # Save inputs
        if isinstance(dummyInput, tuple):
            id = 0
            for input in dummyInput:
                idText = 'input_'+str(id)
                blobDict[idText] = input.view(input.numel()).tolist()
                id += 1
        else:
            inputArray = dummyInput.view(dummyInput.numel()).tolist()
            blobDict['input'] = inputArray

        # save outputs
        outputArray = dummyOutput.view(dummyOutput.numel()).tolist()
        blobDict['output'] = outputArray

        # We want list to be dumped as in-line format, hence the choice of the default_flow_style
        # See https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
        yaml.dump(blobDict, blobFile, default_flow_style=None)
        tracer = Tracer(self.model)
        tracer.traceModel(dummyInput)
        tracer.annotate(numMemRegions=3)
        tracer.dump(dirname, fileBaseName)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="testTracer")
    parser.add_argument('--mode', type=str, choices=['test', 'tiny', 'conv', 'maxpool', 'add', 'avg', 'seq', 'avglinear',
                                                     'restest_cifar10', 'restest_imagenet', 'resnet50_conv12'], default='conv',
                        help='Mode. Valid choices are test, tiny, conv, maxpool, add, avg, seq, restest_cifar10, restest_imagenet,'
                             'resnet50_conv12, and avglinear')
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.mode == 'test':
         fileBaseName = 'testTrace'
    elif args.mode == 'tiny':
        fileBaseName = 'tinyTrace'
    elif args.mode == 'conv':
        fileBaseName = 'convTrace'
    elif args.mode == 'maxpool':
        fileBaseName = 'mp'
    elif args.mode == 'add':
        fileBaseName = 'addbig'
    elif args.mode == 'avg':
        fileBaseName = 'avg'
    elif args.mode == 'seq':
        fileBaseName = 'seq'
    elif args.mode == 'avglinear':
        fileBaseName = 'avglinear'
    elif args.mode == 'restest_cifar10':
        fileBaseName = 'restest_resnet56_like'
    elif args.mode == 'restest_imagenet':
        fileBaseName = 'restest_imagenet'
    elif args.mode == 'resnet50_conv12':
        fileBaseName = 'resnet50_conv12'
    else:
        print(args.mode)
        raise ValueError("Unsupported mode")
    test = TracerTest(mode=args.mode)
    test.trace(dirname='.', fileBaseName=fileBaseName)

