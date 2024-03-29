"""
Implement ResNet variants for CIFAR-10 and ImageNet training tasks
Author: James Liu
Inspired by
https://colab.research.google.com/github/seyrankhademi/ResNet_CIFAR10/blob/master/CIFAR10_ResNet.ipynb#scrollTo=V9Y2hYRwB-qg
Also by
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import custom_modules.custom_modules as cm
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.quantization import QuantStub, DeQuantStub
from typing import Union, List
from copy import deepcopy
import math

# Define all the classes and methods that this module will be able to export
# TODO: Update this list
__all__ = ['ResNet', 'cifar_resnet56', 'BasicBlock', 'BottleneckBlock', 'conv1x1BN', 'conv3x3BN']


def conv3x3BN(in_planes, out_planes, stride=1, groups=1, dilation=1, require_relu=False):
    """3x3 convolution with padding followed by batch-normalization"""
    if require_relu is True:
        return cm.ConvBNReLU(in_planes=in_planes,
                         out_planes=out_planes,
                         kernel_size=3,
                         stride=stride,
                         groups=groups
                         )
    else:
        return cm.ConvBN(in_planes=in_planes,
                         out_planes=out_planes,
                         kernel_size=3,
                         stride=stride,
                         groups=groups
                         )

def conv1x1BN(in_planes, out_planes, stride=1, groups=1, dilation=1, require_relu=False):
    """1x1 convolution with padding followed by batch-normalization"""
    if require_relu is True:
        return cm.ConvBNReLU(in_planes=in_planes,
                             out_planes=out_planes,
                             kernel_size=1,
                             stride=stride,
                             groups=groups
                             )
    else:
        return cm.ConvBN(in_planes=in_planes,
                         out_planes=out_planes,
                         kernel_size=1,
                         stride=stride,
                         groups=groups
                         )


class BasicBlock (nn.Module):
    """
    Basic residual block.
    See Figure 5 LEFT in the original ResNet paper (https://arxiv.org/pdf/1512.03385.pdf)

    If the number of feature planes, or feature map width/height change,
    then short-cut is implemented using version B.
    See section 4.1, sub-heading 'Identity vs. Projection Shortcuts' in the original paper
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Gotcha: channel dimension only can change over the first layer
        self.convBN1 = conv3x3BN(in_planes=in_planes,
                                 out_planes=planes,
                                 stride=stride,
                                 require_relu=True)
        self.convBN2 = conv3x3BN(in_planes=planes,
                                 out_planes=planes * self.expansion,
                                 stride=1)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = conv1x1BN(
                in_planes=in_planes,
                out_planes=planes * self.expansion,
                stride=stride
            )
        self.add = cm.EltwiseAdd(relu=True)

    def forward(self, x):
        out = self.convBN1(x)
        out = self.convBN2(out)
        shortcut = self.shortcut(x)

        # out += shortcut
        # out = torch.nn.functional.relu(out)
        out = self.add(out, shortcut)
        return out


class BottleneckBlock (nn.Module):
    """
    Residual block with bottleneck, used in some ResNet variants for ImageNet
    See Figure 5 RIGHT in the original ResNet paper (https://arxiv.org/pdf/1512.03385.pdf)

    If the number of feature planes, or feature map width/height change,
    then short-cut is implemented using version B.
    See section 4.1, sub-heading 'Identity vs. Projection Shortcuts' in the original paper
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckBlock, self).__init__()
        # Gotcha: channel dimension only can change over the first layer
        self.convBN1 = conv1x1BN(in_planes=in_planes,
                                 out_planes=planes,
                                 stride=1,
                                 require_relu=True)
        self.convBN2 = conv3x3BN(in_planes=planes,
                                 out_planes=planes,
                                 stride=stride,
                                 require_relu=True)
        self.convBN3 = conv1x1BN(in_planes=planes,
                                 out_planes=planes * self.expansion,
                                 stride=1,
                                 require_relu=False)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = conv1x1BN(
                in_planes=in_planes,
                out_planes=planes * self.expansion,
                stride=stride
            )
        self.add = cm.EltwiseAdd(relu=True)

    def forward(self, x):
        out = self.convBN1(x)
        out = self.convBN2(out)
        out = self.convBN3(out)
        shortcut = self.shortcut(x)

        # out += shortcut
        # out = torch.nn.functional.relu(out)
        out = self.add(out, shortcut)
        return out


class ResNet(nn.Module):
    """
    Custom ResNet class with type B short-cuts
    """

    def __init__(self,
                 block:Union[BasicBlock,BottleneckBlock],
                 num_blocks:List[int],
                 family: str='cifar10',
                 _flagTest: bool= False,
                 _stage_planes_override=None,
                 _stage_strides_override=None,
                 _avgpool_2d_size_override=None):
        """
        Construct a resnet according to the specifications
        :param block:
        :param num_blocks: Number of residual blocks in each stage
        :param family: The resnet family. Either 'cifar10' or 'imagenet1k' or 'test'
        Family test constraints the input plane to be 32 by 32
        """
        super(ResNet, self).__init__()
        assert family == 'cifar10' or family == 'imagenet1k' or family == 'test', \
            'Supported families are confined to [cifar10, imagenet1k, test]'
        self.family = family
        if family == 'cifar10':
            self.stage_planes = [16, 32, 64]
            self.stage_strides = [1, 2, 2]
            self.num_classes = 10
            self.fc_input_number = 64
            self.in_planes = 16
            self.in_kernel_size = 3
            self.in_stride = 1
            self.avgpool_2d_size = 8
        elif family == 'imagenet1k':
            self.stage_planes = [64, 128, 256, 512]
            self.stage_strides = [1, 2, 2, 2]
            self.num_classes = 1000
            self.fc_input_number = 512 * block.expansion
            self.in_planes = 64
            self.in_kernel_size = 7
            self.in_stride = 2
            self.avgpool_2d_size = 7
        else:
            raise ValueError('Unsupported ResNet variant')

        if _flagTest == True:
            assert (_avgpool_2d_size_override is not None) and \
                   isinstance(_stage_planes_override, list) and isinstance(_stage_strides_override, list), \
                'Not sufficient arguments provided by the test ResNet variant'
            self.stage_planes = deepcopy(_stage_planes_override)
            self.stage_strides = deepcopy(_stage_strides_override)
            self.num_classes = 1000 if family == 'imagenet1k' else 10
            self.fc_input_number = _stage_planes_override[-1] * block.expansion
            self.in_planes = _stage_planes_override[0]
            self.in_kernel_size = 3
            self.in_stride = 1
            self.avgpool_2d_size = _avgpool_2d_size_override

        self.avgpool_divisor = math.ceil(math.pow(2.0, math.log2(self.avgpool_2d_size*self.avgpool_2d_size)))

        self.inputConvBNReLU = cm.ConvBNReLU(
            in_planes=3,
            out_planes=self.in_planes,
            stride=self.in_stride,
            kernel_size=self.in_kernel_size
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpoolrelu = cm.MaxPool2dRelu(kernel_size=3, stride=2, padding=1, relu=False) if family == 'imagenet1k' else None
        self.fc = nn.Linear(self.fc_input_number, self.num_classes)
        self.averagePool = cm.AvgPool2dRelu(kernel_size=self.avgpool_2d_size, divisor_override=self.avgpool_divisor, relu=False)

        assert (len(self.stage_planes) == len(num_blocks)) and (len(self.stage_planes) == len(self.stage_strides)), \
            'Incompatiable num_blocks, stage_planes, stage_strides'
        num_stages = len(self.stage_planes)

        self.stage1 = None
        if num_stages > 0:
            self.stage1 = self._make_stage(
                    block=block,
                    planes=self.stage_planes[0],
                    num_block=num_blocks[0],
                    stride=self.stage_strides[0]
                )

        self.stage2 = None
        if num_stages > 1:
            self.stage2 = self._make_stage(
                block=block,
                planes=self.stage_planes[1],
                num_block=num_blocks[1],
                stride=self.stage_strides[1]
            )

        self.stage3 = None
        if num_stages > 2:
            self.stage3 = self._make_stage(
                block=block,
                planes=self.stage_planes[2],
                num_block=num_blocks[2],
                stride=self.stage_strides[2]
            )

        self.stage4 = None
        if num_stages > 3:
            self.stage4 = self._make_stage(
                block=block,
                planes=self.stage_planes[3],
                num_block=num_blocks[3],
                stride=self.stage_strides[3]
            )

        self.quant = QuantStub()
        self.deQuant = DeQuantStub()
        self.flatten = cm.Flatten()

        # Parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        # Initialize the last BN in each residual layer s.t.
        # initially inference flows through the short-cuts
        for m in self.modules():
            if isinstance(m, BottleneckBlock):
                nn.init.constant_(m.convBN3[1].weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.convBN2[1].weight, 0)


    def _make_stage(self, block, planes, num_block, stride=1):
        strides = [stride] + [1] * (num_block-1)
        layers = []
        for s in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    s
                )
            )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.quant(x)
        output = self.inputConvBNReLU(output)
        if self.maxpoolrelu is not None:
            output = self.maxpoolrelu(output)

        if self.stage1 is not None:
            output = self.stage1(output)
        if self.stage2 is not None:
            output = self.stage2(output)
        if self.stage3 is not None:
            output = self.stage3(output)
        if self.stage4 is not None:
            output = self.stage4(output)
        #output = nn.functional.avg_pool2d(output, output.size()[3])
        output = self.averagePool(output)
        output = self.flatten(output)
        output = self.fc(output)
        output = self.deQuant(output)

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

def cifar_resnet56() -> ResNet:
    """
    Generate and initialize a ResNet-56 network for CIFAR-10
    :return: The ResNet-56 network
    """
    block = BasicBlock
    # Number of blocks per stage. Each stage has two parametrized layers
    num_blocks = [9, 9, 9]
    network = ResNet(block, num_blocks, 'cifar10')
    return network

def imagenet_resnet50() -> ResNet:
    """
    Generate and initialize a ResNet-50 network for ImageNet-1K
    :return: The ResNet-50 network
    """
    block = BottleneckBlock
    num_blocks = [3, 4, 6, 3]
    network = ResNet(block, num_blocks, 'imagenet1k')
    return network



