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

# Define all the classes and methods that this module will be able to export
# TODO: Update this list
__all__ = []


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
                                 stride=stride)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = conv1x1BN(
                in_planes=in_planes,
                out_planes=planes * self.expansion,
                stride=stride
            )

    def forward(self, x):
        out = self.convBN1(x)
        out = self.convBN2(out)
        shortcut = self.shortcut(x)

        out += shortcut
        out = torch.nn.functional.relu(out)
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
                                 stride=stride,
                                 require_relu=True)
        self.convBN2 = conv3x3BN(in_planes=planes,
                                 out_planes=planes,
                                 stride=stride,
                                 require_relu=True)
        self.convBN3 = conv1x1BN(in_planes=planes,
                                 out_planes=planes * self.expansion,
                                 stride=stride,
                                 require_relu=False)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = conv1x1BN(
                in_planes=in_planes,
                out_planes=planes * self.expansion,
                stride=stride
            )

    def forward(self, x):
        out = self.convBN1(x)
        out = self.convBN2(out)
        out = self.convBN3(out)
        shortcut = self.shortcut(x)

        out += shortcut
        out = torch.nn.functional.relu(out)
        return out


class ResNet(nn.Module):
    """
    Custom ResNet class with type B short-cuts
    """

    def __init__(self,
                 block:Union[BasicBlock,BottleneckBlock],
                 num_blocks:List[int],
                 family:str='cifar10'):
        """
        Construct a resnet according to the specifications
        :param block:
        :param num_blocks: Number of residual blocks in each stage
        :param family: The resnet family. Either 'cifar10' or 'imagenet1k'
        """
        super(ResNet, self).__init__()
        assert family == 'cifar10' or family == 'imagenet1k', \
            'Supported families are confined to [cifar10, imagenet1k]'
        self.family = family
        self.stage_planes = [16, 32, 64] if family == 'cifar10' else [64, 128, 256, 512]
        self.stage_strides = [1, 2, 2] if family == 'cifar10' else [1, 2, 2, 2]
        self.num_classes = 10 if self.family == 'cifar10' else 1000
        self.fc_input_number = 64 if self.family == 'cifar10' else 512 * block.expansion

        self.in_planes = 16 if family == 'cifar10' else 64
        self.in_kernel_size = 3 if family == 'cifar10' else 7
        self.in_stride = 1 if family == 'cifar10' else 2
        self.inputConvBNReLu = cm.ConvBNReLU(
            in_planes=3,
            out_planes=self.in_planes,
            stride=self.in_stride,
            kernel_size=self.in_kernel_size
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if family == 'imagenet1k' else None
        self.avgpool = nn.AvgPool2d(1, 1)
        self.fc = nn.Linear(self.fc_input_number, self.num_classes)

        self.stage1 = self._make_stage(
                block=block,
                planes=self.stage_planes[0],
                num_block=num_blocks[0],
                stride=self.stage_strides[0]
            )

        self.stage2 = self._make_stage(
            block=block,
            planes=self.stage_planes[1],
            num_block=num_blocks[1],
            stride=self.stage_strides[1]
        )

        self.stage3 = self._make_stage(
            block=block,
            planes=self.stage_planes[2],
            num_block=num_blocks[2],
            stride=self.stage_strides[2]
        )

        self.stage4 = None if self.family == 'cifar10' else self._make_stage(
            block=block,
            planes=self.stage_planes[3],
            num_block=num_blocks[3],
            stride=self.stage_strides[3]
        )

        self.quant = QuantStub()
        self.deQuant = DeQuantStub()

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
        output = self.inputConvBNReLu(x)
        if self.maxpool is not None:
            output = self.maxpool(output)
        output = self.relu(output)

        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        if self.stage4 is not None:
            output = self.stage4(output)
        output = self.avgpool(output)
        output = self.flatten(output, 1)
        output = self.fc(output)
        output = self.deQuant(x)

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



