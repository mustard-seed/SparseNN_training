import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    """
    Applies 2d convolution, followed by batch normalization and ReLu to the incoming data

    Args:
      in_planes: Number of input channels
      out_planes: Number of output channels
      kernel_size: Size of the kernel.
        Default: 3
      stride: 2d convolution stride
        Default: 1
      groups: Number of channel groups
        Default: 1
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class ConvReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class LinearReLU(nn.Sequential):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearReLU, self).__init__(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU(inplace=False)
        )
