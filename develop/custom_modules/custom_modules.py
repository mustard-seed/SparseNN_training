import torch.nn as nn
import torch

from torch.quantization import QuantStub, DeQuantStub
import torch.nn.quantized


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
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, bias=False):
        padding = (kernel_size - 1) // 2
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class LinearReLU(nn.Sequential):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearReLU, self).__init__(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU(inplace=False)
        )

class ConvBN(nn.Sequential):
    """
    Applies 2d convolution, followed by batch normalization to the incoming tensor

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
        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1)
        )

# class EltwiseAdd(nn.Module):
#     def __init__(self, relu=False):
#         super().__init__()
#         self.relu = relu
#         # self.quant = QuantStub()
#         self.block = torch.nn.quantized.FloatFunctional()
#
#     def forward(self, left_input: torch.Tensor, right_input: torch.Tensor) -> torch.Tensor:
#         output = self.block.add_relu(left_input, right_input) if self.relu is True else self.block.add(left_input, right_input)
#         return output

class EltwiseAdd(nn.Module):
    def __init__(self, relu=False):
        super().__init__()
        self.relu = relu
        self.quant = QuantStub()
        self.leftInputQuant = QuantStub()
        self.rightInputQuant = QuantStub()
    def forward(self, left_input: torch.Tensor, right_input: torch.Tensor) -> torch.Tensor:
        # Take the shift to match the binary point into account
        left = self.leftInputQuant(left_input)
        right = self.rightInputQuant(right_input)
        if hasattr(self.leftInputQuant, 'qconfig') and hasattr(self.rightInputQuant, 'qconfig'):
            if self.leftInputQuant.qconfig is not None and self.rightInputQuant.qconfig is not None:
                leftScale, _ = self.leftInputQuant.activation_post_process.calculate_qparams()
                rightScale, _ = self.rightInputQuant.activation_post_process.calculate_qparams()
                # Pick the greater of the two to be the scale
                scale = max(leftScale.view(1)[0].item(), rightScale.view(1)[0].item())
                quantMin = self.leftInputQuant.activation_post_process.quant_min
                quantMax = self.leftInputQuant.activation_post_process.quant_max
                left = torch.fake_quantize_per_tensor_affine(left, scale, 0, quantMin, quantMax)
                right = torch.fake_quantize_per_tensor_affine(right, scale, 0, quantMin, quantMax)

        output = left + right
        if self.relu is True:
            output = torch.nn.functional.relu(output)
        # New addition on 20200820: Quantize the output
        output = self.quant(output)

        return output

class MaxPool2dRelu(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, relu=False):
        super().__init__(kernel_size, stride, padding, dilation,
                 return_indices, ceil_mode)
        self.relu = relu
        self.quant = QuantStub()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.relu is True:
            output = torch.nn.functional.relu(output)
        output = self.quant(output)

        return output

class AvgPool2dRelu(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None, relu=False):
        super().__init__(kernel_size, stride, padding, ceil_mode,
                 count_include_pad, divisor_override)
        self.relu = relu
        self.quant = QuantStub()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.relu is True:
            output = torch.nn.functional.relu(output)
        output = self.quant(output)

        return output

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.flatten(input, 1)
        return output
