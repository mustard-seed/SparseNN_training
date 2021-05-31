from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.intrinsic
import torch.nn.intrinsic.qat as nniqat
import torch.nn.qat as nnqat
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

def quantize_bias(module, _bias):
    bias = None
    if _bias is not None:
        if module.qconfig is not None:
            input_qparams = module.input_quantization.calculate_qparams()
            weight_qparams = module.weight_fake_quant.calculate_qparams()
            input_scale = input_qparams[0]
            weight_scale = weight_qparams[0]
            combined_scale = float(input_scale) * float(weight_scale)
            # Assume 16-bit accumulator
            QUANT_MIN = -32768
            QUANT_MAX = 32767
            bias = torch.fake_quantize_per_tensor_affine(_bias, combined_scale, 0, QUANT_MIN, QUANT_MAX)
    return bias

def input_observer_hook(mod: nn.Module, input):
    testInput = input
    if isinstance(input, tuple):
        testInput=input[0]
    return mod.input_quantization(testInput)

def add_input_quantization(mod: nn.Module):
    # Instantiate input observer, for keeping track of the input quantization specs
    if mod.qconfig is not None:
        mod.add_module('input_quantization', mod.qconfig.activation())
        # Only update the observer, but do not apply quantization
        mod.input_quantization.enable_fake_quant(False)
        mod.register_forward_pre_hook(input_observer_hook)

class Linear(nnqat.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None):
        super().__init__(in_features, out_features, bias, qconfig)
        add_input_quantization(self)

    def floatForward(self, input):
        weight = self.weight_fake_quant(self.weight)
        bias = quantize_bias(self, self.bias)
        return F.linear(input, weight, bias)

    def forward(self, input):
        return self.activation_post_process(self.floatForward(input))

class Conv2d(nnqat.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig)
        add_input_quantization(self)

    def floatForward(self, input):
        weight = self.weight_fake_quant(self.weight)
        bias = quantize_bias(self, self.bias)
        conv = None
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            conv = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            conv = F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        return conv

    def forward(self, input):
        return self.activation_post_process(self.floatForward(input))

class ConvBn2d(nniqat.ConvBn2d):
    def __init__(self,
                 # Conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 # bias: None, only support Conv with no bias
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        # Instantiate the base object
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups,
            padding_mode,
            eps, momentum,
            freeze_bn,
            qconfig
        )
        # Instantiate input observer, for keeping track of the input quantization specs
        add_input_quantization(self)

    def _forward(self, input):
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and not self.freeze_bn and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # we use running statistics from the previous batch, so this is an
        # approximation of the approach mentioned in the whitepaper, but we only
        # need to do one convolution in this case instead of two
        running_std = torch.sqrt(self.running_var + self.eps)
        scale_factor = self.gamma / running_std
        scaled_weight = self.weight * scale_factor.reshape([-1, 1, 1, 1])
        conv = self._conv_forward(input, self.weight_fake_quant(scaled_weight))
        bias = None
        if self.training and not self.freeze_bn:
            # recovering original conv to get original batch_mean and batch_var
            conv_orig = conv / scale_factor.reshape([1, -1, 1, 1])
            batch_mean = torch.mean(conv_orig, dim=[0, 2, 3])
            batch_var = torch.var(conv_orig, dim=[0, 2, 3], unbiased=False)
            n = float(conv_orig.numel() / conv_orig.size()[1])
            unbiased_batch_var = batch_var * (n / (n - 1))
            batch_rstd = torch.ones_like(batch_var, memory_format=torch.contiguous_format) \
                         / torch.sqrt(batch_var + self.eps)

            rescale_factor = running_std * batch_rstd
            conv = conv * rescale_factor.reshape([1, -1, 1, 1])
            # conv = conv + (self.beta - self.gamma * batch_mean * batch_rstd).reshape([1, -1, 1, 1])
            bias = (self.beta - self.gamma * batch_mean * batch_rstd).reshape([1, -1, 1, 1])
            self.running_mean = exponential_average_factor * batch_mean.detach() + \
                (1 - exponential_average_factor) * self.running_mean
            self.running_var = exponential_average_factor * unbiased_batch_var.detach() + \
                (1 - exponential_average_factor) * self.running_var
        else:
            # conv = conv + (self.beta - self.gamma * self.running_mean /
            #               running_std).reshape([1, -1, 1, 1])
            bias = (self.beta - self.gamma * self.running_mean / running_std).reshape([1, -1, 1, 1])

        # Quantize the bias
        bias = quantize_bias(self, bias)
        conv = conv + bias
        return conv

class ConvBnReLU2d(ConvBn2d):
    _FLOAT_MODULE = torch.nn.intrinsic.ConvBnReLU2d
    def __init__(self,
                 # Conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 # bias: None, only support Conv with no bias
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups,
                                           padding_mode, eps, momentum,
                                           freeze_bn,
                                           qconfig)

    def forward(self, input):
        return self.activation_post_process(
            F.relu(super(ConvBnReLU2d, self)._forward(input)))


class ConvReLU2d(Conv2d):
    _FLOAT_MODULE = torch.nn.intrinsic.ConvReLU2d
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 qconfig=None):
        super(ConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode,
                                         qconfig=qconfig)

    def forward(self, input):
        return self.activation_post_process(F.relu(
            super().floatForward(input)))

class LinearReLU(Linear):
    _FLOAT_MODULE = torch.nn.intrinsic.LinearReLU
    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None):
        super().__init__(in_features, out_features, bias, qconfig)

    def forward(self, input):
        return self.activation_post_process(F.relu(self.floatForward(input)))