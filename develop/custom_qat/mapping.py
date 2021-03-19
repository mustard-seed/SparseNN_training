from torch import nn

import torch.nn.intrinsic as nni
import torch.nn.intrinsic.qat as nniqat
import torch.nn.qat as nnqat
import custom_qat.modules as customqat

from torch.quantization import QuantStub, DeQuantStub

# Map for swapping float module to qat modules
CUSTOM_QAT_MODULE_MAPPING = {
    nn.Linear: customqat.Linear,
    nn.Conv2d: customqat.Conv2d,
    # Fused modules:
    nni.ConvBn2d: customqat.ConvBn2d,
    nni.ConvBnReLU2d: customqat.ConvBnReLU2d,
    nni.ConvReLU2d: customqat.ConvReLU2d,
    nni.LinearReLU: customqat.LinearReLU
}