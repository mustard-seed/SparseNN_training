"""
Package for tracing quantized-pruned models in their PyTorch execution order
"""
from torch import Tensor as T
from torch import nn as nn

import custom_modules.custom_modules as cm
import quantization.quantization as qt

from typing import List
from easydict import EasyDict as edict

def find_conv1d_output_dim(inputDim: int,
                           inputBorderPadding: int,
                           inputTransConvPadding: int,
                           kernelSize: int,
                           kernelStride: int,
                           ) -> int:
    assert ((inputTransConvPadding + 1) * inputDim + 2 * inputBorderPadding - kernelSize) % kernelStride == 0, \
        "Incompatibility among input dimension, kernel stride, kernel size, trans-conv padding, and border padding"
    outputDim = ((inputTransConvPadding + 1) * inputDim + 2 * inputBorderPadding - kernelSize) // kernelStride + 1

    return outputDim

class LayerInfo(edict):
    """
    Base class data structure class that encapsulates the information of a layer to be exported
    """
    def __init__(self,
                 operationType : str,
                 outputFracBits : int,
                 outputRelu : bool
                 ):
        super().__init__()
        self.operationType = operationType
        self.outputFracBits = outputFracBits
        self.outputRelu = outputRelu

        self.outputChannels = None
        self.outputHeight = None
        self.outputWidth = None
        self.outputNextNumGroups = None
        self.outputCurrentNumGroups = None
        self.outputCanBeSparse = False

        self.outputMemoryLocation = None

        self.inputFracBits = []
        self.inputMemoryLocations = []

class ConvInfo(LayerInfo):
    """
    Used for housing convolution-bn-relu, linear-bn-relu information
    """
    def __init__(self,
                 outputFracBits : int,
                 outputChannels : int,
                 outputRelu : bool,

                 inputFracBits : int,
                 inputHeight : int,
                 inputWidth : int,
                 inputBorderPadding : int,
                 inputTransConvPadding : int,
                 inputChannels : int,

                 weightFracBits : int,
                 kernelSize : int,
                 kernelStride : int,

                 channelGroups : int
                 ):
        super().__init__(operationType='conv',
                         outputFracBits=outputFracBits,
                         outputRelu=outputRelu)
        self.outputWidth = find_conv1d_output_dim(inputDim=inputWidth,
                                             inputBorderPadding=inputBorderPadding,
                                             inputTransConvPadding=inputTransConvPadding,
                                             kernelSize=kernelSize,
                                             kernelStride=kernelStride)
        self.outputHeight = find_conv1d_output_dim(inputDim=inputHeight,
                                             inputBorderPadding=inputBorderPadding,
                                             inputTransConvPadding=inputTransConvPadding,
                                             kernelSize=kernelSize,
                                             kernelStride=kernelStride)
        self.outputChannels = outputChannels

        if len(self.inputFracBits) == 0:
            self.inputFracBits.append(inputFracBits)
        else:
            self.inputFracBits[0] = inputFracBits
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        self.inputBorderPadding = inputBorderPadding
        self.inputTransConvPadding = inputTransConvPadding
        self.inputChannels = inputChannels

        self.weightFracBits = weightFracBits
        self.kernelSize = kernelSize
        self.kernelStride = kernelStride
        self.weightByteFileStartPosition = None
        self.biasByteFileStartPosition = None

        assert (inputChannels % channelGroups == 0) and (outputChannels % channelGroups == 0), \
            'channelGroups does not divide into inputChannels or outputChannels'
        self.outputCurrentNumGroups = channelGroups

class MaxPoolInfo(LayerInfo):
    def __init__(self,
                 outputFracBits: int,
                 outputRelu: bool,

                 inputFracBits: int,
                 inputHeight: int,
                 inputWidth: int,
                 inputBorderPadding: int,
                 inputChannels: int,

                 kernelSize: int,
                 kernelStride: int
                 ):
        super().__init__(operationType='maxpool',
                         outputFracBits=outputFracBits,
                         outputRelu=outputRelu)

        self.outputChannels = inputChannels
        self.outputWidth = find_conv1d_output_dim(inputDim=inputWidth,
                                                  inputBorderPadding=inputBorderPadding,
                                                  inputTransConvPadding=0,
                                                  kernelSize=kernelSize,
                                                  kernelStride=kernelStride)
        self.outputHeight = find_conv1d_output_dim(inputDim=inputHeight,
                                                   inputBorderPadding=inputBorderPadding,
                                                   inputTransConvPadding=0,
                                                   kernelSize=kernelSize,
                                                   kernelStride=kernelStride)
        self.outputCurrentNumGroups = 1

        if len(self.inputFracBits) == 0:
            self.inputFracBits.append(inputFracBits)
        else:
            self.inputFracBits[0] = inputFracBits
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        self.inputBorderPadding = inputBorderPadding
        self.channels = inputChannels

        self.kernelSize = kernelSize
        self.kernelStride = kernelStride

class EltAddInfo(LayerInfo):
    def __init__(self,
                 outputFracBits: int,
                 outputRelu: bool,

                 inputHeight: int,
                 inputWidth: int,
                 inputChannels: int,

                 inputLeftFracBits: int,
                 inputRightFracBits: int
                 ):
        super().__init__(operationType='eltadd',
                         outputFracBits=outputFracBits,
                         outputRelu=outputRelu)

        self.outputChannels = inputChannels
        self.outputWidth = inputWidth
        self.outputHeight = inputHeight
        self.outputCurrentNumGroups = 1

        if len(self.inputFracBits) == 0:
            self.inputFracBits.append(inputLeftFracBits)
            self.inputFracBits.append(inputRightFracBits)
        else:
            self.inputFracBits[0] = inputLeftFracBits
            self.inputFracBits[1] = inputRightFracBits
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        self.channels = inputChannels

class TraceDNN:
    ID_IDX = 0
    PRECISION_IDX = 0
    def __init__(self, module: nn.Module):
        self.module = module

        # Run-time arguments
        # Running-counter of the layer IDs
        self.layerID = 0
        self.layerCount = 0
        # List of handles for forward-hooks. Used to delete the forward hooks after the first run.
        self.hookHandles = []
        # List of layers
        self.layerList = []
        # Input adjacency list of registered layer
        self.inputAdjacencies = []
        # Output adjacency list of registered layers
        self.outputAdjacencies = []


    def insertLayer(self, layer: LayerInfo) -> None:
        """
        Insert a new layer into the graph
        :param layer: An LayerInfo object
        :return: None
        """
        pass

    def addForwardEdge(self, sourceLayerId: int, sinkLayerId: int) -> None:
        """
        Add a forward dependency
        :param sourceLayerId: Id of the layer that generates the tensor during inference.
        :param sinkLayerId: Id of the layer that receives the tensor during inference.
        :return: None
        """
        pass

    def addBackward(self, sourceLayerId: int, sinkLayerId: int) -> None:
        """
        Add a backward endge
        :param sourceLayerId: Id of the layer that generates the tensor during inference. It will be at the end of the edge
        :param sinkLayerId: Id of the layer that receives the tensor during inference. It will be the emitter of the edge
        :return: None
        """
        pass

    def resetAnnotation(self) -> None:
        """
        Clears all memory and sparsification flag of layers in the graph
        :return: Hone
        """
        pass

    def annotate(self, numMemBanks: int) -> None:
        """
        Annotates memory and sparsification fields of layers in the graph
        :param numMemBanks:
        :return: None
        """
        pass

    def dump(self, fileStream = None) -> str:
        """
        Dumps the DNN IR in the inserted execution order as a YAML string.
        If a file stream is provided, the string will be written to the file.
        :param fileStream: Optional. File stream to dump the YAML string to.
        :return: The YAML string
        """
        pass

    def traceHook(self, module, input: T, output: T) -> None:
        """
        Pytorch NN module forward hook. Used to intercept NN layer execution order and dependency. This function will be
            called by PyTorch during inference.
        :param module: The PyTorch NN module to be hooked.
        :param input: Input tensor. Can be a tuple if the module has multiple inputs.
        :param output:
        :return:
        """
        # Modify the output
        # output[0]: layer id of the layer
        # output[1]: INT8 activation precision

        # Extract the information: input precision(s), input id
        inputIDs = []
        inputPrecisions = []
        if isinstance(input, T):
            nElements = input.numel()
            inputIDs.append(input.view(nElements)[self.ID_IDX])
            inputPrecisions.append(input.view(nElements)[self.PRECISION_IDX])
        elif isinstance(input, tuple):
            for tensor in input:
                nElements = tensor.numel()
                inputIDs.append(tensor.view(nElements)[self.ID_IDX])
                inputPrecisions.append(tensor.view(nElements)[self.PRECISION_IDX])
        else:
            raise TypeError('The input argument is neither a tensor nor a tuple of tensors')

        # TODO
        # Instantiate and insert a layer, register input/output adjacencies
        # If this is a convolution-based layer. Even the fused layer types are Conv2d thanks to inheritance
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            pass
        # if this is an average pooling layer
        elif isinstance(module, nn.AvgPool2d):
            pass
        elif isinstance(module, nn.MaxPool2d):
            pass
        elif isinstance(module, cm.EltwiseAdd):
            pass
        else:
            raise TypeError('The tracer hook can only be applied to Conv2d types, Linear types, AvgPool2d types, '
                            'AvgPool2d, and EltwiseAdd types. Input module type is {}'.format(type(module)))

        # Propagate the layer id, and output precision to the next layer
        precisionScale = inputPrecisions[0]
        if hasattr(module, 'activation_post_process'):
            precisionScale = module.activation_post_process.scale.view(1)[0]
        else:
            if isinstance(module, cm.EltwiseAdd):
                # Number of fraction bits = log2(1/scale)
                # Number of interger bits = 8 - number of fraction bits
                # Number of fraction bits new = log2(1/scale_new) = number of fraction bits - 1 = log2(1/scale) - 1
                # log2(scale) + 1 = log2(scale_new)
                # scale_new = 2 * scale
                precisionScale *= 2.0
            else:
                pass
        outputNumel = output.numel()
        output.view(outputNumel)[self.ID_IDX] = self.layerID
        output.view(outputNumel)[self.PRECISION_IDX] = precisionScale

        self.layerID += 1
        self.layerCount += 1


