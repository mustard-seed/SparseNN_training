"""
Package for tracing quantized-pruned models in their PyTorch execution order
"""
import torch
from torch import Tensor as T
from torch import nn as nn
from torch.nn.intrinsic import qat as nniqat
import torch.nn.qat as nnqat
from torch.quantization import QuantStub as QuantStub
from torch.quantization import DeQuantStub as DeQuantStub
from torch.nn.utils import fuse_conv_bn_weights


import custom_modules.custom_modules as cm
import quantization.quantization as custom_quant
# Custom pruning utility, used to extract cluster size, sparsity, prune range
import pruning.pruning as cm_prune

from typing import List, Union, Tuple
from easydict import EasyDict as edict
import yaml
import os
from copy import deepcopy
import numpy as np

def find_conv1d_output_dim(inputDim: int,
                           inputBorderPadding: int,
                           inputTransConvPadding: int,
                           kernelSize: int,
                           kernelStride: int,
                           ) -> int:
    """
    Computes 1D convolution output dimension
    :param inputDim: 1D input length
    :param inputBorderPadding: Padding to be added to the left/right. Assumes symmetrical zero-padding
    :param inputTransConvPadding: How many padding to insert between adjacent input elements to faciliate transpose convolution
    :param kernelSize: Size of the convolution kernel
    :param kernelStride: convolution stride
    :return: 1D convolution output dimension
    """
    # assert (inputDim + (inputDim-1)*inputTransConvPadding + 2 * inputBorderPadding - kernelSize) % kernelStride == 0, \
    #     "Incompatibility among input dimension, kernel stride, kernel size, trans-conv padding, and border padding"
    outputDim = (inputDim + (inputDim-1)*inputTransConvPadding + 2 * inputBorderPadding - kernelSize) // kernelStride + 1

    return outputDim

class LayerInfo(edict):
    """
    Base class data structure class that encapsulates the information of a layer to be exported
    """
    def __init__(self,
                 operationType : str,
                 outputFracBits : int,
                 outputRelu : bool,
                 layerID: int
                 ):
        super().__init__()
        self.operationType: str = operationType
        self.outputFracBits: int = outputFracBits
        self.outputRelu: bool = outputRelu

        self.outputChannels: Union[int, None] = None
        self.outputHeight: Union[int, None] = None
        self.outputWidth: Union[int, None] = None
        self.outputNextNumGroups: Union[int, None] = None
        self.outputCurrentNumGroups: Union[int, None] = None
        self.outputCanBeSparse: bool = False

        self.outputMemoryLocation: Union[int, None] = None

        self.sparseInput: bool = False

        self.inputFracBits: List[int] = []
        self.inputChannels: List[int] = []
        self.inputHeights:  List[int] = []
        self.inputWidths: List[int] = []
        self.inputMemoryLocations: List[int] = []

        self.inputGroupsSeenBySource: List[int] = [1, 1]

        self.layerID = layerID

class QuantStubInfo(LayerInfo):
    def __init__(self,
                 outputFracBits: int,
                 outputChannels: int,
                 outputHeight: int,
                 outputWidth: int,
                 layerID: int
                 ):
        super().__init__(operationType='quantstub',
                         outputFracBits=outputFracBits,
                         outputRelu=False,
                         layerID=layerID)
        self.outputChannels = outputChannels
        self.outputHeight = outputHeight
        self.outputWidth = outputWidth
        self.outputCurrentNumGroups = 1

        self.inputChannels.append(outputChannels)
        self.inputHeights.append(outputHeight)
        self.inputWidths.append(outputWidth)

class DeQuantStubInfo(LayerInfo):
    def __init__(self,
                 inputFracBits: int,
                 inputChannels: int,
                 inputGroupsSeenBySource: int,
                 inputHeight: int,
                 inputWidth: int,
                 layerID: int
                 ):
        super().__init__(operationType='dequantstub',
                         outputFracBits=inputFracBits,
                         outputRelu=False,
                         layerID=layerID)
        self.outputChannels = inputChannels
        self.outputHeight = inputHeight
        self.outputWidth = inputWidth
        self.inputFracBits.append(inputFracBits)
        self.outputCurrentNumGroups = 1
        self.outputNextNumGroups = 1

        self.inputChannels.append(inputChannels)
        self.inputHeights.append(inputHeight)
        self.inputWidths.append(inputWidth)
        self.inputGroupsSeenBySource[0] = inputGroupsSeenBySource

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
                 inputGroupsSeenBySource: int,

                 weightFracBits : int,
                 kernelSize : int,
                 kernelStride : int,
                 hasBias: bool,

                 #SpW arguments:
                 sparsity: float,
                 pruneClusterSize,
                 pruneRangeInCluster,

                 channelGroups : int,

                 layerID: int,
                 isAfterInput: bool
                 ):
        super().__init__(operationType='conv',
                         outputFracBits=outputFracBits,
                         outputRelu=outputRelu,
                         layerID=layerID)
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
        self.inputChannels.append(inputChannels)
        self.inputHeights.append(inputHeight)
        self.inputWidths.append(inputWidth)
        self.inputBorderPadding = inputBorderPadding
        self.inputTransConvPadding = inputTransConvPadding
        self.inputGroupsSeenBySource[0] = inputGroupsSeenBySource

        self.weightFracBits = weightFracBits
        self.kernelSize = kernelSize
        self.kernelStride = kernelStride
        self.weightParameterFileStartPosition: Union[int, None] = None
        self.biasParameterFileStartPosition: Union[int, None] = None
        self.hasBias = hasBias

        self.pruneClusterSize = pruneClusterSize
        self.sparsity = sparsity
        self.pruneRangeInCluster = pruneRangeInCluster

        assert (inputChannels % channelGroups == 0) and (outputChannels % channelGroups == 0), \
            'channelGroups does not divide into inputChannels or outputChannels'
        self.outputCurrentNumGroups = channelGroups
        self.isAfterInput = isAfterInput

class MaxPoolInfo(LayerInfo):
    def __init__(self,
                 outputFracBits: int,
                 outputRelu: bool,

                 inputFracBits: int,
                 inputHeight: int,
                 inputWidth: int,
                 inputBorderPadding: int,
                 inputChannels: int,
                 inputGroupsSeenBySource: int,

                 kernelSize: int,
                 kernelStride: int,

                 layerID: int
                 ):
        super().__init__(operationType='maxpool',
                         outputFracBits=outputFracBits,
                         outputRelu=outputRelu,
                         layerID=layerID)

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
        self.inputChannels.append(inputChannels)
        self.inputHeights.append(inputHeight)
        self.inputWidths.append(inputWidth)
        self.inputBorderPadding = inputBorderPadding
        self.inputGroupsSeenBySource[0] = inputGroupsSeenBySource

        self.kernelSize = kernelSize
        self.kernelStride = kernelStride

class AvgPoolInfo(LayerInfo):
    def __init__(self,
                 outputFracBits: int,
                 outputRelu: bool,

                 inputFracBits: int,
                 inputHeight: int,
                 inputWidth: int,
                 inputBorderPadding: int,
                 inputChannels: int,
                 inputGroupsSeenBySource: int,

                 kernelSize: int,
                 kernelStride: int,

                 divisor: int,

                 layerID: int
                 ):
        super().__init__(operationType='avgpool',
                         outputFracBits=outputFracBits,
                         outputRelu=outputRelu,
                         layerID=layerID)

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
        self.inputChannels.append(inputChannels)
        self.inputHeights.append(inputHeight)
        self.inputWidths.append(inputWidth)
        self.inputBorderPadding = inputBorderPadding
        self.inputGroupsSeenBySource[0] = inputGroupsSeenBySource

        self.kernelSize = kernelSize
        self.kernelStride = kernelStride

        self.divisor = divisor

class EltAddInfo(LayerInfo):
    def __init__(self,
                 outputFracBits: int,
                 outputRelu: bool,

                 inputHeight: int,
                 inputWidth: int,
                 inputChannels: int,
                 inputLeftGroupsSeenBySource: int,
                 inputRightGroupsSeenBySource: int,

                 inputLeftFracBits: int,
                 inputRightFracBits: int,
                 layerID: int
                 ):
        super().__init__(operationType='eltadd',
                         outputFracBits=outputFracBits,
                         outputRelu=outputRelu,
                         layerID=layerID)

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

        self.inputGroupsSeenBySource[0] = inputLeftGroupsSeenBySource
        self.inputGroupsSeenBySource[1] = inputRightGroupsSeenBySource

        for i in range(2):
            self.inputChannels.append(inputChannels)
            self.inputHeights.append(inputHeight)
            self.inputWidths.append(inputWidth)

class TraceDNN:
    ID_IDX = 0
    PRECISION_IDX = 1
    def __init__(self, module: nn.Module,
                 _foldBN: bool=True,
                 _defaultPruneCluster: int=2,
                 _defaultPruneRangeInCluster: int=4):
        self.module = module
        self.module.apply(torch.quantization.disable_observer)

        # Run-time arguments
        # Running-counter of the layer IDs
        self.layerID = 0
        self.layerCount = 0
        # List of handles for forward-hooks. Used to delete the forward hooks after the first run.
        self.hookHandles = []
        # List of layers
        self.layerList : List[LayerInfo] = []
        # Input adjacency list of registered layer. The inputs of a given layer
        self.inputAdjacencies = []
        # Output adjacency list of registered layers. The outputs of a given layer
        self.outputAdjacencies = []

        self.parameterCount: int = 0
        self.parameters: List[T] = []
        self.parameterKeys: List[str] = []

        self.foldBN = _foldBN
        self.defaultPruneCluster = _defaultPruneCluster
        self.defaultPruneRangeInCluster = _defaultPruneRangeInCluster

        self.activation_intercepts = []
        self.interceptLayerID = 0

    def reset(self):
        self.layerID = 0
        self.layerCount = 0
        self.removeHooks()
        self.hookHandles.clear()
        self.layerList.clear()
        self.activation_intercepts.clear()
        self.inputAdjacencies.clear()
        self.outputAdjacencies.clear()
        self.resetAnnotation()
        self.parameterCount = 0

    def removeHooks(self):
        for handle in self.hookHandles:
            handle.remove()

    def insertLayer(self, layer: LayerInfo) -> None:
        """
        Insert a new layer into the graph
        :param layer: An LayerInfo object
        :return: None
        """
        self.layerList.append(layer)

    def addForwardEdge(self, sourceLayerId: int, sinkLayerId: int) -> None:
        """
        Add a forward dependency
        :param sourceLayerId: Id of the layer that generates the tensor during inference.
        :param sinkLayerId: Id of the layer that receives the tensor during inference.
        :return: None
        """
        while len(self.outputAdjacencies) <= sinkLayerId:
            self.outputAdjacencies.append([])
        self.outputAdjacencies[sourceLayerId].append(sinkLayerId)

    def addBackward(self, sourceLayerId: int, sinkLayerId: int) -> None:
        """
        Add a backward endge
        :param sourceLayerId: Id of the layer that generates the tensor during inference. It will be at the end of the edge
        :param sinkLayerId: Id of the layer that receives the tensor during inference. It will be the emitter of the edge
        :return: None
        """
        while len(self.inputAdjacencies) <= sinkLayerId:
            self.inputAdjacencies.append([])
        self.inputAdjacencies[sinkLayerId].append(sourceLayerId)

    def resetAnnotation(self) -> None:
        """
        Clears all memory and sparsification flag of layers in the graph
        :return: Hone
        """
        for module in self.layerList:
            module.outputCanBeSparse = False
            module.outputMemoryLocation = None
            module.inputMemoryLocations.clear()

    def annotate(self, numMemRegions: int) -> None:
        """
        Annotates memory and sparsification fields of layers in the graph
        :param numMemRegions: Number of memory regions available for scheduling
        :return: None
        """
        self.resetAnnotation()
        # List[List[bool]]. Used to indicate how many forward edges of each layer have yet been consumed
        outputConsumedFlags: List[List[bool]] = []
        for i in range(len(self.outputAdjacencies)):
            edges = []
            for j in range(len(self.outputAdjacencies[i])):
                edges.append(False)
            outputConsumedFlags.append(edges)

        # Stack of free memory regions. Initialized as a list of banks
        memoryStack = [i for i in range(numMemRegions)]

        # TODO: The input/output sparse flag annotation mechanism may have some limitations. See descriptions below
        # Output of a given layer can be sparse if all of the children layers are of type Conv2d or Linear
        # If at least one of the input cannot be sparse, then we assume none of the inputs can be sparse
        for idx, layer in enumerate(self.layerList):
            # Memory region allocation and deallocation.

            # Examine the types of the consumer layers
            # 1) to determine whether the output of this layer can be sparse
            # 2) To verify the consistency in the next layers' number of channel groups
            outputCanBeSparse = True
            layer.outputMemoryLocation = -1

            successorList = self.outputAdjacencies[idx]
            # print("Layer {}: type {}, successors {}".format(idx, type(self.layerList[idx]), successorList))
            if len(successorList) > 0:
                # Try to allocate a memory region for the output of the layer
                assert len(memoryStack) > 0, \
                    "Cannot allocate another memory region for the output of layer {}!".format(idx)
                outputMemoryLoc = memoryStack.pop()
                layer.outputMemoryLocation = outputMemoryLoc

                for succIdx in successorList:
                    outputLayer = self.layerList[succIdx]
                    if isinstance(outputLayer, ConvInfo) is False:
                        outputCanBeSparse = False
                    if layer.outputNextNumGroups is None:
                        layer.outputNextNumGroups = outputLayer.outputCurrentNumGroups
                    else:
                        assert layer.outputNextNumGroups == outputLayer.outputCurrentNumGroups, \
                            'Consumer layer\'s number of channel groups are in consistent. Layer: {}'.format(layer)

            if isinstance(layer, DeQuantStubInfo):
                outputCanBeSparse = False

            layer.outputCanBeSparse = outputCanBeSparse

            # Determine whether all the inputs are sparse and dellocate used up inputs
            predecessorList = self.inputAdjacencies[idx]
            sparseInput = True
            for predIdx in predecessorList:
                # Grab the input memory location and append it to the input memory region list of the current layer
                inputLayer = self.layerList[predIdx]
                inputMemoryLoc = inputLayer.outputMemoryLocation
                assert inputMemoryLoc is not None, 'Input memory location is not allocated!'
                layer.inputMemoryLocations.append(inputMemoryLoc)
                outputConsumedFlags[predIdx].pop()
                if len(outputConsumedFlags[predIdx]) == 0:
                    # Push the free region back to the flag
                    memoryStack.append(inputMemoryLoc)
                if inputLayer.outputCanBeSparse is False:
                    sparseInput = False

            layer.sparseInput = sparseInput

    def dumpTrace(self, fileStream) -> str:
        """
        Dumps the DNN trace in the inserted execution order as a YAML string.
        If a file stream is provided, the string will be written to the file.
        :param fileStream: File stream to dump the YAML string to.
        :return: The YAML string
        """
        # Generate a list of item views
        layerDict = {}
        for layer in self.layerList:
            # Need to convert dict.items() to an actual list, otherwise it is not picklable
            # See this SO post: https://stackoverflow.com/questions/54658048/typeerror-cant-pickle-dict-items-objects
            layerDict[layer.layerID] = layer.__dict__

        # We want list to be dumped as in-line format, hence the choice of the default_flow_style
        # See https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
        return yaml.dump(layerDict, fileStream, default_flow_style=None)

    def dumpParameters(self, filename:str) -> None:
        """
        Flatten the parameter tensors, join them, and save the values to a yaml file
        :param fileStream:
        :return:
        """
        # parameterDict: dict = {}
        # for idx, blob in enumerate(self.parameters):
        #     key = self.parameterKeys[idx]
        #     data = blob.view(blob.numel()).tolist()
        #     parameterDict[key] = data

        # We want list to be dumped as in-line format, hence the choice of the default_flow_style
        # See https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
        # yaml.dump(parameterDict, filename, default_flow_style=None)
        np.savez_compressed(filename,
            **{self.parameterKeys[idx]: blob.view(blob.numel()).detach().numpy() for idx, blob in enumerate(self.parameters)})
        np.savez_compressed(filename,
                            **{self.parameterKeys[idx]: blob.view(blob.numel()).detach().numpy() for idx, blob in
                               enumerate(self.parameters)})

    def dump(self, filePath: str, fileNameBase: str) -> None:
        """
        Saves the model trace and parameters as YAML files.
        Trace file will be saved as filePath/fileNameBase_trace.yaml
        Parameter file will be saved as filePath/fileNameBase_parameters.yaml
        :param filePath:
        :param fileNameBase:
        :return: None
        """
        fullTracePath = os.path.join(filePath, fileNameBase+'_trace.yaml')
        fullParameterPath = os.path.join(filePath, fileNameBase+'_parameters')
        traceFile = open(fullTracePath, 'w')
        # parameterFile = open(fullParameterPath, 'w')

        self.dumpTrace(traceFile)
        print("Tracer: saved trace to {}".format(fullTracePath))
        self.dumpParameters(fullParameterPath)
        print("Tracer: saved data to {}".format(fullParameterPath+'.npz'))

        traceFile.close()
        #parameterFile.close()

    def traceHook(self, module, input: Union[T, Tuple[T, T]], output: T):
        """
        Pytorch NN module forward hook. Used to intercept NN layer execution order and dependency. This function will be
            called by PyTorch during inference.
        TODO: Add support for transposed convolution
        TODO: Add support fof MaxPool2d properly
        :param module: The PyTorch NN module to be hooked.
        :param input: Input tensor. Can be a tuple if the module has multiple inputs.
        :param output:
        :return: The modified output.
        """
        # Modify the output
        # output[0]: layer id of the layer
        # output[1]: INT8 activation precision

        print ("Tracing module {}. Type; {}. {}".format(self.layerID, type(module), module))
        # Extract the information: input precision(s), input id
        inputIDs = []
        inputPrecisions = []
        inputChannels = []
        inputGroupsSeenbySource = []
        inputHeights = []
        inputWidths = []
        input0FracBits = None
        outputPrecisionScale = None
        isAfterInput = False
        if isinstance(module, QuantStub) is False:
            if isinstance(input, T):
                nElements = input.numel()
                idx = int(input.view(nElements)[self.ID_IDX].item())
                # print('Input idx: {}'.format(idx))
                inputIDs.append(idx)
                inputPrecision = input.view(nElements)[self.PRECISION_IDX].item()
                # print('Input precision: {}'.format(inputPrecision))
                inputPrecisions.append(inputPrecision)
                inputChannels.append(self.layerList[idx].outputChannels)
                inputGroupsSeenbySource.append(self.layerList[idx].outputCurrentNumGroups)
                inputHeights.append(self.layerList[idx].outputHeight)
                inputWidths.append(self.layerList[idx].outputWidth)
                if (self.layerList[idx].operationType == 'quantstub'):
                    isAfterInput = True
            elif isinstance(input, tuple):
                for tensor in input:
                    nElements = tensor.numel()
                    # print('Input: {}'.format(tensor.view(nElements)[0:2]))
                    idx = int(tensor.view(nElements)[self.ID_IDX].item())
                    # print('Input idx: {}'.format(idx))
                    inputIDs.append(idx)
                    inputPrecision = tensor.view(nElements)[self.PRECISION_IDX].item()
                    # print('Input precision: {}'.format(inputPrecision))
                    inputPrecisions.append(tensor.view(nElements)[self.PRECISION_IDX])
                    inputChannels.append(self.layerList[idx].outputChannels)
                    inputGroupsSeenbySource.append(self.layerList[idx].outputCurrentNumGroups)
                    inputHeights.append(self.layerList[idx].outputHeight)
                    inputWidths.append(self.layerList[idx].outputWidth)

                    if (self.layerList[idx].operationType == 'quantstub'):
                        isAfterInput = True
            else:
                raise TypeError('The input argument is neither a tensor nor a tuple of tensors')

            input0FracBits = torch.round(torch.log2(1.0 / inputPrecisions[0])).view(1)[0].item()
            outputPrecisionScale = inputPrecisions[0]

        # Determine output precision scales
        if hasattr(module, 'activation_post_process'):
            outputPrecisionScale = module.activation_post_process.scale.view(1)
        elif isinstance(module, cm.EltwiseAdd):
            outputPrecisionScale = module.quant.activation_post_process.scale.view(1)
        else:
            if isinstance(module, (cm.MaxPool2dRelu, cm.AvgPool2dRelu)):
                # Number of fraction bits = log2(1/scale)
                # Number of interger bits = 8 - number of fraction bits
                # Number of fraction bits new = log2(1/scale_new) = number of fraction bits - 1 = log2(1/scale) - 1
                # log2(scale) + 1 = log2(scale_new)
                # scale_new = 2 * scale
                # iprecision0 = inputPrecisions[0].view(1)[0].item()
                # iprecision1 = inputPrecisions[1].view(1)[0].item()
                # import math
                # if  math.isclose(iprecision0, iprecision1):
                #     outputPrecisionScale *= 2.0
                # else:
                #     outputPrecisionScale = torch.tensor(iprecision1) if iprecision1 > iprecision0 else torch.tensor(iprecision0)
                outputPrecisionScale = module.quant.activation_post_process.scale.view(1)
            else:
                pass
        outputFracBits = int(torch.round(torch.log2(1.0 / outputPrecisionScale)).view(1)[0].item())

        # Instantiate and insert a layer, register input/output adjacencies
        # If this is a convolution-based layer. Even the fused layer types are Conv2d thanks to inheritance
        newLayer = None
        outputChannels = output.size()[1]
        outputRelu = False

        # For the list that convolution layer-like types after qat is applied,
        # see
        # https://github.com/pytorch/pytorch/blob/20ac7362009dd8e0aca6e72fc9357773136a83b8/torch/quantization/quantization_mappings.py#L54
        if isinstance(module, (nnqat.Linear, nnqat.Conv2d, nniqat.ConvBn2d, nniqat.ConvBnReLU2d, nniqat.ConvReLU2d, nniqat.LinearReLU)):
            if isinstance(module, (nniqat.ConvReLU2d, nniqat.LinearReLU, nniqat.ConvBnReLU2d)):
                outputRelu = True

            # Determine padding and kernel size
            padding = 0
            kernelSize = inputWidths[0]
            kernelStride = kernelSize
            groups = 1
            if isinstance(module, (nnqat.Conv2d, nniqat.ConvBn2d, nniqat.ConvBnReLU2d, nniqat.ConvReLU2d)):
                # Extract the padding for convolution.
                # Assume that the horizontal and vertical paddings are the same
                padding = module.padding[0]
                kernelSize = module.kernel_size[0]
                kernelStride = module.stride[0]
                groups = module.groups

            weight = module.weight
            bias = None
            if module.bias is not None:
                bias = module.bias
            # Perform batchnorm folding and update the quantization parameters if necessary
            if self.foldBN and isinstance(module, (nniqat.ConvBn2d, nniqat.ConvBnReLU2d)):
                """
                Assumptions:
                    - This is a fused model
                    - When the fusion occured, the model was not in eval mode, so its bn parameters are not folded at this moment
                    - Warning: If bn folding have occured, this the following folding will mess up the weights and bias
                    -   Reason: PyTorch's BN folding function only change conv weights and biases, but does affect BN parameters
                """
                weight, bias = fuse_conv_bn_weights(
                    module.weight, module.bias, module.running_mean, module.running_var,
                    module.eps, module.gamma, module.beta)

            # Determine weight frac bits
            # The quantization observer for weight of fused conv_bn already monitors the folded weights
            # see: https://github.com/pytorch/pytorch/blob/7f73f1d591afba823daa4a99a939217fb54d7688/torch/nn/intrinsic/qat/modules/conv_fused.py#L113
            weight_quantizer = module.weight_fake_quant
            # weight_post_process.enable_observer()
            # weight_post_process(weight)
            # weight_post_process.disable_observer()
            # weight_quantizer = custom_quant.RoundedMinMaxObserver()
            # weight_quantizer.forward(weight)
            weightPrecisionScale, _ = weight_quantizer.calculate_qparams()
            weightPrecisionScale = weightPrecisionScale.view(1)
            weightFracBits = int(torch.round(torch.log2(1.0 / weightPrecisionScale)).view(1)[0].item())
            hasBias = False if bias is None else True

            # Determine the SpW parameters
            flagFoundSpWInfo = False
            pruneRangeInCluster = None
            pruneCluster = None
            sparsity = None
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, cm_prune.balancedPruningMethod):
                    flagFoundSpWInfo = True
                    pruneRangeInCluster = hook.pruneRangeInCluster
                    sparsity = hook.sparsity
                    pruneCluster = hook.clusterSize

            if not flagFoundSpWInfo:
                print ("Using default sparsity pruning parameters for module {}".format(module))
                pruneRangeInCluster = self.defaultPruneRangeInCluster
                sparsity = 0.0
                pruneCluster = self.defaultPruneCluster

            # isAfterInput = False
            # nElements = input.numel()
            # preIdx = int(input.view(nElements)[self.ID_IDX].item())
            # if (self.layerList[preIdx].operationType == 'quantstub'):
            #     isAfterInput = True
            newLayer = ConvInfo(
                outputFracBits=int(outputFracBits),
                outputChannels=outputChannels,
                outputRelu=outputRelu,

                inputFracBits=int(input0FracBits),
                inputHeight=inputHeights[0],
                inputWidth=inputWidths[0],
                inputBorderPadding=padding,
                inputTransConvPadding=0,
                inputChannels=inputChannels[0],
                inputGroupsSeenBySource=inputGroupsSeenbySource[0],

                weightFracBits=int(weightFracBits),
                kernelSize=kernelSize,
                kernelStride=kernelStride,
                hasBias=hasBias,

                pruneRangeInCluster=pruneRangeInCluster,
                pruneClusterSize=pruneCluster,
                sparsity=sparsity,

                channelGroups=groups,
                layerID=self.layerID,
                isAfterInput=isAfterInput
            )

            # Extract parameters
            # Extract weights
            newLayer.weightParameterFileStartPosition = self.parameterCount
            self.parameterCount += weight.numel()
            self.parameters.append(weight)
            self.parameterKeys.append(str(self.layerID)+'_weight')

            newLayer.biasParameterFileStartPosition = self.parameterCount
            self.parameterCount += outputChannels
            self.parameterKeys.append(str(self.layerID) + '_bias')
            if hasBias is False:
                bias = torch.zeros([outputChannels])

            self.parameters.append(bias)

        # if this is an average pooling layer
        elif isinstance(module, cm.AvgPool2dRelu):
            #Average pooling layer should be seen as a special case of depth-wise convolution layer
            padding: int = 0
            if hasattr(module.padding, '__getitem__'):
                padding = module.padding[0]
            else:
                padding = module.padding
            newLayer = AvgPoolInfo(
                outputFracBits=outputFracBits,
                outputRelu=False,

                inputFracBits=int(input0FracBits),
                inputHeight=inputHeights[0],
                inputWidth=inputWidths[0],
                inputBorderPadding=padding,
                inputChannels=inputChannels[0],
                inputGroupsSeenBySource=inputGroupsSeenbySource[0],

                kernelSize=module.kernel_size,
                kernelStride=module.stride,

                divisor=module.divisor_override,

                layerID=self.layerID
            )
        elif isinstance(module, cm.MaxPool2dRelu):
            padding: int = 0
            if hasattr(module.padding, '__getitem__'):
                padding = module.padding[0]
            else:
                padding = module.padding
            newLayer = MaxPoolInfo(
                outputFracBits=outputFracBits,
                outputRelu=module.relu,

                inputFracBits=int(input0FracBits),
                inputHeight=inputHeights[0],
                inputWidth=inputWidths[0],
                inputBorderPadding=padding,
                inputChannels=inputChannels[0],
                inputGroupsSeenBySource=inputGroupsSeenbySource[0],

                kernelSize=module.kernel_size,
                kernelStride=module.stride,

                layerID=self.layerID
            )
        elif isinstance(module, cm.EltwiseAdd):
            assert inputHeights[0] == inputHeights[1], "Input heights do not match for eltwise-add operation"
            assert inputWidths[0] == inputWidths[1], "Input widths do not match for eltwise-add operation"
            assert inputChannels[0] == inputChannels[1], "Input channels do not match for eltwise-add operation"
            input1FracBits = int(torch.round(torch.log2(1.0 / inputPrecisions[1])).view(1)[0].item())
            newLayer = EltAddInfo(
                outputFracBits=outputFracBits,
                outputRelu=module.relu,

                inputHeight=inputHeights[0],
                inputWidth=inputWidths[0],
                inputChannels=inputChannels[0],

                inputLeftFracBits=int(input0FracBits),
                inputRightFracBits=int(input1FracBits),
                inputLeftGroupsSeenBySource=int(inputGroupsSeenbySource[0]),
                inputRightGroupsSeenBySource=int(inputGroupsSeenbySource[1]),

                layerID=self.layerID
            )
        elif isinstance(module, QuantStub):
            newLayer = QuantStubInfo(
                outputFracBits=outputFracBits,
                outputChannels=input[0].size()[1],
                outputHeight=input[0].size()[2],
                outputWidth=input[0].size()[3],
                layerID=self.layerID
            )
        elif isinstance(module, DeQuantStub):
            newLayer = DeQuantStubInfo(
                inputFracBits=int(input0FracBits),
                inputChannels=input[0].size()[1],
                inputGroupsSeenBySource=inputGroupsSeenbySource[0],
                inputHeight=input[0].size()[2] if len(input[0].size()) > 2 else 1,
                inputWidth=input[0].size()[3] if len(input[0].size()) > 2 else 1,
                layerID=self.layerID
            )
        else:
            raise TypeError('The tracer hook can only be applied to QuantStub, Conv2d types, Linear types, AvgPool2d types, '
                            'AvgPool2d, and EltwiseAdd types. Input module type is {}'.format(type(module)))

        # Insert the layer and save the connections
        self.insertLayer(newLayer)
        if isinstance(module, QuantStub) is False:
            for idx in inputIDs:
                self.addForwardEdge(sourceLayerId=idx, sinkLayerId=self.layerID)
                self.addBackward(sourceLayerId=idx, sinkLayerId=self.layerID)

        # Propagate the layer id, and output precision to the next layer
        outputNumel = output.numel()
        output.view(outputNumel)[self.ID_IDX] = self.layerID
        output.view(outputNumel)[self.PRECISION_IDX] = outputPrecisionScale

        self.layerID += 1
        self.layerCount += 1

        # print("Layer ID: {} \n Modified output: {}".format(self.layerID-1, output.view(outputNumel)[0:2]))

    def interceptHook(self, module, input: Union[T, Tuple[T, T]], output: T):
        # print ('Intercepting layer {}'.format(self.layerID))
        if self.layerID == self.interceptLayerID:
            self.activation_intercepts.append(deepcopy(output.clone().detach()))
        self.layerID += 1

    def traceModel(self, input: Union[T, Tuple[T, T]]) -> T:
        """
        Applies the tracing hooks to the model, run the model once, and remove the hooks.
        :return: None
        """
        # 1. Reset everything
        self.reset()

        # 2. Apply hooks
        stack = [self.module]
        while len(stack) > 0:
            module = stack.pop()
            # If module is a leaf, then add to the graph
            if isinstance(module, (QuantStub, DeQuantStub, cm.EltwiseAdd, nn.Conv2d, nn.Linear, cm.MaxPool2dRelu, cm.AvgPool2dRelu)):
                handle = module.register_forward_hook(self.traceHook)
                self.hookHandles.append(handle)
            # If the module is not a leaf, then push its children onto the stack
            else:
                for child in module.children():
                    stack.append(child)

        # 3. Run the model once, and the hooks should be called.
        self.module.eval()
        if isinstance(input, tuple):
            self.module(*input)
        else:
            self.module(input)
        #output = self.module(input)

        # 4. Remove the hooks
        self.removeHooks()

    def getOutput(self, input: Union[T, Tuple[T, T]], layerID=-1) -> T:
        # 1. Reset everything
        self.reset()

        self.interceptLayerID = layerID
        # 2. Apply hooks
        stack = [self.module]
        while len(stack) > 0:
            module = stack.pop()
            # If module is a leaf, then add to the graph
            if isinstance(module, (
            QuantStub, DeQuantStub, cm.EltwiseAdd, nn.Conv2d, nn.Linear, cm.MaxPool2dRelu, cm.AvgPool2dRelu)):
                # print('Adding intercept hook to layer{}'.format(module))
                handle = module.register_forward_hook(self.interceptHook)
                self.hookHandles.append(handle)
            # If the module is not a leaf, then push its children onto the stack
            else:
                for child in module.children():
                    stack.append(child)

        # 3. Run the model once, and the hooks should be called.
        self.module.eval()
        output = None
        if isinstance(input, tuple):
            output = self.module(*input)
        else:
            output = self.module(input)

        if layerID == -1:
            # print('Dumping the final output')
            return output

        self.removeHooks()
        print('Dumping output of layer {}'.format(layerID))
        return self.activation_intercepts[0]