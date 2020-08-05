"""
Package for tracing quantized-pruned models in their PyTorch execution order
"""
import torch
from torch import Tensor as T
from torch import nn as nn
from torch.nn import intrinsic as nninstrinsic
from torch.quantization import QuantStub as QuantStub


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
        self.outputCurrentNumGroups = inputChannels

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

class AvgPoolInfo(LayerInfo):
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
        super().__init__(operationType='avgpool',
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
        self.outputCurrentNumGroups = inputChannels

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
        # Input adjacency list of registered layer. The inputs of a given layer
        self.inputAdjacencies = []
        # Output adjacency list of registered layers. The outputs of a given layer
        self.outputAdjacencies = []

    def reset(self):
        self.layerID = 0
        self.layerCount = 0
        self.removeHooks()
        self.hookHandles.clear()
        self.layerList.clear()
        self.inputAdjacencies.clear()
        self.outputAdjacencies.clear()
        self.resetAnnotation()

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
        while len(self.outputAdjacencies) <= sourceLayerId:
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

    def annotate(self, numMemBanks: int) -> None:
        """
        Annotates memory and sparsification fields of layers in the graph
        TODO: Finish this
        :param numMemBanks:
        :return: None
        """
        outputConsumedFlags = []
        for i in range(len(self.outputAdjacencies)):
            edges = []
            for j in range(len(self.outputAdjacencies[i])):
                edges.append(False)
            outputConsumedFlags.append(edges)

        outputMemoryLocations = [-1 for i in range(len(self.layerList))]

        # Prepared a list of banks
        memoryStack = [i for i in range(numMemBanks)]

        for idx, layer in enumerate(self.layerList):
            # Try to allocate a spot for the output of the layer
            assert len(memoryStack) > 0, "Cannot allocate another memory region for the output of layer {}!".format(idx)
            outputMemoryLoc = memoryStack.pop()

            # Deassert one of the consumer flags in the producer
            predecessorList = self.inputAdjacencies[idx]
            for predIdx in predecessorList:
                outputConsumedFlags[predIdx].pop()
                if len(outputConsumedFlags[predIdx])==0:
                    locationToRestore = outputMemoryLocations[predIdx]
                    assert locationToRestore >= 0, 'Memory location to restore is -1!'
                    memoryStack.append(locationToRestore)





    def dump(self, fileStream=None) -> str:
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
        TODO: Add support for transposed convolution
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
        inputChannels = []
        inputHeights = []
        inputWidths = []
        if isinstance(input, T):
            nElements = input.numel()
            idx = input.view(nElements).numpy()[self.ID_IDX]
            inputIDs.append(idx)
            inputPrecisions.append(input.view(nElements)[self.PRECISION_IDX])
            inputChannels.append(self.layerList[idx].outputChannel)
            inputHeights.append(self.layerList[idx].outputHeight)
            inputWidths.append(self.layerList[idx].outputWidth)
        elif isinstance(input, tuple):
            for tensor in input:
                nElements = tensor.numel()
                idx = tensor.view(nElements)[self.ID_IDX]
                inputIDs.append(idx)
                inputPrecisions.append(tensor.view(nElements)[self.PRECISION_IDX])
                inputChannels.append(self.layerList[idx].outputChannel)
                inputHeights.append(self.layerList[idx].outputHeight)
                inputWidths.append(self.layerList[idx].outputWidth)
        else:
            raise TypeError('The input argument is neither a tensor nor a tuple of tensors')

        # Determine output precision scales
        outputPrecisionScale = inputPrecisions[0]
        if hasattr(module, 'activation_post_process'):
            outputPrecisionScale = module.activation_post_process.scale.view(1)
        else:
            if isinstance(module, cm.EltwiseAdd):
                # Number of fraction bits = log2(1/scale)
                # Number of interger bits = 8 - number of fraction bits
                # Number of fraction bits new = log2(1/scale_new) = number of fraction bits - 1 = log2(1/scale) - 1
                # log2(scale) + 1 = log2(scale_new)
                # scale_new = 2 * scale
                outputPrecisionScale *= 2.0
            else:
                pass
        outputFracBits = torch.round(torch.log2(1.0 / outputPrecisionScale)).view(1).numpy()[0]

        # Instantiate and insert a layer, register input/output adjacencies
        # If this is a convolution-based layer. Even the fused layer types are Conv2d thanks to inheritance
        newLayer = None
        outputChannels = output.size()[1]
        outputRelu = False

        input0FracBits = torch.round(torch.log2(1.0 / inputPrecisions[0])).view(1).numpy()[0]
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if isinstance(module, (nninstrinsic.ConvReLU2d, nninstrinsic.LinearReLU, nninstrinsic.ConvBnReLU2d)):
                outputRelu = True

            # Determine padding and kernel size
            padding = 0
            kernelSize = inputWidths[0]
            kernelStride = kernelSize
            groups = 1
            if isinstance(module, nn.Conv2d):
                # Extract the padding for convolution.
                # Assume that the horizontal and vertical paddings are the same
                padding = module.padding[0]
                kernelSize = module.kernel_size[0]
                kernelStride = module.stride[0]
                groups = module.groups

            # Determine weight frac bits
            weightPrecisionScale = module.weight_fake_quant.scale.view(1)
            weightFracBits = torch.round(torch.log2(1.0 / weightPrecisionScale)).view(1).numpy()[0]
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

                weightFracBits=int(weightFracBits),
                kernelSize=kernelSize,
                kernelStride=kernelStride,

                channelGroups=groups
            )
        # if this is an average pooling layer
        elif isinstance(module, nn.AvgPool2d):
            #Average pooling layer should be seen as a special case of depth-wise convolution layer
            padding = module.padding[0]
            newLayer = AvgPoolInfo(
                outputFracBits=outputFracBits,
                outputRelu=False,

                inputFracBits=input0FracBits,
                inputHeight=inputHeights[0],
                inputWidth=inputWidths[0],
                inputBorderPadding=padding,
                inputChannels=inputChannels[0],

                kernelSize=module.kernel_size,
                kernelStride=module.stride
            )
        elif isinstance(module, nn.MaxPool2d):
            padding = module.padding[0]
            newLayer = MaxPoolInfo(
                outputFracBits=outputFracBits,
                outputRelu=False,

                inputFracBits=input0FracBits,
                inputHeight=inputHeights[0],
                inputWidth=inputWidths[0],
                inputBorderPadding=padding,
                inputChannels=inputChannels[0],

                kernelSize=module.kernel_size,
                kernelStride=module.stride
            )
        elif isinstance(module, cm.EltwiseAdd):
            assert inputHeights[0] == inputHeights[1], "Input heights do not match for eltwise-add operation"
            assert inputWidths[0] == inputWidths[1], "Input widths do not match for eltwise-add operation"
            assert inputChannels[0] == inputChannels[1], "Input channels do not match for eltwise-add operation"
            input1FracBits = torch.round(torch.log2(1.0 / inputPrecisions[1])).view(1).numpy()[0]
            newLayer = EltAddInfo(
                outputFracBits=outputFracBits,
                outputRelu=module.relu,

                inputHeight=inputHeights[0],
                inputWidth=inputWidths[0],
                inputChannels=inputChannels[0],

                inputLeftFracBits=input0FracBits,
                inputRightFracBits=input1FracBits
            )
        elif isinstance(module, QuantStub):
            newLayer = LayerInfo(operationType='input', outputFracBits=outputFracBits, outputRelu=False)
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

    def traceModel(self, input: T) -> None:
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
            if isinstance(module, (QuantStub, cm.EltwiseAdd, nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d)):
                handle = module.register_forward_hook(self.traceHook)
                self.hookHandles.append(handle)
            # If the module is not a leaf, then push its children onto the stack
            else:
                for child in module.children():
                    stack.append(child)

        # 3. Run the model once, and the hooks should be called.
        self.module.eval()
        output = self.module(input)

        # 4. Remove the hooks
        self.removeHooks()




