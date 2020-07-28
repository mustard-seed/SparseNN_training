"""
Package for tracing quantized-pruned models in their PyTorch execution order
"""
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
    def __init__(self):
        pass

    def insertLayer(layer: LayerInfo) -> None:
        """
        Insert a new layer into the graph
        :param layer: An LayerInfo object
        :return: None
        """
        pass

    def addForwardEdge(sourceLayerId: int, sinkLayerId: int) -> None:
        """
        Add a forward dependency
        :param sourceLayerId: Id of the layer that generates the tensor during inference.
        :param sinkLayerId: Id of the layer that receives the tensor during inference.
        :return: None
        """
        pass

    def addBackward(sourceLayerId: int, sinkLayerId: int) -> None:
        """
        Add a backward endge
        :param sourceLayerId: Id of the layer that generates the tensor during inference. It will be at the end of the edge
        :param sinkLayerId: Id of the layer that receives the tensor during inference. It will be the emitter of the edge
        :return: None
        """
        pass

    def resetAnnotation() -> None:
        """
        Clears all memory and sparsification flag of layers in the graph
        :return: Hone
        """
        pass

    def annotate(numMemBanks: int) -> None:
        """
        Annotates memory and sparsification fields of layers in the graph
        :param numMemBanks:
        :return: None
        """
        pass

    def dump(fileStream = None) -> str:
        """
        Dumps the DNN IR in the inserted execution order as a YAML string.
        If a file stream is provided, the string will be written to the file.
        :param fileStream: Optional. File stream to dump the YAML string to.
        :return: The YAML string
        """
        pass
