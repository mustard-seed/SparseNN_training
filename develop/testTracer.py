"""
Test file for the NN tracer.
Declare a mini ResNet-like model, prune and quantize it, and then feed it through the tracer to see what comes out of it.
"""
import custom_modules.custom_modules as cm
import custom_modules.resnet as resnet
import pruning.pruning as custom_pruning
import quantization.quantization as custom_quant
from    tracer.tracer import TraceDNN as Tracer

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.quantization import QuantStub, DeQuantStub
from typing import Union, List

import os
import yaml

def toy_resnet() -> resnet.ResNet:
    """
    Generate and initialize a toy resent with only 5 layers.
    Input size is compatiable with CIFAR-10
    :return: The ResNet-56 network
    """
    block = resnet.BasicBlock
    # Number of blocks per stage. Each stage has two parametrized layers
    num_blocks = [1, 1, 1]
    network = resnet.ResNet(block, num_blocks, 'cifar10')
    return network

class TracerTest():
    def __init__(self):
        self.model = toy_resnet()
        self.pruneClusterSize=1
        self.pruneThreshold=1e-4
        qatRoundedConfig = torch.quantization.FakeQuantize.with_args(
            observer=custom_quant.RoundedMovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            averaging_constant=0.01
        )
        self.qatConfig = torch.quantization.QConfig(
            activation=qatRoundedConfig
            , weight=qatRoundedConfig
        )
        # Quantize the model
        self.model.fuse_model()
        self.model.qconfig = self.qatConfig
        torch.quantization.prepare_qat(self.model, inplace=True)

        # Prune the model
        custom_pruning.applyClusterPruning(
            self.model.inputConvBNReLU[0],
            'weight',
            clusterSize=self.pruneClusterSize,
            threshold=self.pruneThreshold
        )
        for m in self.model.modules():
            if isinstance(m, resnet.BasicBlock):
                custom_pruning.applyClusterPruning(m.convBN1[0],
                                                   "weight",
                                                   clusterSize=self.pruneClusterSize,
                                                   threshold=self.pruneThreshold)
                custom_pruning.applyClusterPruning(m.convBN2[0],
                                                   "weight",
                                                   clusterSize=self.pruneClusterSize,
                                                   threshold=self.pruneThreshold)
                if isinstance(m.shortcut, cm.ConvBN):
                    custom_pruning.applyClusterPruning(m.shortcut[0],
                                                       "weight",
                                                       clusterSize=self.pruneClusterSize,
                                                       threshold=self.pruneThreshold)
    def trace(self, dirname: str, fileBaseName: str):
        tracer = Tracer(self.model)
        dummyInput = torch.randn(size=[16, 3, 32, 32])
        dummyOutput = tracer.traceModel(dummyInput)
        tracer.annotate(numMemRegions=3)
        tracer.dump(dirname, fileBaseName)

        """
        Saves the input and output
        """
        print("Saves the dummy input and output")
        inputArray = dummyInput.view(dummyInput.numel()).tolist()
        outputArray = dummyOutput.view(dummyOutput.numel()).tolist()
        blobPath = os.path.join(dirname, fileBaseName + '_inout.yaml')
        blobFile = open(blobPath, 'w')
        blobDict: dict = {}

        blobDict['input'] = inputArray
        blobDict['output'] = outputArray

        # We want list to be dumped as in-line format, hence the choice of the default_flow_style
        # See https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
        yaml.dump(blobDict, blobFile, default_flow_style=None)

if __name__=='__main__':
    torch.manual_seed(0)
    test = TracerTest()
    test.trace(dirname='.', fileBaseName='testTrace')

