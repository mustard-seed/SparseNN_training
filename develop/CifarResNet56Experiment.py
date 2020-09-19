import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed as distributed
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np


import argparse
import os
import shutil


from custom_modules.custom_modules import ConvBNReLU, LinearReLU, ConvReLU, ConvBN
import pruning.pruning as custom_pruning
from custom_modules.resnet import ResNet, cifar_resnet56, BasicBlock, BottleneckBlock, conv1x1BN, conv3x3BN
from utils.meters import ClassificationMeter, TimeMeter
from experiment.experiment import experimentBase, globalActivationDict, globalWeightDict, hook_activation

import horovod.torch as hvd

class experimentCifar10ResNet56(experimentBase):
    def __init__(self, configFile, multiprocessing=False):
        super().__init__(configFile, multiprocessing)
        self.model = cifar_resnet56()

        datasetDir = self.config.dataTrainDir
        """
        See Section 4.2 in the original ResNet paper for data pre-processing and augmentation settings
        """
        train_transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.trainDataSet = datasets.CIFAR10(datasetDir, train=True, download=False,
                                           transform=train_transform)
        self.trainDataSampler = distributed.DistributedSampler(
            self.trainDataSet, num_replicas=hvd.size(), rank=hvd.rank()
        ) if multiprocessing is True \
            else None

        self.trainDataLoader = DataLoader(
            self.trainDataSet,
            batch_size=self.config.batchSizePerWorker,
            sampler=self.trainDataSampler,
            shuffle=True if self.trainDataSampler is None else False
            #,**dataKwargs
        )
        self.valDataSet = datasets.CIFAR10(datasetDir, train=False, download=False,
                                           transform=val_transform)

        self.valDataSampler = distributed.DistributedSampler(
            self.valDataSet, num_replicas=hvd.size(), rank=hvd.rank()
        ) if multiprocessing is True \
            else None

        self.valDataLoader = DataLoader(
            self.valDataSet,
            batch_size=self.config.batchSizePerWorker,
            sampler=self.valDataSampler,
            shuffle=True if self.valDataSampler is None else False
            #,**dataKwargs
        )

        if (multiprocessing is True and hvd.rank() == 0) or multiprocessing is False:
            if not os.path.exists(self.config.logDir):
                os.makedirs(self.config.logDir)
            self.logWriter = SummaryWriter(self.config.logDir)

        self.trainMeter = ClassificationMeter(
            multiprocessing,
            self.logWriter,
            logPrefix='Train'
        )

        self.valMeter = ClassificationMeter(
            multiprocessing,
            self.logWriter,
            logPrefix='Validation'
        )

        self.trainTimeMeter = TimeMeter(
            multiprocessing,
            self.logWriter,
            logPrefix='Train'
        )

        # End of __init__

    def evaluate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(input=output, target=target)

    def apply_hook_activation(self, module: torch.nn.Module, prefix=None) -> dict:
        """
        Apply hook on the model. Cannot be used recursively
        :param module: Not used
        :param prefix: Not used
        :return: Handles to the activation interception hooks
        """
        pruneDict = {
            'inputConvBNReLU': self.model.inputConvBNReLU
        }
        # Add the residual blocks to the dictionary
        # Don't prune the input tensor of the elementwise add operation
        blockId = 0
        for m in self.model.modules():
            if isinstance(m, BasicBlock):
                name = 'block_{}_layer0'.format(blockId)
                pruneDict[name] = m.convBN1
                name = 'block_{}_out'.format(blockId)
                pruneDict[name] = m
                blockId +=1
        # Don't prune the output of the final fc layer
        forwardHookHandlesDict = {}
        for name, m in pruneDict.items():
            handle = m.register_forward_hook(hook_activation)
            forwardHookHandlesDict[name] = handle

        return forwardHookHandlesDict

    def extract_weight(self, module: torch.nn.Module) -> None:
        """
        Helper function that specifies which layers' weight tensors should be extracted.
        This is NOT called recursively (different from the base class implementation
        Prunes the first convolution layer, and the convolution layers in the residual blocks
        Don't prune the last fully-connected layer
        :param module: Not used
        :return: None
        """
        #global globalWeightDict
        globalWeightDict['inputConvBNReLU'] = self.model.inputConvBNReLU[0].weight.clone()

        blockId = 0
        for m in self.model.modules():
            if isinstance(m, BasicBlock):
                name = 'block_{}_layer0'.format(blockId)
                globalWeightDict[name] = m.convBN1[0].weight.clone()
                name = 'block_{}_layer1'.format(blockId)
                globalWeightDict[name] = m.convBN2[0].weight.clone()
                if isinstance(m.shortcut, ConvBN):
                    name = 'block_{}_shortcut'.format(blockId)
                    globalWeightDict[name] = m.shortcut[0].weight.clone()
                blockId += 1

    # Override the evaluate_sparsity function
    def evaluate_sparsity(self) -> (list, list, list):
        """
        Evaluate the activation and weight sparsity of the model on the entire validation set
        :return: Three lists. List one for the average activation sparsity per layer
                List two for the weight sparsity per layer
                List three the relevant layer name list
        """
        # List of activation intercept layers
        activationList = []
        def intercept_activation(module, input, output):
            activationList.append(output)

        # Change this
        def evaluate_setup (model : ResNet, needPrune=False, prefix=None):
            interceptHandleList = []
            weightList = []
            layerNameList = []

            targetList = [model.inputConvBNReLU,
                          model.stage1, model.stage2, model.stage3, model.stage4,
                          model.averagePool, model.fc]

            blockId = 0
            for target in targetList:
                if isinstance(target, ConvBNReLU):
                    # Input convolution layer
                    if needPrune is True:
                        custom_pruning.applyClusterPruning(
                            target[0], name='weight',
                            clusterSize=self.config.pruneCluster,
                            threshold=self.config.pruneThreshold
                        )
                    interceptHandle = target.register_forward_hook(intercept_activation)
                    interceptHandleList.append(interceptHandle)
                    weightList.append(target[0].weight.detach().clone())
                    layerNameList.append('Input ConvBNReLU')
                elif isinstance(target, nn.Sequential):
                    for m in target.modules():
                        if isinstance(m, BasicBlock):
                            for layerId, layer in enumerate([m.convBN1, m.convBN2]):
                                name = 'block_{}_layer{}'.format(blockId, layerId)
                                if needPrune is True:
                                    custom_pruning.applyClusterPruning(
                                        layer[0], name='weight',
                                        clusterSize=self.config.pruneCluster,
                                        threshold=self.config.pruneThreshold
                                    )
                                interceptHandle = layer.register_forward_hook(intercept_activation)
                                interceptHandleList.append(interceptHandle)
                                weightList.append(layer[0].weight.detach().clone())
                                layerNameList.append(name)
                            if isinstance(m.shortcut, ConvBN):
                                name = 'block_{}_shortcut'.format(blockId)
                                if needPrune is True:
                                    custom_pruning.applyClusterPruning(
                                        m.shortcut[0], name='weight',
                                        clusterSize=self.config.pruneCluster,
                                        threshold=self.config.pruneThreshold
                                    )
                                interceptHandle = m.shortcut.register_forward_hook(intercept_activation)
                                interceptHandleList.append(interceptHandle)
                                weightList.append(m.shortcut[0].weight.detach().clone())
                                layerNameList.append(name)

                            interceptHandle = m.register_forward_hook(intercept_activation)
                            interceptHandleList.append(interceptHandle)
                            weightList.append(None)
                            layerNameList.append('block_{}_output'.format(blockId))
                            blockId += 1

                elif isinstance(target, nn.AvgPool2d):
                    interceptHandle = target.register_forward_hook(intercept_activation)
                    interceptHandleList.append(interceptHandle)
                    weightList.append(None)
                    layerNameList.append('average_pool')

                elif isinstance(target, nn.Linear):
                    if needPrune is True:
                        custom_pruning.applyClusterPruning(
                            target, name='weight',
                            clusterSize=self.config.pruneCluster,
                            threshold=self.config.pruneThreshold
                        )
                    interceptHandle = target.register_forward_hook(intercept_activation)
                    interceptHandleList.append(interceptHandle)
                    weightList.append(target.weight.detach().clone())
                    layerNameList.append('final classification')

            return interceptHandleList, weightList, layerNameList
        #End of helper function evaluate_setup

        def generate_sparsity_list(tensorList : list):
            sparsityList = []
            for idx, tensor in enumerate(tensorList):
                if tensor is not None:
                    mask = custom_pruning.compute_mask(tensor, self.config.pruneCluster, self.config.pruneThreshold)
                    mask = mask.byte()
                    reference = torch.ones_like(mask)
                    comparison = torch.eq(mask, reference)
                    numNz = torch.sum(comparison.float())
                    sparsity = numNz.item() / comparison.numel()
                else:
                    sparsity = None

                sparsityList.append(sparsity)

            return sparsityList

        if self.multiprocessing is True:
            assert hvd.size() == 1, "Sparsity evaluation cannot be done in multi-processing mode!"

        # Fuse and quantized the model if this is haven't been done so
        #evaluatedModel = copy.deepcopy(self.model)
        if self.experimentStatus.flagFusedQuantized is False:
            self.quantize_model()
            self.experimentStatus.flagFusedQuantized = True

        evaluatedModel = self.model

        # Apply pruning mask, and activation interception, extract weight
        interceptHandleList, weightList, layerNameList = \
            evaluate_setup(evaluatedModel, self.experimentStatus.flagPruned is False)

        # Compute weight sparsity
        weightSparsityList = generate_sparsity_list(weightList)
        activationSparsityList = None

        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(self.valDataLoader):
                activationList.clear()
                output = evaluatedModel(data)
                batchActivationSparsityList = np.array(generate_sparsity_list(activationList))
                if activationSparsityList is None:
                    activationSparsityList = np.zeros_like(batchActivationSparsityList)

                activationSparsityList = np.add(batchActivationSparsityList, activationSparsityList)
                # End of iteration of all validation data
            activationSparsityList = activationSparsityList / float(len(self.valDataLoader))

        return activationSparsityList, weightSparsityList, layerNameList
        # End of evaluate sparsity


    def prune_network(self) -> None:
        custom_pruning.applyClusterPruning(
            self.model.inputConvBNReLU[0],
            'weight',
            clusterSize=self.config.pruneCluster,
            threshold=self.config.pruneThreshold
        )

        for m in self.model.modules():
            if isinstance(m, BasicBlock):
                custom_pruning.applyClusterPruning(m.convBN1[0],
                                                   "weight",
                                                   clusterSize=self.config.pruneCluster,
                                                   threshold=self.config.pruneThreshold)
                custom_pruning.applyClusterPruning(m.convBN2[0],
                                                   "weight",
                                                   clusterSize=self.config.pruneCluster,
                                                   threshold=self.config.pruneThreshold)
                if isinstance(m.shortcut, ConvBN):
                    custom_pruning.applyClusterPruning(m.shortcut[0],
                                                       "weight",
                                                       clusterSize=self.config.pruneCluster,
                                                       threshold=self.config.pruneThreshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CIFAR10_ResNet56 experiment")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate_sparsity', 'print_model', 'trace_model'],
                        default='train',
                        help='Mode. Valid choices are train, evaluate_sparsity, and print model')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the experiment configuration file. Required')
    parser.add_argument('--load_checkpoint', type=int, choices=[0, 1, 2], default=0,
                        help='Load experiment from checkpoint. Default: 0. 0: start from scratch; 1: load full experiment; 2: load model only')
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Enable multiprocessing (using Horovod as backend). Default: False')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to the checkpoint to be loaded. Required if --load_checkpoint is set as 1 or 2')
    parser.add_argument('--override_cluster_size', type=int,
                       help='Override the cluster size in the experiment config when performing sparsity evaluation')


    args = parser.parse_args()
    if args.multiprocessing is True:
        hvd.init()
    experiment = experimentCifar10ResNet56(configFile=args.config_file,
                                 multiprocessing=args.multiprocessing)
    if args.load_checkpoint == 1 or args.load_checkpoint == 2:
        assert args.checkpoint_path is not None, 'Experiment is required to load from an existing checkpoint, but no path to checkpoint is provided!'
        loadModelOnly = True if args.load_checkpoint == 2 else False
        experiment.restore_experiment_from_checkpoint(checkpoint=args.checkpoint_path,
                                                      loadModelOnly=loadModelOnly)

    if args.mode == 'train':
        experiment.train()
        # Copy the config file into the log directory
        logPath = experiment.config.checkpointSaveDir
        configFileName = os.path.basename(args.config_file)
        newConfigFilePath = os.path.join(logPath, configFileName)
        shutil.copy(args.config_file, newConfigFilePath)
    elif args.mode == 'evaluate_sparsity':
        experiment.save_sparsity_stats(args.override_cluster_size)
    elif args.mode == 'print_model':
        experiment.print_model()
    elif args.mode == 'trace_model':
        experiment.trace_model(dirnameOverride=os.getcwd(), numMemoryRegions=3, modelName='resnet56_cifar10', foldBN=True)