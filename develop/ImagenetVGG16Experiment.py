import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torch.optim as optim
from torchvision import models, datasets, transforms
import torch.utils.data.distributed as distributed
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import yaml


import argparse
import os
import shutil


from custom_modules.custom_modules import ConvBNReLU, LinearReLU, ConvReLU, ConvBN, MaxPool2dRelu
import pruning.pruning as custom_pruning
from custom_modules.vgg16 import VGG16
from utils.meters import ClassificationMeter, TimeMeter
from experiment.experiment import experimentBase, globalActivationDict, globalWeightDict, hook_activation
from tracer.tracer import TraceDNN as Tracer

import horovod.torch as hvd

class experimentImagenetVGG16(experimentBase):
    """
    Train script is based on https://github.com/horovod/horovod/blob/master/examples/pytorch_imagenet_resnet50.py
    """
    def __init__(self, configFile, multiprocessing=False):
        super().__init__(configFile, multiprocessing)
        self.model = VGG16()

        datasetTrainDir = self.config.dataTrainDir
        datasetValDir = self.config.dataValDir
        """
        See https://github.com/pytorch/vision/blob/master/references/classification/presets.py
        """
        # Might've accidentally used ImageNet's settings....
        self.train_transform = transforms.Compose([
                                        transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.trainDataSet = datasets.ImageFolder(datasetTrainDir,
                                           transform=self.train_transform)
        self.trainDataSampler = distributed.DistributedSampler(
            self.trainDataSet, num_replicas=hvd.size(), rank=hvd.rank()
        ) if multiprocessing is True \
            else None

        # TODO: Check whether having multiple workers actually speed up data loading
        dataLoaderKwargs = {'num_workers': 1}

        self.trainDataLoader = DataLoader(
            self.trainDataSet,
            batch_size=self.config.batchSizePerWorker,
            sampler=self.trainDataSampler,
            shuffle=True if self.trainDataSampler is None else False,
            **dataLoaderKwargs
        )
        self.valDataSet = datasets.ImageFolder(datasetValDir,
                                           transform=self.val_transform)

        self.valDataSampler = distributed.DistributedSampler(
            self.valDataSet, num_replicas=hvd.size(), rank=hvd.rank()
        ) if multiprocessing is True \
            else None

        self.valDataLoader = DataLoader(
            self.valDataSet,
            batch_size=self.config.batchSizePerWorker,
            sampler=self.valDataSampler,
            shuffle=True if self.valDataSampler is None else False,
            **dataLoaderKwargs
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

    def initialize_from_pre_trained_model_helper(self) -> None:
        '''
        Download the pre-trained VGG-16 (no BN) model from Torch Vision,
        and use the pre-trained parameters to initialize our custom ResNet-50 model
        :return: None
        '''
        print('Downloading the pretrained ResNet-50 from TorchVision')
        pretrainedModel = models.vgg16(pretrained=True, progress=True)

        '''
        Strategy:
        - Manually match the layers one-by-one
          See https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py#L149
        '''
        print('Start to load parameters from the pre-trained VGG-16 from TorchVision')
        # Match the classifier stages
        destFeatureLayers = [
            self.model.conv1_1,
            self.model.conv1_2,
            self.model.maxpool1,
            self.model.conv2_1,
            self.model.conv2_2,
            self.model.maxpool2,
            self.model.conv3_1,
            self.model.conv3_2,
            self.model.conv3_3,
            self.model.maxpool3,
            self.model.conv4_1,
            self.model.conv4_2,
            self.model.conv4_3,
            self.model.maxpool4,
            self.model.conv5_1,
            self.model.conv5_2,
            self.model.conv5_3,
            self.model.maxpool5
        ]
        sourceFeatures = pretrainedModel.features
        # Iterate through the stages
        featureIdx = 0
        for _, destLayer in enumerate(destFeatureLayers):
            if isinstance(destLayer, ConvReLU):
                destLayer[0].load_state_dict(sourceFeatures[featureIdx].state_dict())
                destLayer[1].load_state_dict(sourceFeatures[featureIdx+1].state_dict())
                featureIdx = featureIdx + 2
            elif isinstance(destLayer, MaxPool2dRelu):
                featureIdx = featureIdx + 1

        # Load the classifier layers
        sourceClassifier = pretrainedModel.classifier
        self.model.fc1[0].load_state_dict(sourceClassifier[0].state_dict())
        self.model.fc2[0].load_state_dict(sourceClassifier[3].state_dict())
        self.model.fc3.load_state_dict(sourceClassifier[6].state_dict())

        # Compenstate for dropouts before fc2 and fc3 in the original models
        # Assuming drop-out is 0.5
        with torch.no_grad():
            modified_weight = torch.mul(self.model.fc1[0].weight, 0.5)
            self.model.fc1[0].weight.copy_(modified_weight)
            modified_weight = torch.mul(self.model.fc2[0].weight, 0.5)
            self.model.fc2[0].weight.copy_(modified_weight)

        print('Finished loading parameters from the pre-trained VGG-16 from TorchVision')

    def initialize_from_pre_trained_model(self) -> None:
        if self.multiprocessing is True:
            if hvd.rank() == 0:
                self.initialize_from_pre_trained_model_helper()
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        else:
            self.initialize_from_pre_trained_model_helper()

    def evaluate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(input=output, target=target)

    def apply_hook_activation(self, module: torch.nn.Module, prefix=None) -> dict:
        # No regularization on activation output
        myDict = {}
        return myDict

    def extract_weight(self, module: torch.nn.Module) -> None:
        """
        Helper function that specifies which layers' weight tensors should be extracted.
        This is NOT called recursively (different from the base class implementation
        Prunes the first convolution layer, and the convolution layers in the residual blocks
        Don't prune the last fully-connected layer
        :param module: Not used
        :return: None
        """
        blockId = 0
        for m in self.model.modules():
            if isinstance(m, ConvReLU):
                name = 'conv_{}'.format(blockId)
                globalWeightDict[name] = m[0].weight.clone()
                blockId += 1
            elif isinstance(m, nn.Linear):
                name = 'fc_{}_shortcut'.format(blockId)
                globalWeightDict[name] = m.weight.clone()
                blockId += 1

    # Override the evaluate_sparsity function
    def evaluate_sparsity(self, numBatches=None) -> (list, list, list):
       pass


    def prune_network(self, sparsityTarget: float=0.5) -> None:
        self.prune_network_method(self.model, sparsityTarget, self.config)

    def prune_network_method(cls, model, sparsityTarget, config):
        # Don't prune the first layer
        # Cap layers that see IC != OC at 0.5
        for m in model.modules():
            sparsity = sparsityTarget
            if isinstance(m, ConvReLU):
                # Hard-code the input layer's characteristics
                if m[0].in_channels == 3:
                    continue
                elif m[0].in_channels != m[0].out_channels:
                    sparsity = 0.5
                custom_pruning.applyBalancedPruning(
                    m[0],
                    'weight',
                    clusterSize=config.pruneCluster,
                    pruneRangeInCluster=config.pruneRangeInCluster,
                    sparsity=sparsity
                )
            elif isinstance(m, nn.Linear):
                custom_pruning.applyBalancedPruning(
                    m,
                    'weight',
                    clusterSize=config.pruneCluster,
                    pruneRangeInCluster=config.pruneRangeInCluster,
                    sparsity=sparsity
                )

        return model

    def trace_model(self, dirnameOverride=None, numMemoryRegions: int = 3, modelName: str = 'model',
                    foldBN: bool = True, outputLayerID: int = -1, custom_image_path=None) -> None:
        """
        Trace the model after pruning and quantization, and save the trace and parameters
        :return: None
        """
        dirname = self.config.checkpointSaveDir if dirnameOverride is None else dirnameOverride
        # Prune and quantize the model
        self.eval_prep()

        # Deepcopy doesn't work, do the following instead:
        # See https://discuss.pytorch.org/t/deep-copying-pytorch-modules/13514/2
        module = VGG16()
        module = self.quantize_model_method(module, self.qatConfig)
        module = self.prune_network_method(module, self.experimentStatus.targetSparsity, self.config)
        module.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            # Hack
            # module.inputConvBNReLU._modules['0'].running_mean.zero_()
            # module.inputConvBNReLU._modules['0'].beta.zero_()
            # end of hack
            module.eval()
            trace = Tracer(module, _foldBN=foldBN, _defaultPruneCluster=self.config.pruneCluster,
                           _defaultPruneRangeInCluster=self.config.pruneRangeInCluster)
            """
            Run inference and save a reference input-output pair
            """
            blobPath = os.path.join(dirname, modelName + '_inout.yaml')
            blobFile = open(blobPath, 'w')
            blobDict: dict = {}
            output = None
            sampleIn = None
            if custom_image_path is None:
                for (data, target) in self.valDataLoader:
                    sampleIn = data[0].unsqueeze(0)
                    print(sampleIn.shape)
                    output = trace.getOutput(sampleIn, outputLayerID)
                    break
            else:
                print('Using custom image for inference tracing: {}'.format(custom_image_path))
                img = Image.open(custom_image_path)
                img = img.convert('RGB')
                # val_transform = transforms.Compose([
                #     transforms.Resize(256),
                #     transforms.CenterCrop(224),
                #     transforms.ToTensor(),
                #     transforms.Normalize(mean=[0.000, 0.000, 0.000],
                #                          std=[0.229, 0.224, 0.225])
                # ])
                sampleIn = self.val_transform(img)
                sampleIn = sampleIn.unsqueeze(0)
                print(sampleIn.shape)
                output = trace.getOutput(sampleIn, outputLayerID)
            inputArray = sampleIn.view(sampleIn.numel()).tolist()
            blobDict['input'] = inputArray

            outputArray = output.view(output.numel()).tolist()
            blobDict['output'] = outputArray
            # We want list to be dumped as in-line format, hence the choice of the default_flow_style
            # See https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
            yaml.dump(blobDict, blobFile, default_flow_style=None)

            trace.traceModel(sampleIn)

            trace.annotate(numMemRegions=numMemoryRegions)
            trace.dump(dirname, fileNameBase=modelName)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ImageNet-VGG16 experiment")
    parser.add_argument('--mode', type=str, choices=['train', 'print_model', 'trace_model', 'validate'],
                        default='train',
                        help='Mode. Valid choices are train, evaluate_sparsity, print model, trace_model, and validate')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the experiment configuration file. Required')
    parser.add_argument('--load_checkpoint', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Load experiment from checkpoint. '
                             'Default: 0. 0: start from scratch; '
                             '1: load full experiment; '
                             '2: load model only'
                             '3: initialize model from pre-trained ResNet-50 from Torch Vision')
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Enable multiprocessing (using Horovod as backend). Default: False')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to the checkpoint to be loaded. Required if --load_checkpoint is set as 1 or 2')
    parser.add_argument('--output_layer_id', type=int, default=-1,
                        help='ID of the layer to intercept the output during model tracing. Default: -1')
    parser.add_argument('--custom_image_path', type=str, default=None, help='Path to the image to run inference on during tracing')
    parser.add_argument('--custom_sparsity', type=float, default=None, help='Override the sparsity target with a custom value')


    args = parser.parse_args()
    if args.multiprocessing is True:
        hvd.init()
    experiment = experimentImagenetVGG16(configFile=args.config_file,
                                 multiprocessing=args.multiprocessing)
    if args.load_checkpoint == 1 or args.load_checkpoint == 2:
        assert args.checkpoint_path is not None, 'Experiment is required to load from an existing checkpoint, but no path to checkpoint is provided!'
        loadModelOnly = True if args.load_checkpoint == 2 else False
        experiment.restore_experiment_from_checkpoint(checkpoint=args.checkpoint_path,
                                                      loadModelOnly=loadModelOnly)
    elif args.load_checkpoint == 3:
        experiment.initialize_from_pre_trained_model()

    if args.mode == 'train':
        experiment.train()
        # Copy the config file into the log directory
        logPath = experiment.config.checkpointSaveDir
        configFileName = os.path.basename(args.config_file)
        newConfigFilePath = os.path.join(logPath, configFileName)
        shutil.copy(args.config_file, newConfigFilePath)
    elif args.mode == 'print_model':
        experiment.print_model()
    elif args.mode == 'trace_model':
        if args.custom_sparsity is not None:
            experiment.experimentStatus.targetSparsity = args.custom_sparsity
        experiment.trace_model(dirnameOverride=os.getcwd(), numMemoryRegions=3, modelName='vgg16_imagenet',
                               foldBN=True, outputLayerID=args.output_layer_id, custom_image_path=args.custom_image_path)
    elif args.mode == 'validate':
        if args.custom_sparsity is not None:
            experiment.experimentStatus.targetSparsity = args.custom_sparsity
        if experiment.multiprocessing is False or (experiment.multiprocessing is True and hvd.rank() == 0):
            print('Running inference on the entire validation set. Target sparsity = {level:.4f}'
                  .format(level=experiment.experimentStatus.targetSparsity))
        experiment.eval_prep()
        experiment.validate(epoch=0)