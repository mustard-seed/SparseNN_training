import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torch.optim as optim
from torchvision import models, datasets, transforms
import torch.utils.data.distributed as distributed
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import yaml


import argparse
import os
import shutil


from custom_modules.custom_modules import ConvBNReLU, LinearReLU, ConvReLU, ConvBN
import pruning.pruning as custom_pruning
from custom_modules.resnet import ResNet, cifar_resnet56, BasicBlock, BottleneckBlock, conv1x1BN, conv3x3BN
from utils.meters import ClassificationMeter, TimeMeter
from experiment.experiment import experimentBase, globalActivationDict, globalWeightDict, hook_activation
from tracer.tracer import TraceDNN as Tracer

import horovod.torch as hvd

class experimentCifar10ResNet56(experimentBase):
    def __init__(self, configFile, multiprocessing=False):
        super().__init__(configFile, multiprocessing)
        self.model = cifar_resnet56()

        datasetDir = self.config.dataTrainDir
        """
        See Section 4.2 in the original ResNet paper for data pre-processing and augmentation settings
        """
        # Might've accidentally used ImageNet's settings....
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
        self.trainDataSet = datasets.CIFAR10(datasetDir, train=True, download=True,
                                           transform=train_transform)
        self.trainDataSampler = distributed.DistributedSampler(
            self.trainDataSet, num_replicas=hvd.size(), rank=hvd.rank()
        ) if multiprocessing is True \
            else None

        self.trainDataLoader = DataLoader(
            self.trainDataSet,
            batch_size=self.config.batchSizePerWorker,
            sampler=self.trainDataSampler,
            shuffle=True if self.trainDataSampler is None else False,
            pin_memory=True if torch.cuda.is_available() else False
            #,**dataKwargs
        )
        self.valDataSet = datasets.CIFAR10(datasetDir, train=False, download=True,
                                           transform=val_transform)

        self.valDataSampler = distributed.DistributedSampler(
            self.valDataSet, num_replicas=hvd.size(), rank=hvd.rank()
        ) if multiprocessing is True \
            else None

        self.valDataLoader = DataLoader(
            self.valDataSet,
            batch_size=self.config.batchSizePerWorker,
            sampler=self.valDataSampler,
            shuffle=True if self.valDataSampler is None else False,
            pin_memory=True if torch.cuda.is_available() else False
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

    def initialize_from_pre_trained_model_helper(self) -> None:
        """
        Download the pre-trained ResNet-56 CIFAR model from Torch Vision,
        and use the pre-trained parameters to initialize the custom ResNet-56 model
        """
        print('ResNet-56 is not available from TorchVision. Need to train from scratch.')
        pass

    def evaluate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(input=output, target=target)

    def apply_hook_activation(self, module: torch.nn.Module, prefix=None) -> dict:
        """
        Apply hook on the model. Cannot be used recursively
        :param module: Not used
        :param prefix: Not used
        :return: Handles to the activation interception hooks
        """
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
    def evaluate_sparsity(self, numBatches=None) -> (list, list, list):
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
                    mask = custom_pruning.compute_group_lasso_mask(tensor, self.config.pruneCluster, self.config.pruneThreshold)
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

        numBatchesToRun = len(self.valDataLoader) if numBatches is None else numBatches
        iterBatch = 0
        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(self.valDataLoader):
                activationList.clear()
                output = evaluatedModel(data)
                batchActivationSparsityList = np.array(generate_sparsity_list(activationList))
                if activationSparsityList is None:
                    activationSparsityList = np.zeros_like(batchActivationSparsityList)

                activationSparsityList = np.add(batchActivationSparsityList, activationSparsityList)
                if iterBatch == numBatchesToRun:
                    break
                # End of iteration of all validation data
            activationSparsityList = activationSparsityList / float(numBatchesToRun)

        return activationSparsityList, weightSparsityList, layerNameList
        # End of evaluate sparsity

    def prune_network_method(cls, model, sparsityTarget, config):
        # Prune the residual blocks
        for m in model.modules():
            if isinstance(m, BasicBlock):
                for layer in [m.convBN1[0], m.convBN2[0], m.shortcut]:
                    if not isinstance(layer, nn.Identity):
                        # Special case for short-cut
                        if isinstance(layer, (ConvBN, ConvBNReLU)):
                            layer = layer[0]
                        sparsity = sparsityTarget
                        # cap the sparsity-level of layers in residual blocks
                        # that see change in the number of channels to 50%
                        if layer.in_channels != layer.out_channels:
                            sparsity = min(0.5, sparsityTarget)
                        custom_pruning.applyBalancedPruning(
                            layer,
                            'weight',
                            clusterSize=config.pruneCluster,
                            pruneRangeInCluster=config.pruneRangeInCluster,
                            sparsity=sparsity
                        )

        # Prune the FC layer at the end
        custom_pruning.applyBalancedPruning(
            model.fc,
            'weight',
            clusterSize=config.pruneCluster,
            pruneRangeInCluster=config.pruneRangeInCluster,
            sparsity=sparsityTarget
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
        module = cifar_resnet56()
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
    parser = argparse.ArgumentParser(description="CIFAR10_ResNet56 experiment")
    parser.add_argument('--mode', type=str,
                        choices=['train', 'evaluate_sparsity', 'print_model', 'trace_model', 'validate'],
                        default='train',
                        help='Mode. Valid choices are train, evaluate_sparsity, print model, trace_model, and validate')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the experiment configuration file. Required')
    parser.add_argument('--load_checkpoint', type=int, choices=[0, 1, 2], default=0,
                        help='Load experiment from checkpoint. '
                             'Default: 0. 0: start from scratch; '
                             '1: load full experiment; '
                             '2: load model only')
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Enable multiprocessing (using Horovod as backend). Default: False')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to the checkpoint to be loaded. Required if --load_checkpoint is set as 1 or 2')
    parser.add_argument('--override_cluster_size', type=int,
                        help='Override the cluster size in the experiment config when performing sparsity evaluation')
    parser.add_argument('--output_layer_id', type=int, default=-1,
                        help='ID of the layer to intercept the output during model tracing. Default: -1')
    parser.add_argument('--custom_image_path', type=str, default=None,
                        help='Path to the image to run inference on during tracing')
    parser.add_argument('--custom_sparsity', type=float, default=None,
                        help='Override the sparsity target with a custom value')


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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        experiment.train(device)
        # Copy the config file into the log directory
        logPath = experiment.config.checkpointSaveDir
        configFileName = os.path.basename(args.config_file)
        newConfigFilePath = os.path.join(logPath, configFileName)
        shutil.copy(args.config_file, newConfigFilePath)
    elif args.mode == 'evaluate_sparsity':
        if args.custom_sparsity is not None:
            experiment.experimentStatus.targetSparsity = args.custom_sparsity
        experiment.save_sparsity_stats(args.override_cluster_size, numBatches=20)
    elif args.mode == 'print_model':
        experiment.print_model()
    elif args.mode == 'trace_model':
        if args.custom_sparsity is not None:
            experiment.experimentStatus.targetSparsity = args.custom_sparsity
        if args.override_cluster_size is not None:
            experiment.experimentStatus.pruneCluster = args.override_cluster_size
        experiment.trace_model(dirnameOverride=os.getcwd(), numMemoryRegions=3, modelName='traces/resnet56_cifar',
                               foldBN=True, outputLayerID=args.output_layer_id,
                               custom_image_path=args.custom_image_path)
    elif args.mode == 'validate':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        if args.custom_sparsity is not None:
            experiment.experimentStatus.targetSparsity = args.custom_sparsity
        if experiment.multiprocessing is False or (experiment.multiprocessing is True and hvd.rank() == 0):
            print('Running inference on the entire validation set. Target sparsity = {level:.4f}'
                  .format(level=experiment.experimentStatus.targetSparsity))
        experiment.eval_prep()
        experiment.validate(epoch=0, device=device)
