import quantization.quantization as custom_quant
import pruning.pruning as custom_prune
from pruning.pruning import compute_mask

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import horovod.torch as hvd
import yaml
from easydict import EasyDict as edict
from abc import ABC, abstractmethod
import numpy as np

import os
import math
import copy

# Make sure to import the global the following globals
# Global list for storing the intermediate activation layers
globalActivationDict = {}


# Function hook used to intercept intermediate activation layers
# To be exported
def hook_activation(module, input, output):
    global globalActivationDict
    globalActivationDict[module] = output.clone().detach()


def remove_hook_activation(forwardHookHandlesDict:dict):
    for name, handle in forwardHookHandlesDict.items():
        handle.remove()


# Global list for storing the weight tensors
globalWeightDict = {}

def generate_experiment_status():
    config = edict()

    config.flagFusedQuantized = False
    config.flagPruned = False
    config.numEpochTrained = 0
    return config

def generate_base_config():
    config = edict()
    config.numEpochToTrain = 1
    config.batchSizePerWorker = 32
    config.numThreadsPerWorker = 28
    config.trainEpochs = 20

    # Learning rate configuration
    # Constant multiplicative LR with optional warmup
    config.lrInitialBase = 1e-7
    config.lrPeakBase = 1e-5
    config.lrWarmupEpochs = 4
    config.lrStepSize = 4
    config.lrReductionFactor = 0.7
    config.lrMomentum = 0.01
    config.lrMPAdjustmentWarmUpEpochs = 5

    # Loss contribution
    config.lossActivationLambda = 0.0
    config.lossWeightL1Lambda = 0.0
    config.lossWeightL2Lambda = 0.0

    # Pruning paramters
    config.prune = False
    config.pruneCluster = 1
    config.pruneThreshold = 1e-5
    config.quantize = False

    # See
    config.seed = 47

    # Dataset location
    config.dataTrainDir = './'
    config.dataValDir = './'

    # Checkpoints
    config.checkpointSaveDir = ''
    config.checkpointSaveFileNameFormat = ''

    # Log Dir
    config.logDir = ''

    return config


# Helper function for parsing configuration file keys
def extractConfigKey(config : dict, key):
    """
    Extracts the value of the key from a dictionary if the key exists
    :param config: The configuration dictionary
    :param key:
    :return: Value of the dictionary if the key exists, else None
    """
    assert isinstance(config, dict), "ValueError: config should be a dictionary object!"
    if key in config.keys():
        return config[key]
    else:
        return None


class experimentBase(object):
    """
    Base experiment class for classifier training and evaluation,
    with multinode processing support
    """

    @abstractmethod
    def __init__(self, configFile, multiprocessing=False):
        """
        Initialize the basic configuration of an experiment object.
        Children initialization implementations should also
        1) Instantiate and initialize model
        :param configFile:
        :param multiprocessing:
        """
        # TODO: Children should provide their own init method
        # 1) Load configuration file
        # 2) Instantiate data loader and samplers
        # 3) Instantiate train and validation meters

        # Experiment states initialization
        status = generate_experiment_status()
        self.experimentStatus = status
        self.multiprocessing = multiprocessing

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

        # Placeholder reference to optimizer state
        # To be populated if restoring the experiment from checkpoint
        self.optimizerStateDict = None

        # TODO: Initialize these in the concrete __init__() method of each derived class
        self.model = None
        self.trainDataSet = None
        self.trainDataLoader = None
        self.trainDataSampler = None
        self.valDataSet = None
        self.valDataLoader = None
        self.valDataSampler = None
        self.logWriter = None
        self.trainMeter = None
        self.valMeter = None

        # Load experiment setting from config file
        config = generate_base_config()
        if (multiprocessing is False) or (multiprocessing is True and hvd.rank() == 0):
            try:
                file = open(configFile, "r")
            except IOError:
                raise ValueError("The provided configuration file cannot be opened.")
            with file:
                yamlConfig = yaml.load(file, Loader=yaml.FullLoader)
                config = edict(yamlConfig)

        # Broadcast the configuration to the workers during multiprocessing
        # TODo: Will this work?
        if multiprocessing is True:
            config = hvd.broadcast_object(obj=config, root_rank=0, name='config')

        self.config = config
        torch.manual_seed(self.config.seed)
        # Set intra-op parallelism threads
        torch.set_num_threads(self.config.numThreadsPerWorker)
        # End of loading configuration file

        # TODOs in the implementation of __init__ methods in the derived classes
        # Instantiate data loader
        # Instantiate model

    @abstractmethod
    def evaluate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the inference loss
        :param output: The neural network output
        :param target: The ground truth
        :return: The loss
        """
        return torch.tensor(0.)

    @abstractmethod
    def apply_hook_activation(self, module: torch.nn.Module, prefix=None) -> dict:
        """
        Specify what layer's activations should be extracted. To be called recursively
        Calls experiment.hook_activation. Make sure to import hook_activaiton
        :param module: Current top level module
        :param prefix: Prefix attached to dictiionary key of the handle of interest
        :return: Dictionary
        """
        forwardHookHandlesDict = {}
        for name, m in module.named_children():
            if prefix:
                modulePrefix = prefix + '/' + name
            else:
                modulePrefix = name
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d)):
                # registeration of the hook shall generate a handle.
                # see https://github.com/pytorch/pytorch/issues/5037#issuecomment-362910897
                handle = m.register_forward_hook(hook_activation)
                forwardHookHandlesDict[modulePrefix] = handle
            else:
                forwardHookHandlesDict.update(self.apply_hook_activation(m, modulePrefix))

        return forwardHookHandlesDict

    @abstractmethod
    def extract_weight(self, module: torch.nn.Module) -> None:
        """
        Helper function that specifies which layers' weight tensors should be extracted.
        To be called recursively
        :param module: torch.nn.Module
        :return: None
        """
        for m in module.children():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d)):
                global globalWeightDict
                globalWeightDict[m] = m.weight.clone().detach()
            else:
                self.extract_weight(m)

    @abstractmethod
    def prune_network(self) -> None:
        """
        Prunes the network according.
        Derived classes should specify which layers should pruned
        :return:
        """

    def prepare_model(self) -> None:
        """"
        Quantize and prune the model and reset the optimizer accordingly if the experiment requires these but haven't been done
        :return None
        """
        if (self.config.prune is True and self.experimentStatus.flagPruned is False) or \
        (self.config.quantize is True and self.experimentStatus.flagFusedQuantized is False):
            if self.config.prune is True and self.experimentStatus.flagPruned is False:
                self.quantize_model()
                self.experimentStatus.flagFusedQuantized = True

            if self.config.quantize is True and self.experimentStatus.flagFusedQuantized is False:
                self.prune_network()
                self.experimentStatus.flagPruned = True


    def adjust_learning_rate(self, epoch, batchIdx, optimizer):
        """
        Compute the learning rate for the current epoch, and
        adjust the learning rate in the optimizer of the experiment
        :param epoch: The current epoch
        :param batchidx: Batch id the current epoch
        :param optimizer: The optimizer to be updated
        :return: The learning rate for the current epoch
        """
        MININUM_LR = 1e-5
        lr = 1e-5
        roundedEpoch = epoch + float(batchIdx + 1) / len(self.trainDataLoader)
        if epoch < self.config.lrWarmupEpochs:
            lr = (roundedEpoch / self.config.lrWarmupEpochs * (self.config.lrPeakBase - self.config.lrInitialBase)) \
                 + self.config.lrInitialBase
        else:
            lr = self.config.lrPeakBase * (self.config.lrReductionFactor ** ((epoch - self.config.lrWarmupEpochs) // self.config.lrStepSize))
            lr = max(lr, MININUM_LR)

        lrAdj = 1.0
        if self.multiprocessing is True:
            if epoch < self.config.lrMPAdjustmentWarmUpEpochs:
                lrAdj = 1. * (roundedEpoch * (hvd.size() - 1) / self.config.lrMPAdjustmentWarmUpEpochs + 1)

        lr = lr * lrAdj
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def quantize_model(self):
        """
        Helper function, quantize the model. It does NOT set the fuseQuantized experiment status flag!
        :param inpLace: Whether the quantization should be done in place
        :return: None
        """
        self.model.fuse_model()
        self.model.qconfig = self.qatConfig
        torch.quantization.prepare_qat(self.model, inplace=True)

    def initialize_optimizer (self):
        """
        Initialize an optimizer based on the current model configuration
        :return: An initialized optimizer
        """
        # Reconfigure the optimizer accordingly
        lrScalar = 1
        if self.multiprocessing is True:
            lrScalar = hvd.size()

        optimizer = optim.SGD(self.model.parameters(),
                              lr=(self.config.lrPeakBase * lrScalar),
                              momentum=self.config.lrMomentum
                              )

        if self.multiprocessing is True:
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=self.model.named_parameters(),
                compression=hvd.Compression.none,
                backward_passes_per_step=1,
                op=hvd.Average)

        return optimizer

    def restore_model_from_state_dict(self, modelStateDict) -> None:
        """
        Restores the experiment's model from a given state dict
        :param modelStateDict: The state dict to be loaded from
        :return:
        """
        model_dict = self.model.state_dict()

        # 1. Filter out unnecessary keys
        saved_model_dict = {k: v for k, v in modelStateDict.items() if k in model_dict}
        # 2. Overwrite entries in the existing model
        model_dict.update(saved_model_dict)
        # 3. Load the new state dicc
        self.model.load_state_dict(saved_model_dict)

    def restore_experiment_from_checkpoint(self, checkpoint, loadModelOnly=False):
        """
        Restores an experiment stats, optimizer, and model from checkpoint
        When restoring a model, look for the status flag 'quantized'
        If the model has been quantized, but the experiment doesn't require quantization,
        then error out.
        :param checkpoint: Checkpoint path
        :return: None
        """

        if self.multiprocessing is True:
            if hvd.rank() == 0:
                state_dict = torch.load(checkpoint)
            else:
                state_dict = None
            state_dict = hvd.broadcast_object(state_dict, root_rank=0, name='state_dict')
        else:
            state_dict = torch.load(checkpoint)

        experimentStatus = state_dict['experimentStatus']
        config = state_dict['experimentConfig']
        if loadModelOnly is False:
            self.experimentStatus = experimentStatus
            self.config = config
            # Save the optimizer state
            self.optimizerStateDict = state_dict['optimizer']

        # Load the model
        # Check for whether it makes sense to load a quantized experiment
        # If so, quantize the model before proceed with loading
        if experimentStatus.flagFusedQuantized is True:
            assert self.config.quantize is True, \
                'Loaded experiment contains quantized model, but the experiment config does not require quantization'
            self.quantize_model()

        if experimentStatus.flagPruned is True:
            assert self.config.prune is True, \
                'Loaded experiment contains pruned model, but the experiment config does not require pruning'
            self.prune_network()

        self.restore_model_from_state_dict(state_dict['model'])

    def save_experiment_to_checkpoint(self, optimizer, filePath):
        """
        Saves experiment stats, optimizer, and model from checkpoint
        :param optimizer: The optimizer object to be saved
        :param filepath: Path to the checkpoint to be saved
        :return: None
        """
        if (self.multiprocessing is False) or (self.multiprocessing is True and hvd.rank() == 0):
            filename = self.config.checkpointSaveFileNameFormat.format(self.experimentStatus.numEpochTrained)
            path = os.path.join(filePath, filename)
            state = {
                'experimentStatus': self.experimentStatus,
                'experimentConfig': self.config,
                'optimizer': optimizer.state_dict(),
                'model': self.model.state_dict()
            }
            torch.save(state, path)

    def evaluate(self, data, target, isTrain=False) -> torch.Tensor:
        """
        Evaluate the model on one batch of data.
        Calculates the losses, and update the repsective meter depends on the isTrain flag
        :param data: The input data
        :param target: ground truth
        :param isTrain: Flag indicating wether the evlaute method is called during training
        :return: The total loss
        """
        # The following dicts are global objects. Need to "declare" them in this variable scope
        global globalActivationDict
        global globalWeightDict
        # Clear the intermediate tensor lists
        globalActivationDict.clear()
        globalWeightDict.clear()

        batchSize = target.size()[0]
        output = self.model(data)

        # Prediction loss is already averaged over the local batch size
        predictionLoss = self.evaluate_loss(output, target)

        # If pruning has not been enacted, obtain weight Lasso and L2 regularization loss
        weightGroupLassoLoss = torch.tensor(0.0)
        weightL2Loss = torch.tensor(0.0)

        self.extract_weight(self.model)
        for key, tensor in globalWeightDict.items():
            weightGroupLassoLoss.add_(custom_prune.calculateChannelGroupLasso(tensor,
                                                                              clusterSize=self.config.pruneCluster))
            weightL2Loss.add_(tensor.pow(2.0).sum())

        weightGroupLassoLoss.mul_(self.config.lossWeightL1Lambda)
        weightL2Loss.mul_(self.config.lossWeightL2Lambda)

        # TODO: Calculate activation regularization loss
        activationGroupLassoLoss = torch.tensor(0.0)
        for _, tensor in globalActivationDict.items():
            activationGroupLassoLoss.add_(custom_prune.calculateChannelGroupLasso(tensor,
                                                                                  clusterSize=self.config.pruneCluster))
        activationGroupLassoLoss.div_(batchSize)
        activationGroupLassoLoss.mul_(self.config.lossActivationLambda)
        # print('predictionLoss: ', predictionLoss)
        # print('activationGroupLassoLoss ', activationGroupLassoLoss)
        totalLoss = predictionLoss + \
                    weightGroupLassoLoss + weightL2Loss + activationGroupLassoLoss

        meter = self.trainMeter if isTrain is True else self.valMeter
        meter.update(
            modelOutput=output,
            target=target,
            totalLoss=totalLoss,
            predictionLoss=predictionLoss,
            weightL2Loss=weightL2Loss,
            weightSparsityLoss=weightGroupLassoLoss,
            activationSparsityLoss=activationGroupLassoLoss
        )

        return totalLoss


    def train_one_epoch(self, optimizer, epoch: int) -> None:
        """
        Train the model for one epoch
        :param optimizer optimizer object
        :param epoch: The current epoch
        :return:
        """
        # Add forward_hook to extract activation from relevant layers
        fowardHookHandles = self.apply_hook_activation(self.model)

        # Reset the train meter stats
        self.trainMeter.reset()
        # Set the model into training mode
        self.model.train()

        # Shuffle the data set if we are using multiprocessing
        if self.trainDataSampler is not None:
            self.trainDataSampler.set_epoch(epoch)

        for batchIdx, (data, target) in enumerate(self.trainDataLoader):
            self.adjust_learning_rate(epoch, batchIdx, optimizer)
            optimizer.zero_grad()
            totalLoss = self.evaluate(data, target, isTrain=True)
            # print('totalLoss: ', totalLoss)
            totalLoss.backward()
            optimizer.step()
            # End of training one iteration
        # End of training one epoch

        # Log the average accuracy, and various losses over the epoch
        self.trainMeter.log(epoch)

        # Remove the activation intercept hooks
        remove_hook_activation(forwardHookHandlesDict=fowardHookHandles)


    def train(self):
        """
        Train a model for multiple epoch and evaluate on the validation set after every epoch
        :return: None
        """
        self.prepare_model()
        optimizer = self.initialize_optimizer()
        if self.optimizerStateDict:
            if (self.multiprocessing is True and hvd.rank() == 0) or self.multiprocessing is False:
                    optimizer.load_state_dict(self.optimizerStateDict)

        if self.multiprocessing is True:
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        resumeFromEpoch = self.experimentStatus.numEpochTrained
        for epoch in range(resumeFromEpoch, self.config.numEpochToTrain):
            print('Epoch {}, train'.format(epoch + 1))
            self.train_one_epoch(optimizer, epoch)
            print('Epoch {}, validate'.format(epoch + 1))
            self.validate(epoch)
            # TODO: save the checkpoint
            self.save_experiment_to_checkpoint(optimizer, self.config.checkpointSaveDir)

            #Update statistics
            self.experimentStatus.numEpochTrained += 1

    def validate(self, epoch: int):
        """
        Validate the model on the validation set
        :return: None
        """
        # Add forward_hook to extract activation from relevant layers
        fowardHookHandles = self.apply_hook_activation(self.model)

        self.valMeter.reset()
        self.model.eval()

        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(self.valDataLoader):
                self.evaluate(data, target, isTrain=False)
                # End of one validation iteration in the epoch
            # End of one validation epoch

            # Log the validation epoch
            self.valMeter.log(epoch)

        remove_hook_activation(forwardHookHandlesDict=fowardHookHandles)

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

        def evaluate_setup (module : torch.nn.Module, prefix=None):
            interceptHandleList = []
            weightList = []
            layerNameList = []

            for name, m in module.named_children():
                if prefix:
                    modulePrefix = prefix + '/' + name
                else:
                    modulePrefix = name
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    custom_prune.applyClusterPruning(m, name='weight',
                                             clusterSize=self.config.pruneCluster,
                                             threshold=self.config.pruneThreshold)
                    interceptHandle = m.register_forward_hook(intercept_activation)
                    interceptHandleList.append(interceptHandle)
                    weightList.append(m.weight.detach().clone())
                    layerNameList.append(modulePrefix)
                else:
                    localInterceptHandleList, localWeightList, localLayerNameList = \
                        evaluate_setup(m, modulePrefix)

                    interceptHandleList.extend(localInterceptHandleList)
                    weightList.extend(localWeightList)
                    layerNameList.extend(localLayerNameList)

            return interceptHandleList, weightList, layerNameList
        #End of helper function evaluate_setup

        def generate_sparsity_list(tensorList : list):
            sparsityList = []
            for idx, tensor in enumerate(tensorList):
                mask = compute_mask(tensor, self.config.pruneCluster, self.config.pruneThreshold)

                mask = mask.byte()
                reference = torch.ones_like(mask)
                comparison = torch.eq(mask, reference)
                numNz = torch.sum(comparison.float())
                sparsity = numNz.item() / comparison.numel()

                sparsityList.append(sparsity)

            return sparsityList

        if self.multiprocessing is True:
            assert hvd.size() == 1, "Sparsity evaluation cannot be done in multi-processing mode!"

        # Fuse and quantized the model if this is haven't been done so
        evaluatedModel = copy.deepcopy(self.model)
        if self.experimentStatus.flagFusedQuantized is False:
            evaluatedModel.fuse_model()
            evaluatedModel.qconfig = self.qatConfig
            torch.quantization.prepare_qat(evaluatedModel, inplace=True)

        # Apply pruning mask, and activation interception, extract weight
        interceptHandleList, weightList, layerNameList = \
            evaluate_setup(evaluatedModel)

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

                np.add(x1=batchActivationSparsityList, x2=activationSparsityList, out=activationSparsityList)
                # End of iteration of all validation data
            activationSparsityList = activationSparsityList / float(len(self.valDataLoader))

        return activationSparsityList, weightSparsityList, layerNameList
        # End of evaluate sparsity






