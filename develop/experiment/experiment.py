import quantization.quantization as custom_quant
import pruning.pruning as custom_prune

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import yaml
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

import os
import math

# Make sure to import the global the following globals
# Global list for storing the intermediate activation layers
globalActivationDict = {}
# Function hook used to intercept intermediate activation layers
def hook_activation(module, input, output):
  global globalActivationDict
  globalActivationDict[module] = output

# Global list for storing the weight tensors
globalWeightDict = {}

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = torch.tensor(0.)
    self.sum = torch.tensor(0.)
    self.count = torch.tensor(0.)

  def update(self, val):
    val = hvd.allreduce(val.detach().cpu(), name=self.name)
    self.val = val
    self.sum += val
    self.count += 1

  @property
  def avg(self):
    return self.sum / self.count


def generate_base_config():
  config = edict()
  config.batchSizePerWorker = 32
  config.numOfThreadsPerWorker = 28
  config.trainEpochs = 20

  # Learning rate configuration
  # Constant multiplicative LR with optional warmup
  config.lrInitialBase = 1e-7
  config.lrPeakBase = 1e-5
  config.lrWarmupEpochs = 4
  config.lrStepSize = 4
  config.lrReductionFactor = 0.7
  config.lrMomentum = 0.01

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
  config.loadCheckpoint = ''
  config.loadPretrained = ''
  config.checkpointLoadPath = ''
  config.checkpointSaveDir = ''
  config.checkpointSaveFileNameFormat = ''

  return config

#Helper function for parsing configuration file keys
def extractConfigKey(config, key):
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

class experimentBase():
  """
  Base experiment class for classifier training and evaluation,
  with multinode processing support
  """

  @abstractmethod
  def __init__(self, configFile, multiprocessing=False):
    ##TODO: Children should provide their own init method
    ##1) Load configuration file
    ##2) Instantiate and initialize model
    ##3) Instantiate optimizer
    ##4) If load from checkpoint is required, then load from checkpoint
    ##5) Prune network if necessary
    # Experiment states initialization
    self.bestAccuracy = 0.0
    self.flagFusedQuantized = False
    self.totalEpochElapsed = 0
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
    self.optimizer = None
    self.model = None
    self.trainDataSet = None
    self.trainDataLoader = None
    self.trainSampler = None
    self.valDataSet = None
    self.valDataLoader = None
    self.valDataSampler = None
    self.logWriter = None

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
    if multiprocessing is True:
      config = hvd.broadcast_object(obj=config, root_rank=0, name='config')

    self.config = config
    torch.manual_seed(self.config.seed)
    #End of loading configuration file

    # TODO in the implementation of __init__ methods in the concrete classes
    # Instantiate data loader
    # Instantiate model
    # Quantize the model if quantization is required

  def adjust_learning_rate(self, epoch):
    """
    Compute the learning rate for the current epoch, and
    adjust the learning rate in the optimizer of the experiment
    :return: The learning rate for the current epoch
    """
    MININUM_LR = 1e-5
    lr = 1e-5
    if epoch < self.config.lrWarmupEpochs:
      lr = (epoch / self.config.lrWarmupEpochs * (self.config.lrPeakBase - self.config.lrInitialBase)) \
           + self.config.lrInitialBase
    else:
      lr = self.config.lrReductionFactor ** ((epoch - self.config.lrWarmupEpochs) // self.config.lrStepSize)
      lr = max(lr, MININUM_LR)

    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr

    return lr

  def quantize_model(self):
    """
    Helper function, quantize the model
    :return: None
    """
    self.model.fuse_model()
    self.model.qconfig = self.qatConfig
    torch.quantization.prepare_qat(self.model, inplace=True)
    self.flagFusedQuantized = True


  def restore_experiment_from_checkpoint(self, checkpoint):
    """
    Restores an experiment stats, optimizer, and model from checkpoint
    When restoring a model, look for the status flag 'quantized'
    If the model has been quantized, but the experiment doesn't require quantization,
    then error out.
    :param checkpoint: Checkpoint path
    :return: None
    """
    assert self.config.loadPretrained is False, \
      "Configuration error: cannot resume an experiment on top of a pre-trained model"

    if self.multiprocessing is True:
      if hvd.rank() == 0:
        state_dict = torch.load(checkpoint)
      else:
        state_dict = None
      state_dict = hvd.broadcast_object(state_dict, root_rank=0, name='state_dict')
    else:
      state_dict = torch.load(checkpoint)

    self.bestAccuracy = state_dict['bestAccuracy']
    self.totalEpochElapsed = state_dict['totalEpochElapsed']
    self.flagFusedQuantized = state_dict['fusedQuantized']
    self.optimizer.load_state_dict(state_dict['optimizer'])

    #Load the model
    #Check for whether it makes sense to load a quantized experiment
    #If so, quantize the model before proceed with loading
    if self.flagFusedQuantized is True:
      assert self.config.configQuantize is True, \
        "Loaded experiment contains quantized model, but the experiment config requires"

    if self.config.configQuantize is True:
      if self.flagFusedQuantized is True:
        self.quantize_model()

    model_dict = self.model.state_dict()

    #1. Filter out unnecessary keys
    saved_model_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    #2. Overwrite entries in the existing model
    model_dict.update(saved_model_dict)
    #3. Load the new state dicc
    self.model.load_state_dict(saved_model_dict)

    if self.config.configQuantize is True:
      if self.flagFusedQuantized is False:
        self.quantize_model()


  def save_experiment_to_checkpoint(self, filepath):
    """
    Saves experiment stats, optimizer, and model from checkpoint
    :param filepath: Path to the checkpoint to be saved
    :return: None
    """
    if (self.multiprocessing is False) or (self.multiprocessing is True and hvd.rank() == 0):
      filename = self.config.checkpointSaveFileNameFormat.format(self.totalEpochElapsed)
      path = os.path.join(self.config.checkpointSaveDir, filename)
      state = {
        'bestAccuracy': self.bestAccuracy,
        'totalEpochElapsed': self.totalEpochElapsed,
        'fusedQuantized': self.flagFusedQuantized,
        'optimizer': self.optimizer.state_dict(),
        'model': self.state_dict()
      }
      torch.save(state, path)

  @abstractmethod
  def apply_hook_activation(self, module: torch.nn.Module) -> None:
    """
    Specify what layer's activations should be extracted. To be called recursively
    Calls experiment.hook_activation. Make sure to import hook_activaiton
    :param module: Current top level module
    :return: None
    """
    for m in module.children():
      if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d)):
        m.register_forward_hook(hook_activation)
      else:
        self.apply_hook_activation(m)

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
        globalWeightDict[m] = m.weight
      else:
        self.extract_weight(m)

  def train_one_epoch(self, epoch: int) -> None:
    """
    Train the model for one epoch
    :return:
    """
    self.model.train()
    # Shuffle the data set
    if self.trainSampler is not None:
      self.trainSampler.set_epoch(epoch)
    # TODO: Initialize loss and accuracy meters
    aggregateAccuracyTop1 = AverageMeter("Top1 Accuracy")
    aggregateLoss = AverageMeter("Train Loss")
    aggregatePredictionLoss = AverageMeter("Prediction Loss")
    aggregateWeightL2Loss = AverageMeter("Weight L2 Regularization Loss")
    aggregateWeightSparsityLoss = AverageMeter("Weight Structured SP loss")
    aggregateActivationSparsityLoss = AverageMeter("Activation Structure SP loss")

    # TODO: Add forward_hook to extract activation from relevant layers
    self.apply_hook_activation(self.model)

    #If pruning is enacted, prune the network
    if self.config.prune is True:
      custom_prune.pruneNetwork(self.model, clusterSize=self.config.pruneCluster)

    for batchIdx, (data, target) in enumerate(self.trainDataLoader):
      global globalActivationDict
      global globalWeightDict
      batchSize = target.size()[0]

      #Clear the intermediate tensor lists
      globalActivationDict.clear()
      globalWeightDict.clear()

      output = self.model(data)
      #Prediction loss is already averaged over the local batch size
      predictionLoss = F.cross_entropy(output, target)

      #TODO: If pruning has not been enacted, obtain weight Lasso and L2 regularization loss
      weightGroupLassoLoss = torch.tensor(0)
      weightL2Loss = torch.tensor(0)
      self.extract_weight(self.model)
      for key, tensor in globalWeightDict.items():
        weightGroupLassoLoss.add_(custom_prune.calculateChannelGroupLasso(tensor,
            clusterSize=self.config.pruneCluster))
        weightL2Loss.add_(tensor.pow(2).sum())

      weightGroupLassoLoss.mul_(self.config.lossWeightL1Lambda)
      weightL2Loss.mul_(self.config.lossWeightL2Lambda)

      #TODO: Calculate activation regularization loss
      activationGroupLassoLoss = torch.tensor(0)
      for _, tensor in globalActivationDict.items():
        activationGroupLassoLoss.add_(custom_prune.calculateChannelGroupLasso(tensor,
            clusterSize=self.config.pruneCluster))
      activationGroupLassoLoss.div_(batchSize)
      activationGroupLassoLoss.mul_(self.config.lossActivationLambda)

      totalLoss = predictionLoss +\
                  weightGroupLassoLoss + weightL2Loss + activationGroupLassoLoss

      acc1 = accuracy(output, target, topk=(1,))

      aggregateAccuracyTop1.update(acc1)
      aggregateLoss.update(totalLoss)
      aggregateActivationSparsityLoss.update(activationGroupLassoLoss)
      aggregateWeightSparsityLoss.update(weightGroupLassoLoss)
      aggregateWeightL2Loss.update(weightL2Loss)

      self.optimizer.zero_grad()
      totalLoss.backward()
      self.optimizer.step()
      # End of training one iteration
    # End of training one epoch

    # TODO: Log the average accuracy, and various losses over the epoch
    self.logWriter.add_scalar("train/acc1", aggregateAccuracyTop1.avg, epoch)
    self.logWriter.add_scalar("train/total_loss", aggregateLoss.avg, epoch)
    self.logWriter.add_scalar("train/weight_sparsity_loss", aggregateWeightSparsityLoss.avg, epoch)
    self.logWriter.add_scalar("train/activation_sparsity_loss", aggregateActivationSparsityLoss.avg, epoch)
    self.logWriter.add_scalar("train/weight_L2_loss", aggregateWeightL2Loss.avg, epoch)

  def train(numEpoch):
    """
    Train a model for multiple epoch and evaluate on the validation set after every epoch
    :return: None
    """
    pass

  def validate(self):
    """
    Validate the model on the validation set
    :return: None
    """
    pass