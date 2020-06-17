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
from abc import ABC, abstractmethod

import os
import math

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


  def train_one_epoch(self):
    """
    Train the model for one epoch
    :return:
    """
    pass

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