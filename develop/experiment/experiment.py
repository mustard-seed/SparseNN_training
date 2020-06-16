import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import yaml
from abc import ABC, abstractmethod

import os
import math

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
  def __init__(self, configFile):
    """
    Instantiates an experiment object. This is an abstract method with implementation
    Children initialization methods should call this,
    and define how 1) the data loader, 2) the model, and 3) the experiment type name
    :param configFile: Path to YAML file that contains the configuration
    """
    #Experiment states initialization
    self.bestAccuracy = 0.0
    self.flagFusedQuantized = False
    self.totalEpochElapsed = 0

    #Load experiment setting from config file
    with open(configFile, 'r') as config:
      self.configPatchSizePerWorker = extractConfigKey(config, 'batch_size')
      self.configNumOfWorkers = extractConfigKey(config, 'num_workers')
      self.configNumOfThreadsPerWorker = extractConfigKey(config, 'threads_per_worker')
      self.configTrainEpochs = extractConfigKey(config, "num_epoch")

      #Learning rate configuration
      #Constant multiplicative LR with optional warmup
      self.configInitialBaseLR = extractConfigKey(config, 'initial_base_lr')
      self.configPeakBaseLR = extractConfigKey(config, 'peak_base_lr')
      self.configLRWarmUp = extractConfigKey(config, 'warm_up_lr')
      self.configLRSchedule = extractConfigKey(config, 'schedule_lr')
      self.configMomentum = extractConfigKey(config, "momentum")

      #Loss contribution
      self.lossActivationLambda = extractConfigKey(config, 'loss_activation_lambda')
      self.lossWeightL1Lambda = extractConfigKey(config, 'loss_weight_l1_lambda')
      self.lossWeightL2Lambda = extractConfigKey(config, 'loss_weight_l2_lambda')

      #Pruning paramters
      self.configPrune = extractConfigKey(config, 'prune')
      self.configPruneCluster = extractConfigKey(config, 'prune_cluster')
      self.configPruneThreshold = extractConfigKey(config, 'prune_threshold')
      self.configQuantize = extractConfigKey(config, 'quantize')

      #See
      self.seed = extractConfigKey(config, 'seed')

      #Path configuration
      #TODO: Add paths for logging, dataset, loading/saving checkpoint,
      # and flag for checking whether loading from checkpoint is needed


  def lr_scheduler(self, epoch):
    """
    Compute the learning rate for the current epoch
    :return: The learning rate for the current epoch
    """
    return 1.0
  def restore_experiment_from_checkpoint(self, checkpoint):
    """
    Restores an experiment stats, optimizer, and model from checkpoint
    :return: None
    """
    #TODO: Implement this later
    pass

  def save_experiment_to_checkpoint(self, filepath):
    """
    Saves experiment stats, optimizer, and model from checkpoint
    :param filepath: Path to the checkpoint to be saved
    :return: None
    """
    #TODO: Implement this later
    pass

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