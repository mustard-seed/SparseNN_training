import torch
import horovod.torch as hvd
from torch.utils.tensorboard import SummaryWriter

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


class BaseMeter(object):
  def __init__(self, flagUseMultiProcessing : bool = False, logDir='', logPrefix=''):
    self.logPrefix = logPrefix
    self.logWriter = None
    if (flagUseMultiProcessing is True and hvd.rank() == 0) or flagUseMultiProcessing is False:
      self.logWriter = SummaryWriter(logDir)
    self.aggregateLoss = AverageMeter(logPrefix + '/Loss')
    self.aggregatePredictionLoss = AverageMeter(logPrefix + '/Prediction Loss')
    self.aggregateWeightL2Loss = AverageMeter(logPrefix + '/Weight L2 Regularization Loss')
    self.aggregateWeightSparsityLoss = AverageMeter(logPrefix + '/Weight Structured SP loss')
    self.aggregateActivationSparsityLoss = AverageMeter(logPrefix + '/Activation Structure SP loss')

  def reset(self) -> None:
    self.aggregateLoss.reset()
    self.aggregatePredictionLoss.reset()
    self.aggregateWeightL2Loss.reset()
    self.aggregateWeightSparsityLoss.reset()
    self.aggregateActivationSparsityLoss.reset()

  def update(self,
             modelOutput: torch.Tensor,
             target: torch.Tensor,
             totalLoss: torch.Tensor,
             predictionLoss: torch.Tensor,
             weightL2Loss: torch.Tensor,
             weightSparsityLoss: torch.Tensor,
             activationSparsityLoss: torch.Tensor) -> None:

    self.aggregateLoss.update(totalLoss)
    self.aggregatePredictionLoss.update(predictionLoss)
    self.aggregateActivationSparsityLoss.update(activationSparsityLoss)
    self.aggregateWeightSparsityLoss.update(weightSparsityLoss)
    self.aggregateWeightL2Loss.update(weightL2Loss)

  def log(self, epoch):
    self.logWriter.add_scalar(self.logPrefix + '/total_loss', self.aggregateLoss.avg, epoch)
    self.logWriter.add_scalar(self.logPrefix + '/weight_sparsity_loss', self.aggregateWeightSparsityLoss.avg, epoch)
    self.logWriter.add_scalar(self.logPrefix + '/activation_sparsity_loss', self.aggregateActivationSparsityLoss.avg, epoch)
    self.logWriter.add_scalar(self.logPrefix + '/weight_L2_loss', self.aggregateWeightL2Loss.avg, epoch)


class ClassificationMeter(BaseMeter):
  def __init__(self, flagUseMultiProcessing : bool = False, logDir='', logPrefix=''):
    super().__init__(flagUseMultiProcessing, logDir, logPrefix)
    self.aggregateAccuracyTop1 = AverageMeter('Top1 Accuracy')

  def reset(self) -> None:
    super().reset()
    self.aggregateAccuracyTop1.reset()

  def update(self,
             modelOutput : torch.Tensor,
             target : torch.Tensor,
             totalLoss : torch.Tensor,
             predictionLoss : torch.Tensor,
             weightL2Loss : torch.Tensor,
             weightSparsityLoss : torch.Tensor,
             activationSparsityLoss : torch.Tensor) -> None:

    super().update(modelOutput, target, totalLoss,
                   predictionLoss, weightL2Loss,
                   weightSparsityLoss, activationSparsityLoss)

    acc1 = accuracy(modelOutput, target, topk=(1,))
    self.aggregateAccuracyTop1.update(acc1)

  def log(self, epoch):
    self.logWriter.add_scalar(self.logPrefix + '/acc1', self.aggregateAccuracyTop1.avg, epoch)
    super().log(epoch)
