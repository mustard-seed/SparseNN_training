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
            res.append(correct_k.mul_(100.0 / batch_size).squeeze())
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, multiprocessing=False, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.mp = multiprocessing
        self.reset()

    def reset(self):
        self.val = torch.tensor(0.)
        self.sum = torch.tensor(0.)
        self.count = torch.tensor(0.)

    def update(self, val):
        if self.mp is True:
            val = hvd.allreduce(val.detach().cpu(), name=self.name)
        self.val = val
        self.sum += val
        self.count += 1

    @property
    def avg(self):
        return self.sum / self.count


class BaseMeter(object):
    def __init__(self, multiprocessing: bool = False, logWriter=None, logPrefix=''):
        self.logPrefix = logPrefix
        self.logWriter = None
        self.logWriter = logWriter
        self.aggregateLoss = AverageMeter(logPrefix + '/Loss', multiprocessing)
        self.aggregatePredictionLoss = AverageMeter(logPrefix + '/Prediction Loss', multiprocessing=multiprocessing)
        self.aggregateWeightL2Loss = AverageMeter(logPrefix + '/Weight L2 Regularization Loss', multiprocessing=multiprocessing)
        self.aggregateWeightSparsityLoss = AverageMeter(logPrefix + '/Weight Structured SP loss', multiprocessing=multiprocessing)
        self.aggregateActivationSparsityLoss = AverageMeter(logPrefix + '/Activation Structure SP loss', multiprocessing=multiprocessing)

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
        if self.logWriter:
            self.logWriter.add_scalar(self.logPrefix + '/total_loss', self.aggregateLoss.avg, epoch)
            self.logWriter.add_scalar(self.logPrefix + '/weight_sparsity_loss', self.aggregateWeightSparsityLoss.avg, epoch)
            self.logWriter.add_scalar(self.logPrefix + '/activation_sparsity_loss',
                                      self.aggregateActivationSparsityLoss.avg, epoch)
            self.logWriter.add_scalar(self.logPrefix + '/weight_L2_loss', self.aggregateWeightL2Loss.avg, epoch)

            print('----{prefix}: {epoch}------\n Average total loss: {total_loss:.2f} \n Average activation loss: {activation_loss:.2f} \n Average prediction loss: {prediction_loss}'.format(
                prefix = self.logPrefix,
                epoch = epoch+1,
                total_loss = self.aggregateLoss.avg,
                activation_loss = self.aggregateActivationSparsityLoss.avg,
                prediction_loss = self.aggregatePredictionLoss.avg
            ))


class ClassificationMeter(BaseMeter):
    def __init__(self, multiprocessing: bool = False, logWriter=None, logPrefix=''):
        super().__init__(multiprocessing, logWriter, logPrefix)
        self.aggregateAccuracyTop1 = AverageMeter('Top1 Accuracy', multiprocessing=multiprocessing)

    def reset(self) -> None:
        super().reset()
        self.aggregateAccuracyTop1.reset()

    def update(self,
               modelOutput: torch.Tensor,
               target: torch.Tensor,
               totalLoss: torch.Tensor,
               predictionLoss: torch.Tensor,
               weightL2Loss: torch.Tensor,
               weightSparsityLoss: torch.Tensor,
               activationSparsityLoss: torch.Tensor) -> None:
        super().update(modelOutput, target, totalLoss,
                       predictionLoss, weightL2Loss,
                       weightSparsityLoss, activationSparsityLoss)

        accList = accuracy(modelOutput, target, topk=(1,))
        self.aggregateAccuracyTop1.update(accList[0])

    def log(self, epoch):
        if self.logWriter:
            super().log(epoch)
            self.logWriter.add_scalar(self.logPrefix + '/acc1', self.aggregateAccuracyTop1.avg, epoch)
            print('top-1 prediction accuracy: {:2f}%\n'.format(self.aggregateAccuracyTop1.avg))