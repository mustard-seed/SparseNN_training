import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torch.optim as optim
import horovod.torch as hvd
from torchvision import datasets, transforms
import torch.utils.data.distributed as distributed
from torch.utils.data import DataLoader


from custom_modules.custom_modules import ConvBNReLU, LinearReLU, ConvReLU
import pruning.pruning as custom_pruning
import experiment.experiment as experiment
from utils.meters import ClassificationMeter
from experiment.experiment import experimentBase, globalActivationDict, globalWeightDict, hook_activation


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.convBN1 = ConvBNReLU(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.convBN2 = ConvBNReLU(6, 16, 5)
        self.fc1 = LinearReLU(16 * 7 * 7, 120)
        self.fc2 = LinearReLU(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.quant = QuantStub()
        self.deQuant = DeQuantStub()

        # weight and BN parameter initialization
        # BN: set gamma (a.k.a weight) to 1, and bias to zero
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        # BN statistics initialization

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(self.convBN1(x))
        x = self.pool(self.convBN2(x))
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.deQuant(x)
        return x

    # Fuse convBNReLU prior to quantization
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                # Fuse the layers in ConvBNReLU module, which is derived from nn.Sequential
                # Use the default fuser function
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            elif type(m) == LinearReLU:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)

class experimentLeNet(experimentBase):
    def __init__(self, configFile, multiprocessing=False):
        super().__init__(configFile, multiprocessing)
        self.model = LeNet()

        # Dataset rootdir relative to the python script
        datasetDir = 'datasets'
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
        self.trainDataSet = datasets.MNIST(datasetDir, train=True, download=False,
                                           transform=transform)
        self.trainDataSampler = distributed.DistributedSampler(
            self.trainDataSet, num_replicas=hvd.size(), rank=hvd.rank()
        ) if multiprocessing is True \
            else None

        self.trainDataLoader = DataLoader(
            self.trainDataSet,
            batch_size=self.config.batchSizePerWorker,
            sampler=self.trainDataSampler,
            shuffle=True if self.trainDataSampler is None else False
        )
        self.valDataSet = datasets.MNIST(datasetDir, train=False, download=False,
                                           transform=transform)

        self.valDataSampler = distributed.DistributedSampler(
            self.valDataSet, num_replicas=hvd.size(), rank=hvd.rank()
        ) if multiprocessing is True \
            else None

        self.valDataLoader = DataLoader(
            self.valDataSet,
            batch_size=self.config.batchSizePerWorker,
            sampler=self.valDataSampler,
            shuffle=True if self.valDataSampler is None else False
        )

        self.trainMeter = ClassificationMeter(
            multiprocessing,
            logDir = self.config.logDir,
            logPrefix='Train'
        )

        self.valMeter = ClassificationMeter(
            multiprocessing,
            logDir=self.config.logDir,
            logPrefix='Train'
        )

        # End of __init__

    def evaluate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(output, target)

    def apply_hook_activation(self, module: torch.nn.Module, prefix=None) -> dict:
        return super().apply_hook_activation(module, prefix)

    def extract_weight(self, module: torch.nn.Module) -> None:
        super().extract_weight(module)

    def prune_network(self) -> None:
        pruneList = [self.model.convBN1, self.model.convBN2, self.model.fc1,
                     self.model.fc2, self.model.fc3]
        for m in pruneList:
            if isinstance(m, (ConvBNReLU, ConvReLU)):
                layer = m.children()[0]
                custom_pruning.applyClusterPruning(layer,
                                                   "weight",
                                                   clusterSize=self.config.pruneCluster,
                                                   threshold=self.config.pruneThreshold)
            elif isinstance(m, nn.Linear):
                layer = m
                custom_pruning.applyClusterPruning(layer,
                                                   "weight",
                                                   clusterSize=self.config.pruneCluster,
                                                   threshold=self.config.pruneThreshold)



