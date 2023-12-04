# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
import torchvision

NAMES = {
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
}

N_CLASSES = {'seq-cifar10': 10, 'seq-cifar100': 100, 'seq-tinyimg': 200}
BACKBONES = {'seq-cifar10': ["lenet", "resnet18", "densenet", "senet", "regnet"],
             'seq-cifar100': ["lenet","lenet", "alexnet", "alexnet", "vgg16", "vgg16", "inception", "inception",  "resnet18", "resnet18", "resnext", "resnext", "densenet", "densenet",  "senet", "senet", "regnet", "regnet", "regnet", "regnet"],
             'seq-tinyimg': ["lenet", "lenet", "resnet18", "resnet18", "resnext", "resnext", "senet", "senet", "regnet", "regnet"],
             }

             
def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset_kwargs['dataset'] in NAMES.keys()
    return NAMES[args.dataset_kwargs['dataset']](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
