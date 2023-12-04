# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from datasets.transforms.denormalization import DeNormalize
from augmentations import get_aug


class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                import gdown
                import zipfile
                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                url = 'https://drive.google.com/uc?id=1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj'
                if not os.path.exists(root): os.makedirs(root)
                gdown.download(url, root, quiet=False, fuzzy=True)
                with zipfile.ZipFile(os.listdir(root), "r") as f:
                        f.extractall(path=root)
                gdown.extractall(root)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class SequentialTinyImagenet(ContinualDataset):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(64, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821))])

    def get_data_loaders(self, args):
        imagenet_norm = [[0.4802, 0.4480, 0.3975], [0.2770, 0.2691, 0.2821]]
        transform = get_aug(train=True, mean_std=imagenet_norm, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, mean_std=imagenet_norm, **args.aug_kwargs)
        
        if args.server:
            train_dataset = TinyImagenet('/tinyimg_data', train=True,
                                  download=False, transform=transform)
            memory_dataset = TinyImagenet('/tinyimg_data', train=True,
                                  download=False, transform=test_transform)
        else:
            train_dataset = TinyImagenet(base_path() + 'TINYIMG',
                                 train=True, download=True, transform=transform)

            memory_dataset = TinyImagenet(base_path() + 'TINYIMG',
                                 train=True, download=True, transform=test_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
            memory_dataset, _ = get_train_val(memory_dataset, test_transform, self.NAME)
        else:
            if args.server:
                test_dataset = TinyImagenet('/tinyimg_data', train=False,
                                   download=False, transform=test_transform)
            else:
                test_dataset = TinyImagenet(base_path() + 'TINYIMG',
                        train=False, download=True, transform=test_transform)

        train, memory, test = store_masked_loaders(train_dataset, test_dataset, memory_dataset, self)
        return train, memory, test

    def get_transform(self, args):
        imagenet_norm = [[0.4802, 0.4480, 0.3975], [0.2770, 0.2691, 0.2821]]
        if args.cl_default:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_norm)
                ])
        else:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_norm)
                ])

        return transform

    def not_aug_dataloader(self, batch_size):
        imagenet_norm = [[0.4802, 0.4480, 0.3975], [0.2770, 0.2691, 0.2821]]
        transform = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize(*imagenet_norm)])

        train_dataset = TinyImagenet(base_path() + 'TINYIMG',
                            train=True, download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader
