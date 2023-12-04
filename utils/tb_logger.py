# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.conf import base_path
import os
from argparse import Namespace
from typing import Dict, Any
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def img_denormlaize(img):
    """Scaling and shift a batch of images (NCHW)
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2615]
    nch = img.shape[1]

    mean = torch.tensor(mean, device=img.device).reshape(1, nch, 1, 1)
    std = torch.tensor(std, device=img.device).reshape(1, nch, 1, 1)

    return img * std + mean


def save_img(img, unnormalize=True, max_num=5, size=32, nrow=5, dataname='imagenet'):
    img = img[:max_num].detach()
    if unnormalize:
        img = img_denormlaize(img)
    images = torch.clamp(img, min=0., max=1.)
    images = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    # print(images.shape)
    # if img.shape[-1] > size:
        # img = F.interpolate(img, size)
    
    return images

class TensorboardLogger:
    def __init__(self, args: Namespace, setting: str,
                 stash: Dict[Any, str]=None) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.settings = [setting]
        if setting == 'class-il':
            self.settings.append('task-il')
        self.loggers = {}
        self.name = args.model.backbone
        for a_setting in self.settings:
            self.loggers[a_setting] = SummaryWriter(
                os.path.join(args.ckpt_dir, 'tensorboard_runs'))
        config_text = ', '.join(
            ["%s=%s" % (name, getattr(args, name)) for name in args.__dir__()
             if not name.startswith('_')])
        for a_logger in self.loggers.values():
            a_logger.add_text('config', config_text)

    def get_name(self) -> str:
        """
        :return: the name of the model
        """
        return self.name

    def log_accuracy(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     args: Namespace, task_number: int) -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """
        mean_acc_common, mean_acc_task_il = all_mean_accs
        for setting, a_logger in self.loggers.items():
            mean_acc = mean_acc_task_il\
                if setting == 'task-il' else mean_acc_common
            index = 1 if setting == 'task-il' else 0
            accs = [all_accs[index][kk] for kk in range(len(all_accs[0]))]
            for kk, acc in enumerate(accs):
                a_logger.add_scalar('acc_task%02d' % (kk + 1), acc,
                                    task_number * args.train.num_epochs)
            a_logger.add_scalar('acc_mean', mean_acc, task_number * args.train.num_epochs)

    def log_loss(self, loss: float, args: Namespace, epoch: int,
                 task_number: int, iteration: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss', loss, task_number * args.train.num_epochs + epoch)


    def log_penalty(self, penalty: float, args: Namespace, epoch: int,
                 task_number: int, iteration: int) -> None:
        """
        Logs the loss penalty value at each iteration.
        :param loss penalty: the loss penalty value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('penalty', penalty, task_number * args.train.num_epochs + epoch)


    def log_lr(self, lr: float, args: Namespace, epoch: int,
                 task_number: int, iteration: int) -> None:
        """
        Logs the lr value at each iteration.
        :param lr: the lr value
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('lr', lr, iteration)
            a_logger.add_scalar('lr', lr, task_number * args.train.num_epochs + epoch)

    def log_images(self, images, args: Namespace, epoch: int,
                 task_number: int, iteration: int) -> None:
        """
        Logs the lr value at each iteration.
        :param lr: the lr value
        :param iteration: the current iteration
        """
        # img_grid = torchvision.utils.make_grid(images)
        # matplotlib_imshow(img_grid)
        images = save_img(images)
        for a_logger in self.loggers.values():
            a_logger.add_image('syn_images', images, task_number * args.train.num_epochs + epoch)

    def log_loss_gcl(self, loss: float, iteration: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss', loss, iteration)

    def close(self) -> None:
        """
        At the end of the execution, closes the logger.
        """
        for a_logger in self.loggers.values():
            a_logger.close()
