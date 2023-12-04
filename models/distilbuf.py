#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import torch
from utils.losses import LabelSmoothing, KL_div_Loss
from datasets import get_dataset

def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))



class DistilBuf(ContinualModel):
    NAME = 'distilbuf'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform, global_model):
        super(DistilBuf, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.global_model = global_model
        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.global_model = global_model
        self.criterion_kl = KL_div_Loss(temperature=1.0).cuda()
        self.soft = torch.nn.Softmax(dim=1)

    def observe(self, inputs1, labels, inputs2, notaug_inputs, task_id):

        self.opt.zero_grad()
        inputs1, labels = inputs1.to(self.device), labels.to(self.device)
        inputs2 = inputs2.to(self.device)
        notaug_inputs = notaug_inputs.to(self.device)
        real_batch_size = inputs1.shape[0]

        if task_id:
            self.global_model.eval()
            outputs = self.net.module.backbone(inputs1)
            with torch.no_grad():
                outputs_teacher = self.global_model.net.module.backbone(inputs1)

            penalty = self.args.train.alpha * self.criterion_kl(outputs, outputs_teacher)
            loss = self.loss(outputs, labels) + penalty
        else:
            outputs = self.net.module.backbone(inputs1)
            loss = self.loss(outputs, labels)

        if not self.global_model.buffer.is_empty():
            buf_inputs, buf_logits = self.global_model.buffer.get_data(
                    self.args.train.batch_size, transform=self.transform)
            buf_outputs = self.net.module.backbone(buf_inputs)
            penalty = 0.3 * self.loss(buf_outputs, buf_logits.long())
            loss += penalty

        if task_id:
            data_dict = {'loss': loss, 'penalty': penalty}
        else:
            data_dict = {'loss': loss, 'penalty': 0.}

        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})
        self.global_model.buffer.add_data(examples=notaug_inputs, labels=labels[:real_batch_size])

        return data_dict
