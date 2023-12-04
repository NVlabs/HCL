#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np
import random
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
from utils.deep_inversion import DeepInversionFeatureHook
from utils.losses import LabelSmoothing, KL_div_Loss
import torchvision.utils as vutils
from datasets import get_dataset


def lr_policy(lr_fn):
    def _alr(optimizer, epoch):
        lr = lr_fn(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        print(lr)
        return lr
    return lr_policy(_lr_fn)


class Qdi(ContinualModel):
    NAME = 'qdi'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform, global_model=None):
        super(Qdi, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.num_classes = 10
        im_size = (32, 32) if args.dataset.name == "seq-cifar10" or args.dataset.name == "seq-cifar100" else (64, 64)
        images_per_class = 20
        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.global_model = global_model
        self.criterion_kl = KL_div_Loss(temperature=1.0).cuda()
        self.lr_scheduler = lr_cosine_policy(args.train.di_lr, 100, args.train.di_itrs)
        self.args = args
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.current_step = 0

    def begin_task(self, task_id, dataset=None):
      if task_id:
        self.sample_inputs = []
        if dataset is not None:
            for i in range(0, dataset.train_loader.dataset.data.shape[0], self.args.train.batch_size):
                inputs = torch.stack([dataset.train_loader.dataset.__getitem__(j)[0][0]
                            for j in range(i, min(i + self.args.train.batch_size, len(dataset.train_loader.dataset)))])
                self.sample_inputs.append(inputs)

            self.sample_inputs = torch.cat(self.sample_inputs)

        rand_idx = torch.randperm(self.sample_inputs.shape[0])
        sample_inputs = self.sample_inputs[rand_idx].to(self.device)
        sample_batch = sample_inputs[:self.args.model.buffer_size * 4].to(self.device)
        statistics = []

        batchnorm_flag = [True if isinstance(module, torch.nn.BatchNorm2d) else False for module in self.global_model.net.module.backbone.modules()]
                   
        if True in batchnorm_flag:
            for module in self.global_model.net.module.backbone.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    statistics.append(DeepInversionFeatureHook(module))

            for item in statistics:
                item.capture_bn_stats = False
                item.use_stored_stats = False
        else:
            for module in self.global_model.net.module.backbone.modules():
                if isinstance(module, torch.nn.Conv2d):
                    statistics.append(DeepInversionFeatureHook(module))

            for item in statistics:
                item.capture_bn_stats = True
                item.use_stored_stats = True
        
            _ = self.global_model.net.module.backbone(sample_batch)    
            print('Finished capturing post conv2d stats. Freezing the stats.')
                
            for item in statistics:
                item.capture_bn_stats = False
                item.use_stored_stats = True

        rand_idx = torch.randperm(self.sample_inputs.shape[0])
        sample_inputs = self.sample_inputs[rand_idx].to(self.device)
        sample_batch = sample_inputs[:self.args.model.buffer_size].to(self.device)
        vutils.save_image(sample_batch.data.clone(),
                f'./di_images_{self.args.dataset.name}/sample_batch_{task_id}.png',
                normalize=True, scale_each=True, nrow=5)
        sample_batch_size, im_size = sample_batch.shape[0], sample_batch.shape[2]
        cls_per_task = task_id * self.cpt
        self.label_syn = torch.tensor([np.ones(sample_batch_size//cls_per_task) * i for i in range(cls_per_task)], dtype=torch.long, requires_grad=False, device=self.device).view(-1)
        rand_idx = torch.randperm(len(self.label_syn))
        label_syn = self.label_syn[rand_idx]
        image_syn = torch.randn(size=(self.label_syn.shape[0], 3, im_size, im_size), dtype=torch.float, requires_grad=True, device=self.device)
        sample_batch = sample_batch[:self.label_syn.shape[0]]
        image_syn.data = sample_batch.data.clone()
        image_opt = torch.optim.Adam([image_syn], lr=self.args.train.di_lr, betas=[0.5, 0.9], eps = 1e-8)
        

        self.global_model.eval()
        self.net.eval()

        for step in range(self.args.train.di_itrs +1):
            self.lr_scheduler(image_opt, step)
            image_opt.zero_grad()
            self.global_model.zero_grad()
            outputs = self.global_model.net.module.backbone(image_syn)
            loss_ce = self.loss(outputs, label_syn.long())

            diff1 = image_syn[:,:,:,:-1] - image_syn[:,:,:,1:]
            diff2 = image_syn[:,:,:-1,:] - image_syn[:,:,1:,:]
            diff3 = image_syn[:,:,1:,:-1] - image_syn[:,:,:-1,1:]
            diff4 = image_syn[:,:,:-1,:-1] - image_syn[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

            loss_distr = self.args.train.di_feature * sum([mod.r_feature for mod in statistics])
            loss_var = self.args.train.di_var * loss_var
            loss_l2 = self.args.train.di_l2 * torch.norm(image_syn, 2)
            loss = loss_ce + loss_distr + loss_l2 + loss_var

            if step % 5 == 0:
                print('\t step', step, '\t ce', loss_ce.item(), '\t r feature', loss_distr.item(), '\tr var', loss_var.item(), '\tr l2', loss_l2.item(), '\t total', loss.item())
                
            loss.backward()
            image_opt.step()
            if step % 5 == 0:
                vutils.save_image(image_syn.data.clone(),
                        f'./di_images_{self.args.dataset.name}/di_generated_{task_id}_{step//5}.png',
                        normalize=True, scale_each=True, nrow=5)

        self.global_model.buffer.add_data(examples=image_syn, labels=label_syn)
        self.image_syn = image_syn.detach().clone()
        self.label_syn = label_syn.detach().clone()
        self.net.train()

    def observe(self, inputs1, labels, inputs2, notaug_inputs, task_id):
        inputs1, labels = inputs1.to(self.device), labels.to(self.device)
        real_batch_size = inputs1.shape[0]

        if task_id:
            outputs_clean = self.net.module.backbone(inputs1)

            outputs = self.net.module.backbone(self.image_syn)
            outputs_teacher = self.global_model.net.module.backbone(self.image_syn)
            outputs_teacher_clean = self.global_model.net.module.backbone(inputs1)

            penalty = self.criterion_kl(outputs_clean, outputs_teacher_clean) + self.criterion_kl(outputs, outputs_teacher)
            loss = self.loss(outputs_clean, labels) + self.args.train.alpha * penalty
        else:
            outputs = self.net.module.backbone(inputs1)
            loss = self.loss(outputs, labels)

        if task_id:
            data_dict = {'loss': loss, 'penalty': penalty}
        else:
            data_dict = {'loss': loss, 'penalty': 0.}
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})
        self.current_step += 1

        return data_dict
