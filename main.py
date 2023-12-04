"""
Training script

Originated from https://github.com/divyam3897/UCL/blob/main/main.py

Hacked together by / Copyright 2023 Divyam Madaan (https://github.com/divyam3897)
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, Logger, file_exist_check
from datasets import get_dataset
from datetime import datetime
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.tb_logger import TensorboardLogger
from typing import Tuple
from datasets import BACKBONES
import wandb
from pytorch_model_summary import summary


def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if classifier is not None:
                outputs = classifier(outputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
        
        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


def main(device, args):

    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)
    wandb.init(project="poc_lwf", sync_tensorboard=True)
    wandb.run.name = f"{args.model.cl_model}_{args.dataset.name}_n_alpha_{args.alpha}"

    # define model
    global_model = get_model(args, device, dataset_copy, dataset.get_transform(args), global_model=None)
    model = get_model(args, device, dataset_copy, dataset.get_transform(args), global_model=global_model)

    logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    tb_logger = TensorboardLogger(args, dataset.SETTING)
    csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, args.model.backbone)
    accuracy = 0 
    results, results_mask_classes = [], []

    for t in range(dataset.N_TASKS):
      train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
      
      global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
      prev_mean_acc = 0.
      best_epoch = 0.

      if args.hcl and BACKBONES[args.dataset.name][t] != BACKBONES[args.dataset.name][t - 1]:
        model = get_model(args, device, dataset_copy, dataset.get_transform(args), task_id=t, global_model=global_model)
        print(summary(model.net.module.backbone, torch.zeros((1, 3, args.dataset.image_size, args.dataset.image_size)).to(device), show_input=True))
      
      if hasattr(model, 'begin_task'):
         model.begin_task(t, dataset)

      if t:
        accs = evaluate(model, dataset, device, last=True)
        results[t-1] = results[t-1] + accs[0]
        results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

      for epoch in global_progress:
        model.train()
        
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, data in enumerate(local_progress):
            (images1, images2, notaug_images), labels = data
            data_dict = model.observe(images1, labels, images2, notaug_images, t)

            logger.update_scalers(data_dict)
            tb_logger.log_loss(data_dict['loss'], args, epoch, t, idx)
            tb_logger.log_penalty(data_dict['penalty'], args, epoch, t, idx)
            tb_logger.log_lr(data_dict['lr'], args, epoch, t, idx)
            
        global_progress.set_postfix(data_dict)
       
        accs = evaluate(model.net.module.backbone, dataset, device)
        mean_acc = np.mean(accs, axis=1)

        epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
        tb_logger.log_accuracy(accs, mean_acc, args, t)

        if (sum(mean_acc)/2.) - prev_mean_acc < -0.2:
            continue
        if args.cl_default:
            best_model = copy.deepcopy(model.net.module.backbone)
        else:
            best_model = copy.deepcopy(model.net.module)
        prev_mean_acc = sum(mean_acc)/2.
        best_epoch = epoch

      accs = evaluate(best_model, dataset, device)
      results.append(accs[0])
      results_mask_classes.append(accs[1])
      mean_acc = np.mean(accs, axis=1)
      print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
      
      if args.cl_default:
        model.global_model.net.module.backbone = copy.deepcopy(best_model)
      else:
        model.global_model.net.module = copy.deepcopy(best_model)
      print(f"Updated global model at epoch {best_epoch} with accuracy {prev_mean_acc}.")    

      model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}.pth")
      torch.save({
        'epoch': best_epoch+1,
        'state_dict': model.global_model.net.state_dict(),
      }, model_path)
      print(f"Task Model saved to {model_path}")
      with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')
      
      if hasattr(model, 'end_task'):
        model.end_task(dataset)

    csv_logger.add_bwt(results, results_mask_classes)
    csv_logger.add_forgetting(results, results_mask_classes)
    csv_logger.write(args.ckpt_dir, vars(args))
    tb_logger.close()
    if args.eval is not False and args.cl_default is False:
        args.eval_from = model_path

if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')


