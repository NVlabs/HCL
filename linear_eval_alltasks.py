"""
Evaluation script

Originated from https://github.com/divyam3897/UCL/blob/main/linear_eval_alltasks.py

Hacked together by / Copyright 2023 Divyam Madaan (https://github.com/divyam3897)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter, knn_monitor
from datasets import get_dataset
from models.optimizers import get_optimizer, LR_Scheduler
from utils.loggers import *
from utils.metrics import forgetting


def evaluate_single(model, dataset, test_loader, memory_loader, device, k, last=False) -> Tuple[list, list, list, list]:
    accs, accs_mask_classes = [], []
    knn_accs, knn_accs_mask_classes = [], []
    correct = correct_mask_classes = total = 0
    knn_acc, knn_acc_mask = knn_monitor(model.net.module.backbone, dataset, memory_loader, test_loader, device, args.cl_default, task_id=k, k=min(args.train.knn_k, len(dataset.memory_loaders[k].dataset))) 

    return knn_acc


def evaluate(model, dataset, device, classifier=None, last=False) -> Tuple[list, list]:
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

    results, results_mask_classes = [], []
    for t in tqdm(range(0, dataset.N_TASKS), desc='Evaluatinng'):
      train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
      model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}.pth")
      save_dict = torch.load(model_path, map_location='cpu')
      mean_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
      model = get_model(args, device, len(train_loader), get_aug(train=False, train_classifier=False, mean_std=mean_norm), task_id=t)

      msg = model.net.module.backbone.load_state_dict({k[16:]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k}, strict=True)
      model = model.to(args.device)

      accs = evaluate(model.net.module.backbone, dataset, device)
      results.append(accs[0])
      results_mask_classes.append(accs[1])
      mean_acc = np.mean(accs, axis=1)
      print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

    ci_mean_fgt = forgetting(results)
    ti_mean_fgt = forgetting(results_mask_classes)
    print(f'CI Forgetting: {ci_mean_fgt} \t TI Forgetting: {ti_mean_fgt}')


if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
