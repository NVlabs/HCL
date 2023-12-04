import os
import importlib
from .simsiam import SimSiam
import torch
from .backbones import resnet18, lenet, vgg16, alexnet, densenet, senet, regnet, inception, swin, resnext
from datasets import N_CLASSES, BACKBONES
from utils.losses import LabelSmoothing, KL_div_Loss


def get_backbone(args, task_id=0):
    if args.hcl:
        backbone = BACKBONES[args.dataset.name][task_id]
    else:
        backbone = args.model.backbone
    
    net = eval(f"{backbone}(num_classes=N_CLASSES[args.dataset.name], args=args)")
    print("Backbone changed to ", backbone)

    net.n_classes = N_CLASSES[args.dataset.name]
    net.output_dim = net.fc.in_features
    if not args.cl_default:
        net.fc = torch.nn.Identity()

    return net


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

def get_model(args, device, dataset, transform, global_model=None, task_id=0):
    allowed_models  = ["distil", "qdi", "distilbuf"]
    if args.model.cl_model in allowed_models:
        loss = LabelSmoothing(smoothing=0.1)
    else:
        loss = torch.nn.CrossEntropyLoss()
    if args.model.name == 'simsiam':
        backbone =  SimSiam(get_backbone(args, task_id=task_id)).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)

    names = {}
    for model in get_all_models():
        mod = importlib.import_module('models.' + model)
        class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
        names[model] = getattr(mod, class_name)
    
    return names[args.model.cl_model](backbone, loss, args, dataset, transform, global_model)

