## Originated from https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py

import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):
    def __init__(self, num_classes, args):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        if args.dataset.image_size == 32:
            self.fc1 = nn.Linear(16*5*5, 120)
        else:
            self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x, return_features=False):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        if return_features:
            return x
        x = self.fc(x)
        return x

def lenet(num_classes, args):
    return LeNet(num_classes, args=args)
