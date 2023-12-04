from torchvision import transforms
from PIL import Image


class Transform_single():
    def __init__(self, image_size, train, mean_std):
        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                # transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*mean_std)
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*mean_std)
            ])

    def __call__(self, x):
        return self.transform(x)
