import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, ChainDataset, Dataset

# DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../Data/omniglot/data')
# DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../Data/miniImagenet/Data')
DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../Data/cifar100/data')
# DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../Data/fc100')


class CustomDataset(Dataset):
    def __init__(self, root_dir, class_paths, transform=None):
        self.root_dir = root_dir
        self.class_paths = class_paths
        self.transform = transform

        self.data = []  # A list to store tuples (image_path, class_label)

        for label, path in enumerate(class_paths):
            class_dir = os.path.join(root_dir, path)
            for filename in os.listdir(class_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(class_dir, filename)
                    self.data.append((image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path) #.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def load_task(classes):
    
    transforms = Compose([
        # Resize((32, 28)),
        ToTensor(),
    ])

    dataset = CustomDataset(
        root_dir=DATA_DIR,
        class_paths=classes,
        transform=transforms)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    return loader