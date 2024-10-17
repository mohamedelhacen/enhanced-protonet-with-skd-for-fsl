from PIL import Image
import cv2
import torch
import numpy as np
from torchvision.transforms import Normalize, RandomResizedCrop, RandomHorizontalFlip, RandomApply, ColorJitter, RandomVerticalFlip, RandomGrayscale
from torchnet.transform import compose


def load_image_path(key, out_field, d):
    image = cv2.imread(d[key])
    d[out_field] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Image.open(d[key]).convert('RGB') #.convert('L')
    # print(d[out_field].shape)
    return d

def convert_tensor(key, d):
    d[key] = torch.from_numpy(np.array(d[key], np.float32, copy=False)).contiguous().view(3, d[key].shape[0], d[key].shape[1])
    # print(d[key].shape)  #.transpose(0, 1)
    return d

def rotate_image(key, rot, d):
    d[key] = d[key].rotate(rot)
    return d

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d

def apply_data_augmentation(key, d):
    tfs = compose([
                    # RandomResizedCrop(32, scale=(0.5, 1.0)),
                    RandomHorizontalFlip(p=0.5), 
                    RandomVerticalFlip(p=0.5), 
                    # RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
                    # RandomApply([RandomGrayscale(p=0.2)], p=0.2)
                    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) or (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ])    
    d[key] = tfs(d[key])
    return d