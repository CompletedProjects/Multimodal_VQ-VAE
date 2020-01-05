"""
Data loading pipeline.

All data should be stored as images in a single directory.
"""

import os
import random

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms


#IMAGE_SIZE = 256

BATCH_SIZE = 16
'''
def load_images(dir_path, batch_size=16): # Nawid - Loads the image data
    images = load_single_images(dir_path)
    while True:
        batch = np.array([next(images) for _ in range(batch_size)]) # Nawwid - Changes the next image into a numpy array
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous() # Nawid - Changes the order
        batch = batch.float() / 255 # Nawid - Divides by 255 
        yield batch
'''

def load_images(train=True): # Nawid - Loads images
    while True:
        for data, _ in create_data_loader(train):
            yield data

       
def create_data_loader(train): # Nawid - Creates a dataloader object
    STL10 = torchvision.datasets.STL10('./data', split='train', download=True,
                                       transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(STL10, batch_size=BATCH_SIZE, shuffle=True)

