"""
Data loading pipeline.

All data should be stored as images in a single directory.
"""

import os
import random

from PIL import Image
import numpy as np
import torch

IMAGE_SIZE = 128


def load_images(dir_path, batch_size=16): # Nawid - Loads the image data
    images = load_single_images(dir_path)
    while True:
        batch = np.array([next(images) for _ in range(batch_size)]) # Nawwid - Changes the next image into a numpy array
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous() # Nawid - Changes the order
        batch = batch.float() / 255 # Nawid - Divides by 255 
        yield batch


def load_single_images(dir_path):
    while True:
        with os.scandir(dir_path) as listing: # Nawid - Opens a list of file names
            for entry in listing: # Nawid - Looks at each entry in the list of filenames
                if not (entry.name.endswith('.png') or entry.name.endswith('.jpg')):
                    continue
                try:
                    img = Image.open(entry.path) # Nawid - Displays an image from the filename
                except OSError:
                    # Ignore corrupt images.
                    continue
                width, height = img.size # Nawid - Obtain the image size
                scale = IMAGE_SIZE / min(width, height) #  Nawid - obtains the scale
                img = img.resize((round(scale * width), round(scale * height))) # Nawid - Resizes the image to make it square 
                img = img.convert('RGB') # Nawid -  Obtains the rgb image
                tensor = np.array(img) # Nawid - Change the image to an np array
                row = random.randrange(tensor.shape[0] - IMAGE_SIZE + 1) # Nawid - Obtains a random number from the between tensor.shape[0]- IMAGE_SIZE + 1  -  However this value should be quite low and therefore the range in which the random number is obtained should also be quite low. 
                col = random.randrange(tensor.shape[1] - IMAGE_SIZE + 1)
                yield tensor[row:row + IMAGE_SIZE, col:col + IMAGE_SIZE] # Nawid - Yield is used to send a sequence of images through, it is used when we want to iterate over a sequence but do not want to store the entire sequence in memory
