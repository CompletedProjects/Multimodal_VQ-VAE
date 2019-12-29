"""
Train a PixelCNN on MNIST using a pre-trained VQ-VAE.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms

from vq_vae_2.examples.mnist.model import Generator, make_vq_vae

BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device('cpu')


def main():
    vae = make_vq_vae()
    vae.load_state_dict(torch.load('vae.pt', map_location='cpu'))
    vae.to(DEVICE)
    vae.eval()

    generator = Generator()
    if os.path.exists('gen.pt'):
        generator.load_state_dict(torch.load('gen.pt', map_location='cpu')) # Nawid - Upload the model for the generator if a model is present
    generator.to(DEVICE)

    optimizer = optim.Adam(generator.parameters(), lr=LR) # Nawid- Optimizer
    loss_fn = nn.CrossEntropyLoss() # Nawid - Loss function

    test_images = load_images(train=False) # Nawid-  Loads the test images
    for batch_idx, images in enumerate(load_images()): # Nawid - Iterates through the training images
        images = images.to(DEVICE)
        losses = []
        for img_set in [images, next(test_images).to(DEVICE)]: # Nawid - next retrieves the next item from the iterator
            _, _, encoded = vae.encoders[0](img_set) # Nawid - Encodes the train image and the test image i believe
            logits = generator(encoded) # Nawid - Generates data from the encoded data
            logits = logits.permute(0, 2, 3, 1).contiguous()
            logits = logits.view(-1, logits.shape[-1])
            losses.append(loss_fn(logits, encoded.view(-1))) # Nawid - Output is the logits and the target is the encoder values. This is used to calculate the loss
        optimizer.zero_grad()
        losses[0].backward()
        optimizer.step()
        print('train=%f test=%f' % (losses[0].item(), losses[1].item()))
        if not batch_idx % 100:
            torch.save(generator.state_dict(), 'gen.pt')


def load_images(train=True): # Nawid - Loads images
    while True:
        for data, _ in create_data_loader(train):
            yield data


def create_data_loader(train): # Nawid - Creates a dataloader object
    mnist = torchvision.datasets.MNIST('./data', train=train, download=True,
                                       transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    main()
