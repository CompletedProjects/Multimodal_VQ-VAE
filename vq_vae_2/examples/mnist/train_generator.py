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
    vae.load_state_dict(torch.load('vae.pt', map_location='cpu')) # Nawid - Loads the trained VAE model
    vae.to(DEVICE)
    vae.eval()

    generator = Generator() # Nawid - Instantiates the generator
    if os.path.exists('gen.pt'):
        generator.load_state_dict(torch.load('gen.pt', map_location='cpu')) # Nawid - Upload the model for the generator if a model is present
    generator.to(DEVICE)

    optimizer = optim.Adam(generator.parameters(), lr=LR) # Nawid- Optimizer for the generator
    loss_fn = nn.CrossEntropyLoss() # Nawid - Loss function -which performs a softmax

    test_images = load_images(train=False) # Nawid-  Loads the test images in the dataloader
    for batch_idx, images in enumerate(load_images()): # Nawid - Iterates through the training images
        images = images.to(DEVICE) # Nawid - Sends to cuda
        losses = []
        for img_set in [images, next(test_images).to(DEVICE)]: # Nawid - next retrieves the next item from the iterator
            _, _, encoded = vae.encoders[0](img_set) # Nawid - The output of encode is embedding, embedding_pt as well as indices. In this case, the indices of the embedding are used
            logits = generator(encoded) # Nawid - Generates data from the indices ( so they are first turned to embeddings) and then passes through a pixelCNN
            logits = logits.permute(0, 2, 3, 1).contiguous() # Nawid-  Change the dimensions so that the last dimension is equal to the number of classes( number of different indices which is required to compute the softmax)
            logits = logits.view(-1, logits.shape[-1]) # Nawid - Change input into the form (minibatch,C) 
            losses.append(loss_fn(logits, encoded.view(-1))) # Nawid - Output is the logits with shape (minibatch, C) and the target is the indices from the encoder. This is used to calculate the loss by undergoing a softmax to find the probabilities and then finding the negative log likelihood
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
