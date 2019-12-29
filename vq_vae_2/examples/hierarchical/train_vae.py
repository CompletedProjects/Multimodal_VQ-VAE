"""
Train a hierarchical VQ-VAE on 256x256 images.
"""

import argparse
import itertools
import os

from PIL import Image
import numpy as np
import torch
import torch.optim as optim

from vq_vae_2.examples.hierarchical.data import load_images
from vq_vae_2.examples.hierarchical.model import make_vae

VAE_PATH = 'vae.pt'


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)
    model = make_vae() # Nawid - Makes the encoder and decoders
    if os.path.exists(VAE_PATH):
        model.load_state_dict(torch.load(VAE_PATH, map_location='cpu')) # Nawid - Loads model if possible
    model.to(device)
    optimizer = optim.Adam(model.parameters()) # Nawid - Optimizer
    data = load_images(args.data) # Nawid- Loads data
    for i in itertools.count(): # Nawid - An infinite iterator
        images = next(data).to(device)
        terms = model(images) # Nawid - Input image into the model
        print('step %d: mse=%f mse_top=%f' %
              (i, terms['losses'][-1].item(), terms['losses'][0].item()))
        optimizer.zero_grad() #Nawid - I think this makes the gradient become zero 
        terms['loss'].backward() # Nawid - Backpropagates the loss I believe 
        optimizer.step() # Nawid- Updates using optimizer
        model.revive_dead_entries()
        if not i % 30:
            torch.save(model.state_dict(), VAE_PATH)
            save_reconstructions(model, images) # Nawid - saves the reconstruction


def save_reconstructions(vae, images):
    vae.eval() # Nawid - Sets the module in evaluation mode. This has any effect only on certain module
    with torch.no_grad(): # Nawid - Not calculating gradients
        recons = [torch.clamp(x, 0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
                  for x in vae.full_reconstructions(images)] # Nawid - performs the full_reconstructions of the image from using the top latent map, as well as the top level latent map combined with the lower level latent map. These values are then  
    vae.train() # Nawid - This sets the nn.module into training mode
    top_recons, real_recons = recons # Nawid - I believe the top_recons is the reconstruction using only the top latent map, whilst real_recons is using both the top latent map and the bottom latent map 
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy()

    columns = np.concatenate([top_recons, real_recons, images], axis=-2)
    columns = np.concatenate(columns, axis=0)
    Image.fromarray((columns * 255).astype('uint8')).save('reconstructions.png') # Nawid - Changes the valeus back to 255 and integers and save the reconstructions


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data directory') # Nawid - Chooses the data
    parser.add_argument('--device', help='torch device', default='cuda') # Nawid - Chooses the device
    return parser


if __name__ == '__main__':
    main()
