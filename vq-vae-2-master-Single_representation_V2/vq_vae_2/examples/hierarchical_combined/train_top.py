"""
Train the top-level prior.
"""

import itertools
import os

import torch
import torch.nn as nn
import torch.optim as optim

from vq_vae_2.examples.hierarchical.data import load_images
from vq_vae_2.examples.hierarchical.model import TopPrior, make_vae
from vq_vae_2.examples.hierarchical.train_vae import VAE_PATH, arg_parser

TOP_PRIOR_PATH = 'top.pt'


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae() # Nawid-  Makes the encoder and decoder
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu')) # Nawid - Uploads the trained vae model
    vae.to(device)
    vae.eval() # Nawid- Set to evaluation mode

    top_prior = TopPrior() # Nawid - Instantiate the top prior
    if os.path.exists(TOP_PRIOR_PATH):
        top_prior.load_state_dict(torch.load(TOP_PRIOR_PATH, map_location='cpu')) # Nawid - Uploads the top prior model if present
    top_prior.to(device)

    optimizer = optim.Adam(top_prior.parameters(), lr=1e-4) # Nawid - Optimizer
    loss_fn = nn.CrossEntropyLoss() # Nawid - Cross entropy loss

    data = load_images(args.data, batch_size=2)
    for i in itertools.count():
        images = next(data).to(device) # Nawid - Iterates through the data
        _, _, encoded = vae.encoders[1](vae.encoders[0].encode(images)) # Nawid-  The top encoder layer ( the second layer) obtains the indices for a particular image 
        logits = top_prior(encoded) # Nawid - Obtains the logits for the encoding
        logits = logits.permute(0, 2, 3, 1).contiguous() 
        logits = logits.view(-1, logits.shape[-1]) # Nawid - Gets to the shape N,C
        loss = loss_fn(logits, encoded.view(-1)) # Nawid - Logits go through a softmax to give the probabilities of the different classses  and then compares the output with indices from the encoder (target)
        print('step %d: loss=%f' % (i, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i % 30:
            torch.save(top_prior.state_dict(), TOP_PRIOR_PATH)


if __name__ == '__main__':
    main()
