"""
Train the bottom-level prior.
"""

import itertools
import os

import torch
import torch.nn as nn
import torch.optim as optim

from vq_vae_2.examples.hierarchical.data import load_images
from vq_vae_2.examples.hierarchical.model import BottomPrior, make_vae
from vq_vae_2.examples.hierarchical.train_vae import VAE_PATH, arg_parser

BOTTOM_PRIOR_PATH = 'bottom.pt'


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae() # Nawid - Make the encoder and decoder
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu')) # Nawid - Instantiate the network
    vae.to(device)
    vae.eval() # Nawid - Turn to evaluation mode

    bottom_prior = BottomPrior() # Nawid- Instantiate a bottom prior
    if os.path.exists(BOTTOM_PRIOR_PATH):
        bottom_prior.load_state_dict(torch.load(BOTTOM_PRIOR_PATH, map_location='cpu')) # Nawid - Load information of bottom prior
    bottom_prior.to(device)

    optimizer = optim.Adam(bottom_prior.parameters(), lr=1e-4)# Nawid - Optimizer for the bottom prior
    loss_fn = nn.CrossEntropyLoss() # Nawid - Cross entropy loss function

    data = load_images(args.data, batch_size=2) # Nawid - Load the data
    for i in itertools.count():
        images = next(data).to(device) # Nawid - Iterate through the data
        bottom_enc = vae.encoders[0].encode(images) # Nawid - Uses the first encoder layer to encoder the bottom latent map - This lowers the dimension
        _, _, bottom_idxs = vae.encoders[0].vq(bottom_enc) # Nawid- Quantises the first (bottom) latent map into indices
        _, _, top_idxs = vae.encoders[1](bottom_enc) # Nawid - Sends the bottom latent mapthrough the top encoder to get the top indices I believe
        logits = bottom_prior(bottom_idxs, top_idxs) # Nawid - Bottom prior uses both the bottom and top indices
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = logits.view(-1, logits.shape[-1]) # Nawid - Gets into the shape N,C in order for it to go through the softmax and cross entropy loss
        loss = loss_fn(logits, bottom_idxs.view(-1)) # Nawid - nn.Crossentropyloss calculates the softmax of the logits and then performs a cross entropy of the output of the logits softmax and the target bottom_indxs. So it finds the cross entropy between the bottom indices predicted by the prior and the target bottom indices from the prior.
        print('step %d: loss=%f' % (i, loss.item()))
        optimizer.zero_grad() 
        loss.backward() # Nawid-  Backpropagate the loss
        optimizer.step()
        if not i % 30:
            torch.save(bottom_prior.state_dict(), BOTTOM_PRIOR_PATH)


if __name__ == '__main__':
    main()
