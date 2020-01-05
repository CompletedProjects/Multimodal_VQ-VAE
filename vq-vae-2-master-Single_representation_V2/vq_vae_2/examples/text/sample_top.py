"""
Produce samples from the top-level prior.
"""

import argparse
import os
import random

import numpy as np
import torch

from vq_vae_2.examples.text.model import TopPrior, make_vae
from vq_vae_2.examples.text.recon_vae import print_bytes
from vq_vae_2.examples.text.train_vae import VAE_PATH
from vq_vae_2.examples.text.train_top import TOP_PRIOR_PATH


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae() # Nawid - Makes the model
    if os.path.exists(VAE_PATH):
        vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu')) # Nawid - Loads the model
    vae.to(device)
    vae.eval()

    top_prior = TopPrior(args.context_len // 8) # Nawid - Instantiate sthe top prior
    if os.path.exists(TOP_PRIOR_PATH):
        top_prior.load_state_dict(torch.load(TOP_PRIOR_PATH, map_location='cpu')) # Nawid - Loads the trained top prior
    top_prior.to(device)

    latents = np.zeros([1, args.context_len // 8], dtype=np.int64)
    for i in range(args.context_len // 8):
        outs = top_prior(torch.from_numpy(latents[:, :i+1]).to(device))
        probs = torch.softmax(outs, dim=1)
        latents[:, i] = [sample_softmax(x) for x in probs.detach().cpu().numpy()[:, :, i]] # Nawid - Samples different embedding indices

    embedded = vae.encoders[-1].vq.embed(torch.from_numpy(latents).to(device)) # Nawid - Changes the indices into embeddings
    decoded = [embedded]
    for encoder, decoder in zip(vae.encoders[:-1][::-1], vae.decoders):
        raw = decoder(decoded) # Nawid - Forward pass of the embeddings through different decoders to get an upsampled version of a continous embedding
        decoded.append(encoder.vq(raw)[0]) # Nawid- Quantises the raw continous embedding to the nearest neighbour
    decoded.append(vae.decoders[-1](decoded)) # Nawid - Passes all the different levels of embeedings through the last decoder.
    print_bytes('Reconstruction', torch.argmax(decoded[-1][0], dim=-1))


def sample_softmax(probs):
    number = random.random()
    for i, x in enumerate(probs):
        number -= x
        if number <= 0:
            return i
    return len(probs) - 1


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='torch device', default='cuda')
    parser.add_argument('--context-len', help='context size in bytes', default=512, type=int)
    return parser


if __name__ == '__main__':
    main()
