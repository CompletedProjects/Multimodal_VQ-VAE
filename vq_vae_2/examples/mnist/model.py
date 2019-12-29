"""
A basic PixelCNN + VQ-VAE model.
"""

import torch.nn as nn

from vq_vae_2.pixel_cnn import PixelCNN, PixelConvA, PixelConvB
from vq_vae_2.vq_vae import QuarterDecoder, QuarterEncoder, VQVAE

LATENT_SIZE = 16 # Nawid - This is the dimensionality of the encoder in this case
LATENT_COUNT = 32 # Nawid - This is the number of the embeddings in this case


def make_vq_vae():
    return VQVAE([QuarterEncoder(1, LATENT_SIZE, LATENT_COUNT)], 
                 [QuarterDecoder(LATENT_SIZE, 1)])
# Nawid - The number of channels as input is latentsize and the number of channels as output is 1

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(LATENT_COUNT, 64) # Nawid -  Used to make look up table that stores embeddings - The latent count is the number of the embeddings and 64 is the embedding dimension in this case
        self.model = PixelCNN(
            PixelConvA(64, 64),
            PixelConvB(64, 64, norm=True),
            PixelConvB(64, 64, norm=True),
            PixelConvB(64, 64, norm=True),
            PixelConvB(64, 64, norm=True),
            PixelConvB(64, 64, norm=True),
            PixelConvB(64, 64, norm=True),
            PixelConvB(64, 64, norm=True),
        )
        self.to_logits = nn.Conv2d(64, LATENT_COUNT, 1) # Nawid- Input is 64 channels which is the output from the PixelCNN, the output is Latent_Count and it uses a 1x1 convolution - The output is 32 channels which is related to the embeddings. This is so the probability of each different embedding can be found for a particular input 

    def forward(self, x):
        x = self.embed(x) # Nawid - Turns x(where x in this case is the indices of the embeddings) into embeddings I believe - The input is 32 dimensional in the number of channel ( or in this case, it is concatenation of two 16 dimensional channels) and the output is 64 dimensional channel 
        x = x.permute(0, 3, 1, 2).contiguous()
        out1, out2 = self.model(x) # Nawid - generates a pixelCNN model and passes through the embeddings an input, there are 2 outputs as they are the outputs of the vertical and the horizontal stack from the last pixelConvB layer
        return self.to_logits(out1 + out2) # Nawid - The output has depth equal to the number of elements in the embedding matrix in order to calculate the cross entropy loss between the different channels
