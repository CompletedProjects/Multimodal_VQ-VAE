"""
A basic PixelCNN + VQ-VAE model.
"""

import torch.nn as nn

from vq_vae_2.pixel_cnn import PixelCNN, PixelConvA, PixelConvB
from vq_vae_2.vq_vae import QuarterDecoder, QuarterEncoder, VQVAE

#LATENT_SIZE = 16 # Nawid - 
#LATENT_COUNT = 32 # Nawid - 

LATENT_SIZE = 32 # Nawid - 
LATENT_COUNT = 64 # Nawid -

def make_vq_vae():
    return VQVAE([QuarterEncoder(3,1, LATENT_SIZE, LATENT_COUNT)],
                 [QuarterDecoder(LATENT_SIZE, 3,1)])


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(LATENT_COUNT, 64) # Nawid -  Used to make look up table that stores embeddings
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
        self.to_logits = nn.Conv2d(64, LATENT_COUNT, 1) # Nawid- Input is 64 channels which is the output from the PixelCNN, the output is Latent_Count and it uses a 1x1 convolution - The output is 32 channels which is related to the embeddings. Therefore it could have the same dimensionality as the embedding or it could have the same number as the output

    def forward(self, x):
        x = self.embed(x) # Nawid - Turns x into embeddings I believe
        x = x.permute(0, 3, 1, 2).contiguous()
        out1, out2 = self.model(x) # Nawid - generates a pixelCNN model and passes through the embeddings an input, there are 2 outputs as they are the outputs of the vertical and the horizontal stack from the last pixelConvB layer
        return self.to_logits(out1 + out2) # Nawid - The output has depth equal to the number of elements in the embedding matrix
