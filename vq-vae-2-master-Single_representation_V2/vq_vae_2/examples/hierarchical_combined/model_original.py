"""
Models for hierarchical image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_vae_2.attention import PixelAttention
from vq_vae_2.pixel_cnn import ChannelNorm, PixelCNN, PixelConvA, PixelConvB
from vq_vae_2.vq_vae import HalfDecoder, HalfQuarterDecoder, HalfEncoder, QuarterEncoder, VQVAE


def make_vae(): # Nawid - Makes the encoder and the decoder
    encoders = [QuarterEncoder(3, 128, 512), HalfEncoder(128, 128, 512)] # Nawid -For the first encoder, the number of input channels is 3 and it produces 128 output. The 512 is the number of latents in the dictionary
    decoders = [HalfDecoder(128, 128), HalfQuarterDecoder(128, 3)]
    return VQVAE(encoders, decoders)


class TopPrior(nn.Module): # Nawid -  As described in the paper, the top prior is equipped with a attention layers so it can benefit from a larger receptive field to capture correlations in spatial locations that are far apart in the image
    def __init__(self, depth=128, num_heads=2): # Nawid - 2 attention heads
        super().__init__()
        self.embed = nn.Embedding(512, depth) # Nawid - Makes the embedding dictionary
        self.pixel_cnn = PixelCNN(
            PixelConvA(depth, depth),

            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelAttention(depth, num_heads=num_heads),

            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelAttention(depth, num_heads=num_heads),

            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelAttention(depth, num_heads=num_heads),

            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelAttention(depth, num_heads=num_heads),
        ) # Nawid - Builds the PixelSNAIL structure which is shown by there being pixel CNN layers followed by attention layers
        self.out_stack = nn.Sequential(
            nn.Conv2d(depth * 2, depth, 1),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            nn.Conv2d(depth, 512, 1),
        ) # Nawid - Obtains the output which as 512 output channels which I believe is related to the number of entries in the codebook/latent space

    def forward(self, x):
        x = self.embed(x) # Nawid - Turns x into an embedding
        x = x.permute(0, 3, 1, 2).contiguous()
        out1, out2 = self.pixel_cnn(x) # Nawid - PixelSnail seems to give two outputs which I believe is due to the last attenion mechanism working on the horizontal stack and the vertical stack so there is 2 output- I do not believe it is due to the number of attention heads being 2
        return self.out_stack(torch.cat([out1, out2], dim=1))


class BottomPrior(nn.Module): # Nawid - Bottom prior is used to encode local information and so operates at a larger resolution. Using an attention network fo this layer would also not be practical due to memory constraints.
    def __init__(self, depth=128, num_heads=2):
        super().__init__()
        self.embed_top = nn.Embedding(512, depth) # Nawid- Makes the embedding dictionary
        self.embed_bottom = nn.Embedding(512, depth) # Nawid- Makes another embedding dictionary
        self.cond_stack = nn.Sequential(
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            nn.ConvTranspose2d(depth, depth, 4, stride=2, padding=1),
        )
        self.pixel_cnn = PixelCNN(
            PixelConvA(depth, depth, cond_depth=depth),

            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),

            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),

            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),

            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
        ) # Nawid - Bottom prior does not seem to have any attention mechanism present
        self.out_stack = nn.Sequential(
            nn.Conv2d(depth * 2, depth, 1),
            nn.Conv2d(depth, 512, 1),
        )

    def forward(self, bottom, top):
        conds = self.embed_top(top) # Nawid - Changes the indices of the top layer into specific embeddings
        conds = conds.permute(0, 3, 1, 2).contiguous()
        conds = self.cond_stack(conds) # Nawid-  Input the embeddings of the top layer into the conditional stack

        out = self.embed_bottom(bottom) # Nawid  - Changes the indices of the bottom layer into an embedding 
        out = out.permute(0, 3, 1, 2).contiguous()
        out1, out2 = self.pixel_cnn(out, conds=conds) # Nawid - Obtains the horizontal and vertical stacks from the pixel CNN
        return self.out_stack(torch.cat([out1, out2], dim=1))


class Residual1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 1) # Nawid - 1x1 convolution
        self.conv2 = nn.Conv2d(num_channels, num_channels, 1) # Nawid - 1x1 convolution
        self.norm = ChannelNorm(num_channels) # Nawid -  Normalisation

    def forward(self, x):
        inputs = x
        x = F.relu(x) 
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return inputs + self.norm(x) # Nawid - Residual connection


class Residual3x3(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1) # Nawid - 3x3 convolution but the shape is the same due to the padding
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1) # Nawid - 3x3 convolution but the shape is the same due to the padding
        self.norm = ChannelNorm(num_channels)

    def forward(self, x): # Nawid - Convolutions followed by normalised residual connection
        inputs = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return inputs + self.norm(x) 
