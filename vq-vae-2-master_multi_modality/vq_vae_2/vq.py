"""
Vector-Quantization for the VQ-VAE itself.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def vq_loss(inputs, embedded, inputs_thermal, embedded_thermal, commitment=0.25):
    """
    Compute the codebook and commitment losses for an
    input-output pair from a VQ layer.
    """
    return (torch.mean(torch.pow(inputs.detach() - embedded, 2)) +
            commitment * torch.mean(torch.pow(inputs - embedded.detach(), 2)) +
            torch.mean(torch.pow(inputs_thermal.detach() - embedded_thermal, 2)) +
            commitment * torch.mean(torch.pow(inputs_thermal - embedded_thermal.detach(), 2))) 

# Nawid - Added to the loss function such such that the loss takes into codebook loss and the commitment loss for thermal input - the codebook loss looks at the difference between the continoust latent vectors and the discrete embeddings

class VQ(nn.Module):
    """
    A vector quantization layer.

    This layer takes continuous inputs and produces a few
    different types of outputs, including a discretized
    output, a commitment loss, a codebook loss, etc.

    Args:
        num_channels: the depth of the input Tensors.
        num_latents: the number of latent values in the
          dictionary to choose from.
        dead_rate: the number of forward passes after
          which a dictionary entry is considered dead if
          it has not been used.
    """

    def __init__(self, num_channels, num_latents, dead_rate=100):
        super().__init__()
        self.num_channels = num_channels # Nawid - Depth of input channel
        self.num_latents = num_latents
        self.dead_rate = dead_rate

        self.dictionary = nn.Parameter(torch.randn(num_latents, num_channels)) # Nawid - Builds an initial random dictionary
        self.usage_count = nn.Parameter(dead_rate * torch.ones(num_latents).long(),
                                        requires_grad=False) # Nawid - Looks at how much one of the entries is used
        self._last_batch = None

    def embed(self, idxs):
        """
        Convert encoded indices into embeddings.

        Args:
            idxs: an [N x H x W] or [N] Tensor.

        Returns:
            An [N x H x W x C] or [N x C] Tensor.
        """
        embedded = F.embedding(idxs, self.dictionary)
        if len(embedded.shape) == 4:
            # NHWC to NCHW
            embedded = embedded.permute(0, 3, 1, 2).contiguous()
        return embedded

    def forward(self, inputs,inputs_thermal): # Nawid - The inputs are the continous embeddings which gets quantised to discrete embeddings
        """
        Apply vector quantization.

        If the module is in training mode, this will also
        update the usage tracker and re-initialize dead
        dictionary entries.

        Args:
            inputs: the input Tensor. Either [N x C] or
              [N x C x H x W].

        Returns:
            A tuple (embedded, embedded_pt, idxs):
              embedded: the new [N x C x H x W] Tensor
                which passes gradients to the dictionary.
              embedded_pt: like embedded, but with a
                passthrough gradient estimator. Gradients
                through this pass directly to the inputs.
              idxs: a [N x H x W] Tensor of Longs
                indicating the chosen dictionary entries.
        """
        channels_last = inputs
        channels_last_thermal = inputs_thermal # Nawid - Added this to represent the thermal data
        if len(inputs.shape) == 4:
            # NCHW to NHWC
            channels_last = inputs.permute(0, 2, 3, 1).contiguous()
            channels_last_thermal  = inputs_thermal.permute(0, 2, 3, 1).contiguous()
            
            

        diffs = embedding_distances(self.dictionary, channels_last) # Nawid - Calculates distances between the input tensor and the embeddings
        diffs_thermal = embedding_distances(self.dictionary,channels_last_thermal) # Nawid - calculates the embedding distancce between the thermal continous embeddings and the actual embeddings   
        
        idxs = torch.argmin(diffs, dim=-1) # Nawid - Obtain the indices of chosen dictionary entries
        embedded = self.embed(idxs)# Nawid - Converts the indices to embeddings
        embedded_pt = embedded.detach() + (inputs - inputs.detach()) # Nawid - Gradients passs through to the inputs only
        
        idxs_thermal = torch.argmin(diffs_thermal, dim=-1)
        embedded_thermal = self.embed(idxs_thermal)
        embedded_pt_thermal = embedded_thermal.detach() + (inputs_thermal - inputs_thermal.detach()) # Nawid - Gradients passs through to the inputs only

        if self.training:
            self._update_tracker(idxs)
            self._last_batch = channels_last.detach()
            
            self._update_tracker(idxs_thermal)
            self._last_batch_thermal = channels_last_thermal.detach()

        return embedded, embedded_pt, idxs, embedded_thermal, embedded_pt_thermal,idxs_thermal

    def revive_dead_entries(self, inputs=None):
        """
        Use the dictionary usage tracker to re-initialize
        entries that aren't being used often.

        Args:
          inputs: a batch of inputs from which random
            values are sampled for new entries. If None,
            the previous input to forward() is used.
        """
        if inputs is None:
            assert self._last_batch is not None, ('cannot revive dead entries until a batch has ' +
                                                  'been run')
            inputs = self._last_batch
        counts = self.usage_count.detach().cpu().numpy()
        new_dictionary = None
        inputs_numpy = None
        for i, count in enumerate(counts):
            if count:
                continue
            if new_dictionary is None:
                new_dictionary = self.dictionary.detach().cpu().numpy()
            if inputs_numpy is None:
                inputs_numpy = inputs.detach().cpu().numpy().reshape([-1, inputs.shape[-1]])
            new_dictionary[i] = random.choice(inputs_numpy)
            counts[i] = self.dead_rate
        if new_dictionary is not None:
            dict_tensor = torch.from_numpy(new_dictionary).to(self.dictionary.device)
            counts_tensor = torch.from_numpy(counts).to(self.usage_count.device)
            self.dictionary.data.copy_(dict_tensor)
            self.usage_count.data.copy_(counts_tensor)

    def _update_tracker(self, idxs): # Nawid - Updates the count of how many times an certain dictionary index has NOT been used
        raw_idxs = set(idxs.detach().cpu().numpy().flatten())
        update = -np.ones([self.num_latents], dtype=np.int)
        for idx in raw_idxs:
            update[idx] = self.dead_rate
        self.usage_count.data.add_(torch.from_numpy(update).to(self.usage_count.device).long())
        self.usage_count.data.clamp_(0, self.dead_rate)


def embedding_distances(dictionary, tensor):
    """
    Compute distances between every embedding in a
    dictionary and every vector in a Tensor.

    This will not generate a huge intermediate Tensor,
    unlike the naive implementation.

    Args:
        dictionary: a [D x C] Tensor.
        tensor: a [... x C] Tensor.

    Returns:
        A [... x D] Tensor of distances.
    """
    dict_norms = torch.sum(torch.pow(dictionary, 2), dim=-1)
    tensor_norms = torch.sum(torch.pow(tensor, 2), dim=-1)

    # Work-around for https://github.com/pytorch/pytorch/issues/18862.
    exp_tensor = tensor[..., None].view(-1, tensor.shape[-1], 1)
    exp_dict = dictionary[None].expand(exp_tensor.shape[0], *dictionary.shape)
    dots = torch.bmm(exp_dict, exp_tensor)[..., 0]
    dots = dots.view(*tensor.shape[:-1], dots.shape[-1])

    return -2 * dots + dict_norms + tensor_norms[..., None]
