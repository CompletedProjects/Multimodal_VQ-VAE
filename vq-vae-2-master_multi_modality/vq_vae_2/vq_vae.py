"""
An implementation of the hierarchical VQ-VAE.
See https://arxiv.org/abs/1906.00446.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq import VQ, vq_loss


class Encoder(nn.Module):
    """
    An abstract VQ-VAE encoder, which takes input Tensors,
    shrinks them, and quantizes the result.

    Sub-classes should overload the encode() method.

    Args:
        num_channels: the number of channels in the latent
          codebook.
        num_latents: the number of entries in the latent
          codebook.
        kwargs: arguments to pass to the VQ layer.
    """

    def __init__(self, num_channels, num_latents, **kwargs):
        super().__init__()
        self.vq = VQ(num_channels, num_latents, **kwargs) # Nawid - I believe this makes the codebook which as a dimensionality of num_channels

    def encode(self, x, x_thermal): # Nawid - Changed so encode uses both different types of inputs
        """
        Encode a Tensor before the VQ layer.

        Args:
            x: the input Tensor.

        Returns:
            A Tensor with the correct number of output
              channels (according to self.vq).
        """
        raise NotImplementedError

    def forward(self, x, x_thermal): # Nawid - Forwards has both types of input tensors
        """
        Apply the encoder.

        See VQ.forward() for return values.
        """
        encoded_x_rgb, encoded_x_thermal = self.encode(x,x_thermal)
        return self.vq(encoded_x_rgb, encoded_x_thermal) # Nawid - Obtains the embedding, embeddingpt and the indices - The encode is from the subclasses

# Nawid - Added in the input for the thermal vector and providing the output as a tuple

class QuarterEncoder(Encoder):
    """
    The encoder from the original VQ-VAE paper that cuts
    the dimensions down by a factor of 4 in both
    directions.
    """

    def __init__(self, in_channels,in_channels_thermal, out_channels, num_latents, **kwargs): # Nawid- Added a channel for the thermal
        super().__init__(out_channels, num_latents, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 4, stride=2) # Nawid - Cuts down the size by half
        self.conv2 = nn.Conv2d(out_channels, out_channels, 4, stride=2) # Nawid - Cuts down the size by half again
        self.residual1 = _make_residual(out_channels)
        self.residual2 = _make_residual(out_channels)
        
        self.conv1_thermal = nn.Conv2d(in_channels_thermal, out_channels, 4, stride=2) # Nawid - Cuts down the size by half and the number of input channels is related ot the thermal input 

    def encode(self, x, x_thermal):
        # Padding is uneven, so we make the right and
        # bottom more padded arbitrarily.
        x = F.pad(x, (1, 2, 1, 2))
        x = self.conv1(x)
        x = F.relu(x)
        x = F.pad(x, (1, 2, 1, 2))
        x = self.conv2(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        
        x_thermal = F.pad(x_thermal, (1, 2, 1, 2))
        x_thermal = self.conv1_thermal(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = F.pad(x_thermal, (1, 2, 1, 2))
        x_thermal = self.conv2(x_thermal)
        x_thermal = x_thermal + self.residual1(x_thermal)
        x_thermal = x_thermal + self.residual2(x_thermal)
        return x, x_thermal

class HalfEncoder(Encoder):
    """
    An encoder that cuts the input size in half in both
    dimensions.
    """

    def __init__(self, in_channels, out_channels, num_latents, **kwargs): # Nawid - Do not need to specify the number of channels for the thermal input as the input of this comes from the quarter encoder and so the
        super().__init__(out_channels, num_latents, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1) # Nawid - Use to cut down the size
        self.residual1 = _make_residual(out_channels)
        self.residual2 = _make_residual(out_channels)
        
        #self.conv_thermal = self.conv = nn.Conv2d(in_channels_thermal, out_channels, 3, stride=2, padding=1) # Nawid - Use to cut down the size

    def encode(self, x, x_thermal):
        x = self.conv(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        
        x_thermal = self.conv(x_thermal)
        x_thermal = x_thermal + self.residual1(x_thermal)
        x_thermal = x_thermal + self.residual2(x_thermal)
        return x, x_thermal


class Decoder(nn.Module):
    """
    An abstract VQ-VAE decoder, which takes a stack of
    (differently-sized) input Tensors and produces a
    predicted output Tensor.

    Sub-classes should overload the forward() method.
    """

    def forward(self, inputs, inputs_thermal):
        """
        Apply the decoder to a list of inputs.

        Args:
            inputs: a sequence of input Tensors. There may
              be more than one in the case of a hierarchy,
              in which case the top levels come first.

        Returns:
            A decoded Tensor.
        """
        raise NotImplementedError


class QuarterDecoder(Decoder):
    """
    The decoder from the original VQ-VAE paper that
    upsamples the dimensions by a factor of 4 in both
    directions.
    """

    def __init__(self, in_channels, out_channels, out_channels_thermal):
        super().__init__()
        self.residual1 = _make_residual(in_channels) # Nawid - Performs a 3x3 followed by a 1x1 convolution
        self.residual2 = _make_residual(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1) # Nawid - Increases the size
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1) # Nawid - Increases the size further
        
        # Nawid - Specific layers for the thermal channel
        self.residual1_thermal = _make_residual(in_channels)
        self.residual2_thermal = _make_residual(in_channels)
        self.conv1_thermal = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1) 
         
        self.conv2_thermal = nn.ConvTranspose2d(in_channels, out_channels_thermal, 4, stride=2, padding=1) # Nawid - Increases the size further - Used to recreate the thermal channel since the number of layers is different  

    def forward(self, inputs, inputs_thermal):
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        
        x_thermal = inputs_thermal[0]
        x_thermal = x_thermal + self.residual1_thermal(x_thermal)
        x_thermal = x_thermal + self.residual2_thermal(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv1_thermal(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv2_thermal(x_thermal)
        
        return x,x_thermal

class HalfDecoder(Decoder):
    """
    A decoder that upsamples by a factor of 2 in both
    dimensions.
    """

    def __init__(self, in_channels, out_channels): # Nawid - Do not need to specify the channels fo the thermal here as this the input of the encoder is input here and the input of the encoder for both the thermal and the rgb have the same number of channels
        super().__init__()
        self.residual1 = _make_residual(in_channels) 
        self.residual2 = _make_residual(in_channels)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1) # Nawid - Increases the size
        #self.conv_thermal = nn.ConvTranspose2d(in_channels, out_channels_thermal, 4, stride=2, padding=1) # Nawid - Increases the size
        
        self.residual1_thermal = _make_residual(in_channels) 
        self.residual2_thermal = _make_residual(in_channels)
        self.conv_thermal = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
    def forward(self, inputs,inputs_thermal):
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv(x)
        
        x_thermal = inputs_thermal[0]
        x_thermal = x_thermal + self.residual1_thermal(x_thermal)
        x_thermal = x_thermal + self.residual2_thermal(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv_thermal(x_thermal)

        return x, x_thermal

class HalfQuarterDecoder(Decoder):
    """
    A decoder that takes two inputs. The first one is
    upsampled by a factor of two, and then combined with
    the second input which is further upsampled by a
    factor of four.
    """

		# Nawid - Combines both levels of the hierarchy together
    def __init__(self, in_channels, out_channels, out_channels_thermal): # Nawid - The number of in channels for the network should be the same for botht the rgb and the thermal since the number of the encoded channels should be equal to the dimensionality of the codebook. However the output dimensions should be the same as the output dimensionality  
        super().__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.residual3 = _make_residual(in_channels)
        self.residual4 = _make_residual(in_channels)
        self.conv3 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        
        
        # Nawid-  Specific decoding for the thermal channel
        self.residual1_thermal = _make_residual(in_channels)
        self.residual2_thermal = _make_residual(in_channels)
        self.conv1_thermal = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2_thermal = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.residual3_thermal = _make_residual(in_channels)
        self.residual4_thermal = _make_residual(in_channels)
        self.conv3_thermal = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv4_thermal = nn.ConvTranspose2d(in_channels, out_channels_thermal, 4, stride=2, padding=1)
    def forward(self, inputs, inputs_thermal):
        assert len(inputs) == 2

        # Upsample the top input to match the shape of the
        # bottom input.
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x) # Nawid - This is a convolution transpose which make the top input match the shape of the bottom input
        x = F.relu(x)

        # Mix together the bottom and top inputs.
        x = torch.cat([x, inputs[1]], dim=1) # Nawid - Concatentate the upsample top feature map with the bottom feature map
        x = self.conv2(x) # Nawid - Downsamples

        x = x + self.residual3(x)
        x = x + self.residual4(x)
        x = F.relu(x)
        x = self.conv3(x) # Nawid - Upsamples
        x = F.relu(x)
        x = self.conv4(x) # Nawid - Upsamples
        
        assert len(inputs_thermal) == 2
        x_thermal = inputs_thermal[0]
        x_thermal = x_thermal + self.residual1_thermal(x_thermal)
        x_thermal = x_thermal + self.residual2_thermal(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv1_thermal(x_thermal) # Nawid - This is a convolution transpose which make the top input match the shape of the bottom input
        x_thermal = F.relu(x_thermal)

        # Mix together the bottom and top inputs.
        x_thermal = torch.cat([x_thermal, inputs_thermal[1]], dim=1) # Nawid - Concatentate the upsample top feature map with the bottom feature map
        x_thermal = self.conv2_thermal(x_thermal) # Nawid - Downsamples

        x_thermal = x_thermal + self.residual3_thermal(x_thermal)
        x_thermal = x_thermal + self.residual4_thermal(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv3_thermal(x_thermal) # Nawid - Upsamples
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv4_thermal(x_thermal) # Nawid - Upsamples where the number of channels is equal to the channels required for the thermal output
                
        return x,x_thermal


class VQVAE(nn.Module):
    """
    A complete VQ-VAE hierarchy.

    There are N encoders, stored from the bottom level to
    the top level, and N decoders stored from top to
    bottom.
    """

    def __init__(self, encoders, decoders):
        super().__init__()
        assert len(encoders) == len(decoders) # Nawid - Make sure the number of encoders and decoders are the same 
        self.encoders = encoders # Nawid - Encoders
        self.decoders = decoders
        for i, enc in enumerate(encoders):
            self.add_module('encoder_%d' % i, enc)
        for i, dec in enumerate(decoders):
            self.add_module('decoder_%d' % i, dec)

    def forward(self, inputs,inputs_thermal, commitment=0.25): # Nawid - Inputted the thermal input separately
        """
        Compute training losses for a batch of inputs.

        Args:
            inputs: the input Tensor. If this is a Tensor
              of integers, then cross-entropy loss will be
              used for the final decoder. Otherwise, MSE
              will be used.
            commitment: the commitment loss coefficient.

        Returns:
            A dict of Tensors, containing at least:
              loss: the total training loss.
              losses: the MSE/log-loss from each decoder.
              reconstructions: a reconstruction Tensor
                from each decoder.
              embedded: outputs from every encoder, passed
                through the vector-quantization table.
                Ordered from bottom to top level.
        """
        all_encoded = [inputs]
        all_encoded_thermal = [inputs_thermal] # Nawid - Values for the encoded_thermal
        all_vq_outs = []
        all_vq_outs_thermal = []
        
        total_vq_loss = 0.0 # Nawid - Initialises total vq loss
        total_recon_loss = 0.0
        
        for encoder in self.encoders: # Nawid -Iterates through the different encoders
            encoded, encoded_thermal = encoder.encode(all_encoded[-1], all_encoded_thermal[-1]) # Nawid- Looks at the last input to make an encoding, Added the encoded for the thermal
            embedded, embedded_pt, _, embedded_thermal,embedded_pt_thermal,_ = encoder.vq(encoded,encoded_thermal) #Nawid - Quantises the last encoding
            all_encoded.append(encoded) # Nawid - Adds the encoded output to the list of all encoded so that it gets used in the next encoder layer 
            all_encoded_thermal.append(encoded_thermal)# Nawid - Updates the all_encoded_thermal list
            
            all_vq_outs.append(embedded_pt)
            all_vq_outs_thermal.append(embedded_pt_thermal)
            
            total_vq_loss = total_vq_loss + vq_loss(encoded, embedded,encoded_thermal, embedded_thermal, commitment=commitment) # Nawid - Updates the total vq loss at eahc of the different layers
        losses = []
        losses_thermal = [] # Nawid - Used to hold the reconstruction loss
        reconstructions = []
        reconstructions_thermal = [] # Nawid- Reconstructions for the thermal case
        
        for i, decoder in enumerate(self.decoders): # Nawid - Iterate through all the decoders
            dec_inputs = all_vq_outs[::-1][:i + 1] # Nawid- as i increases, each decoder uses previous vq_outs as well as another one. Not sure why the -1 is present
            dec_inputs_thermal = all_vq_outs_thermal[::-1][:i + 1]
            
            target = all_encoded[::-1][i + 1] # Nawid - Target of the output
            target_thermal = all_encoded_thermal[::-1][i + 1] # Nawid - Target of the output
            
            recon, recon_thermal = decoder(dec_inputs,dec_inputs_thermal) # Nawid - Reconstruct 
            reconstructions.append(recon)
            reconstructions_thermal.append(recon_thermal)
            if target.dtype.is_floating_point:
                recon_loss = torch.mean(torch.pow(recon - target.detach(), 2))
                recon_loss_thermal = torch.mean(torch.pow(recon_thermal - target_thermal.detach(), 2))
            else:
                recon_loss = F.cross_entropy(recon.view(-1, recon.shape[-1]), target.view(-1)) # Nawid -  Uses cross entropy if a tensor of integers
                recon_loss_thermal = F.cross_entropy(recon_thermal.view(-1, recon.shape[-1]), target_thermal.view(-1))
            total_recon_loss = total_recon_loss + recon_loss +  recon_loss_thermal
            losses.append(recon_loss)
            losses_thermal.append(recon_loss_thermal)
        return {
            'loss': total_vq_loss + total_recon_loss,
            'losses': losses,
            'losses_thermal':losses_thermal,
            'reconstructions': reconstructions,
            'reconstructions_thermal': reconstructions_thermal,
            'embedded': all_vq_outs,
            'embedded_thermal': all_vq_outs_thermal
        }

    def revive_dead_entries(self):
        """
        Revive dead entries from all of the VQ layers.

        Only call this once the encoders have all been
        through a forward pass in training mode.
        """
        for enc in self.encoders:
            enc.vq.revive_dead_entries()

    def full_reconstructions(self, inputs,inputs_thermal):
        """
        Compute reconstructions of the inputs using all
        the different layers of the hierarchy.

        The first reconstruction uses only information
        from the top-level codes, the second uses only
        information from the top-level and second-to-top
        level codes, etc.

        This is not forward(inputs)['reconstructions'],
        since said reconstructions are simply each level's
        reconstruction of the next level's features.
        Instead, full_reconstructions reconstructs the
        original inputs.
        """
        terms = self(inputs,inputs_thermal) # Nawid - I believe this computes a forward pass through the network - Therefore if I want to compute the situation for both the rgb and the thermal, both need to be passed through
        layer_recons = [] # Nawid - List of the reconstruction
        layer_recons_thermal = [] # Nawid - Reconstructs the thermal
        for encoder, recon, recon_thermal in zip(self.encoders[:-1][::-1], terms['reconstructions'][:-1],terms['reconstructions_thermal'][:-1]):
            _, embedded_pt, _, _,embedded_pt_thermal, _ = encoder.vq(recon,recon_thermal) # Nawid - I believe this gives the embedding tensor from the reconstruction of a certain layer 
            layer_recons.append(embedded_pt)
            layer_recons_thermal.append(embedded_pt_thermal)
        hierarchy_size = len(self.decoders) # Nawid - Number of the decoder
        results = []
        results_thermal = []
        for i in range(hierarchy_size - 1):
            num_actual = i + 1
            dec_in = terms['embedded'][-num_actual:][::-1] + layer_recons[num_actual - 1:] # Nawid - Term is a dictionary i believe and it is based on the input. The embedding with the layer reconstruction is used to reconstruct the original inputs  (num_actual-1 : goes from num_actual to the end, where the first term goes from -num_actual to the end so it uses more embedding terms as the input the decoder I believe
            dec_in_thermal = terms['embedded_thermal'][-num_actual:][::-1] + layer_recons_thermal[num_actual -1:]  # Nawid - Uses the embeddings from thermal as well as the actual results from the thermal reconstruction
            result_rgb, result_thermal = self.decoders[-1](dec_in, dec_in_thermal)
            results.append(result_rgb)
            results_thermal.append(result_thermal)
            #results.append(self.decoders[-1](dec_in))
        results.append(terms['reconstructions'][-1])
        results_thermal.append(terms['reconstructions_thermal'][-1]) # Nawid - The reconstructed thermal images
        return results, results_thermal


def _make_residual(channels): # Nawid- Performs a 3x3 convolution followed by a 1x1 convolution - The 3x3 convolution is padded and so the overall shape is the same.
    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, channels, 1),
    )
