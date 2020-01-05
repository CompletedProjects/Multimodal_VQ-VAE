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

    def encode(self, x,x_thermal):
        """
        Encode a Tensor before the VQ layer.

        Args:
            x: the input Tensor.

        Returns:
            A Tensor with the correct number of output
              channels (according to self.vq).
        """
        raise NotImplementedError

    def forward(self, x,x_thermal):
        """
        Apply the encoder.

        See VQ.forward() for return values.
        """
        return self.vq(self.encode(x,x_thermal)) # Nawid - Obtains the embedding, embeddingpt and the indices

class QuarterEncoder(Encoder):
    """
    The encoder from the original VQ-VAE paper that cuts
    the dimensions down by a factor of 4 in both
    directions.
    """

    def __init__(self, in_channels,in_channels_thermal,out_channels, num_latents, **kwargs):
        super().__init__(out_channels, num_latents, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, 4, stride=2) # Nawid - Cuts down the size by half
        self.conv2 = nn.Conv2d(out_channels//2, out_channels//2, 4, stride=2) # Nawid - Cuts down the size by half again
        self.residual1 = _make_residual(out_channels//2)
        self.residual2 = _make_residual(out_channels//2)
        
        self.conv1_thermal = nn.Conv2d(in_channels_thermal, out_channels//2, 4, stride=2) # Nawid - Cuts down the size by half and the number of input channels is related to the thermal input
         

    def encode(self, x,x_thermal):
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
        x = torch.cat([x, x_thermal], dim=1) # Nawid - Concatentates the x and x_thermal into one feature map along the dimension of the number of channels ( so that the overall number of channels equals the dimensionality of the input)
        return x

class HalfEncoder(Encoder):
    """
    An encoder that cuts the input size in half in both
    dimensions.
    """

    def __init__(self, in_channels, out_channels, num_latents, **kwargs):
        super().__init__(out_channels, num_latents, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1) # Nawid - Use to cut down the size
        self.residual1 = _make_residual(out_channels)
        self.residual2 = _make_residual(out_channels)

    def encode(self, x):
        x = self.conv(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        return x


class Decoder(nn.Module):
    """
    An abstract VQ-VAE decoder, which takes a stack of
    (differently-sized) input Tensors and produces a
    predicted output Tensor.

    Sub-classes should overload the forward() method.
    """

    def forward(self, inputs):
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

    def __init__(self, in_channels, out_channels, out_channels_thermal): # Nawid - I believe the decoder takes in the embeddings and produces the outputs
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

    def forward(self, inputs): # Nawid - Input is the concatenated feature map
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        
        x_thermal = inputs[0] # Nawid - Concatenated feature map
        x_thermal = x_thermal + self.residual1_thermal(x_thermal)
        x_thermal = x_thermal + self.residual2_thermal(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv1_thermal(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv2_thermal(x_thermal)
                
        return x, x_thermal # Nawid- Obtains both of the outputs from the decoder from a single encoder input

class HalfDecoder(Decoder):
    """
    A decoder that upsamples by a factor of 2 in both
    dimensions.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual1 = _make_residual(in_channels) 
        self.residual2 = _make_residual(in_channels)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1) # Nawid - Increases the size

    def forward(self, inputs): # Nawid - The input is an encoding feature map (rather than actual inputs)
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv(x)
        #print(' Half decoder currently works',x.size())
        return x

class HalfQuarterDecoder(Decoder):
    """
    A decoder that takes two inputs. The first one is
    upsampled by a factor of two, and then combined with
    the second input which is further upsampled by a
    factor of four.
    """

		# Nawid - Combines both levels of the hierarchy together
    def __init__(self, in_channels, out_channels, out_channels_thermal):
        super().__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.residual3 = _make_residual(in_channels)
        self.residual4 = _make_residual(in_channels)
        self.conv3 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        
        
        self.residual1_thermal = _make_residual(in_channels)
        self.residual2_thermal = _make_residual(in_channels)
        self.conv1_thermal = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2_thermal = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.residual3_thermal = _make_residual(in_channels)
        self.residual4_thermal = _make_residual(in_channels)
        self.conv3_thermal = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv4_thermal = nn.ConvTranspose2d(in_channels, out_channels_thermal, 4, stride=2, padding=1)
        

    def forward(self, inputs):
        assert len(inputs) == 2
        #print('input 1 size',inputs[0].size())
        #print('input 2 size',inputs[1].size())

        # Upsample the top input to match the shape of the
        # bottom input.
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x) # Nawid - This is a convolution transpose which make the top input match the shape of the bottom input
        x = F.relu(x)
        
        #print('x before concatenation', x.size())
        # Mix together the bottom and top inputs.
        x = torch.cat([x, inputs[1]], dim=1) # Nawid - Concatentate the upsample top feature map with the bottom feature map
        #print('x after concatenation', x.size())
        x = self.conv2(x) # Nawid - Downsamples

        x = x + self.residual3(x)
        x = x + self.residual4(x)
        x = F.relu(x)
        x = self.conv3(x) # Nawid - Upsamples
        x = F.relu(x)
        x = self.conv4(x) # Nawid - Upsamples
        
        
				# Upsample the top input to match the shape of the
        # bottom input.
        x_thermal = inputs[0]
        x_thermal = x_thermal + self.residual1_thermal(x_thermal)
        x_thermal = x_thermal + self.residual2_thermal(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv1_thermal(x_thermal) # Nawid - This is a convolution transpose which make the top input match the shape of the bottom input
        x_thermal = F.relu(x_thermal)

        # Mix together the bottom and top inputs.
        x_thermal = torch.cat([x_thermal, inputs[1]], dim=1) # Nawid - Concatentate the upsample top feature map with the bottom feature map
        x_thermal = self.conv2_thermal(x_thermal) # Nawid - Downsamples

        x_thermal = x_thermal + self.residual3(x_thermal)
        x_thermal = x_thermal + self.residual4(x_thermal)
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv3_thermal(x_thermal) # Nawid - Upsamples
        x_thermal = F.relu(x_thermal)
        x_thermal = self.conv4_thermal(x_thermal) # Nawid - Upsamples
				        
        return x, x_thermal




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

    def forward(self, inputs,inputs_thermal, commitment=0.25):
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
        all_encoded = []
        all_vq_outs = []
        total_vq_loss = 0.0 # Nawid - Initialises total vq loss
        total_recon_loss = 0.0
        for i, encoder in enumerate(self.encoders):
        	if i ==0:
        		encoded = encoder.encode(inputs,inputs_thermal)
        	else:
        		encoded = encoder.encode(all_encoded[-1]) # Nawid - Use the encoder output of the last layer as input to the next layer
        	embedded, embedded_pt, _ = encoder.vq(encoded)
        	all_encoded.append(encoded)
        	all_vq_outs.append(embedded_pt)
        	total_vq_loss = total_vq_loss + vq_loss(encoded, embedded, commitment=commitment)
        
        losses = []
        reconstructions = []
        reconstructions_thermal = []
        for i, decoder in enumerate(self.decoders):
        	dec_inputs = all_vq_outs[::-1][:i+1] # Nawid - [:i+1] is related to the number of inputs, for the case of the halfquarter decoder, two inputs are used
        	if i == len(self.decoders)-1: # Nawid - If it is the final decoder, then there are 2 different outputs and 2 targets
        		target = inputs
        		target_thermal = inputs_thermal
        		recon, recon_thermal = decoder(dec_inputs)
        		reconstructions.append(recon)
        		reconstructions_thermal.append(recon_thermal)
        		if target.dtype.is_floating_point:
        			recon_loss = torch.mean(torch.pow(recon - target.detach(), 2)) + torch.mean(torch.pow(recon_thermal - target_thermal.detach(), 2))
        		else:
        			recon_loss = F.cross_entropy(recon.view(-1, recon.shape[-1]), target.view(-1)) + F.cross_entropy(recon_thermal.view(-1, recon_thermal.shape[-1]), target_thermal.view(-1))
        	
        	
        	else:
        		target = all_encoded[::-1][i+1] # Nawid- The target is the encoding one ahead of the current target
        		recon = decoder(dec_inputs)
        		reconstructions.append(recon)
        		reconstructions_thermal.append(recon)
        		if target.dtype.is_floating_point:
        			recon_loss = torch.mean(torch.pow(recon - target.detach(), 2)) 
        		else:
        			recon_loss = F.cross_entropy(recon.view(-1, recon.shape[-1]), target.view(-1))
        	        	
        	total_recon_loss = total_recon_loss + recon_loss
        	losses.append(recon_loss)
        return {
            'loss': total_vq_loss + total_recon_loss,
            'losses': losses,
            'reconstructions': reconstructions,
            'reconstructions_thermal': reconstructions_thermal,
            'embedded': all_vq_outs,
        }
            
        '''
        all_encoded = [inputs]
        #all_encoded_thermal = [inputs_thermal]
        all_vq_outs = []
        total_vq_loss = 0.0 # Nawid - Initialises total vq loss
        total_recon_loss = 0.0
        for encoder in self.encoders: # Nawid -Iterates through the different encoders
            encoded = encoder.encode(all_encoded[-1],all_encoded_thermal[-1]) # Nawid- Looks at the last input to make an encoding - HAVING THE INPUTS SEPARATELY WORKS FOR THE FIRST LAYER BUT IT IS LIKELY NOT GOING TO WORK FOR THE HIGHER LAYERS WHERE THE NUMBER OF INPUTS ARE LIKELY TO BE DIFFERENT
            
            embedded, embedded_pt, _ = encoder.vq(encoded)# Nawid - Quantises the last encoding
            all_encoded.append(encoded) # Nawid - Adds the encoded output to the list of all encoded so that it gets used in the next encoder layer 
            all_vq_outs.append(embedded_pt)
            
            total_vq_loss = total_vq_loss + vq_loss(encoded, embedded, commitment=commitment) # Nawid - Updates the total vq loss at eahc of the different layers
        
        losses = []
        reconstructions = []
        for i, decoder in enumerate(self.decoders): # Nawid - Iterate through all the decoders
            dec_inputs = all_vq_outs[::-1][:i + 1] # Nawid- as i increases, each decoder uses previous vq_outs as well as another one. Not sure why the -1 is present - For the first dimension it starts at zero, stops till the end and stride is minus 1. So it goes through all the elements in reverse.
            target = all_encoded[::-1][i + 1] # Nawid - Target of the output
            recon = decoder(dec_inputs)# Nawid - Reconstruct 
            reconstructions.append(recon)
            if target.dtype.is_floating_point:
                recon_loss = torch.mean(torch.pow(recon - target.detach(), 2))
            else:
                recon_loss = F.cross_entropy(recon.view(-1, recon.shape[-1]), target.view(-1)) # Nawid -  Uses cross entropy if a tensor of integers
            total_recon_loss = total_recon_loss + recon_loss
            losses.append(recon_loss)
        return {
            'loss': total_vq_loss + total_recon_loss,
            'losses': losses,
            'reconstructions': reconstructions,
            'embedded': all_vq_outs,
        }
        '''

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
        terms = self(inputs,inputs_thermal) # Nawid - Initial inputs in the model
        layer_recons = [] # Nawid - List of the reconstruction
        layer_recons_thermal = []
        for encoder, recon, recon_thermal in zip(self.encoders[:-1][::-1], terms['reconstructions'][:-1], terms['reconstructions_thermal'][:-1]): # Nawid- I believe it goes from the first encoder until it reaches the last encoder in reverse order( the last encoder is not taken into account) and it goes from the first reconstruction to the last reconstruciton (last reconstruction not taken into account)
            _, embedded_pt, _ = encoder.vq(recon) # Nawid - I believe this gives the embedding tensor from the reconstruction of a certain layer (quantizes the feature map corresponding to an encoder)- I believe that the only thing that occurs here is the quantisation of different output feature maps - The quantisation procedure should be the same for all the different encoders so I am not entirely sure why  the encoders are in reverse
            _, embedded_pt_thermal,_ = encoder.vq(recon_thermal)
            
            layer_recons.append(embedded_pt)
            layer_recons_thermal.append(embedded_pt_thermal)
        hierarchy_size = len(self.decoders) # Nawid - Number of the decoders
        results = []
        results_thermal = []
        #print('Hierarchy size', hierarchy_size)
        for i in range(hierarchy_size - 1):
            num_actual = i + 1
            dec_in = terms['embedded'][-num_actual:][::-1] + layer_recons[num_actual - 1:] # Nawid - Term is a dictionary i believe and it is based on the input. The embedding with the layer reconstruction is used to reconstruct the original inputs  - The parts of embedding starts at -num_embdding to the end point in reverse order. Therefore this means at first it will use
            rgb_image, _ = self.decoders[-1](dec_in)
            #print('rgb_image', rgb_image)
            results.append(rgb_image)
            
            
            dec_in_thermal = terms['embedded'][-num_actual:][::-1] + layer_recons_thermal[num_actual - 1:]
            _, thermal_image = self.decoders[-1](dec_in_thermal)
            results_thermal.append(thermal_image)
        results.append(terms['reconstructions'][-1])
        results_thermal.append(terms['reconstructions_thermal'][-1])
        return results,results_thermal


def _make_residual(channels): # Nawid- Performs a 3x3 convolution followed by a 1x1 convolution - The 3x3 convolution is padded and so the overall shape is the same.
    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, channels, 1),
    )

