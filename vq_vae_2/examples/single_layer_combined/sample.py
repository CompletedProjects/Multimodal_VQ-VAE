"""
Sample an image from a PixelCNN.
"""

import random

from PIL import Image
import numpy as np
import torch

from vq_vae_2.examples.mnist.model import Generator, make_vq_vae

DEVICE = torch.device('cpu')


def main():
    vae = make_vq_vae() # Nawid- Builds the encoder and decoder
    vae.load_state_dict(torch.load('vae.pt', map_location='cpu')) # Nawid -Loads the vae model
    vae.to(DEVICE)
    vae.eval()
    generator = Generator()
    generator.load_state_dict(torch.load('gen.pt', map_location='cpu'))
    generator.to(DEVICE)

    inputs = np.zeros([4, 7, 7], dtype=np.long) # Nawid - The first input is an array of zeros
    for row in range(7):
        for col in range(7):
            with torch.no_grad(): # Nawid - Stop gradient as this is not training
                outputs = torch.softmax(generator(torch.from_numpy(inputs).to(DEVICE)), dim=1) # Nawid - Performs a softmax on the logits obtained from the array of inputs ( the logits I believe are related to the different embeddings)
                for i, out in enumerate(outputs.cpu().numpy()):
                    probs = out[:, row, col] # Nawid - Obtain the probabilites for  a
                    inputs[i, row, col] = sample_softmax(probs) # Nawid - Sample from the softmax and puts the output in the index array. The output is index values
        print('done row', row)
    embedded = vae.encoders[0].vq.embed(torch.from_numpy(inputs).to(DEVICE)) # Nawid - Converts encoded indices into embeddings.

    decoded = torch.clamp(vae.decoders[0]([embedded]), 0, 1).detach().cpu().numpy() # Nawid - Decodes the indices I believe.
    decoded = np.concatenate(decoded, axis=1)
    Image.fromarray((decoded * 255).astype(np.uint8)[0]).save('samples.png') # Nawid - Saves the decoded image


def sample_softmax(probs):
    number = random.random()
    for i, x in enumerate(probs): # Nawid -i is the counter and x is the value in probs
        number -= x
        if number <= 0:
            return i
    return len(probs) - 1


if __name__ == '__main__':
    main()
