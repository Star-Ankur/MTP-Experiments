import scanpy as sc
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import umap.umap_ as umap
from sklearn.decomposition import PCA

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import torchplot as plt

import cdopt
from cdopt.nn.utils.set_constraints import set_constraint_dissolving
from cdopt.manifold_torch import stiefel_torch
from cdopt.nn import get_quad_penalty

class LatentMerge(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(LatentMerge, self).__init__()
        
        # Define the dimensions
        self.x_dim = x_dim
        self.z_dim = z_dim
        input_dim = x_dim + z_dim
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, x_dim)  # Merge layer

    def forward(self, x, z):
        # Concatenate the inputs along the last dimension
        combined_input = torch.cat((x, z), dim=-1)
        
        return self.fc1(combined_input)

class LadderEncoder(nn.Module):
    def __init__(self, dims):
        """
        The ladder encoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions [input_dim, [hidden_dims], [latent_dims]].
        """
        super(LadderEncoder, self).__init__()
        [x_dim, h_dim, self.z_dim] = dims
        self.in_features = x_dim
        self.out_features = h_dim

        self.linear = nn.Linear(x_dim, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.latent_sample = nn.Linear(h_dim, self.z_dim)
        set_constraint_dissolving(self.latent_sample, 'weight', manifold_class = stiefel_torch, penalty_param= 0.02)

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(self.batchnorm(x), 0.1)
        return x, self.latent_sample(x)


class LadderDecoder(nn.Module):
    def __init__(self, dims):
        """
        The ladder dencoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(LadderDecoder, self).__init__()

        [self.z_dim, h_dim, x_dim] = dims

        self.linear = nn.Linear(x_dim, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.merge = LatentMerge(x_dim, self.z_dim)

    def forward(self, x, z=None):
        if z is not None:
            # Merge x and z
            x = self.merge(x, z)

        # Sample from the decoder and send forward
        z = self.linear(x)
        z = F.leaky_relu(self.batchnorm(z), 0.1)

        return z

class LadderManifoldAutoencoder(nn.Module):
    def __init__(self, dims):
        """
        Ladder Variational Autoencoder as described by
        Ankur Rathore. Adds several Latent 
        layers to improve the Latent representation.

        :param dims: x, z and hidden dimensions of the networks
        """
        super(LadderManifoldAutoencoder, self).__init__()
        [x_dim, z_dim, h_dim] = dims
        
        neurons = [x_dim, *h_dim]
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]
        # decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]
        dec_neurons = [ *h_dim, z_dim[-1]]
        decoder_layers = [LadderDecoder([z_dim[i], h_dim[i], dec_neurons[i+1]]) for i in range(len(h_dim))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = nn.Linear(h_dim[0], x_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        for encoder in self.encoder:
            x, z = encoder(x)
            latents.append(z)

        latents = list(reversed(latents))

        for i, decoder in enumerate(self.decoder):
            if i == 0:
                x = decoder(latents[i])
            else:
                x = decoder(x, latents[i])

        x_mu = self.reconstruction(x)
        return x_mu
