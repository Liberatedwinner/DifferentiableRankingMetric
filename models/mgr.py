import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import itertools

dg = torch.digamma


def dgd(x):
    return torch.polygamma(1, x)


def dgdsqrt(x):
    return torch.sqrt(dgd(x))


def g_eps(a, e):
    return torch.exp(e * dgdsqrt(a) + dg(a))


def r_beta(alpha, beta):
    alpha.clamp(1.0)
    beta.clamp(1.0)
    ns1 = torch.normal(mean=0, std=1, size=alpha.shape).clamp(-2, 2)
    ns2 = torch.normal(mean=0, std=1, size=alpha.shape).clamp(-2, 2)
    a_samples = g_eps(alpha, ns1).clamp(min=1e-9, max=30)
    b_samples = g_eps(beta, ns2).clamp(min=1e-9, max=30)
    return (a_samples / (a_samples + b_samples))



class MGR(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims, dropout=0.5):
        super(MGR, self).__init__()
        self.p_dims = p_dims
        self.q_dims = q_dims

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        self.dd = nn.Dropout(0.1)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        pi = self.get_pi(z)
        return pi, mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        if self.training:
            h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = self.dd(h)
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            eps = eps.clamp(-3, 3)
            return mu + eps * std
        else:
            return mu

    def get_pi(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = self.dd(h)
                h = torch.tanh(h)
        l = self.p_dims[-1] // 2
        alpha = h[:, :l]
        beta = h[:, l:]
        pi = r_beta(alpha, beta)
        return pi

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            #layer.weight.data.normal_(0.0, std)
            torch.nn.init.xavier_uniform(layer.weight)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            torch.clamp(layer.bias.data, min=-0.001, max=0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            #std = np.sqrt(2.0 / (fan_in + fan_out))
            #layer.weight.data.normal_(0.0, std)
            torch.nn.init.xavier_uniform(layer.weight)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        self.p_layers[-1].weight.data.uniform_(1.00, 1.00)
