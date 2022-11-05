import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def kl_gaussian(mu,log_var):
    return torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))

def kl_two_gaussian(mu1, log_var1, mu2, log_var2):
    return torch.sum(0.5 * ((mu1-mu2.detach())**2/torch.exp(log_var2.detach()) + torch.exp(log_var1)/torch.exp(log_var2.detach()) - (log_var1-log_var2.detach()) -1))

def kl_two_gaussian2(mu1, log_var1, mu2, log_var2):
    return torch.sum(0.5 * ((mu1-mu2)**2/torch.exp(log_var2) + torch.exp(log_var1)/torch.exp(log_var2) - (log_var1-log_var2) -1))

def kl_prior_post_loss(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                                   - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                                   - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)))
    return kld

def paded_mm(one_hot_matrix, other_matrix, flatten=True):
    # one_hot_matrix.size: (batch, N), other_matrix: (batch, N, gauss_dim)
    N = one_hot_matrix.size(1)
    gaussian_dim = other_matrix.size(2)
    one_hot_matrix = one_hot_matrix.view(-1, N)
    other_matrix = other_matrix.view(-1, N, gaussian_dim)
    mixture = one_hot_matrix.unsqueeze(-1) * other_matrix
    if flatten:
        mixture = mixture.view(-1, N * gaussian_dim).contiguous()
    return mixture


class condition_VAE5(nn.Module):
    def __init__(self, embed_size, feat_size, h_dim, gaussian_dim, device):
        super(condition_VAE5, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embed_size+feat_size, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, 2*gaussian_dim)
            )

        self.decoder = nn.Sequential(
            nn.Linear(gaussian_dim+feat_size, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, embed_size))

        self.embed_size = embed_size
        self.feat_size = feat_size
        self.h_dim = h_dim
        self.gaussian_dim = gaussian_dim
        self.device = device

    def to_var(self, x):
        return x.to(self.device)
    
    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        if self.training:
            eps = self.to_var(torch.randn(mu.size(0), mu.size(1)))
        else:
            eps = self.to_var(torch.zeros(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return z
    
    def forward(self, embed_x, feat_x):
        input_tensor = torch.cat([embed_x, feat_x], dim=-1)
        gauss_latent = self.encoder(input_tensor)
        mues, logs = torch.chunk(gauss_latent, 2, dim=-1)
        mues = mues.contiguous()
        logs = logs.contiguous()
        mus_logs = (mues, logs)
        z = self.reparametrize(mues, logs)
        input_z = torch.cat([z, feat_x], dim=-1)
        out = self.decoder(input_z)
        return out, mus_logs, z

    def generate(self, embed_x, feat_x_enc, feat_x_dec):
        input_tensor = torch.cat([embed_x, feat_x_enc], dim=-1)
        gauss_latent = self.encoder(input_tensor)
        mues, logs = torch.chunk(gauss_latent, 2, dim=-1)
        mues = mues.contiguous()
        logs = logs.contiguous()
        mus_logs = (mues, logs)
        z = self.reparametrize(mues, logs)
        input_z = torch.cat([z, feat_x_dec], dim=-1)
        out = self.decoder(input_z)
        return out, mus_logs, z

    def generate2(self, embed_x, feat_x):
        input_tensor = torch.cat([embed_x, feat_x], dim=-1)
        gauss_latent = self.encoder(input_tensor)
        mues, logs = torch.chunk(gauss_latent, 2, dim=-1)
        mues = mues.contiguous()
        logs = logs.contiguous()
        mus_logs = (mues, logs)
        z = self.reparametrize(mues, logs)
        input_z = torch.cat([z, feat_x], dim=-1)
        out = self.decoder(input_z)
        return out, mus_logs, z