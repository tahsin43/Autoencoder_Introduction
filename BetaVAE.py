from typing import List, Any

import torch
from torch import nn


from abc import abstractmethod
# This is a base class for all VAEs (Variational Autoencoders)
# This is an abstract class, so it cannot be instantiated
# This class inherits from nn.Module
from BaseVAE import BaseVAE


class BetaVAE(BaseVAE):
    def __init__(self,input_dim,h_dim=200,z_dim=20):
        super(BetaVAE, self).__init__()
        self.input_dim=input_dim
        self.z_dim = z_dim
        self.h_dim = h_dim


    #encoder
        self.image_2hid=nn.Linear(input_dim,h_dim)
        #push it to guassian so latent space is guassian distribution
        self.hid_2mu=nn.Linear(h_dim,z_dim)
        self.hid_2sigma=nn.Linear(h_dim,z_dim)


    #decoder
        self.z_tohid=nn.Linear(z_dim,h_dim)
        self.hid_2img=nn.Linear(h_dim,input_dim)
        self.relu=nn.ReLU()

    def encode(self,x):
        h=self.relu(self.image_2hid(x))
        mu,sigma= self.hid_2mu(h),self.hid_2sigma(h)
        return mu, sigma

    def decode(self,z):
        h=self.relu(self.z_tohid(z))
        return torch.sigmoid(self.hid_2img(h))


    def forward(self,x):
        mu,sigma= self.encode(x)
        epsilon=torch.randn_like(sigma)
        z_reparametrized= mu+sigma*epsilon
        x_reconstructed= self.decode(z_reparametrized)
        return x_reconstructed,mu,sigma

    def loss_function(self,input,recons):
        loss=nn.BCELoss(reduction=sum)
        reconstructed_loss=loss(input,recons)
        return reconstructed_loss

    def kl_divergence(self,mu,sigma):
        return -torch.sum(1+torch.log(sigma**2)-mu**2-sigma**2)

    def sample(self,num_samples,current_device,**kwargs):
        z = torch.randn(num_samples,
                        self.z_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self,x, **kwargs):
        self.forward(x)[0]


if __name__ == "__main__":
    x=torch.randn(4,28*28)
    vae=BetaVAE(input_dim=784)
    x,mu,sigma=vae(x)
    print(x.shape)
    print(mu.shape)
    print(sigma.shape)






























