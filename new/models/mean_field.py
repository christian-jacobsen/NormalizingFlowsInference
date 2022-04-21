# mean-field inference model to be trained
import torch
import torch.nn as nn


class MeanField(nn.Module):
    def __init__(self, D):
        super().__init__()
        
        # D is the dimension of the parameter space

        self.D = D

        self.mu = nn.Parameter(torch.empty(D))
        self.logvar = nn.Parameter(torch.empty(D))

        nn.init.normal_(self.mu)
        nn.init.normal_(self.logvar)


    def forward(self, b):
        # b is the number of samples to return
        mu = self.mu*torch.ones((b, self.D))
        logvar = self.logvar*torch.ones((b, self.D))
        z = self._reparameterize(mu, logvar)

        return z, self.mu, self.logvar


    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).type_as(mu)
        return mu + std * eps
