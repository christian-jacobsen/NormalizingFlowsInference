import torch
import torch.nn as nn
import numpy as np


class GaussianVI(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.D = D

        self.mu = nn.Parameter(torch.empty(D))
        self.logvar = nn.Parameter(torch.empty(D))

        nn.init.normal_(self.mu)
        nn.init.normal_(self.logvar)

        # initialize the parameters
        '''
        mu_init = torch.Tensor([1.0, 7.0, 50.0])
        logvar_init = torch.Tensor([np.log(0.25**2), np.log(1.5**2), np.log(10**2)])

        '''

        '''
        self.mu[0] = 1.0
        self.mu[1] = 7.0
        self.mu[2] = 50.0

        self.logvar[0] = np.log(0.25**2)
        self.logvar[1] = np.log(1.5**2)
        self.logvar[2] = np.log(10**2)
        '''

        '''
        self.mu0 = torch.ones((1,D)) # nn.Parameter(torch.ones((D,))*0.5)
        self.mu0[0,0] = 7.5
        self.mu0[0,1] = 170
        self.mu0[0,2] = 0.5
        self.logvar0 = torch.ones((1,D)) #nn.Parameter(torch.ones((D,))*np.log(1**2))
        self.logvar0[0,0] = np.log(0.1**2)
        self.logvar0[0,1] = np.log(10**2)
        self.logvar0[0,2] = np.log(0.1**2)
        '''

    def forward(self, n):  # zmu, zlogvar, mu_pzk, var_pzk):
        sigma = torch.exp(0.5*self.logvar)  # torch.exp(0.5*zlogvar)
        zlogvar = torch.ones((n, self.D))*self.logvar
        zmu = torch.ones((n, self.D))*self.mu

        z = self._reparameterize(zmu, zlogvar)

        log_prob_z0 = torch.sum(- 0.5 * ((z-zmu)/sigma) ** 2, axis=1)

        return zmu, zlogvar, z, log_prob_z0

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).type_as(mu)
        return mu + std * eps

    def gaussian_log_prob(self, x, mu, logvar):
        return -0.5*(math.log(2*math.pi) + logvar + (x-mu)**2/torch.exp(logvar))
