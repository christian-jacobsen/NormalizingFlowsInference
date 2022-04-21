import torch
import torch.nn as nn
import sys

sys.path.append('../')

from models.flow_layers import *


class Flow(nn.Module):
    def __init__(self, D, flow_layers, mu0, logvar0, act=nn.ReLU()):
        super().__init__()
        self.D = D
        self.flow = nn.Sequential()
        #self.flow.add_module('identity', nn.Identity())

        for i in range(flow_layers):
            #if np.mod(i, 2)==0:
            self.flow.add_module('flow%d'%(i+1), PlanarFlow(D))
            #else:
            #self.flow.add_module('flow%d'%(i+1), RadialFlow(D))
            
        self.flow.add_module('flow_sigmoid', SigmoidFlow(D))


        self.mu0 = mu0
        self.logvar0 = logvar0

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

    def forward(self, n, mu_pzk, var_pzk):  #  zmu, zlogvar, mu_pzk, var_pzk):
        batch_size = n#zmu.shape[0]
        sigma = torch.exp(0.5*self.logvar0)#torch.exp(0.5*zlogvar)
        zlogvar = torch.ones((n, self.D))*self.logvar0
        zmu = torch.ones((n, self.D))*self.mu0
 
        z = self._reparameterize(zmu, zlogvar)
        
        z0 = z.clone()
        
        log_prob_z0 = torch.sum(- 0.5 * ((z-zmu)/sigma) ** 2, axis = 1)
        

        log_det = torch.zeros((batch_size,))
        for layer in self.flow:
            z, ld = layer(z)
            log_det += ld

        log_prob_zk = torch.sum(-0.5 * ((z-mu_pzk)**2/var_pzk), axis = 1) # prior z_k

        return z0, zmu, zlogvar, z, \
               log_prob_z0, log_det, log_prob_zk

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).type_as(mu)
        return mu + std * eps

    def gaussian_log_prob(self, x, mu, logvar):
        return -0.5*(math.log(2*math.pi) + logvar + (x-mu)**2/torch.exp(logvar))

    def comp_prob(self, z, n, ind):
        # takes input n as batch size and ind as the index to compute probability on

        sigma = torch.exp(0.5*self.logvar0)
        zmu = torch.ones((n, self.D))*self.mu0
        
        log_prob_z0 = torch.sum(- 0.5 * ((z-zmu)/sigma) ** 2, axis = 1)
        


        log_det = torch.zeros((n,))
        i = 0
        for layer in self.flow:
            if i==(len(self.flow)-1):
                
                
                # compute the probability of only the ind variable (planar flows)

                dot = torch.sum(layer.w*layer.u)
                u = layer.u + (torch.log(1+torch.exp(dot))-1-dot)*layer.w/torch.sum(layer.w**2)
                lin = (z @ layer.w + layer.b).unsqueeze(1)

                ld = 1 + u[ind] * layer.act_deriv(lin) * layer.w[ind]  
                ld = ld[:,0]  
                log_det += torch.log(torch.abs(ld))
                
                z, _ = layer(z)
                zk = z[:,ind]
            else:
                z, ld = layer(z)
                log_det += ld
                
            i += 1
        
        # return the probability
        return zk, log_prob_z0 - log_det
