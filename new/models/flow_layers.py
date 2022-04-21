import torch
import torch.nn as nn
import numpy as np
import sys


class PlanarFlow(nn.Module):
    def __init__(self, D):
        super().__init__()
        act = torch.tanh
        self.D = D

        self.w = nn.Parameter(torch.empty(D))
        self.b = nn.Parameter(torch.empty(1))
        self.u = nn.Parameter(torch.empty(D))
        self.act = act
        self.act_deriv = lambda x: 1 - torch.tanh(x)**2

        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

    def forward(self, z):
        if torch.isnan(torch.mean(self.w)):
            print("w is nan planar flow!")
            self.w = nn.Parameter(torch.empty(self.D))
            nn.init.normal_(self.w)
            #sys.exit("Exiting Training")

        if torch.isnan(torch.mean(self.u)):
            print("u is nan planar flow!")
            self.u = nn.Parameter(torch.empty(self.D))
            nn.init.normal_(self.u)
            #sys.exit("Exiting Training")

        dot = torch.sum(self.w*self.u)
        if torch.isnan(torch.mean(dot)):
            print("dot is nan planar flow!")
            sys.exit("Exiting Training")

        u = self.u + (torch.log(1+torch.exp(dot) + 1e-20)-1-dot)*self.w/torch.sum(self.w**2)

        if torch.isnan(torch.mean(u)):
            print("u is nan planar flow!")
            sys.exit("Exiting Training")

        if torch.isnan(self.b):
            print("b is nan planar flow!")
            self.b = nn.Parameter(torch.empty(1))
            nn.init.normal_(self.b)

        lin = (z @ self.w + self.b).unsqueeze(1)
        if torch.isnan(torch.mean(lin)):
            print("lin is nan planar flow!")
            sys.exit("Exiting Training")
        f = z + u * self.act(lin)
        phi = self.act_deriv(lin) * self.w
        log_det = torch.log(torch.abs(1+phi@u) + 1e-20)  # add a small constant to avoid nan's

        if torch.isnan(torch.mean(f)):
            print("f is nan planar flow!")
            sys.exit("Exiting Training")
        if torch.isnan(torch.mean(log_det)):
            print("log_det is nan planar flow!")
            sys.exit("Exiting Training")

        

        return f, log_det


class SigmoidFlow(nn.Module):
    def __init__(self, D):
        super().__init__()

        self.D = D
        self.a = 1
        self.b = 0

    def forward(self, z):

        f = self.a*1/(1 + torch.exp(-z)) + self.b

        log_det = np.log(self.a)*self.D + torch.sum(torch.log(f * (1 - f) + 1e-42), dim=1)

        if torch.isnan(torch.mean(f)):
            print("f is nan sigmoid flow!")
            sys.exit("Exiting Training")
        if torch.isnan(torch.mean(log_det)):
            print("log_det is nan sigmoid flow!")
            sys.exit("Exiting Training")

        return f, log_det

class RadialFlow(nn.Module):
    def __init__(self, D):
        super().__init__()
        
        self.D = D
        self.z0 = nn.Parameter(torch.empty(D))
        self.beta = nn.Parameter(torch.empty(1))
        self.alpha = nn.Parameter(torch.empty(1))
        
        nn.init.normal_(self.z0)
        nn.init.normal_(self.alpha)
        nn.init.normal_(self.beta)
        
    def forward(self, z):
        beta = torch.log(1+torch.exp(self.beta)) - torch.abs(self.alpha) # to ensure layer is invertible
        r = torch.linalg.vector_norm(z - self.z0, dim = 1)

        h = beta / (torch.abs(self.alpha) + r)
        h_prime = -beta * r / (torch.abs(self.alpha)+r) ** 2
        f = z + h.unsqueeze(1)*(z-self.z0)

        log_det = (self.D-1) * torch.log(1+h) \
                  + torch.log(1+h+h_prime)       
        return f, log_det 

