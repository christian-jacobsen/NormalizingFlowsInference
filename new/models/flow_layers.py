import torch
import torch.nn as nn


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
        
        dot = torch.sum(self.w*self.u)
        u = self.u + (torch.log(1+torch.exp(dot))-1-dot)*self.w/torch.sum(self.w**2)
        lin = (z @ self.w + self.b).unsqueeze(1)
        f = z + u * self.act(lin)
        phi = self.act_deriv(lin) * self.w
        log_det = torch.log(torch.abs(1+phi@u))
        
        #f = z + self.a
        #log_det = 1
        
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

