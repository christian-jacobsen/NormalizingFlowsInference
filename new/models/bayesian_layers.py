import torch
import torch.nn as nn
import numpy as np
import sys

class LinearForwardwAct(nn.Module):
    def __init__(self, nodes, layers, act=nn.ReLU()):
        super().__init__()

        self.act = act
        self.nodes = nodes
        self.layers = layers

    def forward(self, w, x):
        b = np.shape(x)[0]
        x = x.unsqueeze(1)
        x = self.act(torch.bmm(x, w[:, :self.nodes].reshape(b, 1, self.nodes)))
        for i in range(self.layers - 1):
            x = self.act(torch.bmm(x, w[:, (i+1)*self.nodes + self.nodes**2*(i):(i+1)*self.nodes + self.nodes**2*(i+1)].reshape((b, self.nodes, self.nodes))))
        x = self.act(torch.bmm(x, w[:, -self.nodes:].reshape(b, self.nodes, 1)))

        return x
        

class BayesianLinear(nn.Module):
    def __init__(self, inputs, outputs, act=nn.ReLU()):
        super().__init__()

        self.act = act
        self.inputs = inputs
        self.outputs = outputs

        self.w_mean = nn.Parameter(torch.empty(inputs, outputs))
        self.w_logvar = nn.Parameter(torch.empty(inputs, outputs))

        nn.init.normal_(self.w_mean)
        nn.init.normal_(self.w_logvar)

    def forward(self, x):
        eps = torch.randn((self.inputs, self.outputs))
        w = eps*torch.exp(0.5*self.w_logvar) + self.w_mean

        if self.act is not None:
            return self.act(torch.mm(x, w))
        else:
            return torch.mm(x, w)


class ExpLayer(nn.Module):
    def __init__(self, a, b):
        super().__init__()

        self.a = a
        self.b = b

    def forward(self, x):
        return self.b*torch.exp(self.a*x)


class BayesianFNN(nn.Module):
    def __init__(self, inputs, outputs, layers=1, nodes=10, act=nn.ReLU()):
        super().__init__()

        self.layers = layers
        self.net = nn.Sequential()
        if layers == 1:
            self.net.add_module('BayesLin'+str(0),
                                BayesianLinear(inputs, outputs, act=act))
        else:
            self.net.add_module('BayesLin'+str(0),
                                BayesianLinear(inputs, nodes, act=act))
            for i in range(layers-2):
                self.net.add_module('BayesLin'+str(i),
                                    BayesianLinear(nodes, nodes, act=act))
            self.net.add_module('BayesLin'+str(layers-1),
                                BayesianLinear(nodes, outputs, act=act))
            self.net.add_module('Exp', ExpLayer(-0.1, 8e6))

    def forward(self, x):
        return self.net(x)
