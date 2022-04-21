import torch
import torch.nn as nn
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../")

from models.mean_field import *
from models.forward_models import *
from inputs.inputs_meanfield import *



writer = SummaryWriter('./logs/meanfield')


# load our data

data = np.loadtxt("../data/experimental/Exp_Ramp1V.txt")

thk_data = data[:, -2]

thk_data = torch.from_numpy(np.reshape(thk_data, (1, -1)))

# define forward model for likelihood

ForwardModel = torch_forward(0.0254, 11, 0.1, 50, 0.14, 0.5, 1.0)

# define the infernce model
MF = MeanField(3)

mup = meanfield_config['mu_prior']
logvarp = meanfield_config['logvar_prior']

b = meanfield_config['MC_samples']
epochs = training_config['epochs']

Optimizer = torch.optim.Adam(MF.parameters(), lr = 1e-3)

pred_inds = [5, 20, 30]
data_inds = np.arange(0, len(pred_inds))

loss_vector = np.zeros(epochs)

torch.autograd.set_detect_anomaly(False)


# save the computational graph for visualization

z, mu, logvar = MF.forward(b)
writer.add_graph(ForwardModel, z)

for epoch in range(epochs):

    MF.zero_grad()

    z, mu, logvar = MF.forward(b)

    pred = ForwardModel.forward(z)

    l_rec = 0.5*torch.mean((pred[:,pred_inds] - thk_data[0, data_inds])**2)

    l_prior = 0.5*torch.sum((mu**2 + torch.exp(logvar) - 2*mup*mu + mup**2)/torch.exp(logvarp))

    h = - 0.5*torch.sum(logvar)

    loss = l_rec + l_prior + h

    loss.backward()
    Optimizer.step()

    loss_vector[epoch] = loss.cpu().detach().numpy()

z, mu, logvar = MF.forward(1)

print("Posterior mean: ", mu.cpu().detach().numpy())
print("Posterior logvar: ", logvar.cpu().detach().numpy())
print("Posterior var: ", torch.exp(logvar).cpu().detach().numpy())


