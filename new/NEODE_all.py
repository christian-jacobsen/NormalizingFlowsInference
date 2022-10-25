import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint_event

from utils.utils import *

torch.manual_seed(101)

class rho_nn(nn.Module):
    def __init__(self, nodes, N, act=nn.ReLU()):
        super().__init__()

        self.fnn = nn.Sequential()
        self.fnn.add_module('lin0', nn.Linear(1, N))
        self.fnn.add_module('act0', act)
        for i in range(1, nodes):
            self.fnn.add_module('lin'+str(i), nn.Linear(N, N))
            self.fnn.add_module('act0', act)

        self.fnn.add_module('lin_final', nn.Linear(N, 1))
        self.add_module('relu_final', nn.ReLU())

    def forward(self, j):
        return self.fnn(j)


class ECOAT(nn.Module):
    def __init__(self, L, Sigma, R_film0, VR=1., const=False):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available else 'cpu'

        self.L = L
        self.Sigma = Sigma
        self.R_film0 = R_film0

        self.beta = Sigma * VR / (Sigma*R_film0 + L)

        self.VR = torch.tensor([VR]).to(self.device)

        self.rho_func = rho_nn(2, 10)

        # if const, we use it as a ground truth data generator
        if const:
            self.logCv = torch.nn.Parameter(torch.tensor([7.]))
            self.K = torch.nn.Parameter(torch.tensor([35.]))
            self.jmin = torch.nn.Parameter(torch.tensor([1.]))
        else:
            self.logCv = torch.nn.Parameter(torch.randn(1)*0.5 + 7.)
            self.K = torch.nn.Parameter(torch.randn(1)*5. + 35.)
            self.jmin = torch.nn.Parameter(torch.randn(1)*0.2 + 1.)

        self.y0 = (torch.tensor([[0.], [0.]]).to(self.device),
                   torch.tensor([[R_film0], [R_film0]]).to(self.device),
                   torch.tensor([[0.], [0.]]).to(self.device),
                   torch.tensor([[0.], [0.]]).to(self.device),
                   torch.tensor([[1, 0]]).to(self.device),
                   self.K)

    def forward(self, t, state):
        # w is the switch state

        thk, res, bc_anode, Q, w, k = state
        bc_out = self.VR*(torch.zeros_like(bc_anode) + 1.)
        cur = self.Sigma * bc_anode / (self.Sigma * res + self.L)
        Q_out = cur
        thk_out = w[0, 0] * 0 + w[0, 1]*(10**(-self.logCv)*(cur - self.jmin))
        res_out = w[0, 0] * 0 + w[0, 1]*(10**(-self.logCv)*self.rho(cur)*(cur - self.jmin))
        return thk_out, res_out, bc_out, Q_out, torch.zeros_like(w), torch.zeros_like(k)

    def rho(self, j):
        #return 8e6*torch.exp(-0.1*j)
        #return self.limit(2e6, 8e6*torch.exp(-0.1*j), 1e9)
        return 7.5e6*torch.exp(-self.rho_func(j)) + 1e6

    def limit(self, val, lim, scale):
        return (val - lim) / (1 + torch.exp(-scale*(val-lim))) + lim

    def event(self, t, state):
        # if Q >= Qmin, we output zero or negative value. 
        #   Else output positive
        thk, res, bc_anode, Q, w, k = state
        Qmin = (81/(128*self.beta))**(1/3)*(k**(4/3))
        return Qmin - Q

    def update_state(self, t, state):
        # update the switch w
        thk, res, bc_anode, Q, w, k = state
        w = torch.abs(w - 1) # differentiable binary switch
        return (thk[-1, :, :], res[-1, :, :], bc_anode[-1, :, :], Q[-1, :, :], w[-1, :, :], k)

    def get_event_time(self, t0=0.):
        # get the event time
        t_event, _ = odeint_event(self, self.y0,
                                  torch.tensor(t0).to(self.device),
                                  event_fn=self.event,
                                  odeint_interface=odeint)
        return t_event

    def simulate(self, T, dt=0.1, t0=0.):
        # returns model output from t0 to T by dt

        t_event = self.get_event_time(t0)

        tv1 = torch.arange(t0, float(t_event), dt).to(self.device)
        tv1 = torch.cat([tv1, t_event.reshape(-1)])
        tv2 = torch.arange(float(tv1[-2])+0.1, T+1e-4, dt).to(self.device)
        tv2 = torch.cat([t_event.reshape(-1), tv2])

        out1 = odeint(self, self.y0, tv1, method='scipy_solver', options={'solver': 'LSODA'})
        y1 = self.update_state(0, out1)
        out2 = odeint(self, y1, tv2, method='scipy_solver', options={'solver': 'LSODA'})

        thk1, res1, bc_anode1, _, _, _ = out1
        thk2, res2, bc_anode2, _, _, _ = out2
        tv = torch.cat((tv1[:-1], tv2[1:]))

        res = torch.cat((res1[:-1, :, :], res2[1:, :, :]), dim=0)
        thk = torch.cat((thk1[:-1, :, :], thk2[1:, :, :]), dim=0)
        bc_anode = torch.cat((bc_anode1[:-1, :, :], bc_anode2[1:, :, :]), dim=0)
        cur = self.Sigma * bc_anode / (self.Sigma * res + self.L)

        return tv, 1.6e-3*cur, 1.6e3*res, thk, t_event


device = 'cuda' if torch.cuda.is_available else 'cpu'
print('Running on device: ', device)

Sigma = 0.14
L = 0.0254
R_film0 = 0.5
VR = 1.0

t0 = 0.
T = 50.
dt = 0.1

# generate some data or load the data
data_cur, var_cur, data_res, var_res, t_final, n_trials = load_all_data_torch('./data/experimental')
T = t_final[0].float()
'''
data_model = ECOAT(L, Sigma, R_film0, VR=VR, const=True).to(device)
t_data, cur_data, res_data, thk_data, t_event_data = data_model.simulate(T)
cur_data, res_data, thk_data = cur_data[:, 0, 0].detach(), res_data[:, 0, 0].detach(), thk_data[:, 0, 0].detach()
t_event_data = t_event_data.detach()
del data_model
'''

model = ECOAT(L, Sigma, R_film0, VR=VR).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

epochs = 500
n_batch = 1

loss_v = np.zeros((epochs, ))

model.train()

# train the model
for epoch in range(epochs):
    model.zero_grad()

    t, cur, res, thk, t_event = model.simulate(T)
    data_filled_cur = fill_data_torch(data_cur[0], cur[:, 0, :].reshape(1, -1))
    data_filled_res = fill_data_torch(data_res[0], res[:, 0, :].reshape(1, -1))
    cur1 = torch.transpose(cur, 0, 2)
    res1 = torch.transpose(res, 0, 2)

    loss = torch.mean((torch.tile(data_filled_cur.unsqueeze(0), (n_batch, 1, 1)) - torch.tile(cur1[:, 0, :].unsqueeze(1), (1, int(n_trials[0]), 1)))**2) #torch.mean(100*(t_event-t_event_data)**2)#
    loss = loss + 1e-5*torch.mean((torch.tile(data_filled_res.unsqueeze(0), (n_batch, 1, 1)) - torch.tile(res1[:, 0, :].unsqueeze(1), (1, int(n_trials[0]), 1)))**2) #torch.mean(100*(t_event-t_event_data)**2)#
    loss.backward()

    print("Epoch: ", epoch, " , Loss: ", loss)
    loss_v[epoch] = loss.cpu().detach().numpy()

    optimizer.step()

    if epoch == 0:
        cur_init = cur[:, 0, 0].cpu().detach().numpy()
        res_init = res[:, 0, 0].cpu().detach().numpy()
        thk_init = thk[:, 0, 0].cpu().detach().numpy()

#res_data = res_data.cpu().numpy()
#thk_data = thk_data.cpu().numpy()
print('Final Param Values:')
print('K: ', model.K)
print('logCv: ', model.logCv)
print('jmin: ', model.jmin)

t, cur = t.cpu().detach().numpy(), cur.cpu().detach().numpy()
res, thk = res.cpu().detach().numpy(), thk.cpu().detach().numpy()

plt.figure(1, figsize=(25, 8))
plt.subplot(1, 3, 1)
for i in range(n_trials[0]):
    plt.plot(t, data_cur[0][i, :].cpu().numpy(), 'k.')
plt.plot(t, cur_init, 'b', label='Initial')
plt.plot(t, cur[:, 0, 0], 'r', label='Final')
plt.legend()
plt.title('Current')
plt.subplot(1, 3, 2)
plt.plot(t, thk_init, 'b', label='Initial')
plt.plot(t, thk[:, 0, 0], 'r', label='Final')
#plt.plot(t, thk_data, 'k.', label='Data')
plt.title('Thickness')
plt.subplot(1, 3, 3)
for i in range(n_trials[0]):
    plt.plot(t, data_res[0][i, :].cpu().numpy(), 'k.')
plt.plot(t, res_init, 'b', label='Initial')
plt.plot(t, res[:, 0, 0], 'r', label='Final')
plt.title('Resistance')
plt.savefig('ecoat_neode_test.png')

plt.figure(2)
plt.plot(loss_v)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('loss_curve.png')

plt.figure(3)
j_v = torch.linspace(0, 20, 1000).reshape(-1, 1).to(device)
rho_v = model.rho(j_v)
plt.plot(j_v.cpu().detach().numpy(), rho_v.cpu().detach().numpy())
plt.xlabel('j')
plt.ylabel('rho(j)')
plt.title('FNN rho(j)')
plt.savefig('fnn_rho.png')

