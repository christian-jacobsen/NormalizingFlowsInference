import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint_event

torch.manual_seed(101)


class ECOAT(nn.Module):
    def __init__(self, L, Sigma, R_film0, VR=1., const=False):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available else 'cpu'

        self.L = L
        self.Sigma = Sigma
        self.R_film0 = R_film0

        self.beta = Sigma * VR / (Sigma*R_film0 + L)

        self.VR = torch.tensor([VR]).to(self.device)


        '''
        self.Cv = Cv
        self.K = K
        self.jmin = jmin
        '''

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
                   torch.tensor([[1, 0]]).to(self.device))

    def forward(self, t, state):
        # w is the switch state

        thk, res, bc_anode, Q, w = state
        bc_out = self.VR*(torch.zeros_like(bc_anode) + 1.)
        cur = self.Sigma * bc_anode / (self.Sigma * res + self.L)
        Q_out = cur
        '''
        thk_out = w[0, 0] * 0 + w[0, 1]*(10**(-self.logCv)*F.relu(cur - self.jmin))
        res_out = w[0, 0] * 0 + w[0, 1]*(10**(-self.logCv)*self.rho(cur)*F.relu(cur - self.jmin))
        '''
        thk_out = w[0, 0] * 0 + w[0, 1]*(10**(-self.logCv)*(cur - self.jmin))
        res_out = w[0, 0] * 0 + w[0, 1]*(10**(-self.logCv)*self.rho(cur)*(cur - self.jmin))
        return thk_out, res_out, bc_out, Q_out, torch.zeros_like(w)

    def rho(self, j):
        return 8e6*torch.exp(-0.1*j)
        #return self.limit(2e6, 8e6*torch.exp(-0.1*j), 1e9)

    def limit(self, val, lim, scale):
        return (val - lim) / (1 + torch.exp(-scale*(val-lim))) + lim

    def event(self, t, state):
        # if Q >= Qmin, we output zero or negative value. 
        #   Else output positive
        thk, res, bc_anode, Q, w = state
        Qmin = (81/(128*self.beta))**(1/3)*(self.K**(4/3))
        return Qmin - Q

    def update_state(self, t, state):
        # update the switch w
        thk, res, bc_anode, Q, w = state
        w = torch.abs(w - 1) # differentiable binary switch
        return (thk[-1, :, :], res[-1, :, :], bc_anode[-1, :, :], Q[-1, :, :], w[-1, :, :])

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
        tv2 = torch.arange(float(tv1[-2])+0.1, T+1e-6, dt).to(self.device)
        tv2 = torch.cat([t_event.reshape(-1), tv2])

        out1 = odeint(self, self.y0, tv1, method='scipy_solver', options={'solver': 'LSODA'})
        y1 = self.update_state(0, out1)
        out2 = odeint(self, y1, tv2, method='scipy_solver', options={'solver': 'LSODA'})

        thk1, res1, bc_anode1, _, _ = out1
        thk2, res2, bc_anode2, _, _ = out2
        tv = torch.cat((tv1[:-1], tv2[1:]))

        #cur = torch.cat((cur1[:-1, :, :], cur2[1:, :, :]), dim=0)
        res = torch.cat((res1[:-1, :, :], res2[1:, :, :]), dim=0)
        thk = torch.cat((thk1[:-1, :, :], thk2[1:, :, :]), dim=0)
        bc_anode = torch.cat((bc_anode1[:-1, :, :], bc_anode2[1:, :, :]), dim=0)
        cur = self.Sigma * bc_anode / (self.Sigma * res + self.L)

        return tv, cur, res, thk


device = 'cuda' if torch.cuda.is_available else 'cpu'
print('Running on device: ', device)

Sigma = 0.14
L = 0.0254
R_film0 = 0.5
VR = 1.0

t0 = 0.
T = 100.
dt = 0.1

# generate some data
data_model = ECOAT(L, Sigma, R_film0, VR=VR, const=True).to(device)
t_data, cur_data, res_data, thk_data = data_model.simulate(T)
cur_data, res_data, thk_data = cur_data[:, 0, 0].detach(), res_data[:, 0, 0].detach(), thk_data[:, 0, 0].detach()
del data_model

model = ECOAT(L, Sigma, R_film0, VR=VR).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-2)

epochs = 100

loss_v = np.zeros((epochs, ))

# train the model
for epoch in range(epochs):

    t, cur, res, thk = model.simulate(T)

    loss = torch.mean((cur_data - cur[:, 0, 0])**2)
    loss.backward()

    print("Epoch: ", epoch, " , Loss: ", loss)
    loss_v[epoch] = loss.cpu().detach().numpy()

    optimizer.step()

    if epoch == 0:
        cur_init = cur[:, 0, 0].cpu().detach().numpy()
        res_init = res[:, 0, 0].cpu().detach().numpy()
        thk_init = thk[:, 0, 0].cpu().detach().numpy()

cur_data = cur_data.cpu().numpy()
res_data = res_data.cpu().numpy()
thk_data = thk_data.cpu().numpy()

t, cur = t.cpu().detach().numpy(), cur.cpu().detach().numpy()
res, thk = res.cpu().detach().numpy(), thk.cpu().detach().numpy()

plt.figure(1, figsize=(25, 8))
plt.subplot(1, 3, 1)
plt.plot(t, cur_init, 'b', label='Initial')
plt.plot(t, cur[:, 0, 0], 'r', label='Final')
plt.plot(t, cur_data, 'k.', label='Data')
plt.legend()
plt.title('Current')
plt.subplot(1, 3, 2)
plt.plot(t, thk_init, 'b', label='Initial')
plt.plot(t, thk[:, 0, 0], 'r', label='Final')
plt.plot(t, thk_data, 'k.', label='Data')
plt.title('Thickness')
plt.subplot(1, 3, 3)
plt.plot(t, res_init, 'b', label='Initial')
plt.plot(t, res[:, 0, 0], 'r', label='Final')
plt.plot(t, res_data, 'k.', label='Data')
plt.title('Resistance')
plt.savefig('ecoat_neode_test.png')

plt.figure(2)
plt.plot(loss_v)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('loss_curve.png')