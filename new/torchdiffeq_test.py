import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint_event

torch.manual_seed(101)

class test_function(nn.Module):
    def __init__(self, a):
        super().__init__()
        
        #self.a = torch.nn.Parameter(torch.empty(1))
        self.a = a

        #torch.nn.init.normal_(self.a)

    def forward(self, t, state):
        #return torch.ones(np.shape(y))*torch.tensor([torch.cos(t), torch.tanh(t)])#-self.a * y
        y, w = state
        return -self.a*y, torch.zeros_like(w)

class test_function_switch(nn.Module):
    def __init__(self):#, a):
        super().__init__()
        
        self.a = torch.nn.Parameter(torch.empty(1))
        self.b = torch.nn.Parameter(torch.empty(1))

        torch.nn.init.normal_(self.a)
        torch.nn.init.normal_(self.b)

    def forward(self, t, state):
        #return torch.ones(np.shape(y))*torch.tensor([torch.cos(t), torch.tanh(t)])#-self.a * y
        y, w = state
        return w[0,0]*-self.a*y+w[0,1]*-self.b*y, torch.zeros_like(w)

class event_function(nn.Module):
    def __init__(self):
        super().__init__()

        self.t1 = torch.nn.Parameter(torch.empty(1))

        torch.nn.init.normal_(self.t1)

    def forward(self, t, state):
        y, w = state
        return -t + torch.exp(self.t1)

def event_state(t, state):
    y, w = state
    return -y[0, -1] + 5

class event_deterministic(nn.Module):
    def __init__(self, t_event):
        super().__init__()

        self.t_event = t_event

    def forward(self, t, state):
        #return -t + self.t_event
        y, w = state
        return -t + self.t_event

def state_update(t, state):
    y, w = state
    w = torch.abs(w - 1)
    return (y[-1, :, :], w[-1, :, :])

class ECOAT(nn.Module):
    def __init__(self, L, tF, dt, Sigma, R_film0, Cv, K, jmin, VR=1):
        super().__init__()

        self.L = L
        self.tF = tF
        self.dt = dt
        self.Sigma = Sigma
        self.R_film0 = R_film0
        self.Cv = Cv
        self.K = K
        self.jmin = jmin
        self.VR = VR

    def forward(self, t, state):
        # w is the switch state
        # w1 means no update
        # w2 means update

        thk, res, cur, bc_anode, w1, w2 = state
        bc_anode_out = self.VR
        thk_out = w1*thk + 1
        return thk_out

device = 'cuda' if torch.cuda.is_available else 'cpu'
print('Running on device: ', device)


y0 = (torch.tensor([[1., 2.], [0.1, 1.2]]).to(device), torch.tensor([[1, 0], [1, 0]]).to(device))
#y0 = (torch.tensor([[1., 2.]]).to(device), torch.tensor([[1., 0.]]).to(device))
t0 = 0.
T = 5.

# first generate a single datasample
t_event_true = 2.102
a = -0.5
b = 0.75

func1 = test_function(a).to(device)
func2 = test_function(b).to(device)

#event_instance1 = event_state().to(device)
event_instance1 = event_deterministic(t_event_true).to(device)
t_event_true, output = odeint_event(func1, y0, torch.tensor(t0).to(device), event_fn=event_state, odeint_interface=odeint)
t_event_true1 = t_event_true.cpu()
print('true event time: ', t_event_true.cpu().detach().numpy())

tv1 = torch.arange(t0, t_event_true1, 0.1)
tv1 = torch.cat([tv1, t_event_true1.reshape(-1)]).to(device)
tv2 = torch.arange(float(tv1[-2]) + 0.1, T, 0.1)
tv2 = torch.cat([t_event_true1.reshape(-1), tv2]).to(device)
#print('og tv1: ', np.shape(tv1))
#print('og tv2: ', np.shape(tv2))

out1 = odeint(func1, y0, tv1, method='scipy_solver', options={'solver': 'LSODA'})
y_event = state_update(0, out1)
out2 = odeint(func2, y_event, tv2, method='scipy_solver', options={'solver': 'LSODA'})

pred1, _ = out1
pred1 = pred1[:-1, :, :]
tv1 = tv1[:-1]
pred2, _ = out2
pred2 = pred2[1:, :, :]
tv2 = tv2[1:]

pred_all = torch.cat((pred1, pred2), dim=0)
tv = torch.cat((tv1, tv2))
data = pred_all[:, 0, :]
data_pred1 = pred1[:, 0, :]
data_pred2 = pred2[:, 0, :]

#print('Data shape: ', np.shape(data))
#print(np.shape(tv))

# optimize to find the correct parameters and event times
epochs = 200

tv0 = torch.linspace(0., 2., 100).to(device)

#event_instance = event_deterministic(t_event_true)#event_function().to(device)
event_instance = event_function().to(device)

func1 = test_function_switch().to(device)

'''
for param in event_instance.parameters():
    print('initial event time: ', torch.exp(param))
'''

optimizer_func = torch.optim.Adam(func1.parameters(), lr = 5e-2)
'''
params = list(func1.parameters())
print(params[0])
optimizer_func_a = torch.optim.Adam(iter(func1.a), lr = 1e-1)
optimizer_func_b = torch.optim.Adam(iter(func1.b), lr = 1e-1)
'''
optimizer_event = torch.optim.Adam(event_instance.parameters(), lr = 1e1)

print('Event time initial grad: ', func1.a.grad)


for epoch in range(epochs):

    func1.zero_grad()
    #event_instance.zero_grad()

    event_t, output = odeint_event(func1, y0, tv0[0], event_fn=event_state, odeint_interface=odeint)

    print('event time: ', event_t)

    #loss = (event_t - t_event_true.to(device))**2

    tv1 = torch.arange(t0, float(event_t), 0.1).to(device)
    tv1 = torch.cat([tv1, event_t.reshape(-1)])
    tv2 = torch.arange(float(tv1[-2])+0.1, T+1e-5, 0.1).to(device)
    tv2 = torch.cat([event_t.reshape(-1), tv2])

    #print('tv1: ', np.shape(tv1))
    #print('tv2: ', np.shape(tv2))

    out1 = odeint(func1, y0, tv1, method='scipy_solver', options={'solver': 'LSODA'})
    y1 = state_update(0, out1)
    out2 = odeint(func1, y1, tv2, method='scipy_solver', options={'solver': 'LSODA'})

    pred1, _ = out1
    pred1 = pred1[:-1, :, :]
    #tv1 = tv1[:-1]
    pred2, _ = out2
    pred2 = pred2[1:, :, :]
    #tv2 = tv2[1:]

    pred_all = torch.cat((pred1, pred2), dim=0)
    #tv = torch.cat((tv1, tv2))


    if epoch == 0:
        initial_pred = pred_all

    '''
    if epoch < epochs//2:
        loss = torch.sum((pred1[:, 0, :] - data_pred1)**2)# + (event_t - t_event_true)**2

    else:
        loss = torch.sum((pred2[:, 0, :] - data_pred2)**2)# + (event_t - t_event_true)**2
    '''

    #loss = torch.sum((pred1[:10, 0, :] - data_pred1[:10, :])**2) + (event_t - t_event_true)**2
    loss = torch.sum((pred_all[:, 0, :] - data)**2)# + 1e3*(event_t - t_event_true)**2

    print("Epoch: ", epoch + 1, "  Loss: ", loss.cpu().detach().numpy())
    loss.backward()
    #func1.a.grad = torch.tensor([0.001])
    print('Event time grad: ', func1.a.grad)

    optimizer_func.step()
    #optimizer_event.step()

plt.figure(1)
#plt.subplot(1, 2, 1)
plt.plot(tv.cpu().detach().numpy(), initial_pred[:, 0, 0].cpu().detach().numpy(), 'b')
plt.plot(tv.cpu().detach().numpy(), initial_pred[:, 0, 1].cpu().detach().numpy(), 'b', label='Initial')
plt.plot(tv.cpu().detach().numpy(), pred_all[:, 0, 0].cpu().detach().numpy(), 'r')
plt.plot(tv.cpu().detach().numpy(), pred_all[:, 0, 1].cpu().detach().numpy(), 'r', label='Final')
plt.plot(tv.cpu().detach().numpy(), data[:, 0].cpu().detach().numpy(), 'k.')
plt.plot(tv.cpu().detach().numpy(), data[:, 1].cpu().detach().numpy(), 'k.', label='Data')
plt.legend()
'''
plt.plot(tv1.cpu().detach().numpy(), pred1[:, 0, 0].cpu().detach().numpy())
plt.plot(tv2.cpu().detach().numpy(), pred2[:, 0, 0].cpu().detach().numpy())
plt.plot(tv1.cpu().detach().numpy(), pred1[:, 0, 1].cpu().detach().numpy())
plt.plot(tv2.cpu().detach().numpy(), pred2[:, 0, 1].cpu().detach().numpy())
plt.subplot(1, 2, 2)
plt.plot(tv.cpu().detach().numpy(), pred_all[:, 1, 0].cpu().detach().numpy())
plt.plot(tv.cpu().detach().numpy(), pred_all[:, 1, 1].cpu().detach().numpy())
plt.plot(tv1.cpu().detach().numpy(), pred1[:, 1, 0].cpu().detach().numpy())
plt.plot(tv2.cpu().detach().numpy(), pred2[:, 1, 0].cpu().detach().numpy())
plt.plot(tv1.cpu().detach().numpy(), pred1[:, 1, 1].cpu().detach().numpy())
plt.plot(tv2.cpu().detach().numpy(), pred2[:, 1, 1].cpu().detach().numpy())
'''
plt.savefig('test.png')


