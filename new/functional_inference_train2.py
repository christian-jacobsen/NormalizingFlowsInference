import sys, torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')

from functional_inference_model import *
from utils.utils import *

# generate data for the NF (different from the surrogate)

# load all the data now
# first the voltage ramp experiments
print("Loading Experimental Data...")
n_data_1V = 13
load_path = "./data/experimental/VR_1V.xlsx"
data_1V_cur, std_1V_cur = load_data(load_path, "VR_1V_Current", n_data_1V)
data_1V_res, std_1V_res = load_data(load_path, "VR_1V_Resistance", n_data_1V)

n_data_0_5V = 13
load_path = "./data/experimental/VR_0.5V.xlsx"
data_0_5V_cur, std_0_5V_cur = load_data(load_path, "VR_0.5V_Current", n_data_0_5V)
data_0_5V_res, std_0_5V_res = load_data(load_path, "VR_0.5V_Resistance", n_data_0_5V)

n_data_0_125V = 12
load_path = "./data/experimental/VR_0.125V.xlsx"
data_0_125V_cur, std_0_125V_cur = load_data(load_path, "VR_0.125V_Current", n_data_0_125V)
data_0_125V_res, std_0_125V_res = load_data(load_path, "VR_0.125V_Resistance", n_data_0_125V)

# now constant current experiments
n_data_0_5mA = 10
load_path = "./data/experimental/CC_0.5mA.xlsx"
data_0_5mA_cur, std_0_5mA_cur = load_data(load_path, "CC_0.5mA_Current", n_data_0_5mA)
data_0_5mA_res, std_0_5mA_res = load_data(load_path, "CC_0.5mA_Resistance", n_data_0_5mA)
data_0_5mA_vol, std_0_5mA_vol = load_data(load_path, "CC_0.5mA_Voltage", n_data_0_5mA)

n_data_0_75mA = 10
load_path = "./data/experimental/CC_0.75mA.xlsx"
data_0_75mA_cur, std_0_75mA_cur = load_data(load_path, "CC_0.75mA_Current", n_data_0_75mA)
data_0_75mA_res, std_0_75mA_res = load_data(load_path, "CC_0.75mA_Resistance", n_data_0_75mA)
data_0_75mA_vol, std_0_75mA_vol = load_data(load_path, "CC_0.75mA_Voltage", n_data_0_75mA)

n_data_1mA = 10
load_path = "./data/experimental/CC_1mA.xlsx"
data_1mA_cur, std_1mA_cur = load_data(load_path, "CC_1mA_Current", n_data_1mA)
data_1mA_res, std_1mA_res = load_data(load_path, "CC_1mA_Resistance", n_data_1mA)
data_1mA_vol, std_1mA_vol = load_data(load_path, "CC_1mA_Voltage", n_data_1mA)

# define the data variance
CD_1V_cur = std_1V_cur**2#np.ones((1, np.shape(data)[1]))
CD_1V_res = std_1V_res**2

CD_0_5V_cur = std_0_5V_cur**2
CD_0_5V_res = std_0_5V_res**2

CD_0_125V_cur = std_0_125V_cur**2
CD_0_125V_res = std_0_125V_res**2

CD_0_5mA_cur = std_0_5mA_cur**2
CD_0_5mA_res = std_0_5mA_res**2
CD_0_5mA_vol = std_0_5mA_vol**2

CD_0_75mA_cur = std_0_75mA_cur**2
CD_0_75mA_res = std_0_75mA_res**2
CD_0_75mA_vol = std_0_75mA_vol**2

CD_1mA_cur = std_1mA_cur**2
CD_1mA_res = std_1mA_res**2
CD_1mA_vol = std_1mA_vol**2

# define the time ranges for the experiments
tFinal_1V = 239
tFinal_0_5V = 477
tFinal_0_125V = 639
tFinal_0_5mA = 240
tFinal_0_75mA = 160
tFinal_1mA = 80

# define the prior
mu_prior = np.array([7, 37, 2.5])
var_prior = 0.25**2 * np.array([5**2, 20**2, 5**2])

# define the indices to take so that fowrward model prediction corresponds to data times
data_inds_1V = np.arange(0, 10*tFinal_1V+1)
model_inds_1V = np.arange(0, 10*tFinal_1V+1)
data_inds_0_5V = np.arange(0, 10*tFinal_0_5V+1)
model_inds_0_5V = np.arange(0, 10*tFinal_0_5V+1)
data_inds_0_125V = np.arange(0, 10*tFinal_0_125V+1)
model_inds_0_125V = np.arange(0, 10*tFinal_0_125V+1)
data_inds_1mA = np.arange(0, 10*tFinal_1mA) + 1
model_inds_1mA = data_inds_1mA - 1
data_inds_0_5mA = np.arange(0, 10*tFinal_0_5mA) + 1
model_inds_0_5mA = data_inds_0_5mA - 1
data_inds_0_75mA = np.arange(0, 10*tFinal_0_75mA) + 1
model_inds_0_75mA = data_inds_0_75mA - 1

data_1V_cur = torch.from_numpy(data_1V_cur[:, data_inds_1V]).float()
data_1V_res = torch.from_numpy(data_1V_res[:, data_inds_1V]).float()
data_0_5V_cur = torch.from_numpy(data_0_5V_cur[:, data_inds_0_5V]).float()
data_0_5V_res = torch.from_numpy(data_0_5V_res[:, data_inds_0_5V]).float()
data_0_125V_cur = torch.from_numpy(data_0_125V_cur[:, data_inds_0_125V]).float()
data_0_125V_res = torch.from_numpy(data_0_125V_res[:, data_inds_0_125V]).float()
data_1mA_cur = torch.from_numpy(data_1mA_cur[:, data_inds_1mA]).float()
data_1mA_res = torch.from_numpy(data_1mA_res[:, data_inds_1mA]).float()
data_1mA_vol = torch.from_numpy(data_1mA_vol[:, data_inds_1mA]).float()
data_0_5mA_cur = torch.from_numpy(data_0_5mA_cur[:, data_inds_0_5mA]).float()
data_0_5mA_res = torch.from_numpy(data_0_5mA_res[:, data_inds_0_5mA]).float()
data_0_5mA_vol = torch.from_numpy(data_0_5mA_vol[:, data_inds_0_5mA]).float()
data_0_75mA_cur = torch.from_numpy(data_0_75mA_cur[:, data_inds_0_75mA]).float()
data_0_75mA_res = torch.from_numpy(data_0_75mA_res[:, data_inds_0_75mA]).float()
data_0_75mA_vol = torch.from_numpy(data_0_75mA_vol[:, data_inds_0_75mA]).float()

CD_1V_cur = torch.from_numpy(CD_1V_cur[:, data_inds_1V]).float()
CD_1V_res = torch.from_numpy(CD_1V_res[:, data_inds_1V]).float()
CD_0_5V_cur = torch.from_numpy(CD_0_5V_cur[:, data_inds_0_5V]).float()
CD_0_5V_res = torch.from_numpy(CD_0_5V_res[:, data_inds_0_5V]).float()
CD_0_125V_cur = torch.from_numpy(CD_0_125V_cur[:, data_inds_0_125V]).float()
CD_0_125V_res = torch.from_numpy(CD_0_125V_res[:, data_inds_0_125V]).float()
CD_0_5mA_cur = torch.from_numpy(CD_0_5mA_cur[:, data_inds_0_5mA]).float()
CD_0_5mA_res = torch.from_numpy(CD_0_5mA_res[:, data_inds_0_5mA]).float()
CD_0_5mA_vol = torch.from_numpy(CD_0_5mA_vol[:, data_inds_0_5mA]).float()
CD_0_75mA_cur = torch.from_numpy(CD_0_75mA_cur[:, data_inds_0_75mA]).float()
CD_0_75mA_res = torch.from_numpy(CD_0_75mA_res[:, data_inds_0_75mA]).float()
CD_0_75mA_vol = torch.from_numpy(CD_0_75mA_vol[:, data_inds_0_75mA]).float()
CD_1mA_cur = torch.from_numpy(CD_1mA_cur[:, data_inds_1mA]).float()
CD_1mA_res = torch.from_numpy(CD_1mA_res[:, data_inds_1mA]).float()
CD_1mA_vol = torch.from_numpy(CD_1mA_vol[:, data_inds_1mA]).float()
print("  Data Loaded!")

# training params
epochs = 1505
prior_epochs = 1500
torch.autograd.set_detect_anomaly(True)
b = 1  # batch size

# model params
n_rho_layers = 2
n_rho_nodes = 5

L = 0.0254
#N = 11  # model doesn't need spactial discretization anymore
dt = 0.1
Sigma = 0.14
R_film0 = 0.5

# initialize model / optimizer
model = FunctionalInferenceECOAT(n_rho_layers, n_rho_nodes)
Optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

model.train()

loss_vector = np.zeros((epochs,))

print("Optimization start ...  =====================================")


surr_inds = np.arange(0, 51)
data_inds = np.arange(0, 501, 10)

scales = torch.tensor([0.44, 0.44, 22])
offsets = torch.tensor([1, 7, 50])
perm = [2, 0, 1]

'''
if flow_params['use_exp_data']:
    C_D = C_D[0, data_inds]
'''
mup = torch.Tensor([7.32, 44.2, 0.63])
logvarp = torch.Tensor([np.log(0.5**2), np.log(3**2), np.log(0.1**2)])

print('n_data_1V: ', n_data_1V)

for epoch in range(epochs):

    model.zero_grad()


    if epoch < prior_epochs:
        zmu, zlogvar, z, log_prob_z0 = model.sample(b)

        L = 0.5*(torch.sum(logvarp) - 3 - torch.sum(zlogvar) + torch.sum(torch.exp(zlogvar)/torch.exp(logvarp)) + torch.sum((zmu - mup)**2/torch.exp(logvarp)))

    else:
        zmu, zlogvar, z, log_prob_z0, _, Res_1V, Cur_1V, _ = model(b, L, tFinal_1V+dt, dt, Sigma, R_film0, 'VR', VR=1.0)

        '''
        plt.figure(1)
        plt.plot(Cur_1V[0, :].cpu().detach().numpy())
        plt.show()
        '''

        print('zmu: ', zmu)
        print('zsd: ', torch.exp(0.5*zlogvar))

        # 1VR Experiment
        data_filled = fill_data_torch(data_1V_cur, Cur_1V)
        log_like = torch.sum(-0.5 * ((torch.tile(Cur_1V, (n_data_1V, 1)) - data_filled)**2 / torch.tile(CD_1V_cur, (n_data_1V, 1))))

        '''
        data_filled = fill_data_torch(data_1V_res, Res_1V)
        log_like = torch.sum(-0.5 * ((torch.tile(Res_1V, (n_data_1V, 1)) - data_filled)**2 / torch.tile(CD_1V_res, (n_data_1V, 1))))# + log_like

        # 0.5VR Experiment
        _, _, _, _, _, Res_0_5V, Cur_0_5V, _ = model(b, N, L, tFinal_0_5V+dt, dt, Sigma, R_film0, 'VR', VR=0.5)

        data_filled = fill_data_torch(data_0_5V_cur, Cur_0_5V)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Cur_0_5V, (n_data_0_5V, 1)) - data_filled)**2 / torch.tile(CD_0_5V_cur, (n_data_0_5V, 1))))

        data_filled = fill_data_torch(data_0_5V_res, Res_0_5V)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Res_0_5V, (n_data_0_5V, 1)) - data_filled)**2 / torch.tile(CD_0_5V_res, (n_data_0_5V, 1))))

        # 0.125VR Experiment
        _, _, _, _, _, Res_0_125V, Cur_0_125V, _ = model(b, N, L, tFinal_0_125V+dt, dt, Sigma, R_film0, 'VR', VR=0.125)

        data_filled = fill_data_torch(data_0_125V_cur, Cur_0_125V)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Cur_0_125V, (n_data_0_125V, 1)) - data_filled)**2 / torch.tile(CD_0_125V_cur, (n_data_0_125V, 1))))

        data_filled = fill_data_torch(data_0_125V_res, Res_0_125V)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Res_0_125V, (n_data_0_125V, 1)) - data_filled)**2 / torch.tile(CD_0_125V_res, (n_data_0_125V, 1))))
        '''

        # Constant Current Experiments
        '''
        # 1mA Experiment
        t_in = (torch.arange(0.1, tFinal_1mA+0.1, 0.1).reshape((-1, 1)).float() - 50.4962075) / 28.86910898
        Cur = 1e-3*surr_1mA_cur(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))
        Res = 1e2*surr_1mA_res(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))

        data_filled = fill_data_torch(data_1mA_cur, Cur)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Cur, (n_data_1mA, 1)) - data_filled)**2 / torch.tile(CD_1mA_cur, (n_data_1mA, 1))))

        data_filled = fill_data_torch(data_1mA_res, Res)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Res, (n_data_1mA, 1)) - data_filled)**2 / torch.tile(CD_1mA_res, (n_data_1mA, 1))))

        # 0.75mA Experiment
        t_in = (torch.arange(0.1, tFinal_0_75mA+0.1, 0.1).reshape((-1, 1)).float() - 100.51029687) / 57.74028474
        Cur = 1e-3*surr_0_75mA_cur(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))
        Res = 1e2*surr_0_75mA_res(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))

        data_filled = fill_data_torch(data_0_75mA_cur, Cur)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Cur, (n_data_0_75mA, 1)) - data_filled)**2 / torch.tile(CD_0_75mA_cur, (n_data_0_75mA, 1))))

        data_filled = fill_data_torch(data_0_75mA_res, Res)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Res, (n_data_0_75mA, 1)) - data_filled)**2 / torch.tile(CD_0_75mA_res, (n_data_0_75mA, 1))))

        # 0.5mA Experiment
        t_in = (torch.arange(0.1, tFinal_0_5mA+0.1, 0.1).reshape((-1, 1)).float() - 150.48703708) / 86.60500884
        Cur = 1e-3*surr_0_5mA_cur(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))
        Res = 1e2*surr_0_5mA_res(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))

        data_filled = fill_data_torch(data_0_5mA_cur, Cur)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Cur, (n_data_0_5mA, 1)) - data_filled)**2 / torch.tile(CD_0_5mA_cur, (n_data_0_5mA, 1))))

        data_filled = fill_data_torch(data_0_5mA_res, Res)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Res, (n_data_0_5mA, 1)) - data_filled)**2 / torch.tile(CD_0_5mA_res, (n_data_0_5mA, 1))))
        '''

        for g in Optimizer.param_groups:
            g['lr'] = 1e-3

        L = 0.5*(torch.sum(logvarp) - 3 - torch.sum(zlogvar) + torch.sum(torch.exp(zlogvar)/torch.exp(logvarp)) + torch.sum((zmu - mup)**2/torch.exp(logvarp))) - torch.mean(log_like)

    loss = L  # + ...
    '''

    if torch.isnan(loss):
        print("Loss is nan!")
        print("LR: ", torch.mean(lr))
        print("LPZK: ", torch.mean(lpzk))
        print("LPZ0: ", torch.mean(lpz0))
        print("LogDet: ", torch.mean(ld))

        sys.exit("Stoped Training")
    '''


    '''
    + torch.sum(torch.exp(exp_lim*(z[:,0] - 1)) + torch.exp(-exp_lim*z[:,0]) + \
               torch.exp(exp_lim*(z[:,1] - 1)) + torch.exp(-exp_lim*z[:,1]) + \
               torch.exp(exp_lim*(z[:,2] - 1)) + torch.exp(-exp_lim*z[:,2]))
    '''

    loss.backward(retain_graph=True)
    Optimizer.step()

    loss_vector[epoch] = loss.cpu().detach().numpy()

    # if np.mod(epoch, training_params['epochs']//20) == 0:
    if np.mod(epoch, 1) == 0:
        print('Epoch: ', epoch, ' Loss: ', loss_vector[epoch])

plt.figure(1)
plt.plot(loss_vector)
plt.show()

#data_nf = 0.
#save_vi(NF.state_dict(), flow_params, training_params, loss_vector, data_nf, surrogate_path, save_nf_path)

