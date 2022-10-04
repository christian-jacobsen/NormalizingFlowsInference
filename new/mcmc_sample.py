import sys, torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')

from mcmc_model import *
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
CD_1V_cur = std_1V_cur**2  # np.ones((1, np.shape(data)[1]))
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

# MCMC params
n_particles = 1000
n_iters = 2
n_burn = 10

total_proposals = 0
total_accepted = 0  # to measure acceptance ratio

# model params
n_rho_layers = 2
n_rho_nodes = 5
n_weights = 2*n_rho_nodes + n_rho_nodes**2*(n_rho_layers-1)  # 3
n_params = 3
samples = torch.zeros((n_particles, n_weights + n_params, n_iters))

sig_g_fnn = 1e-4 # proposal standard deviation for the network parameters
#sig_g_fnn = torch.tensor([1e-2, 1e-2, 1e-3])
sig_g_par = torch.tensor([1e-4, 1e-4, 1e-3]) # proposal variance for infernce params

L = 0.0254
#N = 11  # model doesn't need spactial discretization anymore
dt = 0.1
Sigma = 0.14
R_film0 = 0.5

# initialize model / optimizer
model = FunctionalInferenceECOAT_MCMC(n_rho_nodes, n_rho_layers)

print("MCMC start ...  =====================================")

print('cd shape: ', np.shape(CD_1V_cur))


mup = torch.Tensor([7.32, 44.2, 0.63])
logvarp = torch.Tensor([np.log(0.5**2), np.log(3**2), np.log(0.1**2)])
mu_wp = torch.Tensor([2, 8, 0.1])
std_wp = torch.Tensor([1e-1, 1e-1, 1e-2])

proposals = torch.zeros((n_particles, n_weights + n_params))
for i in range(n_iters):

    if i == 0:
        # sample from prior for initial distribution
        #eps = torch.abs(torch.randn((n_particles, n_weights))*std_wp + mu_wp)
        eps = torch.randn((n_particles, n_weights))
        samples[:, :n_weights, 0] = eps
        eps = torch.abs(torch.randn((n_particles, 3))*torch.exp(0.5*logvarp) + mup)
        print(np.shape(eps))
        samples[:, n_weights:, 0] = eps

        # compute f for all samples
        _, Res_1V, Cur_1V, _ = model(n_particles, samples[:, n_weights:, 0], samples[:, :n_weights, 0], L, tFinal_1V+dt, dt, Sigma, R_film0, 'VR', VR=1.0)
        _, Res_0_5V, Cur_0_5V, _ = model(n_particles, samples[:, n_weights:, 0], samples[:, :n_weights, 0], L, tFinal_0_5V+dt, dt, Sigma, R_film0, 'VR', VR=0.5)
        _, Res_0_125V, Cur_0_125V, _ = model(n_particles, samples[:, n_weights:, 0], samples[:, :n_weights, 0], L, tFinal_0_125V+dt, dt, Sigma, R_film0, 'VR', VR=0.125)


    else:
        # propose new samples
        eps = torch.randn((n_particles, n_weights))*sig_g_fnn + samples[:, :n_weights, i-1]
        proposals[:, :n_weights] = eps
        eps = torch.abs(torch.randn((n_particles, 3))*sig_g_par + samples[:, n_weights:, i-1])
        proposals[:, n_weights:] = eps

        # compute f for all samples
        _, Res_1V, Cur_1V, _ = model(n_particles, proposals[:, n_weights:], proposals[:, :n_weights], L, tFinal_1V+dt, dt, Sigma, R_film0, 'VR', VR=1.0)
        _, Res_0_5V, Cur_0_5V, _ = model(n_particles, proposals[:, n_weights:], proposals[:, :n_weights], L, tFinal_0_5V+dt, dt, Sigma, R_film0, 'VR', VR=0.5)
        _, Res_0_125V, Cur_0_125V, _ = model(n_particles, proposals[:, n_weights:], proposals[:, :n_weights], L, tFinal_0_125V+dt, dt, Sigma, R_film0, 'VR', VR=0.125)



    #L = 0.5*(torch.sum(logvarp) - 3 - torch.sum(zlogvar) + torch.sum(torch.exp(zlogvar)/torch.exp(logvarp)) + torch.sum((zmu - mup)**2/torch.exp(logvarp)))  ????



    # 1VR Experiment
    data_filled = fill_data_torch(data_1V_cur, Cur_1V)
    log_like = torch.sum(-0.5 * ((torch.tile(Cur_1V.unsqueeze(1), (1, n_data_1V, 1)) - torch.tile(data_filled.unsqueeze(0), (n_particles, 1, 1)))**2 / torch.tile(CD_1V_cur.unsqueeze(0), (n_particles, n_data_1V, 1))), dim=(1, 2))

    # 0.5VR Experiment
    data_filled = fill_data_torch(data_0_5V_cur, Cur_0_5V)
    log_like = torch.sum(-0.5 * ((torch.tile(Cur_0_5V.unsqueeze(1), (1, n_data_0_5V, 1)) - torch.tile(data_filled.unsqueeze(0), (n_particles, 1, 1)))**2 / torch.tile(CD_0_5V_cur.unsqueeze(0), (n_particles, n_data_0_5V, 1))), dim=(1, 2))

    # 0.125VR Experiment
    data_filled = fill_data_torch(data_0_125V_cur, Cur_0_125V)
    log_like = torch.sum(-0.5 * ((torch.tile(Cur_0_125V.unsqueeze(1), (1, n_data_0_125V, 1)) - torch.tile(data_filled.unsqueeze(0), (n_particles, 1, 1)))**2 / torch.tile(CD_0_125V_cur.unsqueeze(0), (n_particles, n_data_0_125V, 1))), dim=(1, 2))
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

    #L = 0.5*(torch.sum(logvarp) - 3 - torch.sum(zlogvar) + torch.sum(torch.exp(zlogvar)/torch.exp(logvarp)) + torch.sum((zmu - mup)**2/torch.exp(logvarp))) - torch.mean(log_like)

    if i > 0:
        # accept or reject
        ratio = torch.exp(log_like - log_like_prev)  # acceptance probability
        if any(torch.isinf(ratio)):
            print(' acceptance ratio infinite for some samples!')
            ratio[torch.isinf(ratio)] = 1.0

        # print(ratio)
        a = torch.rand((n_particles, ))
        inds_accept = torch.zeros((n_particles, ))
        inds_accept = torch.where(a >= (1-ratio), 1, 0)
        inds_accept = torch.argwhere(inds_accept)
        samples[:, :, i] = samples[:, :, i-1]
        '''
        print('previous sample: ', samples[inds_accept[0], :, i])
        print('accepted sample: ', proposals[inds_accept[0], :])
        '''
        samples[inds_accept, :, i] = proposals[inds_accept, :]

        log_like_prev[inds_accept] = log_like[inds_accept]
        
        total_proposals += n_particles
        total_accepted += np.shape(inds_accept)[0]  # to measure acceptance ratio
        # increment the probabilities

    else:
        log_like_prev = log_like








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


    #loss_vector[epoch] = loss.cpu().detach().numpy()

    # if np.mod(epoch, training_params['epochs']//20) == 0:
    if np.mod(i, 1) == 0:
        if i > 0:
            print('Iter: ', i, ' Acceptance Ratio: ', total_accepted / total_proposals)
        else:
            print('Iter: ', i)


plt.figure(1)
mean_cur = np.mean(Cur_1V.cpu().detach().numpy(), axis=0)
sd_cur = np.sqrt(np.var(Cur_1V.cpu().detach().numpy(), axis=0))
plt.plot(mean_cur, 'k')
plt.plot(mean_cur + 3*sd_cur, 'r')
plt.plot(mean_cur - 3*sd_cur, 'r')

plt.figure(4)
cur = Cur_1V.cpu().detach().numpy()
mean_cur = np.mean(Cur_1V.cpu().detach().numpy(), axis=0)
sd_cur = np.sqrt(np.var(Cur_1V.cpu().detach().numpy(), axis=0))
plt.plot(np.transpose(cur[0:10, :]), 'k')
#plt.plot(mean_cur + 3*sd_cur, 'r')
#plt.plot(mean_cur - 3*sd_cur, 'r')


jvec = torch.linspace(0, 25, 1000)
jtens = torch.zeros((n_particles * 1000, 1))
for i in range(n_particles):
    jtens[1000*i:1000*(i+1), 0] = jvec[i]
rho_out = model.rho(torch.tile(samples[:, :n_weights, -1], (1000, 1)), jtens)

plt.figure(2)
rho_out = rho_out.cpu().detach().numpy()
rho_out = np.reshape(rho_out, (n_particles, 1000))
mean_rho = np.mean(rho_out, axis=0)
sd_rho = np.sqrt(np.var(rho_out, axis=0))
plt.plot(mean_rho, 'k')
#plt.plot(mean_rho + 3*sd_rho, 'r')
#plt.plot(mean_rho - 3*sd_rho, 'r')
'''
plt.figure(2)
jtens = torch.linspace(0, 25, 1000)
jtens = np.reshape(jtens, (-1, 1))
W_mean = torch.mean(samples[:, :3, -1], dim=0).cpu().detach().numpy()
W_samples = samples[:, :3, -1]
rho_out = torch.zeros((n_particles, 1000))
for i in range(n_particles):
    #print('shape rho: ', np.shape(model.rho(W_samples[i, :].reshape((-1, 3)), jtens)))
    rho_out[i, :] = model.rho(W_samples[i, :].reshape((-1, 3)), jtens)[:, 0]
#rho_out = model.rho(W_mean.reshape((-1, 3)), jtens)
jtens = jtens.cpu().detach().numpy()
rho_out = rho_out.cpu().detach().numpy()
plt.plot(jtens, np.mean(rho_out, axis=0), 'r')
plt.plot(jtens, np.mean(rho_out, axis=0) + 2*np.std(rho_out, axis=0), 'k')
plt.plot(jtens, np.mean(rho_out, axis=0) - 2*np.std(rho_out, axis=0), 'k')
plt.xlabel('j')
plt.ylabel('rho(j)')
'''

samples = samples.cpu().detach().numpy()


plt.figure(3)
plt.subplot(1, 3, 1)
plt.hist(samples[:, -3, 0], density=True, color='b', alpha=0.5, label='Initial')
plt.hist(samples[:, -3, -1], density=True, color='r', alpha=0.5, label='Final')
plt.subplot(1, 3, 2)
plt.hist(samples[:, -2, 0], density=True, color='b', alpha=0.5)
plt.hist(samples[:, -2, -1], density=True, color='r', alpha=0.5)
plt.subplot(1, 3, 3)
plt.hist(samples[:, -1, 0], density=True, color='b', alpha=0.5)
plt.hist(samples[:, -1, -1], density=True, color='r', alpha=0.5)

plt.figure(10)
plt.subplot(1, 3, 1)
plt.hist(samples[:, 0, 0], density=True, color='b', alpha=0.5, label='Initial')
plt.hist(samples[:, 0, -1], density=True, color='r', alpha=0.5, label='Final')
plt.subplot(1, 3, 2)
plt.hist(samples[:, 1, 0], density=True, color='b', alpha=0.5)
plt.hist(samples[:, 1, -1], density=True, color='r', alpha=0.5)
plt.subplot(1, 3, 3)
plt.hist(samples[:, 2, 0], density=True, color='b', alpha=0.5)
plt.hist(samples[:, 2, -1], density=True, color='r', alpha=0.5)


plt.show()


'''
plt.figure(1)
plt.plot(loss_vector)
plt.show()
'''

#data_nf = 0.
#save_vi(NF.state_dict(), flow_params, training_params, loss_vector, data_nf, surrogate_path, save_nf_path)

