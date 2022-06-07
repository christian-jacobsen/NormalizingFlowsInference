import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("../")

from models.forward_models import *
from inputs.inputs_surrogate import *
from utils.utils import *


# import our data

# first the voltage ramp experiments
n_data_1V = 13
load_path = "../data/experimental/VR_1V.xlsx"
data_1V_cur, std_1V_cur = load_data(load_path, "VR_1V_Current", n_data_1V)
data_1V_res, std_1V_res = load_data(load_path, "VR_1V_Resistance", n_data_1V)

n_data_0_5V = 13
load_path = "../data/experimental/VR_0.5V.xlsx"
data_0_5V_cur, std_0_5V_cur = load_data(load_path, "VR_0.5V_Current", n_data_0_5V)
data_0_5V_res, std_0_5V_res = load_data(load_path, "VR_0.5V_Resistance", n_data_0_5V)

n_data_0_125V = 12
load_path = "../data/experimental/VR_0.125V.xlsx"
data_0_125V_cur, std_0_125V_cur = load_data(load_path, "VR_0.125V_Current", n_data_0_125V)
data_0_125V_res, std_0_125V_res = load_data(load_path, "VR_0.125V_Resistance", n_data_0_125V)

# now constant current experiments
n_data_0_5mA = 10
load_path = "../data/experimental/CC_0.5mA.xlsx"
data_0_5mA_cur, std_0_5mA_cur = load_data(load_path, "CC_0.5mA_Current", n_data_0_5mA)
data_0_5mA_res, std_0_5mA_res = load_data(load_path, "CC_0.5mA_Resistance", n_data_0_5mA)
data_0_5mA_vol, std_0_5mA_vol = load_data(load_path, "CC_0.5mA_Voltage", n_data_0_5mA)

n_data_0_75mA = 10
load_path = "../data/experimental/CC_0.75mA.xlsx"
data_0_75mA_cur, std_0_75mA_cur = load_data(load_path, "CC_0.75mA_Current", n_data_0_75mA)
data_0_75mA_res, std_0_75mA_res = load_data(load_path, "CC_0.75mA_Resistance", n_data_0_75mA)
data_0_75mA_vol, std_0_75mA_vol = load_data(load_path, "CC_0.75mA_Voltage", n_data_0_75mA)

n_data_1mA = 10
load_path = "../data/experimental/CC_1mA.xlsx"
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

# define the parameter domain
N = 20  # number of points in each dimension
parameters_lower_bound = np.array([6, 0, 0])#[6.5, 36, 2])
parameters_upper_bound = np.array([8.5, 120, 3])#[7.5, 37.5, 2.5])

M1 = np.tile(np.linspace(parameters_lower_bound[0], parameters_upper_bound[0], N), (N, N, 1))
M2 = np.tile(np.linspace(parameters_lower_bound[1], parameters_upper_bound[1], N), (N, N, 1))
M2 = np.transpose(M2, (1, 2, 0))
M3 = np.tile(np.linspace(parameters_lower_bound[2], parameters_upper_bound[2], N), (N, N, 1))
M3 = np.transpose(M3, (2, 0, 1))

# theta_v contains all data points in the grid
theta_v = np.hstack((np.reshape(M1, (-1,1)), np.reshape(M2, (-1,1)), np.reshape(M3, (-1,1))))

# preallocate arrays
n_params = np.shape(theta_v)[0]
log_post = np.zeros((n_params,))
log_like_v = np.zeros((n_params,))

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

data_1V_cur = data_1V_cur[:, data_inds_1V]
data_1V_res = data_1V_res[:, data_inds_1V]
data_0_5V_cur = data_0_5V_cur[:, data_inds_0_5V]
data_0_5V_res = data_0_5V_res[:, data_inds_0_5V]
data_0_125V_cur = data_0_125V_cur[:, data_inds_0_125V]
data_0_125V_res = data_0_125V_res[:, data_inds_0_125V]
data_1mA_cur = data_1mA_cur[:, data_inds_1mA]
data_1mA_res = data_1mA_res[:, data_inds_1mA]
data_1mA_vol = data_1mA_vol[:, data_inds_1mA]
data_0_5mA_cur = data_0_5mA_cur[:, data_inds_0_5mA]
data_0_5mA_res = data_0_5mA_res[:, data_inds_0_5mA]
data_0_5mA_vol = data_0_5mA_vol[:, data_inds_0_5mA]
data_0_75mA_cur = data_0_75mA_cur[:, data_inds_0_75mA]
data_0_75mA_res = data_0_75mA_res[:, data_inds_0_75mA]
data_0_75mA_vol = data_0_75mA_vol[:, data_inds_0_75mA]

CD_1V_cur = CD_1V_cur[:, data_inds_1V]
CD_1V_res = CD_1V_res[:, data_inds_1V]
CD_0_5V_cur = CD_0_5V_cur[:, data_inds_0_5V]
CD_0_5V_res = CD_0_5V_res[:, data_inds_0_5V]
CD_0_125V_cur = CD_0_125V_cur[:, data_inds_0_125V]
CD_0_125V_res = CD_0_125V_res[:, data_inds_0_125V]
CD_0_5mA_cur = CD_0_5mA_cur[:, data_inds_0_5mA]
CD_0_5mA_res = CD_0_5mA_res[:, data_inds_0_5mA]
CD_0_5mA_vol = CD_0_5mA_vol[:, data_inds_0_5mA]
CD_0_75mA_cur = CD_0_75mA_cur[:, data_inds_0_75mA]
CD_0_75mA_res = CD_0_75mA_res[:, data_inds_0_75mA]
CD_0_75mA_vol = CD_0_75mA_vol[:, data_inds_0_75mA]
CD_1mA_cur = CD_1mA_cur[:, data_inds_1mA]
CD_1mA_res = CD_1mA_res[:, data_inds_1mA]
CD_1mA_vol = CD_1mA_vol[:, data_inds_1mA]

print("Starting Inference with ", N**3, " grid points: ")

for i in range(n_params):
    print("Progress: ", i+1, " of ", N**3)
    # compute log likelihood for each experiment

    # VR_1V experiment
    Thk, Res, Cur = forward_model_ford_new(forward_model['L'], forward_model['N'],
            forward_model['dt'], tFinal_1V, forward_model['Sigma'],
            forward_model['R_film0'], 1.0,
            10**(-theta_v[i,0]), theta_v[i,1], theta_v[i,2])

    Cur = Cur[model_inds_1V, 0]*1.6e-3
    Cur = np.reshape(Cur, (1, -1))

    Res = Res[model_inds_1V, 0]/1.6e-3
    Res = np.reshape(Res, (1, -1))

    data_filled = fill_data(data_1V_cur, Cur)
    log_like = np.sum(-0.5 * ((np.tile(Cur, (n_data_1V, 1)) - data_filled)**2 / np.tile(CD_1V_cur, (n_data_1V, 1))))

    data_filled = fill_data(data_1V_res, Res)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_1V, 1)) - data_filled)**2 / np.tile(CD_1V_res, (n_data_1V, 1))))

    # VR_0.5V experiment
    Thk, Res, Cur = forward_model_ford_new(forward_model['L'], forward_model['N'],
            forward_model['dt'], tFinal_0_5V, forward_model['Sigma'],
            forward_model['R_film0'], 0.5,
            10**(-theta_v[i,0]), theta_v[i,1], theta_v[i,2])

    Cur = Cur[model_inds_0_5V, 0]*1.6e-3
    Cur = np.reshape(Cur, (1, -1))

    Res = Res[model_inds_0_5V, 0]/1.6e-3
    Res = np.reshape(Res, (1, -1))

    data_filled = fill_data(data_0_5V_cur, Cur)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_0_5V, 1)) - data_filled)**2 / np.tile(CD_0_5V_cur, (n_data_0_5V, 1))))

    data_filled = fill_data(data_0_5V_res, Res)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_0_5V, 1)) - data_filled)**2 / np.tile(CD_0_5V_res, (n_data_0_5V, 1))))

    # VR_0.125V experiment
    Thk, Res, Cur = forward_model_ford_new(forward_model['L'], forward_model['N'],
            forward_model['dt'], tFinal_0_125V, forward_model['Sigma'],
            forward_model['R_film0'], 0.125,
            10**(-theta_v[i,0]), theta_v[i,1], theta_v[i,2])

    Cur = Cur[model_inds_0_125V, 0]*1.6e-3
    Cur = np.reshape(Cur, (1, -1))

    Res = Res[model_inds_0_125V, 0]/1.6e-3
    Res = np.reshape(Res, (1, -1))

    data_filled = fill_data(data_0_125V_cur, Cur)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_0_125V, 1)) - data_filled)**2 / np.tile(CD_0_125V_cur, (n_data_0_125V, 1))))

    data_filled = fill_data(data_0_125V_res, Res)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_0_125V, 1)) - data_filled)**2 / np.tile(CD_0_125V_res, (n_data_0_125V, 1))))

    # CC_1mA experiment
    Thk, Res, Cur, Vol = forward_model_cc_new(forward_model['L'], forward_model['N'],
            forward_model['dt'], tFinal_1mA, forward_model['Sigma'],
            forward_model['R_film0'], 10.0,
            10**(-theta_v[i,0]), theta_v[i,1], theta_v[i,2], 300)

    Cur = Cur[model_inds_1mA, 0]*1.6e-3
    Cur = np.reshape(Cur, (1, -1))

    Res = Res[model_inds_1mA, 0]/1.6e-3
    Res = np.reshape(Res, (1, -1))

    data_filled = fill_data(data_1mA_cur, Cur)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_1mA, 1)) - data_filled)**2 / np.tile(CD_1mA_cur, (n_data_1mA, 1))))

    data_filled = fill_data(data_1mA_res, Res)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_1mA, 1)) - data_filled)**2 / np.tile(CD_1mA_res, (n_data_1mA, 1))))

    # CC_0_75mA experiment
    Thk, Res, Cur, Vol = forward_model_cc_new(forward_model['L'], forward_model['N'],
            forward_model['dt'], tFinal_0_75mA, forward_model['Sigma'],
            forward_model['R_film0'], 7.5,
            10**(-theta_v[i,0]), theta_v[i,1], theta_v[i,2], 300)

    Cur = Cur[model_inds_0_75mA, 0]*1.6e-3
    Cur = np.reshape(Cur, (1, -1))

    Res = Res[model_inds_0_75mA, 0]/1.6e-3
    Res = np.reshape(Res, (1, -1))

    data_filled = fill_data(data_0_75mA_cur, Cur)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_0_75mA, 1)) - data_filled)**2 / np.tile(CD_0_75mA_cur, (n_data_0_75mA, 1))))

    data_filled = fill_data(data_0_75mA_res, Res)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_0_75mA, 1)) - data_filled)**2 / np.tile(CD_0_75mA_res, (n_data_0_75mA, 1))))

    # CC_0_5mA experiment
    Thk, Res, Cur, Vol = forward_model_cc_new(forward_model['L'], forward_model['N'],
            forward_model['dt'], tFinal_0_5mA, forward_model['Sigma'],
            forward_model['R_film0'], 5.0,
            10**(-theta_v[i,0]), theta_v[i,1], theta_v[i,2], 300)

    Cur = Cur[model_inds_0_5mA, 0]*1.6e-3
    Cur = np.reshape(Cur, (1, -1))

    Res = Res[model_inds_0_5mA, 0]/1.6e-3
    Res = np.reshape(Res, (1, -1))

    data_filled = fill_data(data_0_5mA_cur, Cur)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_0_5mA, 1)) - data_filled)**2 / np.tile(CD_0_5mA_cur, (n_data_0_5mA, 1))))

    data_filled = fill_data(data_0_5mA_res, Res)
    log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_0_5mA, 1)) - data_filled)**2 / np.tile(CD_0_5mA_res, (n_data_0_5mA, 1))))

    # log prior
    prior = np.sum(-0.5 * ((mu_prior - theta_v[i, :])**2 / var_prior))

    log_post[i] = log_like + prior
    log_like_v[i] = log_like

log_post = np.reshape(log_post, (N, N, N))
log_like_v = np.reshape(log_like_v, (N, N, N))

log_post_1 = np.sum(log_post, axis=(0, 1))
log_post_2 = np.sum(log_post, axis=(0, 2))
log_post_3 = np.sum(log_post, axis=(1, 2))

log_like_1 = np.sum(log_like_v, axis=(0, 1))
log_like_2 = np.sum(log_like_v, axis=(0, 2))
log_like_3 = np.sum(log_like_v, axis=(1, 2))

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(M1[0, 0, :], log_post_1)
plt.subplot(3, 1, 2)
plt.plot(M2[0, :, 0], log_post_2)
plt.subplot(3, 1, 3)
plt.plot(M3[:, 0, 0], log_post_3)
plt.savefig("./bayes_post_marginals"+str(N)+".png")


Thk, Res, Cur = forward_model_ford_new(forward_model['L'], forward_model['N'],
        forward_model['dt'], tFinal_1V, forward_model['Sigma'],
        forward_model['R_film0'], forward_model['VR'],
        10**(-7.43), 93, 0.214)

plt.figure(2)
plt.plot(data_1V_cur[0, :], label="Data")
plt.plot(Cur[model_inds_1V, 0]*1.6e-3, label = "Predictive")
plt.legend()
plt.savefig("./data.png")

plt.figure(3)
plt.plot(Cur[model_inds_1V, 0]*1.6e-3)
plt.savefig("./forward_model.png")

plt.figure(4)
plt.subplot(3, 1, 1)
plt.plot(M1[0, 0, :], log_like_1)
plt.subplot(3, 1, 2)
plt.plot(M2[0, :, 0], log_like_2)
plt.subplot(3, 1, 3)
plt.plot(M3[:, 0, 0], log_like_3)
plt.savefig("./log_like_marginals"+str(N)+".png")

with open('posterior'+str(N)+'.npy', 'wb') as f:
    np.save(f, log_post)
    np.save(f, theta_v)
    np.save(f, N)
