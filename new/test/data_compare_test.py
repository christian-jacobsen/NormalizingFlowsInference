# this test plots the forward models against the data given some input parameters
# to the forward models
#
# Model paramters given by inputs.inputs_surrogate
# inferred parameters specified in the script

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")

from inputs.inputs_surrogate import *
from models.forward_models import *
from utils.utils import *

# define the inferred parameters
Cv = 10**(-7.3351)
K = 44.42
jmin = 0.5993

# import all of the data for all experiments
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
data_0_125V_res, std_0_125V_res = load_data(load_path, "VR_0.125V_Resistance",n_data_0_125V)

n_data_0_5mA = 10
load_path = "../data/experimental/CC_0.5mA.xlsx"
data_0_5mA_cur, std_0_5mA_cur = load_data(load_path, "CC_0.5mA_Current", n_data_0_5mA)
data_0_5mA_res, std_0_5mA_res = load_data(load_path, "CC_0.5mA_Resistance", n_data_0_5mA)
data_0_5mA_vol, std_0_5mA_vol = load_data(load_path, "CC_0.5mA_Voltage", n_data_0_5mA)

n_data_0_75mA = 10
load_path = "../data/experimental/CC_0.75mA.xlsx"
data_0_75mA_cur, std_0_75mA_cur = load_data(load_path, "CC_0.75mA_Current", n_data_0_75mA)
data_0_75mA_res, std_0_75mA_res = load_data(load_path, "CC_0.75mA_Resistance",                 n_data_0_75mA)
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

# define time range for each experiment
tFinal_1V = 239
tFinal_0_5V = 477
tFinal_0_125V = 639
tFinal_0_5mA = 240
tFinal_0_75mA = 160
tFinal_1mA = 80

# indices to take all data and forward model output
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

# truncate the data with the indices above
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

# now compute forward models and plot over the data
plt.figure(1, figsize=(18, 12))

# VR_1V Experiment
Thk, Res, Cur = forward_model_ford_new(forward_model['L'], forward_model['N'],
             forward_model['dt'], tFinal_1V, forward_model['Sigma'],
             forward_model['R_film0'], 1.0,
             Cv, K, jmin)

Cur = Cur[model_inds_1V, 0]*1.6e-3
Cur = np.reshape(Cur, (1, -1))

Res = Res[model_inds_1V, 0]/1.6e-3 
Res = np.reshape(Res, (1, -1))

data_filled = fill_data(data_1V_cur, Cur)
log_like = np.sum(-0.5 * ((np.tile(Cur, (n_data_1V, 1)) - data_filled)**2 / np.tile(CD_1V_cur, (n_data_1V, 1))))

data_filled = fill_data(data_1V_res, Res)
log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_1V, 1)) - data_filled)**2 / np.tile(CD_1V_res, (n_data_1V, 1))))

# Current
plt.subplot(3, 4, 1)
plt.xlabel("Time")
plt.ylabel("Current (A)")
plt.title("VR 1V Experiment")
for i in range(np.shape(data_1V_cur)[0]):
    plt.plot(data_1V_cur[i,:], 'k.')
plt.plot(Cur[0,:], c='r', linewidth=2)

# Resistance
plt.subplot(3, 4, 2)
plt.xlabel("Time")
plt.ylabel("Resistance")
plt.title("VR 1V Experiment")
for i in range(np.shape(data_1V_res)[0]):
    plt.plot(data_1V_res[i,:], 'k.')
plt.plot(Res[0,:], c='r', linewidth=2)

# VR_0_5V Experiment
Thk, Res, Cur = forward_model_ford_new(forward_model['L'], forward_model['N'],
             forward_model['dt'], tFinal_0_5V, forward_model['Sigma'],
             forward_model['R_film0'], 0.5,
             Cv, K, jmin)

Cur = Cur[model_inds_0_5V, 0]*1.6e-3
Cur = np.reshape(Cur, (1, -1))

Res = Res[model_inds_0_5V, 0]/1.6e-3 
Res = np.reshape(Res, (1, -1))

data_filled = fill_data(data_0_5V_cur, Cur)
log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_0_5V, 1)) - data_filled)**2 / np.tile(CD_0_5V_cur, (n_data_0_5V, 1))))

data_filled = fill_data(data_0_5V_res, Res)
log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_0_5V, 1)) - data_filled)**2 / np.tile(CD_0_5V_res, (n_data_0_5V, 1))))

# Current
plt.subplot(3, 4, 3)
plt.xlabel("Time")
plt.ylabel("Current (A)")
plt.title("VR 0.5V Experiment")
for i in range(np.shape(data_0_5V_cur)[0]):
    plt.plot(data_0_5V_cur[i,:], 'k.')
plt.plot(Cur[0,:], c='r', linewidth=2)

# Resistance
plt.subplot(3, 4, 4)
plt.xlabel("Time")
plt.ylabel("Resistance")
plt.title("VR 0.5V Experiment")
for i in range(np.shape(data_0_5V_res)[0]):
    plt.plot(data_0_5V_res[i,:], 'k.')
plt.plot(Res[0,:], c='r', linewidth=2)

# VR_0_125V Experiment
Thk, Res, Cur = forward_model_ford_new(forward_model['L'], forward_model['N'],
             forward_model['dt'], tFinal_0_125V, forward_model['Sigma'],
             forward_model['R_film0'], 0.125,
             Cv, K, jmin)

Cur = Cur[model_inds_0_125V, 0]*1.6e-3
Cur = np.reshape(Cur, (1, -1))

Res = Res[model_inds_0_125V, 0]/1.6e-3 
Res = np.reshape(Res, (1, -1))

data_filled = fill_data(data_0_125V_cur, Cur)
log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_0_125V, 1)) - data_filled)**2 / np.tile(CD_0_125V_cur, (n_data_0_125V, 1))))

data_filled = fill_data(data_0_125V_res, Res)
log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_0_125V, 1)) - data_filled)**2 / np.tile(CD_0_125V_res, (n_data_0_125V, 1))))

# Current
plt.subplot(3, 4, 5)
plt.xlabel("Time")
plt.ylabel("Current (A)")
plt.title("VR 0.125V Experiment")
for i in range(np.shape(data_0_125V_cur)[0]):
    plt.plot(data_0_125V_cur[i,:], 'k.')
plt.plot(Cur[0,:], c='r', linewidth=2)

# Resistance
plt.subplot(3, 4, 6)
plt.xlabel("Time")
plt.ylabel("Resistance")
plt.title("VR 0.125V Experiment")
for i in range(np.shape(data_0_125V_res)[0]):
    plt.plot(data_0_125V_res[i,:], 'k.')
plt.plot(Res[0,:], c='r', linewidth=2)

# CC_1mA Experiment
Thk, Res, Cur, Vol = forward_model_cc_new(forward_model['L'], forward_model['N'],
             forward_model['dt'], tFinal_1mA, forward_model['Sigma'],
             forward_model['R_film0'], 10.0,
             Cv, K, jmin, 300)

Cur = Cur[model_inds_1mA, 0]*1.6e-3
Cur = np.reshape(Cur, (1, -1))

Res = Res[model_inds_1mA, 0]/1.6e-3 
Res = np.reshape(Res, (1, -1))

data_filled = fill_data(data_1mA_cur, Cur)
log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_1mA, 1)) - data_filled)**2 / np.tile(CD_1mA_cur, (n_data_1mA, 1))))

data_filled = fill_data(data_1mA_res, Res)
log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_1mA, 1)) - data_filled)**2 / np.tile(CD_1mA_res, (n_data_1mA, 1))))

# Current
plt.subplot(3, 4, 7)
plt.xlabel("Time")
plt.ylabel("Current (A)")
plt.title("CC 1mA Experiment")
for i in range(np.shape(data_1mA_cur)[0]):
    plt.plot(data_1mA_cur[i,:], 'k.')
plt.plot(Cur[0,:], c='r', linewidth=2)

# Resistance
plt.subplot(3, 4, 8)
plt.xlabel("Time")
plt.ylabel("Resistance")
plt.title("CC 1mA Experiment")
for i in range(np.shape(data_1mA_res)[0]):
    plt.plot(data_1mA_res[i,:], 'k.')
plt.plot(Res[0,:], c='r', linewidth=2)

# CC_0_75mA Experiment
Thk, Res, Cur, Vol = forward_model_cc_new(forward_model['L'], forward_model['N'],
             forward_model['dt'], tFinal_0_75mA, forward_model['Sigma'],
             forward_model['R_film0'], 7.5,
             Cv, K, jmin, 300)

Cur = Cur[model_inds_0_75mA, 0]*1.6e-3
Cur = np.reshape(Cur, (1, -1))

Res = Res[model_inds_0_75mA, 0]/1.6e-3 
Res = np.reshape(Res, (1, -1))

data_filled = fill_data(data_0_75mA_cur, Cur)
log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_0_75mA, 1)) - data_filled)**2 / np.tile(CD_0_75mA_cur, (n_data_0_75mA, 1))))

data_filled = fill_data(data_0_75mA_res, Res)
log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_0_75mA, 1)) - data_filled)**2 / np.tile(CD_0_75mA_res, (n_data_0_75mA, 1))))

# Current
plt.subplot(3, 4, 9)
plt.xlabel("Time")
plt.ylabel("Current (A)")
plt.title("CC 0.75mA Experiment")
for i in range(np.shape(data_0_75mA_cur)[0]):
    plt.plot(data_0_75mA_cur[i,:], 'k.')
plt.plot(Cur[0,:], c='r', linewidth=2)

# Resistance
plt.subplot(3, 4, 10)
plt.xlabel("Time")
plt.ylabel("Resistance")
plt.title("CC 0.75mA Experiment")
for i in range(np.shape(data_0_75mA_res)[0]):
    plt.plot(data_0_75mA_res[i,:], 'k.')
plt.plot(Res[0,:], c='r', linewidth=2)

# CC_0_5mA Experiment
Thk, Res, Cur, Vol = forward_model_cc_new(forward_model['L'], forward_model['N'],
             forward_model['dt'], tFinal_0_5mA, forward_model['Sigma'],
             forward_model['R_film0'], 5.0,
             Cv, K, jmin, 300)

Cur = Cur[model_inds_0_5mA, 0]*1.6e-3
Cur = np.reshape(Cur, (1, -1))

Res = Res[model_inds_0_5mA, 0]/1.6e-3 
Res = np.reshape(Res, (1, -1))

data_filled = fill_data(data_0_5mA_cur, Cur)
log_like = log_like + np.sum(-0.5 * ((np.tile(Cur, (n_data_0_5mA, 1)) - data_filled)**2 / np.tile(CD_0_5mA_cur, (n_data_0_5mA, 1))))

data_filled = fill_data(data_0_5mA_res, Res)
log_like = log_like + np.sum(-0.5 * ((np.tile(Res, (n_data_0_5mA, 1)) - data_filled)**2 / np.tile(CD_0_5mA_res, (n_data_0_5mA, 1))))

print('Log-likelihood: ', log_like)

# Current
plt.subplot(3, 4, 11)
plt.xlabel("Time")
plt.ylabel("Current (A)")
plt.title("CC 0.5mA Experiment")
for i in range(np.shape(data_0_5mA_cur)[0]):
    plt.plot(data_0_5mA_cur[i,:], 'k.')
plt.plot(Cur[0,:], c='r', linewidth=2)

# Resistance
plt.subplot(3, 4, 12)
plt.xlabel("Time")
plt.ylabel("Resistance")
plt.title("CC 0.5mA Experiment")
for i in range(np.shape(data_0_5mA_res)[0]):
    plt.plot(data_0_5mA_res[i,:], 'k.')
plt.plot(Res[0,:], c='r', linewidth=2)

plt.savefig("data_model_comparison.png")

# plt.show()










