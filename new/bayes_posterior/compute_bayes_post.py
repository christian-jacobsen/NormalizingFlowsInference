import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("../")

from models.forward_models import *
from inputs.inputs_surrogate import *
from utils.utils import *


# import our data
load_path = "../data/experimental/VR_1V.xlsx"
data_VR1V = pd.read_excel(load_path)
data_VR1V = np.array(data_VR1V)
#print(np.shape(data_VR1V))
data = np.transpose(data_VR1V[:, 5:18])
std_data = np.reshape(data_VR1V[:, 2], (1, -1))
n_data = 13
tFinal = 239

'''
loaded_data = np.loadtxt("../data/experimental/VR_1V.txt")

Cur = loaded_data[:, -2]
Cur = np.reshape(Cur, (-1, 1))
n_data = 1
data = np.reshape(Cur, (1, -1))
'''

# define the data variance
C_D = std_data**2#np.ones((1, np.shape(data)[1]))

# define the prior
mu_prior = np.array([7, 37, 2.5])
var_prior = 0.25**2 * np.array([5**2, 20**2, 5**2])

# define the parameter domain
N = 20  # number of points in each dimension
parameters_lower_bound = np.array([6, 0, 0])#[6.5, 36, 2])
parameters_upper_bound = np.array([8.5, 50, 3])#[7.5, 37.5, 2.5])

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
data_inds = np.arange(0, 10*tFinal+1)
model_inds = np.arange(0, 10*tFinal+1)

data = data[:, data_inds]
C_D = C_D[:, data_inds]

print("Starting Inference with ", N**3, " grid points: ")

for i in range(n_params):
    print("Progress: ", i+1, " of ", N**3)
    Thk, Res, Cur = forward_model_ford_new(forward_model['L'], forward_model['N'],
            forward_model['dt'], tFinal, forward_model['Sigma'],
            forward_model['R_film0'], forward_model['VR'],
            10**(-theta_v[i,0]), theta_v[i,1], theta_v[i,2])

    Cur = Cur[model_inds, 0]*1.6e-3
    Cur = np.reshape(Cur, (1, -1))

    data_filled = fill_data(data, Cur)
    #print(data_filled)
    log_like = np.sum(-0.5 * ((np.tile(Cur, (n_data, 1)) - data_filled)**2 / np.tile(C_D, (n_data, 1))))
    #print(np.tile(Cur, (n_data, 1)))
    #print(data_filled)
    #print(np.tile(C_D, (n_data, 1)))
    prior = np.sum(-0.5 * ((mu_prior - theta_v[i, :])**2 / var_prior))

    log_post[i] = log_like  # + prior
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
        forward_model['dt'], tFinal, forward_model['Sigma'],
        forward_model['R_film0'], forward_model['VR'],
        10**(-7.41), 60, 0.154)

plt.figure(2)
plt.plot(data[0, :], label="Data")
plt.plot(Cur[model_inds, 0]*1.6e-3, label = "Predictive")
plt.legend()
plt.savefig("./data.png")

plt.figure(3)
plt.plot(Cur[model_inds, 0]*1.6e-3)
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
