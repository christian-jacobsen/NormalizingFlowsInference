# this file details the hyperparameters of the normalizing flow

import torch
import numpy as np

D = 3

# define the initial distribution (before the flow)
mu0 = torch.ones((1, D))
mu0[0, 0] = 0
mu0[0, 1] = 0
mu0[0, 2] = 0
logvar0 = torch.zeros((1, D))
logvar0[0, 0] = np.log(1**2)
logvar0[0, 1] = np.log(1**2)
logvar0[0, 2] = np.log(1**2)

mu_prior = torch.Tensor([0, 0, 0])
var_prior = torch.Tensor([1, 1, 1])

n_data_nf = 10 # number of data to generate for training the flow
sig_thk = 8e-8 # thickness data noise

flow_params = {'flow_layers': 10,
               'D': D,
               'mu0': mu0,
               'logvar0': logvar0,
               'mu_prior': mu_prior,
               'var_prior': var_prior,
               'true_params': np.array([1e-7, 50, 1.0]),  #parameters to infer
               'use_exp_data': True,  # False --> generate synthetic data. True --> use experimental data
               'use_thk_data': False,  # Use thickness data in inference? (for synthetic or experimental data)
               'use_res_data': False,  # Use resistance data?
               'use_cur_data': True    # Use current data?
               }

training_params = {'epochs': 4000,
                   'MC_samples': 10,  # number of MC samples to approximate expectation
                   'use_surrogate': True,  # train with surrogate as forward model
                   'R': 1e0,  # greater R increases entropy of posterior
                   'B': 1e0  # greater B increases reconstruction accuracy
                   }

surrogate_path = "../models/surrogate_models/SurrogateFNN_new_current1.pth"  # path to load surrogate from
save_nf_path = "../models/vi_models/VI_ALLEXP_NEW_3.pth"
