# this file details the hyperparameters of the normalizing flow


import torch
import torch.nn as nn

# Define the training data

generate_data = True  # True = generate artificial data from forward_model
                      # False = Load data from data_path
n_data_sur = 500
data_path = './'

# define the forward model for artificial data generation

forward_model = {'name': 'forward_model_ford_new',
                 'L': 0.0254,
                 'N': 11,
                 'dt': 0.01,
                 'tFinal': 50,
                 'Sigma': 0.14,
                 'R_film0': 0.5,
                 'VR': 1,
                 'Cv': 1e-7,
                 'Qmin': 150,
                 'jmin': 1,
                 }

# define the surrogate model

surrogate_model = {'name': 'FNN',    # 'FNN' or 'LSTM'
                   'input_size': 3,  # number of parameters to infer 
                   'output_size': forward_model['tFinal'] + 1,
                   'n_layers': 5, 
                   'n_nodes': 64,
                   'act': nn.SiLU()
                   }

# define the training parameters

training_params = {'optimizer': 'Adam',
                   'lr': 1e-3,
                   'loss': 'MSE',
                   'epochs': int(2e5)
                   }

surrogate_name = "SurrogateFNN_new_current1.pth"
# save the data

# save the surrogate model
save_path_surrogate = '../models/surrogate_models'





