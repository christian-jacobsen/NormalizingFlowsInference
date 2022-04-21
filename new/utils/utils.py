import torch
import sys

sys.path.append('../')

from models.surrogates import *
from models.normalizing_flow import *

def save_surrogate(surrogate_params, surrogate_model, forward_model, training_params, training_loss, data, save_dir, file_name):
    # save the surrogate model with parameters surrogate_params and configuration
    # surrogate_model
    # save the parameters and data it was trained with
    # save the training losses
    # save at save_dir

    config = {'surrogate_model': surrogate_model,
              'surrogate_state_dict': surrogate_params,
              'forward_model': forward_model,
              'training_loss': training_loss,
              'training_params': training_params,
              'data': data}
    torch.save(config, save_dir + '/' + file_name)


def save_nf(nf_state_dict, flow_params, training_params, training_loss, nf_data, surrogate_path, save_path):
    # save the normalizing flow model
    
    config = {'flow_params': flow_params,
              'flow_state_dict': nf_state_dict,
              'training_params': training_params,
              'training_loss': training_loss,
              'data': nf_data,
              'surrogate_path': surrogate_path
              }

    torch.save(config, save_path)


def load_surrogate(file_path):
    config = torch.load(file_path)
    
    surrogate_model = config['surrogate_model']  # load the surrogate configuration
    training_loss = config['training_loss']  # load the training losses
    training_params = config['training_params']  # load the training parameters
    forward_model = config['forward_model']
    data = config['data']

    if surrogate_model['name'] == 'FNN':
        Surrogate = SurrogateFNN(surrogate_model['input_size'], surrogate_model['output_size'],
                                    surrogate_model['n_layers'], surrogate_model['n_nodes'],
                                    act = surrogate_model['act'])

    Surrogate.load_state_dict(config['surrogate_state_dict'])

    return Surrogate, training_loss, surrogate_model, training_params, forward_model, data


def load_nf(file_path):
    config = torch.load(file_path)

    flow_params = config['flow_params']
    NF = Flow(flow_params['D'], flow_params['flow_layers'], flow_params['mu0'], flow_params['logvar0'])
    NF.load_state_dict(config['flow_state_dict'])
    training_params = config['training_params']
    training_loss = config['training_loss']
    data = config['data']
    surrogate_path = config['surrogate_path']

    return NF, training_loss, flow_params, training_params, data, surrogate_path