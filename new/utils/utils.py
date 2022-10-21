import torch
import sys
import pandas as pd
import numpy as np
import os.path as osp

sys.path.append('../')

from models.surrogates import *
from models.normalizing_flow import *
from models.gaussian_vi import *

def load_data(load_path, sheet, n_data):
    # load data from .xlsx at load_path with sheet_name
    dataM = pd.read_excel(load_path, sheet_name=sheet)
    dataM = np.array(dataM)

    data = np.transpose(dataM[:, 5:5+n_data])
    std = np.reshape(dataM[:, 2], (1, -1))

    return data, std

def load_all_data_torch(load_path):
    # experiments are: VR: 1V, 0.5V, 0.125V
    #                  CC: 1mA, 0.75mA, 0.5mA
    # define number of trials in each experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_data = torch.tensor([13, 12, 12, 10, 10, 10])
    t_final = torch.tensor([239, 477, 639, 240, 160, 80])
    load_paths = [osp.join(load_path, 'VR_1V.xlsx'),
                  osp.join(load_path, 'VR_0.5V.xlsx'),
                  osp.join(load_path, 'VR_0.125V.xlsx'),
                  osp.join(load_path, 'CC_1mA.xlsx'),
                  osp.join(load_path, 'CC_0.75mA.xlsx'),
                  osp.join(load_path, 'CC_0.5mA.xlsx')]
    sheet_names_cur = ['VR_1V_Current', 'VR_0.5V_Current', 'VR_0.125V_Current',
                       'CC_1mA_Current', 'CC_0.75mA_Current', 'CC_0.5mA_Current']
    sheet_names_res = ['VR_1V_Resistance', 'VR_0.5V_Resistance', 'VR_0.125V_Resistance',
                       'CC_1mA_Current', 'CC_0.75mA_Current', 'CC_0.5mA_Current']

    mean_cur = []
    var_cur = []
    mean_res = []
    var_res = []

    for i in range(6):
        # load the current data
        m_temp, sd_temp = load_data(load_paths[i], sheet_names_cur[i], n_data[i])
        sd_temp = sd_temp**2
        mean_cur.append(torch.from_numpy(m_temp[:, 0:(10*int(t_final[i]) + 1)]).float().to(device))
        var_cur.append(torch.from_numpy(sd_temp[:, 0:(10*int(t_final[i]) + 1)]).float().to(device))
        # load the resistance data
        m_temp, sd_temp = load_data(load_paths[i], sheet_names_res[i], n_data[i])
        sd_temp = sd_temp**2
        mean_res.append(torch.from_numpy(m_temp[:, 0:(10*int(t_final[i]) + 1)]).float().to(device))
        var_res.append(torch.from_numpy(sd_temp[:, 0:(10*int(t_final[i]) + 1)]).float().to(device))

    return mean_cur, var_cur, mean_res, var_res, t_final, n_data

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

def save_vi(nf_state_dict, flow_params, training_params, training_loss, nf_data, surrogate_path, save_path):
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

def load_vi(file_path):
    config = torch.load(file_path)

    flow_params = config['flow_params']
    NF = GaussianVI(flow_params['D'])
    NF.load_state_dict(config['flow_state_dict'])
    training_params = config['training_params']
    training_loss = config['training_loss']
    data = config['data']
    surrogate_path = config['surrogate_path']

    return NF, training_loss, flow_params, training_params, data, surrogate_path

def fill_data(data, pred):
    # this function deals with the fact that we have different ending times for each experiment 
    # Any missing data is filled with the prediction data such that it will not effect the likelihood

    n_data = np.shape(data)[0]
    data_filled = data + 0.

    for i in range(n_data):
        data_filled[i, np.isnan(data[i, :])] = pred[0, np.isnan(data[i, :])]

    return data_filled

def fill_data_torch(data, pred):
    # this function deals with the fact that we have different ending times for each experiment 
    # Any missing data is filled with the prediction data such that it will not effect the likelihood

    n_data = np.shape(data)[0]
    data_filled = data + 0.

    for i in range(n_data):
        data_filled[i, torch.isnan(data[i, :])] = pred[0, torch.isnan(data[i, :])]

    return data_filled

def gaussian_pdf(x, mu, logv):
    return 1/np.sqrt(2*np.pi*np.exp(logv))*np.exp(-0.5*(x-mu)**2/np.exp(logv))
