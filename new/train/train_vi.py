import sys
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('../')

from inputs.inputs_nf import *
from models.gaussian_vi import *
from models.forward_models import *
from models.surrogates import *
from utils.utils import *


'''
# load the surrogate model:
print("Loading Surrogate Model ==============================")
Surrogate, _, surrogate_config, surrogate_training, forward_model, surrogate_data = load_surrogate(surrogate_path) 
scales = surrogate_data['scales']
offset = surrogate_data['offset']
'''


# generate data for the NF (different from the surrogate)

if not flow_params['use_exp_data']:
    print("Generating Infernce Data =============================")

    if forward_model['name'] == 'forward_model_ford_old':
        print("     Generating data for old model ")
        Thk, Res, Cur = forward_model_ford_old(forward_model['L'], forward_model['N'], 
               forward_model['dt'], forward_model['tFinal'], forward_model['Sigma'],
               forward_model['R_film0'], forward_model['VR'],
               flow_params['true_params'][0], flow_params['true_params'][1], flow_params['true_params'][2])
    elif forward_model['name'] == 'forward_model_ford_new':
        print("     Generating data for new model ")
        Thk, Res, Cur = forward_model_ford_new(forward_model['L'], forward_model['N'], 
               forward_model['dt'], forward_model['tFinal'], forward_model['Sigma'],
               forward_model['R_film0'], forward_model['VR'],
               flow_params['true_params'][0], flow_params['true_params'][1], flow_params['true_params'][2])

    data_nf = np.zeros((n_data_nf, surrogate_config['output_size']))
    for i in range(n_data_nf):
        data_nf[i, :] = Cur[:, 0] + np.random.normal(0, sig_thk, surrogate_config['output_size'])

# load ALL experimental data
else:
    # load the saved surrogate models
    surr_1V_cur = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_current_1.p", "rb" ) )
    surr_0_5V_cur = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_current_05.p", "rb") )
    surr_0_125V_cur = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_current_0125.p", "rb") )
    surr_1V_res = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_resistance_1.p", "rb" ) )
    surr_0_5V_res = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_resistance_05.p", "rb") )
    surr_0_125V_res = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_resistance_0125.p", "rb") )

    surr_1mA_cur = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_current_cc_1.p", "rb" ) )
    surr_0_75mA_cur = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_current_cc_075.p", "rb" ) )
    surr_0_5mA_cur = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_current_cc_05.p", "rb" ) )
    surr_1mA_res = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_resistance_cc_1.p", "rb" ) )
    surr_0_75mA_res = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_resistance_cc_075.p", "rb" ) )
    surr_0_5mA_res = pickle.load( open( "../models/surrogate_models/New_Forward_Models/model_resistance_cc_05.p", "rb" ) ) 
    '''
    surr_1V_cur = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Current_Old_1_2.p", "rb" ) )
    surr_0_5V_cur = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Current_Old_05_2.p", "rb") )
    surr_0_125V_cur = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Current_Old_0125_2.p", "rb") )
    surr_1V_res = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Resistance_Old_1_2.p", "rb" ) )
    surr_0_5V_res = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Resistance_Old_05_2.p", "rb") )
    surr_0_125V_res = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Resistance_Old_0125_2.p", "rb") )

    surr_1mA_cur = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Current_CC_Old_1_2.p", "rb" ) )
    surr_0_75mA_cur = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Current_CC_Old_075_2.p", "rb" ) )
    surr_0_5mA_cur = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Current_CC_Old_05_2.p", "rb" ) )
    surr_1mA_res = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Resistance_CC_Old_1_2.p", "rb" ) )
    surr_0_75mA_res = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Resistance_CC_Old_075_2.p", "rb" ) )
    surr_0_5mA_res = pickle.load( open( "../models/surrogate_models/Old_Forward_Models/model_Resistance_CC_Old_05_2.p", "rb" ) ) 
    '''
    surr_1V_cur.eval()
    surr_0_5V_cur.eval()
    surr_0_125V_cur.eval()
    surr_1V_res.eval()
    surr_0_5V_res.eval()
    surr_0_125V_res.eval()
    surr_1mA_cur.eval()
    surr_0_75mA_cur.eval()
    surr_0_5mA_cur.eval()
    surr_1mA_res.eval()
    surr_0_75mA_res.eval()
    surr_0_5mA_res.eval()

    # load all the data now
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

    '''
    Cur = loaded_data[:, -2]
    Cur = np.reshape(Cur, (-1, 1))
    n_data_nf = 1
    data_nf = np.reshape(Cur, (1, -1))
    '''

'''
dataset = TensorDataset(torch.tensor(data_nf).float(), torch.tensor(data_nf).float())
dataloader = DataLoader(dataset, batch_size = n_data_nf, shuffle = False)
'''

#max_output = surrogate_data['max_output']

'''
if flow_params['use_exp_data']:
    C_D = torch.from_numpy(np.ones((1, np.shape(data_nf)[1])))  # / max_output)
else:
    C_D = torch.from_numpy(np.var(data_nf / max_output, axis = 0))
'''


d = n_data_nf
b = training_params['MC_samples'] 

#ForwardModel = Surrogate

NF = GaussianVI(flow_params['D'])
Optimizer = torch.optim.Adam(NF.parameters(), lr = 1e-1)

NF.train()

loss_vector = np.zeros((training_params['epochs'],))

R = training_params['R']
B = training_params['B']
R1 = 1

exp_lim = 1e0

print("Training the Flow =====================================")

surr_inds = np.arange(0, 51)
data_inds = np.arange(0, 501, 10)

scales = torch.tensor([0.44, 0.44, 22])
offsets = torch.tensor([1, 7, 50])
perm = [2, 0, 1]

'''
if flow_params['use_exp_data']:
    C_D = C_D[0, data_inds]
'''
mup = torch.Tensor([0.63, 7.32, 44.2])
logvarp = torch.Tensor([np.log(0.1**2), np.log(0.5**2), np.log(3**2)])

for epoch in range(training_params['epochs']):
    for i in range(1):#n, (_, out_d) in enumerate(dataloader):

        NF.zero_grad()

        lr = 0

        zmu, zlogvar, z, lpz0 = NF.forward(1)

        p_in = (z * torch.ones((10*tFinal_0_125V + 1, 3)) - offsets ) / scales

        # 1VR Experiment

        t_in = (torch.arange(0, tFinal_1V+0.1, 0.1).reshape((-1, 1)).float() - 150.48703708) / 86.60500884
        Cur = 1e-3*surr_1V_cur(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))
        Res = 1e2*surr_1V_res(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))

        data_filled = fill_data_torch(data_1V_cur, Cur)
        log_like = torch.sum(-0.5 * ((torch.tile(Cur, (n_data_1V, 1)) - data_filled)**2 / torch.tile(CD_1V_cur, (n_data_1V, 1))))

        data_filled = fill_data_torch(data_1V_res, Res)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Res, (n_data_1V, 1)) - data_filled)**2 / torch.tile(CD_1V_res, (n_data_1V, 1))))

        # 0.5VR Experiment
        t_in = (torch.arange(0, tFinal_0_5V+0.1, 0.1).reshape((-1, 1)).float() - 300.515167) / 173.204833
        Cur = 1e-3*surr_0_5V_cur(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))
        Res = 1e2*surr_0_5V_res(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))
        
        data_filled = fill_data_torch(data_0_5V_cur, Cur)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Cur, (n_data_0_5V, 1)) - data_filled)**2 / torch.tile(CD_0_5V_cur, (n_data_0_5V, 1))))

        data_filled = fill_data_torch(data_0_5V_res, Res)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Res, (n_data_0_5V, 1)) - data_filled)**2 / torch.tile(CD_0_5V_res, (n_data_0_5V, 1))))

        # 0.125VR Experiment
        t_in = (torch.arange(0, tFinal_0_125V+0.1, 0.1).reshape((-1, 1)).float() - 400.470386) / 230.953803
        Cur = 1e-3*surr_0_125V_cur(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))
        Res = 1e2*surr_0_125V_res(torch.hstack((p_in[:len(t_in), :], t_in))).reshape((1, -1))

        data_filled = fill_data_torch(data_0_125V_cur, Cur)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Cur, (n_data_0_125V, 1)) - data_filled)**2 / torch.tile(CD_0_125V_cur, (n_data_0_125V, 1))))

        data_filled = fill_data_torch(data_0_125V_res, Res)
        log_like = log_like + torch.sum(-0.5 * ((torch.tile(Res, (n_data_0_125V, 1)) - data_filled)**2 / torch.tile(CD_0_125V_res, (n_data_0_125V, 1))))

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
        # trains on only recon loss first
        if epoch < training_params['epochs']//2:
            K = 0
        else:
            K = 1

        # slowly increase the entropy weighting
        if epoch < (training_params['epochs']//2 + training_params['epochs']//6) and \
           epoch > (training_params['epochs']//2):
            R1 = R1 + 0.1
            if R1 > R:
                R1 = R
        elif epoch > (training_params['epochs']//2 + training_params['epochs']//6):
            R1 = R

        K = 1
        '''

        if epoch < 1000:
            L = 0.5*(torch.sum(logvarp) - 3 - torch.sum(zlogvar) + torch.sum(torch.exp(zlogvar)/torch.exp(logvarp)) + torch.sum((zmu - mup)**2/torch.exp(logvarp)))
        else:
            for g in Optimizer.param_groups:
                g['lr'] = 1e-2

            L = 0.5*(torch.sum(logvarp) - 3 - torch.sum(zlogvar) + torch.sum(torch.exp(zlogvar)/torch.exp(logvarp)) + torch.sum((zmu - mup)**2/torch.exp(logvarp))) - torch.mean(log_like)

        loss = L  # + ...

        if torch.isnan(loss):
            print("Loss is nan!")
            print("LR: ", torch.mean(lr))
            print("LPZK: ", torch.mean(lpzk))
            print("LPZ0: ", torch.mean(lpz0))
            print("LogDet: ", torch.mean(ld))

            sys.exit("Stoped Training")


        '''
        + torch.sum(torch.exp(exp_lim*(z[:,0] - 1)) + torch.exp(-exp_lim*z[:,0]) + \
                   torch.exp(exp_lim*(z[:,1] - 1)) + torch.exp(-exp_lim*z[:,1]) + \
                   torch.exp(exp_lim*(z[:,2] - 1)) + torch.exp(-exp_lim*z[:,2]))
        '''

        loss.backward()
        Optimizer.step()

        loss_vector[epoch] = loss.cpu().detach().numpy()

        # if np.mod(epoch, training_params['epochs']//20) == 0:
        if np.mod(epoch, 50) == 0:
            print('Epoch: ', epoch, ' Loss: ', loss_vector[epoch])

data_nf = 0.
save_vi(NF.state_dict(), flow_params, training_params, loss_vector, data_nf, surrogate_path, save_nf_path)

