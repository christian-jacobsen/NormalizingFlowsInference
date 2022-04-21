import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('../')

from inputs.inputs_nf import *
from models.normalizing_flow import *
from models.forward_models import *
from utils.utils import *



# load the surrogate model:
print("Loading Surrogate Model ==============================")
Surrogate, _, surrogate_config, surrogate_training, forward_model, surrogate_data = load_surrogate(surrogate_path) 
scales = surrogate_data['scales']
offset = surrogate_data['offset']


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
else:
    loaded_data = np.loadtxt("../data/experimental/VR_1V.txt")

    Cur = loaded_data[:, -2]
    Cur = np.reshape(Cur, (-1, 1))
    n_data_nf = 1
    data_nf = np.reshape(Cur, (1, -1))



dataset = TensorDataset(torch.tensor(data_nf).float(), torch.tensor(data_nf).float())
dataloader = DataLoader(dataset, batch_size = n_data_nf, shuffle = False)

max_output = surrogate_data['max_output']

if flow_params['use_exp_data']:
    C_D = torch.from_numpy(np.ones((1, np.shape(data_nf)[1])))  # / max_output)
else:
    C_D = torch.from_numpy(np.var(data_nf / max_output, axis = 0))


d = n_data_nf
b = training_params['MC_samples'] 

ForwardModel = Surrogate

NF = Flow(flow_params['D'], flow_params['flow_layers'], flow_params['mu0'], flow_params['logvar0'])
Optimizer = torch.optim.Adam(NF.parameters(), lr = 1e-3)

NF.train()

loss_vector = np.zeros((training_params['epochs'],))

R = training_params['R']
B = training_params['B']
R1 = 1

exp_lim = 1e0

print("Training the Flow =====================================")

surr_inds = np.arange(0, 51)
data_inds = np.arange(0, 501, 10)

if flow_params['use_exp_data']:
    C_D = C_D[0, data_inds]

for epoch in range(training_params['epochs']):
    for n, (_, out_d) in enumerate(dataloader):

        NF.zero_grad()

        _, _, _, z, lpz0, ld, lpzk = NF.forward(b, flow_params['mu_prior'], flow_params['var_prior'])

        '''
        inp = torch.zeros((b, flow_params['D']))

        inp[:, 0] = (z[:, 0] - offset[0, 0]) / scales[0, 0]
        inp[:, 1] = (z[:, 1] - offset[0, 1]) / scales[0, 1]
        inp[:, 2] = (z[:, 2] - offset[0, 2]) / scales[0, 2]
        '''

        pred = Surrogate(z)

        if flow_params['use_exp_data']:
            pred = pred[:, surr_inds]
            out_d = out_d[:, data_inds]

        lr = torch.sum(-0.5 * ((torch.tile(pred, (d, 1, 1)) - torch.tile(out_d / max_output, (b, 1, 1)).swapaxes(0, 1))**2 /
                    torch.tile(C_D, (d, b, 1))), (0, 2))

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

        L = - K*(torch.mean(lpzk) + R1 * torch.mean(lpz0 - ld)) - B * torch.mean(lr)

        loss = L

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

        if np.mod(epoch, training_params['epochs']//20) == 0:
            print('Epoch: ', epoch, ' Loss: ', loss_vector[epoch])


save_nf(NF.state_dict(), flow_params, training_params, loss_vector, data_nf, surrogate_path, save_nf_path)

