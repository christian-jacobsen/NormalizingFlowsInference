import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('../')

from inputs.inputs_surrogate import *
from models.surrogates import *
from models.forward_models import *
from utils.utils import *

# first load or generate the training data

if not generate_data:
    Data = np.load(data_path)

else:
        
    print("Generating Training Data =============================")
    samps = np.random.uniform(0, 1, (n_data_sur, surrogate_model['input_size']))
    scales = np.ones((1, surrogate_model['input_size']))
    scales[0, 0] = 5  # -log Cv
    scales[0, 1] = 20  # K
    scales[0, 2] = 5  # jmin
    offset = np.zeros((1, 3))
    offset[0, 0] = 4.5
    offset[0, 1] = 27.0
    samps = samps * scales + offset

    output_data = np.zeros((n_data_sur, surrogate_model['output_size']))

    for i in range(n_data_sur):
        if forward_model['name'] == 'forward_model_ford_old':
            T, R, C = forward_model_ford_old(forward_model['L'], forward_model['N'],
                              forward_model['dt'], forward_model['tFinal'], forward_model['Sigma'],
                              forward_model['R_film0'], forward_model['VR'], 
                              10**(-samps[i, 0]), samps[i, 1], samps[i, 2])
            output_data[i, :] = C[:, 0]

        elif forward_model['name'] == 'forward_model_ford_new':
            T, R, C = forward_model_ford_new(forward_model['L'], forward_model['N'],
                              forward_model['dt'], forward_model['tFinal'], forward_model['Sigma'],
                              forward_model['R_film0'], forward_model['VR'], 
                              10**(-samps[i, 0]), samps[i, 1], samps[i, 2])
            output_data[i, :] = C[:, 0]


    output_data = np.reshape(output_data, (surrogate_model['output_size'], n_data_sur, 1))
    input_data = (samps - offset) / scales

    input_data = np.tile(input_data, (surrogate_model['output_size'], 1, 1))

    max_output = np.max(output_data)
    output_data = output_data / max_output


    input_data = torch.from_numpy(input_data).float()
    output_data = torch.from_numpy(output_data).float()

    if surrogate_model['name'] == 'FNN':
        input_data = input_data[0,:,:]
        output_data = np.reshape(output_data, (n_data_sur, surrogate_model['output_size']))

    data_save = {'output_data': output_data,
                 'max_output': max_output,
                 'scales': scales,
                 'offset': offset,
                 'input_data': input_data}

    dataset = TensorDataset(input_data, output_data)
    data_loader = DataLoader(dataset, batch_size = n_data_sur // 10, shuffle = True)


# Now train the surrogate model

if surrogate_model['name'] == 'FNN':
    Surrogate = SurrogateFNN(surrogate_model['input_size'], surrogate_model['output_size'],
                             surrogate_model['n_layers'], surrogate_model['n_nodes'], 
                             act = surrogate_model['act'])
else:
    print('No valid surrogate model specified')

if training_params['optimizer'] == 'Adam':
    Optimizer = torch.optim.Adam(Surrogate.parameters(), lr = training_params['lr'])
else:
    print('No valid optimizer specified')


loss_vector = np.zeros(training_params['epochs'])
print("Training Surrogate Model =============================")
for epoch in range(training_params['epochs']):
    for n, (in_d, out_d) in enumerate(data_loader):
        if epoch > 1e5:
            Optimizer.param_groups[0]['lr'] = 5e-4
        elif epoch > 1.5e5:
            Optimizer.param_groups[0]['lr'] = 1e-4

        Surrogate.zero_grad()
        pred = Surrogate.forward(in_d)

        loss = torch.mean((out_d - pred)**2)
        loss.backward()
        Optimizer.step()

        
    loss_vector[epoch] = loss.cpu().detach().numpy()

    if epoch % (training_params['epochs']//10) == 0:
        print("Epoch: ", epoch, " Loss: ", loss)


# Save the trained surrogate model
if surrogate_model['name'] == 'FNN':
   save_surrogate(Surrogate.state_dict(), surrogate_model, forward_model, training_params, 
           loss_vector, data_save, save_path_surrogate, surrogate_name)
