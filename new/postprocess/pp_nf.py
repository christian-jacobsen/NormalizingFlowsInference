# postprocess a trained flow

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../')

from utils.utils import *

plt.close('all')

# define the path to the model to evaluate
load_path = "../models/nf_models/"
model = "NF_ALLEXP_1"

NF, training_loss, flow_params, training_params, data, surrogate_path = load_nf(load_path + model + ".pth")
Surrogate, _, _, _, _, surr_data = load_surrogate(surrogate_path)
scales = surr_data['scales']
offset = surr_data['offset']

# sample from the posterior and plot histograms
n_samples_post = 50000

_, _, _, z, _, _, _ = NF.forward(n_samples_post, flow_params['mu_prior'], flow_params['var_prior'])


D = flow_params['D']
nbins = 100

l_lim = np.array([6.95, 36.75, 0.95])
r_lim = np.array([7.05, 37.25, 1.05])

scales = np.array([0.44, 0.44, 66])
offset = np.array([1, 7, 150])

z = z.cpu().detach().numpy()

plt.figure(figsize=(8,14))
for i in range(D):
    plt.subplot(D, 1, i + 1)
    #plt.xlim(l_lim[i], r_lim[i])
    #plot_z = z[z[:,i] < r_lim[i],i]
    #plot_z = plot_z[plot_z > l_lim[i]]
    plt.hist(z[:, i]*scales[i] + offset[i], bins = nbins)
    plt.title("Posterior Samples Histogram")

plt.savefig("flows/posterior_samples_" + model + ".png")

print(z[:,1])

# plot training losses
plt.figure(2)
#plt.plot(training_loss)
plt.semilogy(training_loss)
plt.xlabel('epoch')
plt.ylabel('NF Loss')
plt.savefig("flows/training_loss_" + model + ".png")

# plot the surrogate model output at the posterior mean
post_mean = torch.tensor(np.mean(z, axis = 0))
#post_mean = torch.tensor([0.5, 0.5, 0.5])
post_pred = Surrogate(post_mean)
plt.figure(3)
plt.plot(post_pred.cpu().detach().numpy())
plt.savefig("flows/posterior_predictive_" + model + ".png")


# plot the inference data 
n = np.shape(data)[0]

plt.figure(4)
for i in range(n):
    plt.plot(data[i, :])

plt.savefig("flows/inference_data_" + model + ".png")

# plot correlations
plt.figure(figsize=(8,14))

plt.subplot(3, 1, 1)
plt.plot(z[:,0]*scales[0,0]+offset[0,0], z[:,1]*scales[0,1]+offset[0,1], 'k.')
plt.xlabel('-log Cv')
plt.ylabel('K')
plt.title('Posterior Joint Samples')

plt.subplot(3, 1, 2)
plt.plot(z[:,0]*scales[0,0]+offset[0,0], z[:,2]*scales[0,2]+offset[0,2], 'k.')
plt.xlabel('-log Cv')
plt.ylabel('jmin')
plt.title('Posterior Joint Samples')

plt.subplot(3, 1, 3)
plt.plot(z[:,1]*scales[0,1]+offset[0,1], z[:,2]*scales[0,2]+offset[0,2], 'k.')
plt.xlabel('K')
plt.ylabel('jmin')
plt.title('Posterior Joint Samples')

plt.savefig("flows/joints_" + model + ".png")



