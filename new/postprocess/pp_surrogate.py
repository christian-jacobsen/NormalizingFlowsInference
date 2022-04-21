import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

from utils.utils import *

plt.close('all')

# define the saved surrogate path
surrogate_file = "SurrogateFNN_new1"
surrogate_path = "../models/surrogate_models/" + surrogate_file + ".pth"

# load the surrogate
Surrogate, training_loss, surrogate_model, training_params, forward_model, data = load_surrogate(surrogate_path)

# plot the training losses
plt.figure(1)
plt.semilogy(training_loss)
plt.xlabel('epoch')
plt.ylabel('MSE Error')
plt.title('Training Losses')
plt.savefig("surrogates/training_loss_" + surrogate_file + ".png")

# plot the training data
plt.figure(2)
output_data = data['output_data']
n = np.shape(output_data)[0]

for i in range(n):
    plt.plot(data['output_data'][i, :])

plt.savefig("surrogates/training_data_" + surrogate_file + ".png")
