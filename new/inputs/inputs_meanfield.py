import torch
import numpy as np

meanfield_config = {'mu_prior': torch.tensor([7.0, 37.0, 1.0]),
                    'logvar_prior': torch.tensor([np.log(1**2), np.log(1**2), np.log(1**2)]),
                    'MC_samples': 1,
                    'generate_data': True,
                    'true_params': torch.tensor([7.0, 37.0, 1.0]) # only used if generating data
                    }

training_config = {'epochs': int(1e1)}
