import sys, torch, time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(".")

from functional_inference_model import *

n_rho_layers = 2
n_rho_nodes = 5

model = FunctionalInferenceECOAT(n_rho_layers, n_rho_nodes)

'''
Cv = torch.tensor([[1e-7], [1.2e-7], [0.9e-7]])
K = torch.tensor([[37.0], [38.0], [37.5]])
jmin = torch.tensor([[1.0], [1.1], [0.95]])
'''
n = 2
Cv = torch.ones((n, 1))*1e-7
K = torch.ones((n, 1))*37
jmin = torch.ones((n, 1))*1.0

t1 = time.time()
thk2, res2, cur2, tv2 = model.forward_model(0.0254, 50, 0.1, 0.14, 0.5, Cv, K, jmin, 'VR', VR=0.5)
t2 = time.time()
thk, res, cur, tv = model.forward_model(0.0254, 50, 0.1, 0.14, 0.5, Cv, K, jmin, 'VR', VR=1)
t3 = time.time()
#_, _, _, _, thk, res, cur ,tv = model(12, 0.0254, 11, 50, 0.1, 0.14, 0.5, 'VR')

print("time 0.01: ", t2 - t1)
print("time 0.1 : ", t3 - t2)

plt.figure(1)
plt.subplot(1, 3, 1)
plt.plot(tv[0, :], cur2[0, :, 0].detach().numpy())
plt.plot(tv2[0, :], cur[0, :, 0].detach().numpy())
plt.subplot(1, 3, 2)
plt.plot(tv[0, :], res2[0, :, 0].detach().numpy())
plt.plot(tv2[0, :], res[0, :, 0].detach().numpy())
plt.subplot(1, 3, 3)
plt.plot(tv[0, :], thk2[0, :, 0].detach().numpy())
plt.plot(tv2[0, :], thk[0, :, 0].detach().numpy())
plt.show()
