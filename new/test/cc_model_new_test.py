import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")

from models.forward_models import *
from inputs.inputs_surrogate import *

tF = 300 # tFinal

Thk, Res, Cur, VAn = forward_model_cc_new(forward_model["L"], forward_model["N"],
        forward_model["dt"], tF, forward_model["Sigma"],
        forward_model["R_film0"], 10, 1e-7, 50, 1, 300)


plt.figure(1)
plt.plot(Thk)

plt.figure(2)
plt.plot(Cur)

plt.show()

