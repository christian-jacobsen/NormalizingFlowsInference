import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append("../")



load_path = "../bayes_posterior/posterior20.npy"

with open(load_path, 'rb') as f:
    log_post = np.load(f)
    theta_v = np.load(f)
    N = np.load(f)

save_folder = "./bayes_post/"+str(N)
if not(os.path.exists(save_folder)):
    print("CREATING FOLDER: " + save_folder)
    os.mkdir(save_folder)
else:
    print("WARNING: Folder already exists! " + save_folder)
    
# create the 3D tensors of parameter grid
M1 = np.reshape(theta_v[:, 0], (N, N, N))
M2 = np.reshape(theta_v[:, 1], (N, N, N))
M3 = np.reshape(theta_v[:, 2], (N, N, N))

# compute log marginals
Cv_log_marg = np.sum(log_post, axis=(0, 1))
K_log_marg = np.sum(log_post, axis=(0, 2))
jmin_log_marg = np.sum(log_post, axis=(1, 2))

# compute joints
CvK_log_joint = np.sum(log_post, axis=0)
CvJmin_log_joint = np.sum(log_post, axis=1)
KJmin_log_joint = np.sum(log_post, axis=2)


# take exp to get the posteriors
Cv_marg = np.exp(Cv_log_marg + 25)  # + 1.2e3)
K_marg = np.exp(K_log_marg + 35)  #1.2e3)
jmin_marg = np.exp(jmin_log_marg + 30)  #1.2e3)

plt.close('all')

# plot log marginals
plt.figure(1, figsize=(7, 15))
plt.subplot(3, 1, 1)
plt.plot(M1[0, 0, :], Cv_log_marg)
plt.title("Bayesian Log-Posterior (Marginal)")
plt.xlabel(r'$-\log C_v$')
plt.subplot(3, 1, 2)
plt.plot(M2[0, :, 0], K_log_marg)
plt.title("Bayesian Log-Posterior (Marginal)")
plt.xlabel(r'$K$')
plt.subplot(3, 1, 3)
plt.plot(M3[:, 0, 0], jmin_log_marg)
plt.title("Bayesian Log-Posterior (Marginal)")
plt.xlabel(r'$j_{min}$')
plt.savefig(save_folder+"/log_marginals.png")

# plot marginals
plt.figure(2, figsize=(7, 15))
plt.subplot(3, 1, 1)
plt.plot(M1[0, 0, :], Cv_marg)
plt.title("Bayesian Posterior (Marginal)")
plt.xlabel(r'$-\log C_v$')
plt.subplot(3, 1, 2)
plt.plot(M2[0, :, 0], K_marg)
plt.title("Bayesian Posterior (Marginal)")
plt.xlabel(r'$K$')
plt.subplot(3, 1, 3)
plt.plot(M3[:, 0, 0], jmin_marg)
plt.title("Bayesian Posterior (Marginal)")
plt.xlabel(r'$j_{min}$')
plt.savefig(save_folder+"/marginals.png")

# plot log joints
plt.figure(3, figsize=(7, 15))
plt.subplot(3, 1, 1)
plt.contour(M1[0, :, :], M2[0, :, :], CvK_log_joint)
plt.title("Bayesian Log-Posterior (Joint)")
plt.xlabel(r'$-\log C_v$')
plt.ylabel(r'$K$')
plt.subplot(3, 1, 2)
plt.contour(M2[:, :, 0], M3[:, :, 0], KJmin_log_joint)
plt.title("Bayesian Log-Posterior (Joint)")
plt.xlabel(r'$K$')
plt.ylabel(r'$j_{min}$')
plt.subplot(3, 1, 3)
plt.contour(M1[:, 0, :], M3[:, 0, :], CvJmin_log_joint)
plt.title("Bayesian Log-Posterior (Joint)")
plt.xlabel(r'$-\log C_v$')
plt.ylabel(r'$j_{min}$')
plt.savefig(save_folder+"/log_joints.png")
