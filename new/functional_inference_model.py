import torch, sys
import torch.nn as nn
import numpy as np

sys.path.append(".")

from models.bayesian_layers import BayesianFNN

class FunctionalInferenceECOAT(nn.Module):
    def __init__(self, rho_layers, rho_nodes):
        super().__init__()
        self.D = 3 # 3 base parameters to infer

        self.mu = nn.Parameter(torch.empty(self.D))
        self.logvar = nn.Parameter(torch.empty(self.D))

        nn.init.normal_(self.mu)
        nn.init.normal_(self.logvar)

        # rho is a function of current, parameterized by neural network
        self.rho = BayesianFNN(1, 1, layers=rho_layers, nodes=rho_nodes,
                               act=nn.ReLU())

    def forward_model(self, L, tFinal, dt, Sigma, R_film0, Cv, K, jmin, model, VR=1, CC=1):
        # Compute the forward model
        # Args:
        #   b (int) : batch size
        #   L (float) : domain length (1D)
        #   N (int)   : number of nodes in domain
        #   tFinal (int) : final simulation time
        #   dt (float) : time step
        #   Sigma (float) : model parameter
        #   R_film0 (float) : initial film resistance
        #
        #   Cv (float) : parameter to infer
        #   K (float) : parameter to infer
        #   jmin (float) : parameter to infer
        #
        #   model (str) : model type ('CC', 'VR')
        #
        # Returns:
        #   resistance (for all time)
        #   thickness
        #   current

        # Initialize all variables and allocate memory

        '''
        def bf(N, hx, Sigma):
            inds = torch.arange(N).double() + 1
            vec = torch.zeros((N, )).double()
            vec[1:] = (-1)**(N+inds[1:])/((hx**2)**(N-inds[1:]))
            vec[0] = -(-1)**(N+1)*2*Sigma/((hx**2)**(N-1))
            vec[-1] = 1
            return vec

        def c(hx, inds, Sigma):
            c1 = (inds - 1)*hx**2  # (-1)**(inds + 1)*hx**2*inds
            c2 = 2*(inds - 2)*Sigma  # (-1)**inds*2*(inds - 1)*Sigma
            return c1, c2

        def theta(b, a11, ind_mult, vec1, vec2, const1, const2):
            const = (const1*torch.reshape(a11, (-1, 1)) + const2)
            theta = torch.tile(ind_mult, (b, 1))*(torch.tile(vec1, (b, 1))*torch.reshape(a11, (-1, 1)) + torch.tile(vec2, (b, 1))) / const
            return theta
        '''

        b = np.shape(Cv)[0]  # batch size

        nt = int(tFinal / dt)  # number of time steps at dt

        resistance = torch.zeros((b, nt, 1))
        thickness = torch.zeros((b, nt, 1))
        current = 1e-3*torch.ones((b, nt, 1))

        #h_x = L/(N-1)

        # the following vectors are precomputed for efficiently
        #  computing phi = inv(A)*S without using torch's inverse function
        # bvec = bf(N, h_x, Sigma)
        '''
        inds = torch.arange(2) + 1
        c1, c2 = c(h_x, inds, Sigma)
        c1[0] = 0
        c1[1] = 1
        c2[0] = 1
        c2[1] = 0
        ind_mult = torch.ones(np.shape(inds)) * h_x**2 / (-2*Sigma)
        ind_mult[0] = h_x**2
        ind_mult[1] = - h_x**4 / (2*Sigma)
        const1, const2 = c(h_x, N, Sigma)
        '''
        # to compute phi (function of a11 and :
        # phi[i] = bvec*theta(a11[i], ind_mult, c1, c2, const1, const2)*BC_anode[i]*SN_fac
        # where a11[i] = SN_face*(1 - h_x / a)
        # and a = -Rfilm[i] * Sigma
        # Therefore, only store BC_anode, Rfilm, phi

        '''
        # Assemble A(phi) = S
        a = torch.ones(N)
        b = torch.ones(N-1)
        A = Sigma*(-2*torch.diag(a, 0) + torch.diag(b, -1)
                   + torch.diag(b, 1)) / h_x**2
        AM = A * torch.ones((b, np.shape(A)[0], np.shape(A)[1]))

        SN_fac = -2*Sigma/h_x**2
        j_fac = Sigma/h_x

        A[-1, -2] = 0
        A[0, 1] = -SN_fac
        '''

        #S = torch.zeros((N, 1))

        BC_anode = torch.zeros((nt, ))
        #R_film = torch.zeros((b, nt))
        resistance[:, 0] = R_film0

        #phi = torch.zeros((b, N, nt, 1))
        #phi = torch.zeros((b, 2, nt, 1))

        j = torch.zeros((b, nt, 1))
        Q = torch.zeros((b, nt, 1))

        time = torch.zeros((b, nt))
        inds = torch.arange(nt)

        beta = Sigma*VR / (Sigma*R_film0 + L)
        Qmin = (81/(128*beta))**(1/3)*(K.reshape((1, -1))**(4/3))

        for i in range(nt - 1):

            time[:, inds[i+1]] = dt*(inds[i+1])
            BC_anode[inds[i+1]] = VR*dt*inds[i+1]  # BC_anode[inds[i]] + VR*dt
            #S[-1] = SN_fac*BC_anode[i+1]

            #print('rfilm: ', resistance[:, i], ' i: ', i)


            '''
            phi[:, :, i+1, 0] = theta(b, -2*Sigma*(1 - (h_x / (-resistance[:, i]*Sigma)))/h_x**2, ind_mult,
                                        c1, c2, const1,
                                        const2)*SN_fac*BC_anode[i+1]

            j[:, i+1, 0] = j_fac * (phi[:, 1, i+1, 0] - phi[:, 0, i+1, 0])
            '''
            #current[:, inds[i+1], 0] = Sigma*BC_anode[inds[i+1]] / (Sigma*resistance[:, inds[i], 0] + L)

            #print('phi: ', phi[:, :, i+1, 0], " i: ", i)

            '''
            if i == 0:
                beta = Sigma*VR / (Sigma*R_film0 + L)
                Qmin = (81/(128*beta))**(1/3)*(K.reshape((1, -1))**(4/3))
            '''

            #if i == 1000:
            '''
            print('ind_mult: ', ind_mult)
            print('a11: ', -2*Sigma*(1-(h_x / (-resistance[:, i]*Sigma)))/h_x**2)
            print('phi: ', phi[:, :, i+1, 0])
            print('j: ', j[:, i+1, 0])
            print('beta: ', beta)
            '''
            Q[:, inds[i+1], 0] = Q[:, inds[i], 0] + Sigma*BC_anode[inds[i+1]] / (Sigma*resistance[:, inds[i], 0] + L)*dt

            '''
            print('thki: ', thickness[:, i, 0])
            print(jmin.reshape((1, -1)))
            print('cvprod: ', thickness[:, i, 0] + Cv.reshape((1, -1))*(j[:, i+1, 0] - jmin.reshape((1, -1))))
            print('jip1: ', j[:, i+1, 0])
            '''
            thickness[:, inds[i+1], 0] = self.tradeoff(self.limit(thickness[:, inds[i], 0] + Cv.reshape((1, -1))*(Sigma*BC_anode[inds[i+1]] / (Sigma*resistance[:, inds[i], 0] + L) - jmin.reshape((1, -1)))*dt, 0, 1e9), thickness[:, inds[i], 0], Q[:, inds[i+1], 0], Qmin, 1e9)
            resistance[:, inds[i+1], 0] = self.tradeoff(self.limit(resistance[:, inds[i], 0] + self.rho((Sigma*BC_anode[inds[i+1]] / (Sigma*resistance[:, inds[i], 0] + L)).reshape((-1, 1))).reshape((1, -1))*(Sigma*BC_anode[inds[i+1]] / (Sigma*resistance[:, inds[i], 0] + L)-jmin.reshape((1, -1)))*dt*Cv.reshape((1, -1)), R_film0, 1e9), resistance[:, inds[i], 0], Q[:, inds[i+1], 0], Qmin, 1e9)
            #resistance[:, inds[i+1], 0] = self.tradeoff(resistance[:, inds[i], 0] + self.rho((Sigma*BC_anode[inds[i+1]] / (Sigma*resistance[:, inds[i], 0] + L)).reshape((-1, 1))).reshape((1, -1))*(Sigma*BC_anode[inds[i+1]] / (Sigma*resistance[:, inds[i], 0] + L)-jmin.reshape((1, -1)))*dt*Cv.reshape((1, -1)), resistance[:, inds[i], 0], Q[:, inds[i+1], 0], Qmin, 2e-1)
            #current[:, inds[i+1], 0] = j[:, inds[i+1], 0]

        current[:, 1:, :] = Sigma*torch.tile(BC_anode[1:].unsqueeze(0).unsqueeze(2), (b, 1, 1)) / (Sigma*resistance[:, :-1, :] + L)

        return thickness, resistance, current, time

    '''
    def rho(self, j):
        return self.limit(2e6, 8e6*torch.exp(-0.1*j))
    '''

    def limit(self, val, lim, scale):
        # differentiable maximum approximation
        return (val - lim) / (1 + torch.exp(-scale*(val - lim))) + lim

    def tradeoff(self, val1, val2, coord, lim, scale):
        # differentiable approximation to
        # if coord > lim: val1 else: val2
        return (val1 - val2) / (1 + torch.exp(-scale*(coord - lim))) + val2

    def sample(self, b):
        sigma = torch.exp(0.5*self.logvar)
        zlogvar = torch.ones((b, self.D))*self.logvar
        zmu = torch.ones((b, self.D))*self.mu
        z = self._reparameterize(zmu, zlogvar)
        log_prob_z0 = torch.sum(- 0.5 * ((z-zmu)/sigma) ** 2, axis=1)
        return zmu, zlogvar, z, log_prob_z0

    def forward(self, b, L, tFinal, dt, Sigma, R_film0, model, VR=1, CC=1):
        zmu, zlogvar, z, log_prob_z0 = self.sample(b)
        thk, res, cur, tv = self.forward_model(L, tFinal, dt, Sigma, R_film0,
                                               10**(-z[:, 0]), z[:, 1], z[:, 2],
                                               model, VR=VR, CC=CC)

        return zmu, zlogvar, z, log_prob_z0, thk[:, :, 0], 1e2*res[:, :, 0], 1e-3*cur[:, :, 0], tv

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).type_as(mu)
        return mu + std * eps

    def gaussian_log_prob(self, x, mu, logvar):
        return -0.5*(math.log(2*math.pi) + logvar + (x-mu)**2/torch.exp(logvar))
