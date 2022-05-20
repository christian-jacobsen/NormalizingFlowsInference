import numpy as np
import math
import torch
import torch.nn as nn



# define our forward model
def forward_model_ford_old(L,N,dt,tFinal,Sigma,R_film0, VR, Cv, Qmin, jmin):
    # INPUTS:
    #   L       = domain length (1D)
    #   N       = number of nodes in domain
    #   dt      = time step
    #   Sigma   = model parameter

    #   R_film0 = model parameter; initial film resistance

    #   VR      = model parameter; voltage ramp
    #   Cv      = model parameter
    #   Qmin    = model parameter
    #   jmin    = model parameter
    #
    # OUTPUTS:
    #   Thickness
    #   Resistance
    #   Current
    #   Xi          = indicator of identifiability of Qmin, jmin
    #                 0 -> both identifiable
    #                 1 -> jmin unidentifiable
    #                -1 -> Qmin unidentifiable


    # Initialize necessary variables
    BC_anode = 0 
    R_film = R_film0
    Q = 0
    h = 0
    i = 0
    Resistance = np.zeros((math.floor(tFinal)+1,1))
    Thickness  = np.zeros((math.floor(tFinal)+1,1))
    Current    = np.zeros((math.floor(tFinal)+1,1))

    h_x = L/(N-1)
    e = np.ones((N,1))
    
    # Assemble A(phi) = S
    a = np.ones(N)
    b = np.ones(N-1)
    A = Sigma*(-2*np.diag(a,0)+np.diag(b,-1)+np.diag(b,1))/h_x**2
    A[-1,-2]=0
    A[0,1] = 2*Sigma/h_x**2

    SN_fac = -2*Sigma/h_x**2
    j_fac = Sigma/h_x

    S = np.zeros((N,1))
    depositionStart=False
    t = 0
    chk_tol = dt/10
    while (t<tFinal):
        i += 1
        BC_anode = BC_anode + VR*dt
        S[-1] = SN_fac*BC_anode
        a = -R_film*Sigma
        A[0,0] = SN_fac*(1-h_x/a)
        
        phi = np.linalg.solve(A,S)
        j = j_fac*(phi[1]-phi[0])
        Q = Q + j*dt

        '''
        if (Q>Qmin):
            depositionStart = True
        if depositionStart: # threshold criterion
            h = np.maximum(h + Cv*(j-jmin)*dt, 0) # film thickness
            rho_j = np.maximum(8e6*np.exp(-0.1*j),2e6) # resistivity of film
            R_film = np.maximum(R_film + rho_j*Cv*(j-jmin)*dt, 0) # film resistance
        '''
        if (Q>Qmin) and (j > jmin): # threshold criterion
            h = h + Cv*j*dt # film thickness
            rho_j = np.maximum(8e6*np.exp(-0.1*j),2e6) # resistivity of film
            R_film = R_film + rho_j*Cv*j*dt # film resistance        
        
        t += dt
        if ((t%1)<chk_tol)or(((t%1)-1)>-chk_tol):
            tind = int(np.rint(t))
            Resistance[tind,0] = R_film
            Thickness[tind,0] = h
            Current[tind,0] = j
        

    return Thickness, Resistance, Current


def forward_model_ford_new(L,N,dt,tFinal,Sigma,R_film0, VR, Cv, K, jmin):
    # this model uses a parameter K and the choice of experiment (given by exp_num)
    # exp_num = 0 --> voltage ramp
    # exp_num = 1 --> constant current
    # INPUTS:
    #   L       = domain length (1D)
    #   N       = number of nodes in domain
    #   dt      = time step
    #   Sigma   = model parameter

    #   R_film0 = model parameter; initial film resistance

    #   VR      = model parameter; voltage ramp
    #   Cv      = model parameter
    #   Qmin    = model parameter
    #   jmin    = model parameter
    #
    # OUTPUTS:
    #   Thickness
    #   Resistance
    #   Current
    #   Xi          = indicator of identifiability of Qmin, jmin
    #                 0 -> both identifiable
    #                 1 -> jmin unidentifiable
    #                -1 -> Qmin unidentifiable


    # Initialize necessary variables
    Qmin = 1e10
    BC_anode = 0
    R_film = R_film0
    Q = 0
    h = 0
    i = 0
    Resistance = np.zeros((10*(tFinal)+1,1))
    Thickness  = np.zeros((10*(tFinal)+1,1))
    Current    = np.zeros((10*(tFinal)+1,1))

    h_x = L/(N-1)
    e = np.ones((N,1))
    
    # Assemble A(phi) = S
    a = np.ones(N)
    b = np.ones(N-1)
    A = Sigma*(-2*np.diag(a,0)+np.diag(b,-1)+np.diag(b,1))/h_x**2
    A[-1,-2]=0
    A[0,1] = 2*Sigma/h_x**2

    SN_fac = -2*Sigma/h_x**2
    j_fac = Sigma/h_x

    S = np.zeros((N,1))
    depositionStart=False
    t = 0
    dt0 = dt
    nextTime = 1
    chk_tol = dt/10
    tind = 0
    count = 0
    while (t<=tFinal):
        i += 1
        BC_anode = BC_anode + VR*dt
        BC_anode = np.minimum(BC_anode, 300)
        S[-1] = SN_fac*BC_anode
        a = -R_film*Sigma
        if a != 0.0:
            A[0,0] = SN_fac*(1-h_x/a)
            A[0,1] = -SN_fac
        else:
            A[0,0] = SN_fac
            A[0,1] = 0
        
        phi = np.linalg.solve(A,S)
        j = j_fac*(phi[1]-phi[0])

        if i==1:
            #j1 = j
            beta = j/dt
            Qmin = (81/(128*beta))**(1/3)*(K**(4/3))
        #elif i==2:
            #j2 = j
            #beta = (j2-j1)/dt
            #j0 = j1 - beta*dt
            #Qmin = (81/(128*beta))**(1/3)*(K**(4/3))

        Q = Q + j*dt

        if (Q>Qmin):
            depositionStart = True
        if (depositionStart): # threshold criterion
            h = np.maximum(h + Cv*(j-jmin)*dt, 0) # film thickness
            rho_j = np.maximum(8e6*np.exp(-0.1*j), 2e6) # resistivity of film
            R_film = np.maximum(R_film + rho_j*Cv*(j-jmin)*dt, R_film0) # film resistance
        '''
        if (Q>Qmin) and (j > jmin): # threshold criterion
            h = h + Cv*j*dt # film thickness
            rho_j = np.maximum(8e6*np.exp(-0.1*j),2e6) # resistivity of film
            R_film = R_film + rho_j*Cv*j*dt # film resistance        
        '''

        #if (t + dt - nextTime <= 0):
            #dt = dt0
        #else:
            #dt = nextTime - t

        t += dt0

        #if (np.mod(t,0.1)==0):
        if count==9:
            #print('hit')
            #tind = int(np.rint(t))
            tind += 1
            Resistance[tind,0] = R_film
            Thickness[tind,0] = h
            Current[tind,0] = j
            nextTime += 1
            count = 0
        else:
            count += 1

    return Thickness, Resistance, Current

def forward_model_ford_new_unident(L,N,dt,tFinal,Sigma,R_film0, VR, Cv, K, jmin):
    # this model uses a parameter K and the choice of experiment (given by exp_num)
    # exp_num = 0 --> voltage ramp
    # exp_num = 1 --> constant current
    # INPUTS:
    #   L       = domain length (1D)
    #   N       = number of nodes in domain
    #   dt      = time step
    #   Sigma   = model parameter

    #   R_film0 = model parameter; initial film resistance

    #   VR      = model parameter; voltage ramp
    #   Cv      = model parameter
    #   Qmin    = model parameter
    #   jmin    = model parameter
    #
    # OUTPUTS:
    #   Thickness
    #   Resistance
    #   Current
    #   Xi          = indicator of identifiability of Qmin, jmin
    #                 0 -> both identifiable
    #                 1 -> jmin unidentifiable
    #                -1 -> Qmin unidentifiable


    # Initialize necessary variables
    Qmin = 1e10
    BC_anode = 0 
    R_film = R_film0
    Q = 0
    h = 0
    i = 0
    Resistance = np.zeros((math.floor(tFinal)+1,1))
    Thickness  = np.zeros((math.floor(tFinal)+1,1))
    Current    = np.zeros((math.floor(tFinal)+1,1))

    h_x = L/(N-1)
    e = np.ones((N,1))
    
    # Assemble A(phi) = S
    a = np.ones(N)
    b = np.ones(N-1)
    A = Sigma*(-2*np.diag(a,0)+np.diag(b,-1)+np.diag(b,1))/h_x**2
    A[-1,-2]=0
    A[0,1] = 2*Sigma/h_x**2

    SN_fac = -2*Sigma/h_x**2
    j_fac = Sigma/h_x

    S = np.zeros((N,1))
    depositionStart=False
    t = 0
    dt0 = dt
    nextTime = 1
    chk_tol = dt/10
    while (t<tFinal):
        i += 1
        BC_anode = BC_anode + VR*dt
        BC_anode = np.minimum(BC_anode, 300)
        S[-1] = SN_fac*BC_anode
        a = -R_film*Sigma
        if a != 0.0:
            A[0,0] = SN_fac*(1-h_x/a)
            A[0,1] = -SN_fac
        else:
            A[0,0] = SN_fac
            A[0,1] = 0
        
        phi = np.linalg.solve(A,S)
        j = j_fac*(phi[1]-phi[0])

        if i==1:
            j1 = j
        elif i==2:
            j2 = j
            beta = (j2-j1)/dt
            j0 = j1 - beta*dt
            Qmin = (81/(128*beta))**(1/3)*(K**(4/3))

        Q = Q + j*dt
        '''
        if (Q>Qmin):
            depositionStart = True
        if (depositionStart): # threshold criterion
            h = np.maximum(h + Cv*(j-jmin)*dt, 0) # film thickness
            rho_j = np.maximum(8e6*np.exp(-0.1*j), 2e6) # resistivity of film
            R_film = np.maximum(R_film + rho_j*Cv*(j-jmin)*dt, R_film0) # film resistance
        '''
        if (Q>Qmin) and (j > jmin): # threshold criterion
            h = h + Cv*j*dt # film thickness
            rho_j = np.maximum(8e6*np.exp(-0.1*j),2e6) # resistivity of film
            R_film = R_film + rho_j*Cv*j*dt # film resistance        
        
        if (t + dt - nextTime <= 0):
            dt = dt0
        else:
            dt = nextTime - t

        t += dt

        if (np.mod(t, 1) == 0):
            tind = int(np.rint(t))
            Resistance[tind, 0] = R_film
            Thickness[tind, 0] = h
            Current[tind, 0] = j
            nextTime += 1

    return Thickness, Resistance, Current


class torch_forward(nn.Module):
    def __init__(self, L, N, dt, tFinal, Sigma, R_film0, VR):
        super().__init__()

        self.L = L
        self.N = N
        self.dt = dt
        self.tFinal = tFinal
        self.Sigma = Sigma
        self.R_film0 = R_film0
        self.VR = VR

    def forward(self, z):
        # Define all of the parameters
        Sigma = self.Sigma
        R_film0 = self.R_film0   # Initial film resistance
        VR = self.VR     # Voltage Ramp
        L = self.L  # Domain length
        N = self.N  # Number of spatial domain nodes
        dt = self.dt   # Forward model time step
        tFinal = self.tFinal  # Forward model final time

        # infer Cv, K, jmin
        Cv = z[:, 0]
        K = z[:, 1]
        jmin = z[:, 2]

        batch_size = np.shape(Cv)[0]

        Cv = torch.pow(10.0, -Cv)  # [:,0])

        Thk_d = torch.zeros((batch_size, tFinal+1))

        h_x = L/(N-1)
        SN_fac = -2*Sigma/h_x**2
        j_fac = Sigma/h_x
        e = torch.ones((N, 1))
        al = torch.ones(N)
        b = torch.ones(N-1)
        
        for k in range(batch_size):
            
            # Initialize necessary variables
            Qmin = 1e10
            BC_anode = 0 
            R_film = R_film0
            Q = 0
            h = 0 
            i = 0
            Resistance = torch.zeros((math.floor(tFinal)+1,1))
            Thickness  = torch.zeros((math.floor(tFinal)+1,1))
            Current    = torch.zeros((math.floor(tFinal)+1,1))

            
            # Assemble A(phi) = S

            A = Sigma*(-2*torch.diag(al,0)+torch.diag(b,-1)+torch.diag(b,1))/h_x**2
            A[-1,-2]=0
            A[0,1] = 2*Sigma/h_x**2
            A[0,0] = 0
            A = A.repeat(1002, 1, 1)

            R_film = R_film0*torch.ones((501,))
            h = torch.zeros((501, ))
            rho_j = torch.zeros((501, ))
            j = torch.zeros((501, ))
            Q = torch.zeros((501, ))
            weight = torch.zeros((501, ))
            phi = torch.zeros((501, N))
            
            S = torch.zeros((501, N))
            Si = S + 0.
            depoStart=False
            t = 0
            dt0 = dt
            chk_tol = dt/10
            while (t<tFinal):
                i = i + 1

                BC_anode = BC_anode + VR*dt
                S[i-1,-1] = SN_fac*BC_anode
                a = -R_film[i-1]*Sigma
                if a != 0.0:
                    A[i,0,0] = SN_fac*(1-h_x/a)
                    A[i, 0, 1] = -SN_fac
                else:
                    A[0, 0] = SN_fac
                    A[0, 1] = 0.5

                #U, s, Vh = torch.linalg.svd(A[:,:,i])
                #Ai = torch.transpose(Vh,0,1)@torch.diag(1/s)@torch.transpose(U,0,1)

                phi[i-1,:] = torch.linalg.solve(A[i,:,:], S[i-1,:])#torch.matmul(Ai, S[i-1,:])
                j[i] = j_fac*(phi[i-1,1]-phi[i-1,0])
                
                if i==1:
                    #j1 = j[i]
                    beta = j/dt
                    Qmin = (81/(128*beta))**(1/3)*(K[k]**(4/3))
                #elif i == 2:
                    #j2 = j[i]
                    #beta = (j2 - j1)/dt
                    #j0= j1 - beta*dt
                    #Qmin = (81/(128*beta))**(1/3)*(K[k]**(4/3))
                Q[i] = Q[i-1] + j[i]*dt

                '''
                if (Q[i]>Qmin):
                    depoStart = True
                '''
                weight[i] = self.soft_step(Q[i], Qmin)
                #if depoStart: # threshold criterion
                h[i] = weight[i]*torch.maximum(h[i-1] + Cv[k]*(j[i]-jmin[k])*dt, torch.zeros(1))+ (1-weight[i])*h[i-1] # film thickness
                rho_j[i] = weight[i]*torch.maximum(8e6*torch.exp(-0.1*j[i]),2e6*torch.ones(1)) # resistivity of film
                R_film[i] = weight[i]*torch.maximum(R_film[i-1] + rho_j[i]*(j[i]-jmin[k])*Cv[k]*dt,torch.zeros(1))\
                             + (1-weight[i])*R_film[i-1]# film resistance
                #else:
                #R_film[i] = R_film[i-1]
                #h[i] = h[i-1]
                    
                t = t + dt
                if ((t%1)<chk_tol)or(((t%1)-1)>-chk_tol):
                    tind = int(np.rint(t))
                    Resistance[tind,0] = R_film[i]
                    Thickness[tind,0] = h[i]
                    Current[tind,0] = j[i]

            Thk_d[k,:] = Thickness[:,0]
            
        return Thk_d

    def soft_step(self, Q, Qmin):

        return 1/ (1 + torch.exp(-1e8*(Q - Qmin)))
