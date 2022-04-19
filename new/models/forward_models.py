import numpy as np
import math



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
