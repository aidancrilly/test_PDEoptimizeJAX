import jax.numpy as jnp
import jax
import optax
import diffrax

import matplotlib.pyplot as plt

def P1_equations(t,y,args):
    """ P1 closure of transport equation 
    
    Reflective boundary on the left
    Dirichlet boundary on the right (N=0)
    
    """
    # Extract density and flux
    D,nu_N,nu_F,dx = args
    y = y.reshape(2,Nx)
    N,F = y[0,:],y[1,:]
    # dNdt = R - nu_N N - div(F)
    dNdt = R-nu_N*nu_s*N-jnp.diff(F,n=1,prepend=0.0)/dx
    # dFdt = - grad(N) - nu_F F
    dFdt = -jnp.diff(D*N,n=1,append=0.0)/dx-nu_F*nu_s*F
    dydt = jnp.vstack((dNdt,dFdt))
    return dydt.flatten()

def P1_solve(params,y0,tmax,dt0,saveat,solver,adjoint):
    """ Differential equation solve using diffrax """
    term = diffrax.ODETerm(P1_equations)
    solution = diffrax.diffeqsolve(term, solver, t0=0, t1=tmax, dt0=dt0, y0=y0, saveat=saveat, args = params,adjoint=adjoint)
    return solution.ys

# Initialise problem
Nx = 100
L = 1.0
xb = jnp.linspace(0.0,L,Nx+1)
x = 0.5*(xb[1:]+xb[:-1])
dx = x[1]-x[0]

# Source
R    = jnp.exp(-0.5*((x-L/2)/(0.05*L))**2)
# Absorption coefficient
nu_s = 2.0-jnp.exp(-0.5*((x-L/2)/(0.05*L))**2)

dtsave = 0.05
tmax  = 4.0
tsave  = jnp.arange(0.0,tmax+dtsave,dtsave)

# Diffrax set up
y0 = jnp.zeros((2,Nx)).flatten()
dt0 = dtsave/10
saveat = diffrax.SaveAt(ts=tsave)
solver = diffrax.Heun()
adjoint = diffrax.RecursiveCheckpointAdjoint()

params_P1_solve = lambda params : P1_solve(params,y0,tmax,dt0,saveat,solver,adjoint)

# Physical parameters
D = 1./3.
nu_N = 0.5
nu_F = 1.0

p0 = [D,nu_N,nu_F,dx]
sol = params_P1_solve(p0)
sol = sol.reshape(tsave.shape[0],2,Nx)
Nsol = sol[:,0,:]
Fsol = sol[:,1,:]

for i in range(tsave.shape[0]):
    fig = plt.figure(dpi=200)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212,sharex=ax1)

    ax1.plot(jnp.concatenate((-x[::-1],x)),jnp.concatenate((Nsol[i,::-1],Nsol[i,:])),'k')
    ax2.plot(jnp.concatenate((-x[::-1],x)),jnp.concatenate((Fsol[i,::-1],Fsol[i,:])),'k')

    ax1.set_ylabel("N")
    ax2.set_ylabel("F")
    ax2.set_xlabel("x")
    ax1.set_xlim(-x[-1],x[-1])

    fig.tight_layout()
    fig.savefig(f'plots/plot-{str(i).zfill(3)}.png')
    plt.close()