from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, RecursiveCheckpointAdjoint
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import optax

def Dloss(model,D,ytruth):
    """ Compute the RMS distance between prediction of y and truth """
    ypred = model(D)
    return jnp.mean(jnp.sum((ypred-ytruth)**2,axis=0))

def diffusion(t, y, args):
    """ Defining the diffusion equation """
    # dy/dt = D d^2 y/ dx^2
    D, dx = args[0],args[1]
    y_ghost = jnp.append(y,0.0)
    y_ghost = jnp.insert(y_ghost,0,0.0)
    dydt = D*(y_ghost[:-2]-2*y_ghost[1:-1]+y_ghost[2:])/(dx**2)
    return dydt

def solve(D,y0,dx,saveat,term,solver,adjoint):
    """ Differential equation solve using diffrax """
    CFL = dx**2/D
    solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.5*CFL, y0=y0, saveat=saveat, args = ([D,dx]),adjoint=adjoint)
    return solution.ys

term = ODETerm(diffusion)
solver = Dopri5()
adjoint = RecursiveCheckpointAdjoint()

x  = jnp.linspace(-1.0,1.0,100)
dx = x[1]-x[0]
saveat = SaveAt(ts=jnp.linspace(0.0,1.0,10))

# Initial condition
y0 = jnp.exp(-0.5*(x/0.1)**2)

Dsolve = lambda D : solve(D,y0,dx,saveat,term,solver,adjoint)

# Solution for known D
Dtruth = 0.1
ytruth = Dsolve(Dtruth)

# Perform gradient descent with backprop to find Dtruth
learning_rate = 0.01
optimizer = optax.adam(learning_rate)
params = {'D': 0.5*jnp.ones((1,))}
opt_state = optimizer.init(params)

# Set up gradient descent loop
fig = plt.figure(dpi = 200,figsize=(6,4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223,sharex=ax1)

ax3 = fig.add_subplot(122)

nsteps = 50
D_opt = 0.0
loss_opt = 1e10
for i in range(nsteps):
    loss, grad_loss = jax.value_and_grad(lambda params : Dloss(Dsolve,params['D'][0],ytruth))(params)

    updates, opt_state = optimizer.update(grad_loss, opt_state)
    params = optax.apply_updates(params, updates)
    ax1.plot(i,loss,'bo')
    ax2.plot(i,params['D'][0],'ro')
    if(loss < loss_opt):
        loss_opt = loss*1.0
        D_opt = params['D'][0]

ax3.plot(x,Dsolve(D_opt).T)
ax3.plot(x,ytruth.T,c='k',ls='--')

ax1.set_ylabel("Loss")
ax2.set_ylabel("D")
ax2.set_xlabel("Iterations")
ax3.set_ylabel('y')
ax3.set_xlabel('x')

ax2.axhline(Dtruth,c='k',ls='--')
fig.tight_layout()
plt.show()