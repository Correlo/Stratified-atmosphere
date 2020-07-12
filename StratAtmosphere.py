#Author: Martín Manuel Gómez Míguez
#GitHub: @Correlo
#Date: 23/05/2020

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from configparser import ConfigParser

# ***************** #
# Stratified fields #
# ***************** #

# Temperature

def T(x, T0, alpha):

    '''
    Input
    
    x     -> position array
    T0    -> Temperature on the bottom
    alpha -> Temperature field parameter
    
    Output
    
    T     -> Temperature field
    '''

    return T0 + alpha*x


def Tig(mu, P, rho, adim = False):

    '''
    Input

    mu    -> Average molecular weight
    P     -> Pressure
    rho   -> Density
    adim  -> Bool, select adimensional mode

    Constants
    mH    -> Hidrogen atomic mass
    kB    -> Boltzmann constant

    Output

    Tig     -> Temperature of ideal gas
    '''

    ''' Constants that can be needed '''
    mH = 1.00797 * 1.660538e-27  # Hidrogen atomic mass in kg
    kB = 1.380645e-23  # Boltzmann constant in J/K

    if adim: mH, kB = 1, 1

    return mu*mH/kB*P/rho

# Pressure

def Pressure(x, P0, T0, alpha, mu, g, adim = False):
    
    '''
    Input

    x     -> position array
    P0    -> Presure on the bottom
    T0    -> Temperature on the bottom
    alpha -> Temperature field parameter
    mu    -> Average molecular weight
    g     -> Gravity acceleration
    adim  -> Bool, select adimensional mode

    Constants
    mH    -> Hidrogen atomic mass
    kB    -> Boltzmann constant

    Output

    P     -> Pressure
    '''

    ''' Constants that can be needed '''
    mH = 1.00797 * 1.660538e-27  # Hidrogen atomic mass in kg
    kB = 1.380645e-23  # Boltzmann constant in J/K
    
    if adim: mH, kB = 1, 1
    
    return P0*((T0 + alpha*x)/T0)**((mu*mH*g)/(kB*alpha))

# Density

def Density(x, P0, alpha, T0, mu, g, adim = False):
    '''
    Input

    x     -> position array
    P0    -> Presure on the bottom
    T0    -> Temperature on the bottom
    alpha -> Temperature field parameter
    mu    -> Average molecular weight
    g     -> Gravity acceleration
    adim  -> Bool, select adimensional mode

    Constants
    mH    -> Hidrogen atomic mass
    kB    -> Boltzmann constant

    Output

    rho   -> Density
    '''

    ''' Constants that can be needed '''
    mH = 1.00797 * 1.660538e-27  # Hidrogen atomic mass in kg
    kB = 1.380645e-23  # Boltzmann constant in J/K
    
    if adim: mH, kB = 1, 1
    
    return (mu*mH)/(kB)*P0/T0*((T0 + alpha*x)/T0)**((mu*mH*g)/(kB*alpha) - 1)

# Perturbation

def GaussP(x, A, eps, xd, dx):
    '''
    Input

    x     -> position array
    A     -> Amplitud of the perturbation
    eps   -> epsilon parameter
    xd    -> center of the perturbation
    dx    -> grid size

    Output

    GaussP   -> Density Gaussian perturbation
    '''

    return A*np.e**(-0.5*(x - xd)**2/(eps*dx)**2)

# Crossing-time

def Crosst(xi, xf, P0, alpha, T0, mu, g, gamma, adim = False):
    '''
    Input

    xi    -> position of the bottom
    xf    -> position of the top
    P0    -> Presure on the bottom
    T0    -> Temperature on the bottom
    alpha -> Temperature field parameter
    mu    -> Average molecular weight
    g     -> Gravity acceleration
    gamma -> adiabatic constant
    adim  -> Bool, select adimensional mode

    Constants
    mH    -> Hidrogen atomic mass
    kB    -> Boltzmann constant

    Output

    tc    -> Crossing time
    '''

    ''' Constants that can be needed '''
    mH = 1.00797 * 1.660538e-27  # Hidrogen atomic mass in kg
    kB = 1.380645e-23  # Boltzmann constant in J/K
    
    if adim: mH, kB = 1, 1
    
    return 2*np.sqrt((mu*mH)/(gamma*kB*alpha))*(np.sqrt(T0 + alpha*xf) - np.sqrt(T0 + alpha*xi))

# *********************************************** #
# Implementation of Lax-Wendroff-Ritchmyer scheme #
# *********************************************** #

### First step ###
def LWR_FS(u, f, dx, dt):
    
    '''
    Input

    u  -> u field
    f  -> f field
    dx -> spacial grid space
    dt -> timestep

    Output

    uh -> u^(n + 0.5)_(i + 0.5)
    '''
    
    # Array to fill
    uh = np.ones_like(u)
    uh[:-1] = 0.5*(u[1:] + u[:-1]) - 0.5*dt/dx*(f[1:] - f[:-1])
    
    return uh

### Second step ###
def LWR_SS(u, f, dx, dt):
    
    '''
    Input

    u  -> u field
    f  -> f field
    dx -> spacial grid space
    dt -> timestep

    Output

    uf -> u^(n + 1)_i
    '''

    uf = np.ones_like(u)
    uf[1:-1] = u[1:-1] - dt/dx*(f[1:-1] - f[:-2])
    
    return uf


# *********************** #
# Eulerian forward scheme #
# *********************** #

### Forward ###

def Eul_forw(u, f, dt):

    '''
    Input

    u  -> u function
    f  -> f function
    dt -> timestep

    Output

    uf -> function in future
    '''

    # Measure derivative
    uf = np.ones_like(u)
    uf = u + dt*f
    
    return uf


# This function will be the main loop of our simulation:
#   * i : frame number sent by FuncAnimation
#   * rho : function that we will plot, take into account that was to be
#           initialized outside!
#   * t : time variable
#   * next_show : time at which we will show the plot, I do it this way as we could have
#           variable dt in the simulations, but we always want the output at the same physical time!
#

def animate(i):
    
    global x, P0, rh0, e0, P1, rho1, v1, e, t, tc, next_show, dt_show, Line, title
    
    while t < next_show:
        
        # Compute time step
        cs = np.sqrt(gamma*P1[:]/rho1[:]) # Sound speed
        dt = 0.5*dx/max(max(abs(cs + v1[:])), max(abs(cs - v1[:])))
        
        # *********************************************** #
        # Implementation of Lax-Wendroff-Ritchmyer scheme #
        # *********************************************** #
        
        ### First step ###
        
        # Solve continuity equation
        uc1 = LWR_FS(rho1[:], rho1[:]*v1[:], dx, dt)
        
        # Solve movement equation
        um1 = LWR_FS(rho1[:]*v1[:], rho1[:]*v1[:]**2 + P1[:], dx, dt)
        
        # Solve energy equation
        ue1 = LWR_FS(rho1[:]*e[:], (rho1[:]*e[:] + P1[:])*v1[:], dx, dt)
        
        # Obtain v1, e, P and rho
        rho1_h = uc1[:].copy()
        v1_h = um1[:]/rho1_h[:]
        e_h = ue1[:]/rho1_h[:]
        P1_h = (e_h[:] - 0.5*v1_h[:]**2)*(gamma - 1)*rho1_h[:]
        
        
        ### Second step ###

        # Solve continuity equation
        uc2 = LWR_SS(rho1[:], rho1_h[:]*v1_h[:], dx, dt)
        
        # Solve movement equation
        um2 = LWR_SS(rho1[:]*v1[:], rho1_h[:]*v1_h[:]**2 + P1_h[:], dx, dt)
        
        # Solve energy equation
        ue2 = LWR_SS(rho1[:]*e[:], (rho1_h[:]*e_h[:] + P1_h[:])*v1_h[:], dx, dt)
        
        # Impose boundary conditions to density (df/dx|0 = 0, df/dx|x_N-1 = 0)
        rho1[:]  = uc2[:].copy()
        rho1[0]  = rho_0
        rho1[-1] = rho_f
        
        # Impose boundary conditions to momentum (df/dx|0 = 0, df/dx|x_N-1 = 0)
        um2[0]  = um2[1]
        um2[-1] = um2[-2]
        # Measure velocity
        v1[:] = um2[:]/rho1[:]

        # Measure energy
        e[:] = ue2[:]/rho1[:]
        
        # Measure pressure
        P1[:] = (e[:] - 0.5*v1[:]**2)*(gamma - 1)*rho1[:]
        # Impose boundary conditions to pressure (df/dx|0 = 0, df/dx|x_N-1 = 0)
        P1[0] = P_0
        P1[-1] = P_f

        # Measure energy again
        e[:] = P1[:]/((gamma - 1)*rho1[:]) + 0.5*v1[:]**2

        # *********************** #
        # Eulerian forward scheme #
        # *********************** #

        ueg = Eul_forw(rho1[:]*e[:], rho1[:]*v1[:]*g, dt)
        umg = Eul_forw(rho1[:]*v1[:], rho1[:]*g, dt)
        
        # Impose boundary conditions to momentum (df/dx|0 = 0, df/dx|x_N-1 = 0)
        umg[0]  = umg[1]
        umg[-1] = umg[-2]
        # Measure velocity
        v1[:] = umg[:]/rho1[:]
        
        # Measure energy
        e[:] = ueg[:]/rho1[:]
        
        # Measure pressure
        P1[:] = (e[:] - 0.5*v1[:]**2)*(gamma - 1)*rho1[:]
        # Impose boundary conditions to pressure (df/dx|0 = 0, df/dx|x_N-1 = 0)
        P1[0] = P_0
        P1[-1] = P_f
        
        # Measure again energy
        e[:] = P1[:]/((gamma - 1)*rho1[:]) + 0.5*v1[:]**2

        t += dt
    
    next_show = t + dt_show

    Pp = (P1[:] - P[:])/P[:]
    rhop = (rho1[:] - rho[:])/rho[:]

    # We redraw the line with the new information
    Line[0].set_data(x, v1[:]/cs[:])
    Line[1].set_data(x, Pp[:])
    Line[2].set_data(x, rhop[:])
    Line[3].set_data(x, (Tig(mu, P1[:], rho1[:], adim = adim) - T(x[:], T0, alpha))/T(x[:], T0, alpha))
    title.set_text('Stratified Atmosphere\n t/tc = %.2f' % (t/tc))
    
    return Line[0], Line[1], Line[2], Line[3], title


''' Basic parameters '''
#Read params.ini
params = ConfigParser()
params.sections()
params.read('params.ini')
Params = params['params']

xi = float(Params['xi'])
xf = float(Params['xf'])
N = int(Params['N'])  # number of xgrid points
phi = float(Params['phi']) # phase
k = int(Params['k'])   # mode
P0 = float(Params['P0']) # Constant preassure
Arho = float(Params['Arho']) # Density pertrubation amplitude
eps = float(Params['eps']) # Espilon parameter
xd = float(Params['xd']) # Center of the perturbation
gamma = float(Params['gamma']) # Adiabatic constant
g = float(Params['g']) # Gravity in units of Earth gravity
mu = float(Params['mu']) # Average molecular weight
alpha = float(Params['alpha']) # Average
T0 = float(Params['T0']) # Temperature
adim = bool(int(Params['adim'])) # Units

''' Preliminary calculus '''

# X array
dx = (xf - xi)/N
x = np.arange(xi - dx/2, xf + 3*dx/2, dx)

# Crossing time
tc = Crosst(xi, xf, P0, alpha, T0, mu, g, gamma, adim = adim)

### Initial fields ###
# Density
rho = Density(x, P0, T0, alpha, mu, g, adim = adim)
rho_0, rho_f = rho[0], rho[-1] # Fix to boundary conditions
rho1 = rho + rho*GaussP(x, Arho, eps, xd, dx)

# Pressure
P = Pressure(x, P0, T0, alpha, mu, g, adim = adim)
P_0, P_f = P[0], P[-1] # Fix to boundary conditions
P1 = P + P*GaussP(x, gamma*Arho, eps, xd, dx)

# Velocitys
cs = np.sqrt(gamma*P/rho) # Sound speed
v = np.zeros_like(x)
v1 = GaussP(x, cs*Arho, eps, xd, dx)


# Fields with dependence on fileds above
e = P1/((gamma - 1)*rho1) + 0.5*v1**2

# Animation parameters
dt_show = 0.1
t = 0
next_show = dt_show + t

# create a figure with two subplots
plt.close()
fig, ax = plt.subplots(4, 1, sharex = True, figsize = (10, 8))
fig.subplots_adjust(hspace = 0)
Yaxis = [r'$v/c_s$', r'$(P_1 - P_0)$/$P_0$', r'$(\rho_1 - \rho_0)$/$\rho_0$',
        r'$(T_1 - T_0)$/$T_0$']
#Yaxis = ['Velocity', r'$P_1$', r'$\rho_1$', 'Energy']
color = ['r', 'g', 'b', 'k']

Line = []

for i in range(4):

    # intialize two line objects (one in each axes)
    line, = ax[i].plot([], [], color = color[i])
    ax[i].set_xlim(xi, xf)
    ax[i].set_ylim(-4e-3, 4e-3)
    ax[i].grid()
    ax[i].set_ylabel(Yaxis[i], fontsize = 13)
    Line.append(line)

title = ax[0].set_title('Stratified Atmosphere\n t/tc = %.2f ' % (t/tc), fontsize = 15, pad = 14)
ax[-1].set_xlabel(r'$x$', fontsize = 13)
ax[0].set_ylim(-2e-3, 2e-3)

ani = animation.FuncAnimation(fig, animate, interval = 100)
plt.show()



