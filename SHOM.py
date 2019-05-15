#January 27th 2016, Mads Poulsen

"""
Two-dimensional shallow water model. Governing equations are:

\frac{\partial u}{\partial t} = -g \frac{\partial \eta}{\partial x} + fv - ru
\fra{\partial v}{\partial t} = -g \frac{\partial \eta}{\partial y} -fu -rv
\frac{\partial \eta}{\partial t} = - \frac{\partial uh}{\partial x} - \frac{\partial vh}{\partial y} + Q - l *\eta

where u,v is the horizontal velocity, \eta is the sea surface height, g is the gravitational acceleration and h = \eta +D where D is the order-zero water depth. f is the coriolis parameter, r is the rayleigh friction parameter, Q is the local vertical velocity at the surface (a mass flux into the domain) and l governs the amount of upwelling (Kawases damping term). Solution can either be found on an f-plane (f=f_0=const.), on a beta plane (f=f_0+\beta y) or on an equatorial beta plane (f=\beta y), where \beta is the change of f with latitude. 

The equations are solved on a staggered grid. 
"""

#Load modules
import numpy as np, matplotlib.pyplot as plt

#Set plot properties
clevseta = np.linspace(-0.2,0.2,40)
cbartickseta = np.linspace(-0.2,0.2,10)
clevsvel = np.linspace(-0.5,0.5,100)
cbarticksvel = np.linspace(-0.5,0.5,10)

#Define dimensions and discretization 
Lx = 3000e3 #Length of model in x direction (meters)
Ly = 4000e3 #Length of model in y direction (meters)
nx = 151 #Number of cells in x direction.
ny = 201 #Number of cells in y direction.
xu = np.linspace(0, Lx, nx) #grid for u
yu = np.linspace(0, Ly, ny) #grid for v
dx = xu[1] - xu[0] #Space increment in x
dy = yu[1] - yu[0] #Space increment in y
xeta = xu - dx/2 #Grid for eta in x
yeta = yu - dy/2 #Grid for eta in y
T = 0.5e7 #Run time of the model in seconds
nt = 5000 #temporal grid points
dt = T/nt #Time step size

#Define grid
yyu, xxu = np.meshgrid(yeta, xu) #Grid for zonal velocity 
yyv, xxv = np.meshgrid(yu, xeta) #Grid for meridional velocity
yyeta, xxeta = np.meshgrid(yeta,xeta) #Grid for sea surface height

#Output arrays
hovmoeller = np.zeros([nt+1,nx]) #Matrix to hold equatorial interface height as function of time

#Operators and boundary conditions
kx = np.eye(nx, k = 0) #Identity matrix for x
ky = np.eye(ny, k = 0) #Identity matrix for y
kupx = np.eye(nx, k = 1) # k + 1 matrix for x
kupy = np.eye(ny, k = 1) # k + 1 matrix for y
kupx[-1,-1] = 1 #Boundary cond. 
kupy[-1,-1] = 1 #Boundary cond. 
kdnx = np.eye(nx, k = -1) # k -1 matrix for x
kdny = np.eye(ny, k = -1) # k -1 matrix for y
kdnx[0,0]= 1 #Boundary cond.
kdny[0,0]= 1 #Boundary cond.
fwdx = - np.eye(nx, k = 0) + np.eye(nx, k = 1) #Forward differential for x
fwdy = - np.eye(ny, k = 0) + np.eye(ny, k = 1) #Forward differential for y
fwdx[0,0:2] = 0 #Boundary cond.
fwdx[-1,-2:] = 0
fwdy[0,0:2] = 0 #Boundary cond.
fwdy[-1,-2:] = 0
shx = np.eye(nx, k = 1) + np.eye(nx, k = -1)
shx[0,1] = 0
shx[1,0] = 0
shx[1,1] = 1
shx[-1,-1] = 1
shy = np.eye(ny, k = 1) + np.eye(ny, k = -1)
shy[0,1] = 0
shy[1,0] = 0
shy[1,1] = 1
shy[-1,-1] = 1


#Parameters
eps = 0.00 #Strength of Shapiro filter
D0 = 2000 #Water depth (meters)
g = 9.81 / 4000.0 #Gravitational acceleration
r = 1e-7 #Friction coefficient
Q = 0.0 #Vertical velocity at source region
l = 0.0 #Damping coefficient
f_0 = 0.0  #Coriolis parameter about which beta plane is centered. Set to zero for equatorial dynamics. 
beta =  3e-11 #The incline of the beta-plane i.e. f = f_0 + beta * (y - y_0)
y0 = Ly/2 #The latitude of f = f_0
fu = f_0 + beta * (yyv - y0) #The coriolis parameter evaluated at the grid points valid for u
fv = f_0 + beta * (yyv - y0) #The coriolis parameter evaluated at the grid points valid for v
alphau = fu * dt #Alpha term in the semi-implicit scheme used to update coriolis force
kappau = 0.25 * alphau**2 #Beta term in the semi-implicit scheme used to update coriolis force
alphav = fv * dt #Alpha term in the semi-implicit scheme used to update coriolis force
kappav = 0.25 * alphav**2 #Beta term in the semi-implicit scheme used to update coriolis force

#Initial conditions
gauss_width = 300e3 #Width of Gaussian bump
eta0 = np.exp(-((xxeta)**2 + (yyeta-Ly/2)**2)/(gauss_width**2)) #Gaussian bump
eta0[:,0] = 0
eta0[0,:] = 0
#eta0 = np.zeros((nx,ny)) #start model at rest
u0 = np.zeros((nx,ny)) #fluid at rest in the zonal
v0 = u0 #Fluid at rest in the meridional
u = u0 #Working solution for u
v = v0 #Working solution for v
eta = eta0 #Working solution for eta
D = D0 * np.ones((nx,ny))
Q = Q * np.exp(-((xxeta)**2 + (yyeta)**2)/(gauss_width**2)) #Gaussian bump
Q[:,0] = 0
Q[0,:] = 0

#Counter
i = 0

#Dynamical core
for n in range(nt+1): #Runs over all time steps
    unew = u - g * dt/dx * np.dot(fwdx,eta) - r * u * dt #Update u with respect to pressure gradient and rayleigh friction
    vnew = v - g * dt/dy * np.dot(fwdy,eta.T).T - r * v * dt  #Update v with respect to pressure gradient and rayleigh friction
    unew2 = ( unew - kappau * u + alphav * v)/(1 + kappau) #Update u with respect to coriolis force
    vnew2 = ( vnew - kappav * v - alphau * u)/(1 + kappav) #Update v with respect to coriolis force
    u = unew2
    v = vnew2
    u[-1,:] = 0 #Invoke boundary conditions
    u[0,:] = 0 #see above
    v[:,-1] = 0 #Invoke boundary conditions
    v[:,0] = 0 #see above
    uplus = 0.5 * (u + abs(u)) #Generate matrix that holds positive zonal vel
    vplus = 0.5 * (v + abs(v)) #Generate matrix that holds positive meridional vel
    uminus = 0.5 * (u - abs(u)) #Generate matrix that holds negative zonal vel
    vminus = 0.5 * (v - abs(v)) #Generate matrix that holds negative meridional vel
    eta = eta - dt/dx * (np.dot(kx,uplus) * np.dot(kx,eta + D) + np.dot(kx,uminus) * np.dot(kupx,eta + D) - np.dot(kdnx,uplus) * np.dot(kdnx, eta + D) - np.dot(kdnx,uminus) * np.dot(kx, eta + D))  - dt/dy * (np.dot(ky,vplus.T) * np.dot(ky,eta.T + D.T) + np.dot(ky,vminus.T) * np.dot(kupy,eta.T + D.T) - np.dot(kdny,vplus.T) * np.dot(kdny, eta.T + D.T) - np.dot(kdny,vminus.T) * np.dot(ky, eta.T + D.T)).T + Q * dt - l * eta * dt #Add forcing and damping
#Updates the sea surface height field
    eta = (1 - eps) * eta + 0.25 * eps * (np.dot(shx,eta) + np.dot(shy,eta.T).T) #Shapiro filter smoothing
    hovmoeller[n,:] = eta[:,100].T #Save array to hovmoeller matrix 
    if n % 100 == 0: #Below we plot
        plt.figure(figsize=(15,8))
        cs1 = plt.contourf((xxeta)/1e3,(yyeta)/1e3,eta,clevseta, aspect = 'auto', origin = 'lower', cmap = 'bwr', extend = 'both')
        c1 = plt.colorbar(cs1, ticks = cbartickseta)
        c1.set_label('Interface height [m]')
        plt.plot(np.array([0,xeta[-1]/1e3]),np.array([1e-3 * Ly/2, 1e-3 * Ly/2]),'k--')
        plt.grid('on')
        plt.ylabel('Y direction [km]')
        plt.xlabel('X direction [km]')
        plt.savefig('2dwave{0:03d}.png'.format(i))
        plt.close()
        i = i +1 #Update counter

#Plot evolution of model solution via a hovmoeller diagram
hovlevs = np.linspace(-0.15,0.15,50)
plt.figure(figsize = (15,8))
cs5 = plt.contourf(xeta/1e3,np.linspace(0,T,nt+1)/(3600 * 24),hovmoeller,hovlevs,cmap = 'bwr', origin = 'lower', aspect = 'auto', extend = 'both')
c5 = plt.colorbar(cs5,label = 'Interface height [m]')
plt.ylabel('Time [days]',)
plt.xlabel('X direction [km]')
plt.savefig('hovmoeller.png')
