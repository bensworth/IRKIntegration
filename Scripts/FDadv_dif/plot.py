# ---------------------------------- #
# -------------- README ------------ #
# ---------------------------------- #
# Compute errors between  numerical and exact solutions of FD advection problems,
# can also plot both the numerical and  exact solutions at the final time


import numpy as np

import matplotlib
import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sys import argv
from numpy.linalg import norm

import pprint

import glob


# plt.rc("font", family="serif")
# plt.rc("text", usetex=True)
#matplotlib.rcParams["font.size"] = (9)


# Pass a 1 for PIT, or a 0 for sequential...
# Additionally, pass a second argument as a 1 or 0 for the solution to be plotted or not
# if len(argv) > 1:
#     suff = argv[1]
#     if len(argv) > 2:
#         doaplot = bool(int(argv[2]))
#     else:
#         doaplot = False
# else:
#    suff = 0
doaplot = True

U_filename = "data/U"
U_exact_filename = "data/U_exact"

if not glob.glob(U_filename + ".*"):
    raise ValueError('No data exists at: ' + U_filename)

U_exact_exists = True
if not glob.glob(U_exact_filename + ".*"):
    U_exact_exists = False
    print("============================================")
    print('WARNING: No exact solution data exists at: ' + U_exact_filename)


print("============================================")
print("Reading data from: " + U_filename)
if doaplot: 
    print("PLOTTING DATA...")
    # Sit there here to enable latex-style font in plots...
    plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['text.usetex'] = True
    fs = {"fontsize": 18, "usetex": True}
if not doaplot: print("NOT PLOTTING DATA...")

# Read data in and store in dictionary
params = {}
with open(U_filename) as f:
    for line in f:
       (key, val) = line.split()
       params[key] = val
       
# Type cast the parameters from strings into their original types
params["P"]               = int(params["P"])
params["nt"]              = int(params["nt"])
params["dt"]              = float(params["dt"])
params["problemID"]       = int(params["problemID"])
params["nx"]              = int(params["nx"])
params["dx"]              = float(params["dx"])
params["space_dim"]       = int(params["space_dim"])
#params["spatialParallel"] = int(params["spatialParallel"])

# Total number of DOFS in space
if params["space_dim"] == 1:
    NX = params["nx"] 
elif params["space_dim"] == 2:
    NX = params["nx"] ** 2
    #NX = params["nx"] * (params["nx"] + 7)
    
print("--INPUT--")
pprint.pprint(params)


###############################
#  --- Sequential in time --- #
###############################
# Get names of all procs holding data
PuT = []
PuT_exact = []
for P in range(0, params["P"]):
    PuT.append(U_filename + "." + str(P))
    PuT_exact.append(U_exact_filename + "." + str(P))

uT = np.zeros(NX)
uT_exact = np.zeros(NX)

def readInData(filenames, u):
    ind = 0
    for count, filename in enumerate(filenames):
        # Read all data from the proc
        with open(filename) as f:
            dims = f.readline()
        dims.split(" ")
        dims = [int(x) for x in dims.split()] 
        # Get data from lines > 0
        temp = np.loadtxt(filename, skiprows = 1, usecols = 0, dtype = np.float64) # Ignore the 1st column since it's just indices..., top row has junk in it we don't need
        DOFsOnProc = temp.shape[0]
        u[ind:ind+DOFsOnProc] = temp
        ind += DOFsOnProc
    return u
    
uT = readInData(PuT, uT)
if U_exact_exists: uT_exact = readInData(PuT_exact, uT_exact)

### -------------------------------------------- ###
### --- PLOTTING: uT against exact solution ---  ###
### -------------------------------------------- ###

### ----------------------------------- ###
### --------------- 1D  --------------- ###
### ----------------------------------- ###
if params["space_dim"] == 1:
    nx = NX
    nt = params["nt"]
    x = np.linspace(-1, 1, nx+1)    
    if (params["problemID"] <= 100):
        x = x[:-1] # nx points in [-1, 1) for periodic
    else:
        x = x[1:] # nx points in (-1, 1] for inflow
    T = params["dt"] * nt

    # Plot data if requested...
    if doaplot:
        if U_exact_exists: plt.plot(x, uT_exact, linestyle = "--", marker = "o", markerfacecolor = "none", color = "r", label = "$u_{{\\rm{exact}}}$")
        plt.plot(x, uT, linestyle = "--", marker = "x", color = "b", label = "$u_{{\\rm{num}}}$")
        #plt.semilogy(x, np.abs(uT_exact - uT), linestyle = "--", marker = "o", markerfacecolor = "none", color = "r", label = "$u_{{\\rm{exact}}}$")
        plt.legend(fontsize = fs["fontsize"]-2)
        plt.title("$u(x,{:.2f})$".format(T), **fs)
        plt.xlabel("$x$", **fs)
        plt.show()
    
    
### ----------------------------------- ###
### --------------- 2D  --------------- ###
### ----------------------------------- ###
if params["space_dim"] == 2:
    nx = params["nx"]
    ny = params["nx"]
    nt = params["nt"]
    x = np.linspace(-1, 1, nx+1)
    y = np.linspace(-1, 1, ny+1)
    x = x[:-1] # nx points in [-1, 1)
    y = y[:-1] # ny points in [-1, 1)
    [X, Y] = np.meshgrid(x, y)
    T = params["dt"] * nt

    # If used spatial parallelism, DOFs are not ordered in row-wise lexicographic, but instead
    # are blocked by proc, with procs in row-wise lexicographic order and DOFs on proc ordered
    # in row-wise lexicographic order
    if (params["P"] > 1):
        perm = np.zeros(nx*ny, dtype = "int32")
        # Extract dimensions of processor grid if they were given
        if ("np0" in params):
            npInX = int(params["np0"])
            npInY = int(params["np1"])
        # Otherwise assume square processor grid
        else:
            npInX = int(np.sqrt(params["P"])) 
            npInY = npInX 
        count = 0
        
        nxOnProcInt = int(nx / npInX)
        nyOnProcInt = int(ny / npInY)
        nxOnProcBnd = nx - (npInX-1)*int(nx / npInX) 
        nyOnProcBnd = ny - (npInY-1)*int(ny / npInY) 
        
        # Loop over DOFs in ascending order (i.e., ascending in their current order)
        for py in range(0, npInY):
            if (py < npInY-1):
                nyOnProc = nyOnProcInt
            else:
                nyOnProc = nyOnProcBnd
            for px in range(0, npInX):
                if (px < npInX - 1):
                    nxOnProc = nxOnProcInt 
                else:
                    nxOnProc = nxOnProcBnd
                for yIndOnProc in range(0, nyOnProc):
                    for xIndOnProc in range(0, nxOnProc):
                        xIndGlobal = px * nxOnProcInt + xIndOnProc # Global x-index for row we're in currently
                        globalInd  = py * (nyOnProcInt * nx) + yIndOnProc * nx + xIndGlobal # Global index of current DOF in ordering we want
                        perm[globalInd] = count
                        count += 1
        uT = uT[perm] # Permute solution array into correct ordering
        if U_exact_exists: uT_exact = uT_exact[perm]
        
    # Map 1D array into 2D for plotting.
    uT = uT.reshape(ny, nx)
    uT_exact = uT_exact.reshape(ny, nx)
    
    # Plot data if requested...
    if doaplot:
        cmap = plt.cm.get_cmap("coolwarm")
        # ax = fig.gca(projection='3d') 
        # surf = ax.plot_surface(X, Y, uT, cmap = cmap)
        
        ### --- Numerical solution --- ###
        fig = plt.figure(1)
        levels = np.linspace(np.amin(uT, axis = (0,1)), np.amax(uT, axis = (0,1)), 20)
        plt.contourf(X, Y, uT, levels=levels,cmap=cmap)
        plt.colorbar(ticks=np.linspace(np.amin(uT), np.amax(uT), 7), format='%0.1f')	
        
        #### For setting up colour bar between (0,1)
        # levels = np.linspace(0, 1, 20)
        # plt.contourf(X, Y, uT, levels=levels, vmin=0, vmax=1, cmap=cmap)
        # plt.colorbar(ticks=np.linspace(0, 1, 7), format='%0.1f')	
        
        plt.title("$u_{{\\rm{{num}}}}(x,y,{:.2f})$".format(T), **fs)
        plt.xlabel("$x$", **fs)
        plt.ylabel("$y$", **fs)
        #plt.savefig('out.pdf', bbox_inches='tight')
        
        ### --- Analytical solution --- ###
        if U_exact_exists:
            fig = plt.figure(2)
            levels = np.linspace(np.amin(uT_exact, axis = (0,1)), np.amax(uT_exact, axis = (0,1)), 20)
            plt.contourf(X, Y, uT_exact, levels=levels,cmap=cmap)
            plt.colorbar(ticks=np.linspace(np.amin(uT_exact), np.amax(uT_exact), 7), format='%0.1f')	
            
            plt.title("$u_{{\\rm exact}}(x,y,{:.2f})$".format(T), **fs)
            plt.xlabel("$x$", **fs)
            plt.ylabel("$y$", **fs)
            
        # fig = plt.figure(2)
        # ax = fig.gca(projection='3d') 
        # surf = ax.plot_surface(X, Y, uT_exact, cmap = cmap)
        # plt.title("uTexact", fontsize = fs)
        # plt.xlabel("$x$", fontsize = fs)
        # plt.ylabel("$y$", fontsize = fs)
        plt.show()    
        
        # plt.title("UW2-2D: u(x,y, t = {:.2f})".format(t[-1]), fontsize = 15)
        # plt.xlabel("x", fontsize = 15)
        # plt.ylabel("y", fontsize = 15)


print("============================================\n")