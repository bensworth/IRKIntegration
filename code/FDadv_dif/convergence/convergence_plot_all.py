#Plot error data that's saved in files.
#
#NOTES:
# Assumes text files are stored in the format:
# <DIR>/IRK<TIMEDISC>_dt<REFINE>_d<DIMENSION>_FD<FD>

import numpy as np

import matplotlib
import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sys
from sys import argv
from numpy.linalg import norm
import os.path

import argparse

'''
1D problem.
python convergence_plot_all.py -dir data/ -d 1 -ex 1 -t 12 14 -l_min 4 4 -l_max 10 10

2D problem.
python convergence_plot_all.py -dir data/ -d 2 -ex 1 -t 12 14 -l_min 4 4 -l_max 8 8
'''

# Dictionary holding order of SDIRK schemes
IRK_order = {12 : 2,
             14 : 4}
             
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-dir','--dir', help = 'Directory of data', required = True)
parser.add_argument('-t','--SDIRK', nargs = "+", help = 'IDs of IRK methods', required = True)
parser.add_argument('-l_min','-l_min', nargs = "+", help = 'Min refinement', required = True)
parser.add_argument('-l_max','-l_max', nargs = "+", help = 'Max refinement', required = True)
parser.add_argument('-d','--d', help = 'Spatial dimension', required = True)
parser.add_argument('-ex','--problem', help = 'Example problem ID', required = True)
parser.add_argument('-n','--norm', help = 'Error norm (1, 2, inf)', required = False, default = 'inf')
parser.add_argument('-s','--save', help = 'Save figure', required = False, default = False)
args = vars(parser.parse_args())

print(args)

xlims = [None, None]
ylims = [None, None]
fs = {"fontsize": 18}

colours = ["g", "m", "b", "c", "y"]
markers = ["o", "v", "s", "*", "d"]


# Only turn on this setting if saving since it makes the script take ages...
if int(args["save"]):
    plt.rc("text", usetex = True)


# Template for filenames
def filenametemp(scheme, t, refine):
    return args["dir"] + "SDIRK" + t + "_l" + str(refine) + "_d" + args["d"] + "_ex" + args["problem"]


for scheme, t in enumerate(args["SDIRK"]):
    eL1  = []
    eL2  = []
    eLinf  = []
    dt = []
    
    for dt_refine in range(int(args["l_min"][scheme]), int(args["l_max"][scheme])+1):
        filename = filenametemp(scheme, t, dt_refine)
        print(filename)
    
        # If output filename exists, open it, otherwise create it
        if os.path.isfile(filename):
            params = {}
            with open(filename) as f:
                for line in f:
                   (key, val) = line.split()
                   params[key] = val
                   
        else:
            sys.exit("What are you doing? I cannot fine the file: ('{}')".format(filename))
    
    
        # Type cast the parameters from strings into their original types
        params["dt"]    = float(params["dt"])
        params["eL1"]   = float(params["eL1"])
        params["eL2"]   = float(params["eL2"])
        params["eLinf"] = float(params["eLinf"])
    
        dt.append(params["dt"])
        eL1.append(params["eL1"])
        eL2.append(params["eL2"])
        eLinf.append(params["eLinf"])
        

    # Cast from lists to numpy arrays for plotting, etc.
    dt    = np.array(dt)
    eL1   = np.array(eL1)
    eL2   = np.array(eL2)
    eLinf = np.array(eLinf)
    
    # Choose norm 
    if args["norm"] == "1":
        e = eL1; norm_str = "L_{{1}}"
    elif args["norm"] == "2":
        e = eL2; norm_str = "L_{{2}}"
    elif args["norm"] == "inf":    
        e = eLinf; norm_str = "L_{{\\infty}}"
    
    order = IRK_order[int(args["SDIRK"][scheme])]

    # Plot errors for given solves and line indicating expected asymptotic convergence rate
    anchor = -2 # Achor theoretical line to end-1th data point
    plt.loglog(dt, 0.5*e[anchor]*(dt/float(dt[anchor]))**(order), linestyle = '--', color = 'k')
    plt.loglog(dt, e, label = "$p={}$".format(order), marker = markers[scheme], color = colours[scheme], basex = 2)


axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)


plt.legend(fontsize = fs["fontsize"]-2)
plt.xlabel("$\delta t$", **fs)
plt.ylabel("$\\Vert \\mathbf{{e}} \\Vert_{{{}}}$".format(norm_str), **fs)
plt.title("$\\rm{{SDIRK}}$$(p)$+$\\rm{{C}}$$(p)$, ${{{}}}$$\\rm{{D-space}}$".format(int(args["d"])), **fs)



if int(args["save"]):    
    # Generate name to save figure with...
    out = "figures/SDIRK"
    out += "d" + args["d"] + "_ex" + args["problem"]
    plt.savefig('{}.pdf'.format(out), bbox_inches='tight')


plt.show()






