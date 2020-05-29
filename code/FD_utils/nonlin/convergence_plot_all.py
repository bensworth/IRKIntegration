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
1D problems...

2D problems...
python convergence_plot_all.py -dir data/ -d 2 -ex 1 -g 0 -t 1 2 3 4 -dt_min 2 2 2 2 -dt_max 5 5 5 5
python convergence_plot_all.py -dir data/ -d 2 -ex 1 -g 1 -t 12 14 16 18 110 -dt_min 2 2 2 2 2 -dt_max 5 5 5 4 4
python convergence_plot_all.py -dir data/ -d 2 -ex 1 -g 2 -t 23 25 27 29 -dt_min 2 2 2 2 -dt_max 5 5 4 4
python convergence_plot_all.py -dir data/ -d 2 -ex 1 -g 3 -t 32 34 36 38 -dt_min 2 2 2 2 -dt_max 5 5 4 4
'''

# IRK types
IRK_type = {0 : "SDIRK", 
            1 : "Gauss",
            2 : "Radau\,IIA",
            3 : "Lobatto\,IIIC"}

# Dictionary holding order of IRK schemes
IRK_order = {1 : 1,
                2 : 2,
                3 : 3,
                4 : 4,
                12 : 2,
                14 : 4,
                16 : 6,
                18 : 8,
                110 : 10,
                23 : 3,
                25 : 5,
                27 : 7,
                29 : 9,
                32 : 2,
                34 : 4,
                36 : 6,
                38 : 8}

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-dir','--dir', help = 'Directory of data', required = True)
parser.add_argument('-g','--IRK_type', nargs = 1, help = 'Group ID of common IRK methods', required = False)
parser.add_argument('-t','--IRK', nargs = "+", help = 'IDs of IRK methods', required = True)
parser.add_argument('-dt_min','-dt_min', nargs = "+", help = 'Min temporal refinement', required = True)
parser.add_argument('-dt_max','-dt_max', nargs = "+", help = 'Max temporal refinement', required = True)
parser.add_argument('-d','--d', help = 'Spatial dimension', required = True)
parser.add_argument('-ex','--ex', help = 'Example problem ID', required = True)
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
def filenametemp(scheme, t, dt_refine):
    return args["dir"] + "IRK" + t + "_dt" + str(dt_refine) + "_d" + args["d"] + "_ex" + args["ex"]


for scheme, t in enumerate(args["IRK"]):
    eL1  = []
    eL2  = []
    eLinf  = []
    dt = []
    
    for dt_refine in range(int(args["dt_min"][scheme]), int(args["dt_max"][scheme])+1):
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
    
    order = IRK_order[int(args["IRK"][scheme])]

    # Plot errors for given solves and line indicating expected asymptotic convergence rate
    anchor = -1 # Achor theoretical line to end-1th data point
    plt.loglog(dt, 0.5*e[anchor]*(dt/float(dt[anchor]))**(order), linestyle = '--', color = 'k')
    
    if args["IRK_type"] is None:
        plt.loglog(dt, e, label = "IRK{}".format(args["IRK"][scheme]), marker = markers[scheme], color = colours[scheme], basex = 2)
    else:
        plt.loglog(dt, e, label = "$p={}$".format(order), marker = markers[scheme], color = colours[scheme], basex = 2)


axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)


plt.legend(fontsize = fs["fontsize"]-2)
plt.xlabel("$\delta t$", **fs)
plt.ylabel("$\\Vert \\mathbf{{e}} \\Vert_{{{}}}$".format(norm_str), **fs)


# Set the title to the common IRK group if one was passed
if args["IRK_type"] is not None:
    plt.title("$\\rm{{{}}}$$(p)$".format(IRK_type[int(args["IRK_type"][0])]), **fs)

if int(args["save"]):    
    # Generate name to save figure with...
    out = "convergence_plots/"

    if args["IRK_type"] is not None:
        out += IRK_type[int(args["IRK_type"][0])].replace("\,", "") + "_"
    else:
        out += "unknown_IRK_group_"

    out += "d" + args["d"] + "_ex" + args["ex"]
    plt.savefig('{}.pdf'.format(out), bbox_inches='tight')


plt.show()




# 
# if int(args["pit"]):
#     title = "Space-time: "
# else:
#     title = "Time-stepping: "
# 
# title += args["d"] + "D"
# title += ", " + args["FD"]
# print(title)
# plt.title(title, **fs)
# 
# if int(args["save"]):    
#     # Generate name to save figure with...
#     filenameOUT = "plots/" + params["time"] + "/"
#     if int(args["pit"]):
#         filenameOUT += "spaceTime_"
#     else:
#         filenameOUT += "timeStepping_"
# 
#     if int(params["implicit"]):
#         filenameOUT += "implicit_"
#     else:
#         filenameOUT += "explicit_"    
# 
# 
#     filenameOUT += "d" + args["d"] + "_FD" + args["FD"]
#     plt.savefig('{}.pdf'.format(filenameOUT), bbox_inches='tight')
# plt.show()  




