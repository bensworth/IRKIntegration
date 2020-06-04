
import numpy as np

import matplotlib
import matplotlib.pylab as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

import sys
from sys import argv
from numpy.linalg import norm
import os.path

import argparse

'''
Plot Krylov and Newton iterations and discretization errors

--------- 1D ----------
--- 4th-order schemes
python iter_plots.py -dir data_1d/ -d 1 -ex 1 -t 14 34 23 -14 4 -dt_min 2 2 2 2 2 -dt_max 8 8 8 8 8 

--------- 2D ----------
--- 4th-order schemes
python iter_plots.py -dir data_2d/ -d 2 -ex 1 -t 14 34 23 -14 4 -dt_min 2 2 2 2 2 -dt_max 5 5 5 5 5

--- 8th-order schemes
python iter_plots.py -dir data_2d/ -d 2 -ex 1 -t 18 38 27 -dt_min 2 2 2 -dt_max 4 4 4 

'''

# IRK types
IRK_type = {-1: "ASDIRK",
            0 : "LSDIRK", 
            1 : "Gauss",
            2 : "Radau\,IIA",
            3 : "Lobatto\,IIIC"}

# Dictionary holding order of IRK schemes
IRK_order = { -14 : 4,
                1 : 1,
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
                
IRK_label = { -14 : "A\\rm{-}SDIRK(4)",
                1 : "L\\rm{-}SDIRK(1)",
                2 : "L\\rm{-}SDIRK(2)",
                3 : "L\\rm{-}SDIRK(3)",
                4 : "L\\rm{-}SDIRK(4)",
                12 : "Gauss(2)",
                14 : "Gauss(4)",
                16 : "Gauss(6)",
                18 : "Gauss(8)",
                110 : "Gauss(10)",
                23 : "Radau\, IIA(3)",
                25 : "Radau\, IIA(5)",
                27 : "Radau\, IIA(7)",
                29 : "Radau\, IIA(9)",
                32 : "Lobatto\, IIIC(2)",
                34 : "Lobatto\, IIIC(4)",
                36 : "Lobatto\, IIIC(6)",
                38 : "Lobatto\, IIIC(8)"}                

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-dir','--dir', help = 'Directory of data', required = True)
parser.add_argument('-t','--IRK', nargs = "+", help = 'IDs of IRK methods', required = True)
parser.add_argument('-dt_min','--dt_min', nargs = "+", help = 'Min temporal refinement', required = True)
parser.add_argument('-dt_max','--dt_max', nargs = "+", help = 'Max temporal refinement', required = True)
parser.add_argument('-d','--d', help = 'Spatial dimension', required = True)
parser.add_argument('-ex','--ex', help = 'Finite difference problem ID', required = True)
parser.add_argument('-n','--norm', help = 'Error norm (1, 2, inf)', required = False, default = 'inf')
parser.add_argument('-s','--save', help = 'Save figure', required = False, default = False)
args = vars(parser.parse_args())

print(args)

xlims = [None, None]
ylims = [None, None]
fs = {"fontsize": 18}

colours = ["g", "m", "b", "c", "y"]
markers = ["o", "v", "s", "*", "d"]
linestyles = ["-", ":", "-.", "--", "-"]


# Only turn on this setting if saving since it makes the script take ages...
if int(args["save"]):
    plt.rc("text", usetex = True)


# Template for filenames
def filenametemp(scheme, t, dt_refine):
    return args["dir"] + "IRK" + t + "_dt" + str(dt_refine) + "_d" + args["d"] + "_ex" + args["ex"]

orders_plotted = []
order = 0
for scheme, t in enumerate(args["IRK"]):

    eL1  = []
    eL2  = []
    eLinf  = []
    dt = []
    amg_iters = []
    newton_iters = []
    
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
            sys.exit("What are you doing? I cannot find the file: ('{}')".format(filename))
    
        # Cast from lists to numpy arrays for plotting, etc.
        dt.append(float(params["dt"]))
        eL1.append(float(params["eL1"]))
        eL2.append(float(params["eL2"]))
        eLinf.append(float(params["eLinf"]))
        newton_iters.append(int(params["newton_iters"]))
    
        nsys = int(params["nsys"])
        solves = 0
        for system in range(0,nsys):
            if (int(params["sys" + str(system + 1) + "_type"]) == 1):
                solves += int(params["sys" + str(system + 1) + "_iters"])
                #print(t, int(params["sys" + str(system + 1) + "_iters"]))
            else:
                #print(t, int(params["sys" + str(system + 1) + "_iters"]))
                solves += 2*int(params["sys" + str(system + 1) + "_iters"])
        amg_iters.append(solves)

        order = IRK_order[int(args["IRK"][scheme])]


#### Finished with current IRK scheme ###

    
    dt = np.array(dt)
    eL1   = np.array(eL1)
    eL2   = np.array(eL2)
    eLinf = np.array(eLinf)
    newton_iters = np.array(newton_iters)
    amg_iters = np.array(amg_iters)

    scheme_label = "$\\rm{{{}}}$".format(IRK_label[int(args["IRK"][scheme])])

    plt.figure(1)
    plt.semilogx(dt, newton_iters, label = scheme_label, marker = markers[scheme], 
                    color = colours[scheme], linestyle = linestyles[scheme], basex = 2)
        
    plt.figure(2)
    plt.semilogx(dt, amg_iters, label = scheme_label, marker = markers[scheme], 
                    color = colours[scheme], linestyle = linestyles[scheme], basex = 2)

    # Choose norm 
    if args["norm"] == "1":
        e = eL1; norm_str = "L_{{1}}"
    elif args["norm"] == "2":
        e = eL2; norm_str = "L_{{2}}"
    elif args["norm"] == "inf":    
        e = eLinf; norm_str = "L_{{\\infty}}"
    
    plt.figure(3)
    order = IRK_order[int(args["IRK"][scheme])]
    if (order not in orders_plotted):
        orders_plotted.append(order)
        # Plot errors for given solves and line indicating expected asymptotic convergence rate
        anchor = -1 # Achor theoretical line to end-1th data point
        plt.loglog(dt, 0.5*e[anchor]*(dt/float(dt[anchor]))**(order), linestyle = '--', color = 'k')
    plt.semilogx(dt, e, label = scheme_label, marker = markers[scheme], 
                    color = colours[scheme], linestyle = linestyles[scheme], basex = 2)


plt.figure(1)
axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)

plt.legend(fontsize = fs["fontsize"]-2)
plt.xlabel("$\delta t$", **fs)
plt.title("$\\rm{Newton\, iterations\, per\, time\, step}$", **fs)

if int(args["save"]):    
    out = "convergence_plots/newton_iters_O" + str(order) + "_dim" + args["d"] + ".pdf"
    plt.savefig(out, bbox_inches='tight')


plt.figure(2)
axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)

plt.legend(fontsize = fs["fontsize"]-2)
plt.xlabel("$\delta t$", **fs)
plt.title("$\\rm{AMG\, iterations\, per\, time\, step}$", **fs)

if int(args["save"]):    
    out = "convergence_plots/amg_iters_O" + str(order) + "_dim" + args["d"] + ".pdf"
    plt.savefig(out, bbox_inches='tight')
    
    
plt.figure(3)
axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)

plt.legend(fontsize = fs["fontsize"]-2)
plt.xlabel("$\delta t$", **fs)
plt.title("${{{}}} \, \, \\rm{{discretization\\ error}}$".format(norm_str), **fs)

if int(args["save"]):    
    out = "convergence_plots/errors_O" + str(order) + "_dim" + args["d"] + ".pdf"
    plt.savefig(out, bbox_inches='tight')    

plt.show()



