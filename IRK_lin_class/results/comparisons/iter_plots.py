
import IRK_info
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
Plot AMG iterations, timings, and discretization errors


4th-order schemes

8th-order schemes

python iter_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -d 2 -ex 1 -t 18 -h_min 3 -h_max 7
python iter_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -d 2 -ex 1 -t 27 -h_min 3 -h_max 7
python iter_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -d 2 -ex 1 -t 38 -h_min 3 -h_max 7


If you want to save the plots as PDFs, also pass the option "-s 1"
'''
               
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-dir','--dir', nargs = "+", help = 'Directory of data', required = True)
parser.add_argument('-t','--IRK', nargs = 1, help = 'IDs of IRK methods', required = True)
parser.add_argument('-h_min','--h_min', nargs = "+", help = 'Min temporal refinement', required = True)
parser.add_argument('-h_max','--h_max', nargs = "+", help = 'Max temporal refinement', required = True)
parser.add_argument('-d','--d', help = 'Spatial dimension', required = True)
parser.add_argument('-ex','--ex', help = 'Finite difference problem ID', required = True)
parser.add_argument('-n','--norm', help = 'Error norm (1, 2, inf)', required = False, default = 'inf')
parser.add_argument('-s','--save', help = 'Save figure', required = False, default = False)
args = vars(parser.parse_args())

print(args)

# Get IRK information
IRKOrder = IRK_info.Orders()
IRKStage = IRK_info.Stages()
IRKIndividualLabel = IRK_info.IndividualLabels()

xlims = [None, None]
ylims = [None, None]
labelsize = {"fontsize": 22}
legendsize = 20
ticksize = 20

colours = ["g", "m", "b", "c", "y"]
markers = ["o", "v", "s", "*", "d"]
linestyles = ["-", ":", "-.", "--", "-"]


# Only turn on this setting if saving since it makes the script take ages...
if int(args["save"]):
    plt.rc("text", usetex = True)


# Template for filenames
def filenametemp(dir, scheme, t, h_refine):
    return dir + "IRK" + t + "_h" + str(h_refine) + "_d" + args["d"] + "_ex" \
     + args["ex"] + ".probinfo.out"

orders_plotted = []
order = 0
runtime_all = {}
insert_order = {}
for scheme, t in enumerate(args["IRK"]):
    for alg, dir in enumerate(args["dir"]):
        eL1  = []
        eL2  = []
        eLinf  = []
        dt = []
        runtime = []
        amg_iters = []
        
        for h_refine in range(int(args["h_min"][scheme]), int(args["h_max"][scheme])+1):
            filename = filenametemp(dir, scheme, t, h_refine)
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
            runtime.append(float(params["runtime"]))
            dt.append(float(params["dt"]))
            eL1.append(float(params["eL1"]))
            eL2.append(float(params["eL2"]))
            eLinf.append(float(params["eLinf"]))
        
            # CC-preconditioning
            if int(params["IRK_alg"]) == 0:    
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
            
            # Block preconditioning
            else:
                solves = int(params["iters"]) * IRKStage[int(args["IRK"][scheme])]
                amg_iters.append(solves)
            

        #### Finished with current IRK scheme ###
        dt = np.array(dt)
        runtime = np.array(runtime)
        eL1   = np.array(eL1)
        eL2   = np.array(eL2)
        eLinf = np.array(eLinf)
        amg_iters = np.array(amg_iters)

        #scheme_label = "$\\rm{{{}}}$".format(IRKIndividualLabel[int(args["IRK"][scheme])])
        #scheme_label = "$\\rm{{{}}}$".format(IRKIndividualLabel[int(args["IRK"][scheme])])

        if "CC" in dir: 
            scheme_label = "CC"
        elif "PREC1" in dir:
            scheme_label = "GSL"
        elif "PREC4" in dir:
            scheme_label = "LD"
        
        
        # Insert the current alg's runtime into global dict
        insert_order[scheme_label] = alg
        # Keep track of the index with which it was inserted
        runtime_all[scheme_label] = runtime
        
        
        # Choose norm 
        if args["norm"] == "1":
            e = eL1; norm_str = "L_{{1}}"
        elif args["norm"] == "2":
            e = eL2; norm_str = "L_{{2}}"
        elif args["norm"] == "inf":    
            e = eLinf; norm_str = "L_{{\\infty}}"    

        plt.figure(2)
        plt.semilogx(dt, amg_iters, label = scheme_label, marker = markers[alg], 
                        color = colours[alg], linestyle = linestyles[alg], basex = 2)

        
        plt.figure(3)
        order = IRKOrder[int(args["IRK"][scheme])]
        if (order not in orders_plotted):
            orders_plotted.append(order)
            # Plot errors for given solves and line indicating expected asymptotic convergence rate
            anchor = -1 # Achor theoretical line to end-1th data point
            plt.loglog(dt, 0.5*e[anchor]*(dt/float(dt[anchor]))**(order), linestyle = '--', color = 'k')
        plt.semilogx(dt, e, label = scheme_label, marker = markers[alg], 
                        color = colours[alg], linestyle = linestyles[alg], basex = 2)


        # Runtime plot
        plt.figure(4)
        plt.loglog(dt, runtime, label = scheme_label, marker = markers[alg], 
                        color = colours[alg], linestyle = linestyles[alg], basex = 2, basey = 2)


# Relative runtime plot
if "CC"in runtime_all and "GSL" in runtime_all:    
    plt.figure(5)
    plt.semilogx(dt, runtime_all["GSL"]/runtime_all["CC"], label = "CC/GSL", marker = markers[insert_order["CC"]], 
                color = colours[insert_order["CC"]], linestyle = linestyles[insert_order["CC"]], basex = 2)
if "LD"in runtime_all and "GSL" in runtime_all:
    plt.figure(5)
    plt.semilogx(dt, runtime_all["GSL"]/runtime_all["LD"], label = "LD/GSL", marker = markers[insert_order["GSL"]], 
                color = colours[insert_order["GSL"]], linestyle = linestyles[insert_order["GSL"]], basex = 2)


 

# IRK scheme's label
IRK_label = "$\\rm{{{}}}$".format(IRKIndividualLabel[int(args["IRK"][0])])

if int(args["save"]):
    schemes_label = "_".join(args["IRK"])

plt.figure(2)
axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)
axes.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend(fontsize=legendsize)
plt.xlabel("$\delta t$", **labelsize)
plt.ylabel("$\\rm{AMG\,\, iterations}$", **labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.title(IRK_label, **labelsize)

if int(args["save"]):    
    out = "amg_iters_" + args["IRK"][0] + "_d" + args["d"] + "_ex" + args["ex"] + ".pdf"
    plt.savefig(out, bbox_inches='tight')
    
    
plt.figure(3)
axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)

plt.legend(fontsize=legendsize)
plt.xlabel("$\delta t$", **labelsize)
plt.ylabel("$\\rm{Discretization\, error}$", **labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.title(IRK_label, **labelsize)

if int(args["save"]):    
    out = "errors_iters_" + args["IRK"][0] + "_d" + args["d"] + "_ex" + args["ex"] + ".pdf"
    plt.savefig(out, bbox_inches='tight')  

plt.figure(4)
axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)

plt.legend(fontsize=legendsize)
plt.xlabel("$\delta t$", **labelsize)
plt.ylabel("$\\rm{runtime\, (s)}$", **labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.title(IRK_label, **labelsize)

if int(args["save"]):    
    out = "runtime_" + args["IRK"][0] + "_d" + args["d"] + "_ex" + args["ex"] + ".pdf"
    plt.savefig(out, bbox_inches='tight')   
    
plt.figure(5)
axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)

plt.legend(fontsize=legendsize)
plt.xlabel("$\delta t$", **labelsize)
plt.ylabel("$\\rm{relative\,\,runtime}$", **labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.title(IRK_label, **labelsize)

if int(args["save"]):    
    out = "rel_runtime_" + args["IRK"][0] + "_d" + args["d"] + "_ex" + args["ex"] + ".pdf"
    plt.savefig(out, bbox_inches='tight')        

plt.show()



