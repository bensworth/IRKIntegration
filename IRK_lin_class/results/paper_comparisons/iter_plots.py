
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
python iter_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -runs 3 -d 2 -ex 1 -t 14 -h_min 3 -h_max 7
python iter_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -runs 3 -d 2 -ex 1 -t 23 -h_min 3 -h_max 7
python iter_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -runs 3 -d 2 -ex 1 -t 34 -h_min 3 -h_max 7

8th-order schemes
python iter_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -runs 3 -d 2 -ex 1 -t 18 -h_min 3 -h_max 7
python iter_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -runs 3 -d 2 -ex 1 -t 27 -h_min 3 -h_max 7
python iter_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -runs 3 -d 2 -ex 1 -t 38 -h_min 3 -h_max 7



### ---------------------- ###
### --- 4th-order data --- ###
### ---------------------- ###
IRK: Gauss(4)  (i.e., IRK ID= 14 )
dt: [0.5     0.25    0.125   0.0625  0.03125]
Runtime GSL/CC:  [1.39807762 1.31715614 1.20377757 1.2020509  1.09324209]
Runtime LD/CC:  [1.47455081 1.27568863 1.12122409 1.12464528 1.0750736 ]
Mean runtime GSL/CC for all dt: 1.24
Mean runtime LD/CC for all dt: 1.21

IRK: Radau\, IIA(3)  (i.e., IRK ID= 23 )
dt: [0.5     0.25    0.125   0.0625  0.03125]
Runtime GSL/CC:  [2.25393043 1.404976   1.33924277 1.09334795 1.09681703]
Runtime LD/CC:  [1.50054506 0.98092899 1.16971995 1.14273367 1.0213346 ]
Mean runtime GSL/CC for all dt: 1.44
Mean runtime LD/CC for all dt: 1.16

IRK: Lobatto\, IIIC(4)  (i.e., IRK ID= 34 )
dt: [0.5     0.25    0.125   0.0625  0.03125]
Runtime GSL/CC:  [1.95057277 2.05680154 1.97814484 2.29358559 2.16837668]
Runtime LD/CC:  [1.61698076 1.99988276 1.69701897 1.94038248 2.06944646]
Mean runtime GSL/CC for all dt: 2.09
Mean runtime LD/CC for all dt: 1.86

### ---------------------- ###
### --- 8th-order data --- ###
### ---------------------- ###
IRK: Gauss(8)  (i.e., IRK ID= 18 )
dt: [0.5     0.25    0.125   0.0625  0.03125]
Runtime GSL/CC:  [1.35146597 1.83188366 1.6617188  1.65773106 1.71697052]
Runtime LD/CC:  [1.50896057 1.71635166 1.5291348  1.53970988 1.55030776]
Mean runtime GSL/CC for all dt: 1.64
Mean runtime LD/CC for all dt: 1.57

IRK: Radau\, IIA(7)  (i.e., IRK ID= 27 )
dt: [0.5     0.25    0.125   0.0625  0.03125]
Runtime GSL/CC:  [1.85320559 1.95770638 1.90902953 1.85187822 2.01632721]
Runtime LD/CC:  [1.79980109 1.61526363 1.47513455 1.56519029 1.74140747]
Mean runtime GSL/CC for all dt: 1.92
Mean runtime LD/CC for all dt: 1.64

IRK: Lobatto\, IIIC(8)  (i.e., IRK ID= 38 )
dt: [0.5     0.25    0.125   0.0625  0.03125]
Runtime GSL/CC:  [2.37764233 3.17064294 3.00095761 2.93785829 3.37818054]
Runtime LD/CC:  [1.9401637  2.22211367 2.20338145 2.23018212 2.49129144]
Mean runtime GSL/CC for all dt: 2.97
Mean runtime LD/CC for all dt: 2.22


If you want to save the plots as PDFs, also pass the option "-s 1"
'''
               
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-dir','--dir', nargs = "+", help = 'Directory of data', required = True)
parser.add_argument('-t','--IRK', nargs = 1, help = 'IDs of IRK methods', required = True)
parser.add_argument('-runs','--runs', nargs = 1, help = 'Number of runs to average', required = True)
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
def filenametemp(dir, scheme, t, h_refine, run):
    return dir + "IRK" + t + "_h" + str(h_refine) + "_d" + args["d"] + "_ex" \
     + args["ex"] + "_run" + str(run) + ".probinfo.out"

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
            # Loop over the different runs
            for run in range(1, int(args["runs"][0])+1):
                filename = filenametemp(dir, scheme, t, h_refine, run)
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
            
                # Extract runtime from given run
                # Append runtime to list if it's the first one
                if run == 1:
                    runtime.append(float(params["runtime"]))
                # If we're on a later run, add it to the runtime appended above    
                else:
                    runtime[-1] += float(params["runtime"])
            # Now average all of the above runtimes        
            runtime[-1] /= int(args["runs"][0])    
               
            # Now extract all of these other quantities that will be the same across all runs.
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
if "CC" in runtime_all and "GSL" in runtime_all:    
    plt.figure(5)
    plt.semilogx(dt, runtime_all["GSL"]/runtime_all["CC"], label = "GSL/CC", marker = markers[insert_order["CC"]], 
                color = colours[insert_order["GSL"]], linestyle = linestyles[insert_order["CC"]], basex = 2)
if "LD"in runtime_all and "GSL" in runtime_all:
    plt.figure(5)
    plt.semilogx(dt, runtime_all["LD"]/runtime_all["CC"], label = "LD/CC", marker = markers[insert_order["LD"]], 
                color = colours[insert_order["LD"]], linestyle = linestyles[insert_order["LD"]], basex = 2)
    
# Print out the stuff for table entries
print("IRK:", IRKIndividualLabel[int(args["IRK"][0])], " (i.e., IRK ID=", args["IRK"][0], ")")                
print("dt:", dt)
print("Runtime GSL/CC: ", runtime_all["GSL"]/runtime_all["CC"])
print("Runtime LD/CC: ", runtime_all["LD"]/runtime_all["CC"])
print("Mean runtime GSL/CC for all dt: {:.2f}".format(np.mean(runtime_all["GSL"]/runtime_all["CC"])))
print("Mean runtime LD/CC for all dt: {:.2f}".format(np.mean(runtime_all["LD"]/runtime_all["CC"])))

 

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



