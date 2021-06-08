
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
Plot discretization error as a function of relative GMRES halting tolerance


python error_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -t 110 -l 6 -d 2 -ex 1 -r_min 4 -r_max 12
python error_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -t 29 -l 6 -d 2 -ex 1 -r_min 4 -r_max 12
python error_plots.py -dir data/CC_ data/PREC1_ data/PREC4_ -t 38 -l 6 -d 2 -ex 1 -r_min 4 -r_max 12

If you want to save the plots as PDFs, also pass the option "-s 1"
'''
               
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-dir','--dir', nargs = "+", help = 'Directory of data', required = True)
parser.add_argument('-t','--IRK', help = 'IDs of IRK methods', required = True)
parser.add_argument('-l','--l_refine', help = 'Spatial refinement', required = True)
parser.add_argument('-r_min','--reltol_min', nargs = "+", help = 'Min reltol refinement', required = True)
parser.add_argument('-r_max','--reltol_max', nargs = "+", help = 'Max reltol refinement', required = True)
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
def filenametemp(dir, reltol_refine):
    return dir + "IRK" + args["IRK"] + "_l" + args["l_refine"] + "_d" + args["d"] + "_ex" + args["ex"] + "_r" + str(reltol_refine)


for alg_idx, alg in enumerate(args["dir"]):

    eL1  = []
    eL2  = []
    eLinf  = []
    dt = []
    runtime = []
    amg_iters = []
            
    reltol_refine_array = np.arange(int(args["reltol_min"][0]), int(args["reltol_max"][0])+1)
    print(reltol_refine_array)
            
    for reltol_refine in reltol_refine_array:
        filename = filenametemp(alg, reltol_refine)
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
            
                
    #### Finished with current IRK scheme ###
    dt = np.array(dt)
    runtime = np.array(runtime)
    eL1   = np.array(eL1)
    eL2   = np.array(eL2)
    eLinf = np.array(eLinf)
    amg_iters = np.array(amg_iters)
            
    # Choose norm 
    if args["norm"] == "1":
        e = eL1; norm_str = "L_{{1}}"
    elif args["norm"] == "2":
        e = eL2; norm_str = "L_{{2}}"
    elif args["norm"] == "inf":    
        e = eLinf; norm_str = "L_{{\\infty}}"           
            
    if "CC" in alg: 
        scheme_label = "CC"
    elif "PREC1" in alg:
        scheme_label = "GSL"
    elif "PREC4" in alg:
        scheme_label = "LD"
            

    # Plot discretization error as a function of GMRES stopping tol             
    plt.figure(1)
    plt.loglog(np.power(10.0, -reltol_refine_array), e, label = scheme_label, marker = markers[alg_idx], 
                color = colours[alg_idx], linestyle = linestyles[alg_idx], basey = 10)
     
    plt.figure(2)
    plt.semilogy(runtime, e, label = scheme_label, marker = markers[alg_idx], 
                color = colours[alg_idx], linestyle = linestyles[alg_idx], basey = 10)
     
    
            
plt.figure(1)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(fontsize=legendsize)
plt.xlabel("$\\rm{GMRES}\,\,\Vert \mathbf{r}_{\\rm{final}} \Vert/ \Vert \mathbf{r}_0 \Vert$", **labelsize)
plt.ylabel("$\\rm{Discretization\, error}$", **labelsize)
IRK_label = "$\\rm{{{}}}$".format(IRKIndividualLabel[int(args["IRK"])])
plt.title(IRK_label, **labelsize)

if int(args["save"]):    
    out = "discerror_rtol" + args["IRK"] + "_l" + args["l_refine"] + "_d" + args["d"] + "_ex" + args["ex"] + ".pdf"
    plt.savefig(out, bbox_inches='tight')  

plt.figure(2)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(fontsize=legendsize)
plt.xlabel("$\\rm{Run time\,(s)}$", **labelsize)
plt.ylabel("$\\rm{Discretization\, error}$", **labelsize)
IRK_label = "$\\rm{{{}}}$".format(IRKIndividualLabel[int(args["IRK"])])
plt.title(IRK_label, **labelsize)

if int(args["save"]):    
    out = "discerror_runtime" + args["IRK"] + "_l" + args["l_refine"] + "_d" + args["d"] + "_ex" + args["ex"] + ".pdf"
    plt.savefig(out, bbox_inches='tight')  

plt.show()



