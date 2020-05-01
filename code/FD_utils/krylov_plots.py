
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
fiddling around to plot Krylov iters for various problems... some of the code
here doesn't make much sense since I've not settled on the format to plot the data

python test.py -dir data3/ -d 1 -FD 4 -g 0 -t 4 -dt_min 3 -dt_max 7 -cfl 1 2 4 



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
parser.add_argument('-t','--IRK', nargs = 1, help = 'IDs of IRK methods', required = True)
parser.add_argument('-dt_min','--dt_min', nargs = 1, help = 'Min temporal refinement', required = True)
parser.add_argument('-dt_max','--dt_max', nargs = 1, help = 'Max temporal refinement', required = True)
parser.add_argument('-cfl','--cfls', nargs = "+", help = 'CFLs', required = True)
parser.add_argument('-d','--d', help = 'Spatial dimension', required = True)
parser.add_argument('-FD','--FD', help = 'Finite difference problem ID', required = True)
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
def filenametemp(scheme, dt_refine, cfl):
    return args["dir"] + "IRK" + str(scheme) + "_dt" + str(dt_refine) + "_d" + args["d"] + "_FD" + args["FD"] + "_cfl" + str(cfl)


for i, cfl in enumerate(args["cfls"]):
    scheme=0
    eL1  = []
    eL2  = []
    eLinf = []
    dt = []
    
    reltol = []
    iters = []
    eig_ratio = []
    
    for dt_refine in range(int(args["dt_min"][0]), int(args["dt_max"][0])+1):
        filename = filenametemp(args["IRK"][0], dt_refine, cfl)
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
    
        dt.append(float(params["dt"]))
        eL1.append(float(params["eL1"]))
        eL2.append(float(params["eL2"]))
        eLinf.append(float(params["eLinf"]))

        reltol.append(float(params["krtol"]))

        # Size double list on first dt for given scheme
        nsys = int(params["nsys"])
        if (dt_refine == int(args["dt_min"][scheme])):
            for system in range(0,nsys):
                iters.append([])
                # Just get eig ratios once, these are independent of dt
                eig_ratio.append(float(params["sys" + str(system + 1) + "_eig_ratio"]))        
        
        # Add info for every system        
        for system in range(0, nsys):
            iters[system].append(int(params["sys" + str(system + 1) + "_iters"]))


#### Finished with current IRK scheme ###

    # Cast from lists to numpy arrays for plotting, etc.
    dt    = np.array(dt)
    eL1   = np.array(eL1)
    eL2   = np.array(eL2)
    eLinf = np.array(eLinf)
    
    iters = np.array(iters)
    eig_ratio = np.array(eig_ratio)
    
    # Choose norm 
    if args["norm"] == "1":
        e = eL1; norm_str = "L_{{1}}"
    elif args["norm"] == "2":
        e = eL2; norm_str = "L_{{2}}"
    elif args["norm"] == "inf":    
        e = eLinf; norm_str = "L_{{\\infty}}"
    
    order = IRK_order[int(args["IRK"][scheme])]

    # Plot errors for given solves and line indicating expected asymptotic convergence rate
    anchor = -3 # Achor theoretical line to end-1th data point
    plt.figure(1)
    plt.loglog(dt, 0.5*e[anchor]*(dt/float(dt[anchor]))**(order), linestyle = '--', color = 'k')
    
    
    if args["IRK_type"] is None:
        plt.loglog(dt, e, label = "IRK{}".format(args["IRK"][scheme]), marker = markers[scheme], color = colours[i], basex = 2)
    else:
        plt.loglog(dt, e, label = "$\\delta t/\\delta x={{{}}} $".format(cfl[0]), marker = markers[scheme], color = colours[i], basex = 2)

    plt.figure(2)
    for system in range(0, nsys):
        if int(args["IRK_type"][0]) == 0 and system > 0:
            plt.semilogx(dt, iters[system], marker = markers[system], color = colours[i], basex = 2)
        else:
            plt.semilogx(dt, iters[system], label = "$\\beta/\\eta=${:.2f}".format(eig_ratio[system]), marker = markers[system], color = colours[i], basex = 2)
            
        
    # plt.figure(3)
    # for system in range(0, nsys):
    #     rho = reltol[system]**(1.0/iters[system])
    #     print(rho[-1])
    #     plt.semilogx(dt, rho, label = "$\\beta/\\eta=${:.2f}".format(eig_ratio[system]), marker = markers[system], color = colours[scheme], basex = 2)
    #     if (eig_ratio[system] > 0):
    #         r = (eig_ratio[system])**2
    #         rho_bound = r/(2 + r)
    #         plt.semilogx(dt, rho_bound*np.ones_like(dt), marker = markers[system], linestyle = "--", color = colours[scheme], basex = 2)
    # 


plt.figure(1)
axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)

plt.legend(fontsize = fs["fontsize"]-2)
plt.xlabel("$\delta t$", **fs)
plt.ylabel("$\\Vert \\mathbf{{e}} \\Vert_{{{}}}$".format(norm_str), **fs)


# Set the title to the common IRK group if one was passed
if args["IRK_type"] is not None:
    plt.title("$\\rm{{{}}}$$({{{}}})$".format(IRK_type[int(args["IRK_type"][0])], IRK_order[int(args["IRK"][0])]), **fs)

# if int(args["save"]):    
#     # Generate name to save figure with...
#     out = "convergence_plots/"
# 
#     if args["IRK_type"] is not None:
#         out += IRK_type[int(args["IRK_type"][0])].replace("\,", "") + "_"
#     else:
#         out += "unknown_IRK_group_"
# 
#     out += "d" + args["d"] + "_FD" + args["FD"]
#     plt.savefig('{}.pdf'.format(out), bbox_inches='tight')

plt.savefig("convergence_plots/fig1.pdf", bbox_inches='tight')


plt.figure(2)
axes = plt.gca()
axes.set_xlim(xlims)
axes.set_ylim(ylims)

plt.legend(fontsize = fs["fontsize"]-2)
plt.xlabel("$\delta t$", **fs)
plt.ylabel("GMRES iterations", **fs)

ax = plt.figure(2).gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig("convergence_plots/fig2.pdf", bbox_inches='tight')

# 
# plt.figure(3)
# axes = plt.gca()
# axes.set_xlim(xlims)
# axes.set_ylim(ylims)
# 
# plt.legend(fontsize = fs["fontsize"]-2)
# plt.xlabel("$\delta t$", **fs)
# plt.ylabel("approx. convergence factor".format(norm_str), **fs)


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




