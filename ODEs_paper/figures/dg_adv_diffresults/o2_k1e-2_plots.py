import matplotlib as mpl 
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import pdb

fig_width_pt = 400.0
fontsize = 30
fontfamily = 'serif'
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*2      # height in inches
fig_size = [fig_width, fig_height]
params = {'backend': 'ps',
          'font.family': fontfamily,
          'font.serif':  'cm',
          'font.sans-serif': 'arial',
          'axes.labelsize': fontsize,
          'font.size': fontsize,
          'axes.titlesize': fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'text.usetex': True,
          # 'figure.figsize': fig_size,
          'lines.linewidth': 2}
plt.rcParams.update(params)
lw = 4

# Number of outer FGMRES iterations
AMG_iters = [1,2,3,4,5,6,7,8]
gmres = [1011, 2*266, 3*44, 4*31, 5*25, 6*22, 7*16, 8*14]
nogmres = [2000, 2*1007, 3*101, 4*41, 5*31, 6*27, 7*25, 8*22]

fig1, axL = plt.subplots(figsize=(11,8))
line1, = plt.plot(AMG_iters, nogmres, '-o', lw=lw, markersize=12, label='Inner fixed-point')
line2, = plt.plot(AMG_iters, gmres, '-*', lw=lw, markersize=16, label='Inner GMRES')

axL.patch.set_visible(False)
axL.set_xlabel("AIR iterations/FGMRES iteration", va='bottom')
axL.yaxis.tick_left()
axL.set_ylabel("Total AIR iterations/time step")
axL.set_ylim((0,500))
axL.xaxis.labelpad = 28
axL.xaxis.grid(True)
axL.yaxis.grid(True)

# plt.title("4th-order elements")
plt.grid(True)
axL.legend(handles=[line1,line2], ncol=1, frameon=True, loc='upper right')
plt.savefig('dg_advdiff_o2_1e-2.pdf', bbox_inches='tight', transparent=True)
plt.show()


