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

# Number of outer GMRES iterations
dt4 = np.array([0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625])
iter4 = np.array([20,20,20,20,21,25])
iter14 = np.array([32,32,32,33,34,34])

dt7 = np.array([0.3048,0.2050,0.1380,0.0929,0.0625,0.0421])
dt7_dt0 = np.array([0.4101,0.3049,0.2264,0.1682,0.1250,0.0928])
iter27 = np.array([120,134,140,142,142,148])

dt8 = np.array([0.3536,0.2500,0.1768,0.1250,0.0884,0.0625])
dt8_dt0 = np.array([0.3535,0.2500,0.1768,0.1250,0.0884,0.0625])
iter18 = np.array([94,100,106,110,110,118])

dt9 = np.array([0.3969,0.2916,0.2143,0.1575,0.1157,0.0850])
dt9_dt0 = np.array([0.3149,0.2143,0.1458,0.0992,0.0675,0.0460])
iter29 = np.array([184,210,215,226,217,242])

dt10 = np.array([0.4353,0.3300,0.2500,0.1895,0.1436,0.1088])
dt10_dt0 = np.array([0.2872,0.1894,0.1250,0.0825,0.0544,0.0359])
iter110 = np.array([138,155,165,168,169,179])


# Raw AIR iterations
fig1, axL = plt.subplots(figsize=(11,8))
line1, = plt.plot(dt4, iter4, '-o', lw=lw, markersize=16, label='SDIRK4')
line2, = plt.plot(dt4, iter14, '-*', lw=lw, markersize=20, label='Gauss4')
line3, = plt.plot(dt4, iter27, '-X', lw=lw, markersize=16, label='Radau7')
line4, = plt.plot(dt4, iter18, '-s', lw=lw, markersize=16, label='Gauss8')
line5, = plt.plot(dt4, iter29, '-^', lw=lw, markersize=16, label='Radau9')
line6, = plt.plot(dt4, iter110, '->', lw=lw, markersize=16, label='Gauss10')
axL.set_xscale('log', basex=2)

axL.patch.set_visible(False)
axL.set_xlabel("Mesh spacing, h", va='bottom')
axL.yaxis.tick_left()
axL.set_ylabel("AIR iterations/time step")
axL.set_ylim((0,250))
axL.xaxis.labelpad = 28
axL.xaxis.grid(True)
axL.yaxis.grid(True)

# plt.title("4th-order elements")
plt.grid(True)
# axL.legend(handles=[line1,line2,line3,line4,line5,line6], ncol=1, frameon=True, loc='upper right')
plt.savefig('dg_advdiff_o4_1e-6.pdf', bbox_inches='tight', transparent=True)
# plt.show()



# Relative AIR iterations
fig1, axL = plt.subplots(figsize=(11,8))
line1, = plt.plot(dt4, iter4/iter4, '-o', lw=lw, markersize=16, label='SDIRK4')
line2, = plt.plot(dt4, iter14/iter4, '-*', lw=lw, markersize=20, label='Gauss4')
line3, = plt.plot(dt4, iter27*dt7_dt0/iter4, '-X', lw=lw, markersize=16, label='Radau7')
line4, = plt.plot(dt4, iter18*dt8_dt0/iter4, '-s', lw=lw, markersize=16, label='Gauss8')
line5, = plt.plot(dt4, iter29*dt9_dt0/iter4, '-^', lw=lw, markersize=16, label='Radau9')
line6, = plt.plot(dt4, iter110*dt10_dt0/iter4, '->', lw=lw, markersize=16, label='Gauss10')
axL.set_xscale('log', basex=2)

axL.patch.set_visible(False)
axL.set_xlabel("Mesh spacing, h", va='bottom')
axL.yaxis.tick_left()
axL.set_ylabel("Relative AIR iterations/time step")
axL.set_ylim((0,2))
axL.xaxis.labelpad = 28
axL.xaxis.grid(True)
axL.yaxis.grid(True)

# plt.title("4th-order elements")
plt.grid(True)
axL.legend(handles=[line1,line2,line3,line4,line5,line6], ncol=2, frameon=True, loc='lower right')
plt.savefig('dg_advdiff_o4_1e-6_rel.pdf', bbox_inches='tight', transparent=True)
# plt.show()
