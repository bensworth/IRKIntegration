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
iter4_tot = np.array([20,20,20,20,21,24])
iter14 = np.array([10,11,11,11,11,11])
iter14_tot = 2*iter14;

dt7 = np.array([0.3048,0.2050,0.1380,0.0929,0.0625,0.0421])
dt7_dt0 = np.array([0.4101,0.3049,0.2264,0.1682,0.1250,0.0928])
iter27_easy = np.array([7,8,8,8,8,9])
iter27_hard = np.array([19,19,20,20,21,22])
iter27_tot = 2*(iter27_easy + iter27_hard)

dt8 = np.array([0.3536,0.2500,0.1768,0.1250,0.0884,0.0625])
dt8_dt0 = np.array([0.3535,0.2500,0.1768,0.1250,0.0884,0.0625])
iter18_easy = np.array([7,7,8,8,9,11])
iter18_hard = np.array([17,17,18,18,19,20])
iter18_tot = 2*(iter18_easy + iter18_hard)

dt9 = np.array([0.3969,0.2916,0.2143,0.1575,0.1157,0.0850])
dt9_dt0 = np.array([0.3149,0.2143,0.1458,0.0992,0.0675,0.0460])
iter29_easy = np.array([10,10,10,10,11,12])
iter29_hard = np.array([22,22,23,24,25,25])
iter29_single = np.array([4,5,5,6,7,8])
iter29_tot = 2*(iter29_easy + iter29_hard) + iter29_single

dt10 = np.array([0.4353,0.3300,0.2500,0.1895,0.1436,0.1088])
dt10_dt0 = np.array([0.2872,0.1894,0.1250,0.0825,0.0544,0.0359])
iter110_easy = np.array([10,10,10,10,11,14])
iter110_hard = np.array([19,19,20,21,22,24])
iter110_single = np.array([4,5,5,6,7,9])
iter110_tot = 2*(iter110_easy + iter110_hard) + iter110_single


# Raw AIR iterations
fig1, axL = plt.subplots(figsize=(11,8))
line1, = plt.plot(dt4, iter4_tot, '-o', lw=lw, markersize=16, label='SDIRK4')
line2, = plt.plot(dt4, iter14_tot, '-*', lw=lw, markersize=20, label='Gauss4')
line3, = plt.plot(dt4, iter27_tot, '-X', lw=lw, markersize=16, label='Radau7')
line4, = plt.plot(dt4, iter18_tot, '-s', lw=lw, markersize=16, label='Gauss8')
line5, = plt.plot(dt4, iter29_tot, '-^', lw=lw, markersize=16, label='Radau9')
line6, = plt.plot(dt4, iter110_tot, '->', lw=lw, markersize=16, label='Gauss10')
axL.set_xscale('log', basex=2)

axL.patch.set_visible(False)
axL.set_xlabel("Mesh spacing (h)", va='bottom')
axL.yaxis.tick_left()
axL.set_ylabel("AIR iterations/time step")
axL.set_ylim((0,100))
axL.xaxis.labelpad = 28
axL.xaxis.grid(True)
axL.yaxis.grid(True)

plt.xticks(np.array([1.0/8,1.0/16,1.0/32,1.0/64,1.0/128,1.0/256]))
axL.xaxis.set_tick_params(pad=8)
axL.set_xticklabels(('1/8','1/16','1/32','1/64','1/128','1/256'))

# plt.title("4th-order elements")
plt.grid(True)
# axL.legend(handles=[line1,line2,line3,line4,line5,line6], ncol=1, frameon=True, loc='upper right')
plt.savefig('dg_advdiff_o4_1e-6.pdf', bbox_inches='tight', transparent=True)
# plt.show()



# Relative AIR iterations
fig1, axL = plt.subplots(figsize=(11,8))
line1, = plt.plot(dt4, iter4_tot/iter4_tot, '-o', lw=lw, markersize=16, label='SDIRK4')
line2, = plt.plot(dt4, iter14_tot/iter4_tot, '-*', lw=lw, markersize=20, label='Gauss4')
line3, = plt.plot(dt4, iter27_tot*dt7_dt0/iter4_tot, '-X', lw=lw, markersize=16, label='Radau7')
line4, = plt.plot(dt4, iter18_tot*dt8_dt0/iter4_tot, '-s', lw=lw, markersize=16, label='Gauss8')
line5, = plt.plot(dt4, iter29_tot*dt9_dt0/iter4_tot, '-^', lw=lw, markersize=16, label='Radau9')
line6, = plt.plot(dt4, iter110_tot*dt10_dt0/iter4_tot, '->', lw=lw, markersize=16, label='Gauss10')
axL.set_xscale('log', basex=2)

axL.patch.set_visible(False)
axL.set_xlabel("Mesh spacing (h)", va='bottom')
axL.yaxis.tick_left()
axL.set_ylabel("Relative AIR iterations/time step")
axL.set_ylim((0,2))
axL.xaxis.labelpad = 28
axL.xaxis.grid(True)
axL.yaxis.grid(True)

plt.xticks(np.array([1.0/8,1.0/16,1.0/32,1.0/64,1.0/128,1.0/256]))
axL.xaxis.set_tick_params(pad=8)
axL.set_xticklabels(('1/8','1/16','1/32','1/64','1/128','1/256'))

# plt.title("4th-order elements")
plt.grid(True)
axL.legend(handles=[line1,line2,line3,line4,line5,line6], ncol=2, frameon=True, loc='upper right')
plt.savefig('dg_advdiff_o4_1e-6_rel.pdf', bbox_inches='tight', transparent=True)
# plt.show()


pdb.set_trace()