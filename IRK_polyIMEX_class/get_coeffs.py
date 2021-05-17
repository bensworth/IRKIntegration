import numpy as np
from numpy.linalg import inv
from scipy.linalg import schur
import pdb


exp_first = True
z0 = np.array([-1,-1./3., 1])
A0 = np.array( [[0.0, 0.0, 0.0],[0.0, 5.0/6.0, -1.0/6.0],[0.0, 1.5, 0.5]] )
expA0 = np.array( [[0.0, 0.0, 0.0],[8.0/27.0, -11.0/18.0, 53.0/54.0], [4.0, -15.0/2.0, 11.0/2.0]] )
expA0it = np.array( [[0.0, 0.0, 0.0],[8.0/27.0, 7.0/18.0, -1.0/54.0], [0.0, 3.0/2.0, 1.0/2.0]] )
if exp_first:
	A0_imp = A0[1:,:][:,1:]
else:
	A0_imp = A0[0:-1,:][:,0:-1]
	A0it_imp = A0it[0:-1,:][:,0:-1]

A0inv = inv(A0_imp)
R0,Q0 = schur(A0inv)
s = A0.shape[0]

# Get indices of 1x1 and 2x2 diagonal blocks in R
test = np.empty_like(R0)
test[:] = R0
test[np.where(test != 0)]=1
s0 = int(np.ceil((s-1)/2))
if np.abs((s-1) % 2)<1e-12:
	bsize = 2*np.ones((s0,))
else:
	bsize = np.zeros((s0,))
	ind = 0
	i=0
	pdb.set_trace()
	while i<(s-1):
		if i==0:
			if np.abs(test[i+1,i]) > 1e-12:
				bsize[ind] = 2
				i += 2
				ind += 1
				continue
			else:
				bsize[ind] = 1
				i += 1
				ind += 1
				continue
		elif i==(s-2):
			bsize[ind] = 1
			i += 1
			ind += 1
			continue			
		elif np.abs(test[i,i-1]) > 1e-12 or np.abs(test[i+1,i]) > 1e-12:
			bsize[ind] = 2
			i += 2
			ind += 1
			continue
		else:
			bsize[ind] = 1
			ind += 1
			i += 1
			continue

print("is_imex = true;")
print("s =",s-1,";")
print("s_eff =",s0,";")
print("SizeData();")
print("/* --- A --- */")
for i in range(0,s):
	for j in range(0,s):
		print( "A0(",i,",",j,") =",A0[i,j],";" )
print("/* --- exp A --- */")
for i in range(0,s):
	for j in range(0,s):
		print( "expA0(",i,",",j,") =",expA0[i,j],";" )
print("/* --- expA_it --- */")
for i in range(0,s):
	for j in range(0,s):
		print( "expA0_it(",i,",",j,") =",expA0it[i,j],";" )
print("/* --- inv(A) --- */")
for i in range(0,s-1):
	for j in range(0,s-1):
		print( "invA0(",i,",",j,") =",A0inv[i,j],";" )
print("/* --- Q --- */")
for i in range(0,s-1):
	for j in range(0,s-1):
		print( "Q0(",i,",",j,") =",Q0[i,j],";" )
print("/* --- R --- */")
for i in range(0,s-1):
	for j in range(0,s-1):
		print( "R0(",i,",",j,") =",R0[i,j],";" )
print("/* --- z --- */")
for i in range(0,s):
	print( "z0(",i,") =",z0[i],";" )
print("/* --- R block sizes --- */")
for i in range(0,s0):
	print( "R0_block_sizes[",i,"] =",bsize[i],";" )

pdb.set_trace()





# B = np.zeros((5,5));
# B[0, 0] = +0.072998864317903;
# B[0, 1] = -0.026735331107946;
# B[0, 2] = +0.018676929763984;
# B[0, 3] = -0.012879106093306;
# B[0, 4] = +0.005042839233882;
# B[1, 0] = +0.153775231479182;
# B[1, 1] = +0.146214867847494;
# B[1, 2] = -0.036444568905128;
# B[1, 3] = +0.021233063119305;
# B[1, 4] = -0.007935579902729;
# B[2, 0] = +0.140063045684810;
# B[2, 1] = +0.298967129491283;
# B[2, 2] = +0.167585070135249;
# B[2, 3] = -0.033969101686618;
# B[2, 4] = +0.010944288744192;
# B[3, 0] = +0.144894308109535;
# B[3, 1] = +0.276500068760159;
# B[3, 2] = +0.325797922910421;
# B[3, 3] = +0.128756753254910;
# B[3, 4] = -0.015708917378805;
# B[4, 0] = +0.143713560791226;
# B[4, 1] = +0.281356015149462;
# B[4, 2] = +0.311826522975741;
# B[4, 3] = +0.223103901083571;
# B[4, 4] = +0.040000000000000;

# Binv = inv(B)
# Br0,Bq0 = schur(Binv)
# test = Br0
# test[np.where(test != 0)]=1
# s = test.shape[0]+1
# s0 = int(np.ceil((s-1)/2))
# if np.abs((s-1) % 2)<1e-12:
# 	bsize = 2*np.ones((s0,1))
# else:
# 	bsize = np.zeros((s0,1))
# 	ind = 0
# 	i=0
# 	pdb.set_trace()
# 	while i<(s-1):
# 		if i==0:
# 			if np.abs(test[i+1,i]) > 1e-12:
# 				bsize[ind] = 2
# 				i += 2
# 				ind += 1
# 				continue
# 			else:
# 				bsize[ind] = 1
# 				i += 1
# 				ind += 1
# 				continue
# 		elif i==(s-2):
# 			bsize[ind] = 1
# 			i += 1
# 			ind += 1
# 			continue			
# 		elif np.abs(test[i,i-1]) > 1e-12 or np.abs(test[i+1,i]) > 1e-12:
# 			bsize[ind] = 2
# 			i += 2
# 			ind += 1
# 			continue
# 		else:
# 			bsize[ind] = 1
# 			ind += 1
# 			i += 1
# 			continue
#
# pdb.set_trace()
