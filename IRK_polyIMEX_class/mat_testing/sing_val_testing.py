import pdb
import numpy as np
from scipy import io
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt


Ai_dat = np.loadtxt("./A_imp_o1.mm");
Ae_dat = np.loadtxt("./A_exp_o1.mm");
M_dat = np.loadtxt("./M_mat_o1.mm");

Ai =  csr_matrix((Ai_dat[1:,2],(Ai_dat[1:,0],Ai_dat[1:,1])),shape=[int(Ai_dat[0,1]+1),int(Ai_dat[0,1]+1)])
Ae =  csr_matrix((Ae_dat[1:,2],(Ae_dat[1:,0],Ae_dat[1:,1])),shape=[int(Ae_dat[0,1]+1),int(Ae_dat[0,1]+1)])
M =  csr_matrix((M_dat[1:,2],(M_dat[1:,0],M_dat[1:,1])),shape=[int(M_dat[0,1]+1),int(M_dat[0,1]+1)])

Ai0 = np.asarray(Ai.todense())
Ae0 = np.asarray(Ae.todense())
M0 = np.asarray(M.todense())

# Ai <- M^{-1}Ai, Ae <- M^{-1}Ae
Minv = np.linalg.inv(M0)
Ai0 = np.dot(Minv, Ai0)
Ae0 = np.dot(Minv, Ae0)

e = np.eye(Ai0.shape[0])
dt = 0.1

# (I - dt*Ai)^{-1} (I + dt*Ae)
m1 = np.linalg.inv(e - dt*Ai0)
m2 = e + dt*Ae0
rk111 = np.dot(m1, m2)

# I + dt*(Ae + Ai)(I - dt*Ai)^{-1} (I + dt*Ae)
m0 = dt*(Ai0 + Ae0)
rk121 = e + np.dot(m0, np.dot(m1,m2))

eig111 = np.linalg.eigvals(rk111)
eig121 = np.linalg.eigvals(rk121)

pdb.set_trace()

plt.plot(np.real(eig111), np.imag(eig111), '*')
plt.show()

plt.plot(np.real(eig121), np.imag(eig121), '*')
plt.show()

# plt.semilogy(s_op, color='b',label="s(Ai^{-1}Ae)")
# plt.set_title(labels[i])
# plt.set_ylabel("singular values")
# plt.show()

pdb.set_trace()