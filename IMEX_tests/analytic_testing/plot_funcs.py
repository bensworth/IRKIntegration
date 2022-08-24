import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import pdb

# def fun(x, y):
#     return y*np.exp(x) / (1.0 + y*(np.exp(x) - 1.0) )

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = y = np.arange(0.0, 5.0, 0.01)
# X, Y = np.meshgrid(x, y)
# zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)
# ax.plot_surface(X, Y, Z)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()




# Solution to reaction test w/o forcing for given time t and
# reaction coefficient eta
eta = 0.5
t = 3
def fun2(x, y):
    u0 = np.sin(2.0*np.pi*x*(1.0-y))*np.sin(2.0*np.pi*(1.0-x)*y)
    return u0*np.exp(eta*t) / ( 1.0 + u0*(np.exp(eta*t) - 1.0) )

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0.0, 1.0, 0.01)
X, Y = np.meshgrid(x, y)
zs = np.array([fun2(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

pdb.set_trace()


