#!/usr/bin/env python
"""
plotsetup.py : A Python script to set up plotting environment
"""
import dgpython as dg
import os, os.path, time
from dgpython.apps import dgnavierstokes as dgns
from dgpython.core import readsolution

def loadsoln(i):
    return readsolution(msh, 'plt/evsol' + str(i) + '.dat')

def plot(u, comp='r'):
    plt.dgplot(msh, u, comp, eqn='ns')

def plotsoln(i, comp='r'):
    u = loadsoln(i)
    plot(u, comp)

def ploterr(i, comp='r'):
    uE = readsolution(msh, 'plt/exact.dat')
    diff = uE - loadsoln(i)
    plot(diff, comp)

def loaddata(porder):
    global msh, uE
    meshfile = 'data/evmsh' + str(porder) + '.h5'
    msh = dg.Mesh(meshfile)
    
    uE = readsolution(msh, 'plt/exact.dat')

plt = dg.import_plt()
plt.ion()

import ConfigParser
class FakeSecHead(object):
    def __init__(self, fp):
        self.fp = fp
        self.sechead = '[sec]\n'

    def readline(self):
        if self.sechead:
            try: 
                return self.sechead
            finally: 
                self.sechead = None
        else: 
            return self.fp.readline()

cp = ConfigParser.SafeConfigParser()
cp.readfp(FakeSecHead(open('ev.cfg')))
porder = int(cp.get('sec','porder'))

loaddata(porder)
