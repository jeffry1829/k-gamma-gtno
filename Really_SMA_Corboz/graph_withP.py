#python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
#python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --seed 123 --j2 0. --Kz 1.5 --num_h 10
#python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --Kz 1.5 --num_h 10
#python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim 2 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
#python SMA_stored_mat_withP.py --hx 2.5 --GLOBALARGS_dtype complex128 --bond_dim 2 --size 12 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
#python graph_withP.py
import context
import time
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic.rdm import *
# from ctm.generic import ctmrg_ex
from ctm.generic import ctmrg
from ctm.generic.ctm_projectors import *
# from Stat_ori import *
from Norm_ori import *
# from Hami_ori import *
from Localsite_Hami_ori import *
# from Test import *
# from models import j1j2
from models import ising
from groups.pg import *
import groups.su2 as su2
from optim.ad_optim_lbfgs_mod import optimize_state
from tn_interface import contract, einsum
from tn_interface import conj
from tn_interface import contiguous, view, permute
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import scipy.io
import unittest
import logging
from ctm.generic import rdm
import os
log = logging.getLogger(__name__)

tStart = time.time()

from models import aniso_k
from GTNOs import *
import scipy as scipy

# Plot spectral weight
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

cnt = -1
Es = []
SWs = []
points = []
values = []
with open("SW_P.txt", "r") as f:
    for line in f:
        if line[0] == "#":
            cnt += 1
            continue
        Es.append(float(line.split()[0]))
        SWs.append(float(line.split()[1]))
        points.append((cnt, float(line.split()[0])))
        values.append(float(line.split()[1]))

# x is (kx,ky)
# y is w
grid_x, grid_y = np.mgrid[0:cnt, np.amin(Es):\
    np.amax(Es):100j]

grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

# plt.subplot(221)
# plt.imshow(grid_z2.T, extent=(0,1,np.amin(Es),np.amax(Es)), origin='lower')
plt.subplot()
plt.imshow(grid_z2.T, extent=(0,cnt,np.amin(Es),np.amax(Es)), origin='lower')
plt.title('(momentum, energy) to spectral weight')
plt.xlabel('momentum')
plt.ylabel('lowest-excitation energy')
plt.gcf().set_size_inches(6, 6)
plt.colorbar()
plt.show()

plt.subplot()

Es = []
cntlst = []
cnt = -1
with open("excitedE_P.txt", "r") as f:
    for line in f:
        if line[0] == "#":
            cnt += 1
            continue
        cntlst.append(cnt)
        Es.append(float(line.split()[0]))

plt.plot(cntlst, Es, 'o')
plt.show()

Es = []
cntlst = []
cnt = -1
with open("eigN_P.txt", "r") as f:
    for line in f:
        if line[0] == "#":
            cnt += 1
            continue
        for e in line.split():
            cntlst.append(cnt)
            Es.append(float(e))

plt.plot(cntlst, Es, 'o')
plt.show()

