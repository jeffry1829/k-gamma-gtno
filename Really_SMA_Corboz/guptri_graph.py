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

import sys
datadir = sys.argv[1]

# Plot spectral weight
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

plt.subplot()

Es = []
cntlst = []
cnt = -1
with open(datadir+"guptri_excitedE.txt", "r") as f:
    for line in f:
        if line[0] == "#":
            cnt += 1
            continue
        tmp = np.repeat(cnt, len(line.split()))
        cntlst.extend(list(tmp))
        tmp = [float(_) for _ in line.split()]
        Es.extend(tmp)

plt.xticks([0,12,18,24,36,42],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$',r'$M(\pi,0)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
plt.xlabel('momentum')
plt.ylabel('excitation energy')
plt.plot(cntlst, Es, 'o')
plt.show()