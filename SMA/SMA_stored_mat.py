#python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
#python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --seed 123 --j2 0. --Kz 1.5 --num_h 10
#python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --Kz 1.5 --num_h 10
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
from Stat_ori import *
from Norm_ori import *
from Hami_ori import *
# from Test import *
from models import j1j2
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
parser = cfg.get_args_parser()
parser.add_argument("--h", type=float, default=1.,
                    help="On site field in <1,1,1> direction.")
parser.add_argument("--Kx", type=float, default=1,
                    help="Kitaev coupling on x bond.")
parser.add_argument("--Ky", type=float, default=1,
                    help="Kitaev coupling on y bond.")
parser.add_argument("--Kz", type=float, default=1,
                    help="Kitaev coupling on z bond.")
parser.add_argument("--num_h", type=int, default=0,
                    help="index of h list")
parser.add_argument("--size", type=int, default=10, help="effective size")
parser.add_argument("--SSFfn", type=str, default="datas/aniksdg_noLG_sweepLG_SSF/SSF_h0_0.6toCSL_K${i}.txt",
                    help="filename for static structure factor")
args, unknown_args = parser.parse_known_args()

SSFfn = args.SSFfn
os.makedirs(os.path.dirname(SSFfn), exist_ok=True)

cfg.configure(args)
cfg.print_config()
torch.set_num_threads(args.omp_cores)
torch.set_num_interop_threads(120)  # Inter-op parallelism
torch.set_num_threads(120)  # Intra-op parallelism
num_params = 13+1+4
model = aniso_k.ANISO_K(Kx=args.Kx, Ky=args.Ky, Kz=args.Kz, h=args.h)
energy_f = model.energy_2x2
bond_dim = args.bond_dim
# kx,ky,kz,h = 1,1,1.5,1
kx, ky, kz, h = args.Kx, args.Ky, args.Kz, args.h
def _cast_to_real(t):
    return t.real
folder = "../AD_GTNO/datas/aniksdg_noLG_sweepLG/"
csinput = folder+"cs_h0_0.6toCSL_K{}.txt".format(args.Kz)

# NormMat = np.load("NormMat.npy")
# HamiMat = np.load("HamiMat.npy")
# NormMat = NormMat.reshape(np.prod(NormMat.shape[:5]), np.prod(NormMat.shape[5:]))
# HamiMat = HamiMat.reshape(np.prod(HamiMat.shape[:5]), np.prod(HamiMat.shape[5:]))
# Es, Bs = scipy.linalg.eig(HamiMat, NormMat)
# np.save("Es.npy", Es)
# np.save("Bs.npy", Bs)
Es = np.load("Es.npy")
Bs = np.load("Bs.npy")
Es = Es.reshape(4, 4, 4, 4, 4)
print(Es)
# print(Bs)
