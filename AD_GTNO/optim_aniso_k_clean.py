#python optim_ising_c4v.py --bond_dim 1 --chi 16 --seed 1234 --hx 3.1 --CTMARGS_fwd_checkpoint_move --OPTARGS_tolerance_grad 1.0e-8 --out_prefix ex-hx31D2 --params_out paramsh31D2 --opt_max_iter 1000 --instate ex-hx31D2_state.json --params_in paramsh31D2
import context
import torch
import argparse
import config as cfg

from ctm.generic.env import *
from ctm.generic import ctmrg
from ipeps.ipeps import *
from ctm.generic import rdm
from models import aniso_k
# from ipeps.ipeps_c4v import *
from groups.pg import make_c4v_symm
from optim.ad_optim_lbfgs_mod import optimize_state

from GTNOs import *
import unittest
import logging
import json

# 引入 time 模組
import time
import numpy as np

from gtno_model import gtno_model

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--h", type=float, default=0., help="On site field in <1,1,1> direction.")
parser.add_argument("--Kx", type=float, default=0, help="Kitaev coupling on x bond.")
parser.add_argument("--Ky", type=float, default=0, help="Kitaev coupling on y bond.")
parser.add_argument("--Kz", type=float, default=0, help="Kitaev coupling on z bond.")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice", \
    choices=["2SITE"])
args, unknown_args = parser.parse_known_args()


#-------- filenames --------
Efn = f"datas/anik_E1_iso2.txt"
magfn = f"datas/anik_mag1_iso2.txt"
csfn = f"datas/anik_cs1_green_iso2.txt"
#-------- filenames --------

def to_bloch(A_):
    A = A_[:,0,0,0]
    a = A[0].real
    b = A[0].imag
    c = A[1].real
    d = A[1].imag        
    N =  A.conj().dot(A)
    theta = 2*np.arccos((a))
    phi = np.arccos((c)/(np.sin(theta/2)))
    return theta, phi

def main():

    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.set_num_interop_threads(4) # Inter-op parallelism
    torch.set_num_threads(4) # Intra-op parallelism
    # A = state_m1m1m1()
    # print(to_bloch(A))
    # exit()
    num_params = 13+1+4
    model = aniso_k.ANISO_K(Kx = args.Kx, Ky = args.Ky, Kz = args.Kz, h = args.h)
    bond_dim = args.bond_dim
    kx,ky,kz,h = -1,-1,-1,0
    #praw=[1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,  0, 5.3280 ,2.3564, 5.3280 ,2.3564] # |1,1,1> h->inf limit
    #praw=[1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4, 0, 1e-4,1e-4,3.14,1e-4] # |0,0,1> |0,0,-1> kz->inf limit
    #praw=[1e-4,1e-1,1e-4,1e-4,1e-1,1e-4,1e-1,1e-4,1e-1,1e-4,1e-1,1e-4,1e-4, 0.99, 1e-4,1e-4,2,1e-4] # |0,0,1> |0,0,-1> kz->inf limit
    praw=[0,1,0,1,0,1,0,0,0,0,0,0,0,  1, 5.3280 ,2.3564, 5.3280 ,2.3564,] # FM kitaev
    # praw=[0,-0.9,0,-0.9,0,-0.9,0,0,0,0,0,0,0,  1, 5.3280 ,2.3564, 4.0969 , 0.7854] # AFM kitaev

    anisok = gtno_model(model, args.chi)
    anisok.set_params(praw)
    anisok.initialize()
    anisok.optimize()
    
    # hs = np.linspace(-2, 0, 40)
    # save(state_sym, ctm_env, params)NNNNBXCCCCCC
    return 0

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()