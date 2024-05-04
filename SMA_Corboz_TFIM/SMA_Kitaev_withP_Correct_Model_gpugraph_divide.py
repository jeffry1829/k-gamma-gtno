# python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
# python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --seed 123 --j2 0. --Kz 1.5 --num_h 10
# python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 1 --hx 3.0 --chi 8 --statefile D=2TFIM_output_state.json --size 11 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python SMA_withP.py --GLOBALARGS_dtype complex128 --bond_dim 2 --hx 3.0 --chi 8 --statefile D=2TFIM_output_state.json --size 11 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 2 --hx 2.5 --chi 8 --statefile D=2TFIM_output_state.json --size 11 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu

#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='

# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣤⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⠀⠀⠀⢀⣴⠟⠉⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣷⣀⢀⣾⠿⠻⢶⣄⠀⠀⣠⣶⡿⠶⣄⣠⣾⣿⠗⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⢻⣿⣿⡿⣿⠿⣿⡿⢼⣿⣿⡿⣿⣎⡟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡟⠉⠛⢛⣛⡉⠀⠀⠙⠛⠻⠛⠑⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣧⣤⣴⠿⠿⣷⣤⡤⠴⠖⠳⣄⣀⣹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣀⣟⠻⢦⣀⡀⠀⠀⠀⠀⣀⡈⠻⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⡿⠉⡇⠀⠀⠛⠛⠛⠋⠉⠉⠀⠀⠀⠹⢧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⡟⠀⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠈⠑⠪⠷⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣾⣿⣿⣿⣦⣼⠛⢦⣤⣄⡀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠑⠢⡀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⢀⣠⠴⠲⠖⠛⠻⣿⡿⠛⠉⠉⠻⠷⣦⣽⠿⠿⠒⠚⠋⠉⠁⡞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢦⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⢀⣾⠛⠁⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢣⠀⠀⠀
# ⠀⠀⠀⠀⣰⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣑⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡇⠀⠀
# ⠀⠀⠀⣰⣿⣁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣧⣄⠀⠀⠀⠀⠀⠀⢳⡀⠀
# ⠀⠀⠀⣿⡾⢿⣀⢀⣀⣦⣾⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⣫⣿⡿⠟⠻⠶⠀⠀⠀⠀⠀⢳⠀
# ⠀⠀⢀⣿⣧⡾⣿⣿⣿⣿⣿⡷⣶⣤⡀⠀⠀⠀⠀⠀⠀⠀⢀⡴⢿⣿⣧⠀⡀⠀⢀⣀⣀⢒⣤⣶⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇
# ⠀⠀⡾⠁⠙⣿⡈⠉⠙⣿⣿⣷⣬⡛⢿⣶⣶⣴⣶⣶⣶⣤⣤⠤⠾⣿⣿⣿⡿⠿⣿⠿⢿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇
# ⠀⣸⠃⠀⠀⢸⠃⠀⠀⢸⣿⣿⣿⣿⣿⣿⣷⣾⣿⣿⠟⡉⠀⠀⠀⠈⠙⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇
# ⠀⣿⠀⠀⢀⡏⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⠿⠿⠛⠛⠉⠁⠀⠀⠀⠀⠀⠉⠠⠿⠟⠻⠟⠋⠉⢿⣿⣦⡀⢰⡀⠀⠀⠀⠀⠀⠀⠁
# ⢀⣿⡆⢀⡾⠀⠀⠀⠀⣾⠏⢿⣿⣿⣿⣯⣙⢷⡄⠀⠀⠀⠀⠀⢸⡄⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣿⣻⢿⣷⣀⣷⣄⠀⠀⠀⠀⢸⠀
# ⢸⠃⠠⣼⠃⠀⠀⣠⣾⡟⠀⠈⢿⣿⡿⠿⣿⣿⡿⠿⠿⠿⠷⣄⠈⠿⠛⠻⠶⢶⣄⣀⣀⡠⠈⢛⡿⠃⠈⢿⣿⣿⡿⠀⠀⠀⠀⠀⡀
# ⠟⠀⠀⢻⣶⣶⣾⣿⡟⠁⠀⠀⢸⣿⢅⠀⠈⣿⡇⠀⠀⠀⠀⠀⣷⠂⠀⠀⠀⠀⠐⠋⠉⠉⠀⢸⠁⠀⠀⠀⢻⣿⠛⠀⠀⠀⠀⢀⠇
# ⠀⠀⠀⠀⠹⣿⣿⠋⠀⠀⠀⠀⢸⣧⠀⠰⡀⢸⣷⣤⣤⡄⠀⠀⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡆⠀⠀⠀⠀⡾⠀⠀⠀⠀⠀⠀⢼⡇
# ⠀⠀⠀⠀⠀⠙⢻⠄⠀⠀⠀⠀⣿⠉⠀⠀⠈⠓⢯⡉⠉⠉⢱⣶⠏⠙⠛⠚⠁⠀⠀⠀⠀⠀⣼⠇⠀⠀⠀⢀⡇⠀⠀⠀⠀⠀⠀⠀⡇
# ⠀⠀⠀⠀⠀⠀⠻⠄⠀⠀⠀⢀⣿⠀⢠⡄⠀⠀⠀⣁⠁⡀⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⢀⣐⡟⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⢠⡇

from GTNOs import *
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
from Norm_ori_withP import *
# from Norm_ori_withP_divide import *
# from Norm_ori_withP_divide_newfromweilin import *
# from Hami_ori import *

from Localsite_Hami_ori_withP import *
# from Localsite_Hami_ori_withP_divide import *
# from Localsite_Hami_ori_withP_divide_newfromweilin import *

# from Test import *
# from models import j1j2
# from models import ising
# from models import aniso_k_HsuKe as aniso_k
from models import kitaev as aniso_k
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
from pathlib import Path
from torchviz import make_dot
log = logging.getLogger(__name__)

tStart = time.time()

# from models import aniso_k_HsuKe
parser = cfg.get_args_parser()
parser.add_argument("--kx", type=float, default=0.,
                    help="kx")
parser.add_argument("--ky", type=float, default=0.,
                    help="ky")
parser.add_argument("--Jx", type=float, default=1.,
                    help="Jx of Kitave")
parser.add_argument("--Jy", type=float, default=1.,
                    help="Jy of Kitave")
parser.add_argument("--Jz", type=float, default=1.,
                    help="Jz of Kitave")
parser.add_argument("--h", type=float, default=0.,
                    help="h of Kitave")
parser.add_argument("--num_h", type=int, default=0,
                    help="index of h list")
parser.add_argument("--size", type=int, default=10, help="effective size")
parser.add_argument("--statefile", type=str, default="TFIM_output_state.json",
                    help="filename for TFIM input state")
parser.add_argument("--datadir", type=str, default="data/h3.0chi8/",
                    help="datadir")
parser.add_argument("--MultiGPU", type=str, default="False", help="MultiGPU")
parser.add_argument("--reuseCTMRGenv", type=str, default="True", help="Whether to reuse the ENV after CTMRG optimization\
    or not. If True, the ENV after CTMRG optimization will be saved in the same directory, and named as statefile+ENVC or ENVT+chi+size.pt")
parser.add_argument("--removeCTMRGenv", type=str, default="False", help="Whether to remove the ENV after CTMRG optimization\
    or not. If True, the ENV after CTMRG optimization will be removed.")
parser.add_argument("--NormMat", type=str, default="True",
                    help="Whether to calculate NormMat")
parser.add_argument("--HamiMat", type=str, default="True",
                    help="Whether to calculate HamiMat")
parser.add_argument("--UseVUMPSansazAC", type=str, default="False")
args, unknown_args = parser.parse_known_args()

cfg.configure(args)

if args.MultiGPU == "True":
    args.MultiGPU = True
else:
    args.MultiGPU = False
if args.reuseCTMRGenv == "True":
    args.reuseCTMRGenv = True
else:
    args.reuseCTMRGenv = False
if args.removeCTMRGenv == "True":
    args.removeCTMRGenv = True
else:
    args.removeCTMRGenv = False
if args.NormMat == "True":
    args.NormMat = True
else:
    args.NormMat = False
if args.HamiMat == "True":
    args.HamiMat = True
else:
    args.HamiMat = False
if args.UseVUMPSansazAC == "True":
    args.UseVUMPSansazAC = True
else:
    args.UseVUMPSansazAC = False
# cfg.print_config()
# torch.set_num_threads(args.omp_cores)
torch.set_num_threads(64)
torch.set_num_interop_threads(64)  # Inter-op parallelism
torch.set_num_threads(64)  # Intra-op parallelism
model = aniso_k.KITAEV(kx=args.Jx, ky=args.Jy, kz=args.Jz, h=args.h)
energy_f = model.energy_2x2
bond_dim = args.bond_dim


def _cast_to_real(t):
    return t.real


state = read_ipeps(args.datadir+args.statefile)
# print("Replace state to random state")
# state.sites[(0, 0)] = torch.randn_like(
#     state.sites[(0, 0)], dtype=torch.complex128).to(torch.complex128)
# print("state:", state.sites[(0, 0)])

# # try to normalize state
# print("state norm: ", torch.einsum('abcde,abcde', state.sites[(0, 0)], conj(
#     state.sites[(0, 0)])))
# print("Before normalization: ", state.sites[(0, 0)])
# state.sites[(0, 0)] = state.sites[(0, 0)]/torch.einsum('abcde,abcde', state.sites[(0, 0)],
#                                                        conj(state.sites[(0, 0)]))
# print("After normalization: ", state.sites[(0, 0)])
# print(state.sites[(0, 0)])

Numpyfilename = args.datadir+Path(args.statefile).stem+"Jx"+str(args.Jx)+"Jy"+str(args.Jy) + \
    "Jz"+str(args.Jz)+"h"+str(args.h)+"dtype"+str(cfg.global_args.dtype)+".npy"
if not os.path.exists(Numpyfilename):
    print("Saving state.sites((0,0)) to Numpy file: "+Numpyfilename)
    np.save(Numpyfilename, state.sites[(0, 0)].detach().cpu().numpy())
    # state.sites[(0,0)].detach().cpu().numpy().tofile(Numpyfilename)
ENVfilenameC = args.datadir+Path(args.statefile).stem+"ENVC"+"chi" + \
    str(args.chi)+"size"+str(args.size)+".pt"
ENVfilenameT = args.datadir+Path(args.statefile).stem+"ENVT"+"chi" + \
    str(args.chi)+"size"+str(args.size)+".pt"
for key, site in state.sites.items():
    state.sites[key] = site.type(
        cfg.global_args.torch_dtype).to(cfg.global_args.device)
state.dtype = cfg.global_args.torch_dtype
state.device = cfg.global_args.device
sitesDL = dict()
for coord, A in state.sites.items():
    dimsA = A.size()
    a = contiguous(einsum('mefgh,mabcd->eafbgchd', A, conj(A)))
    a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    sitesDL[coord] = a
stateDL = IPEPS(sitesDL, state.vertexToSite)


@torch.no_grad()
def ctmrg_conv_rdm2x1(state, env, history, ctm_args=cfg.ctm_args):
    if not history:
        history = dict({"log": []})

    rdm2x1 = rdm.rdm2x1((0, 0), state, env)

    dist = float('inf')
    if len(history["log"]) > 0:
        dist = torch.dist(rdm2x1, history["rdm"], p=2).item()
    history["rdm"] = rdm2x1
    history["log"].append(dist)
    if dist < ctm_args.ctm_conv_tol or len(history["log"]) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(
            history['log']), "history": history['log']})
        return True, history
    return False, history


difftograph = []


def ctmrg_conv_energy(state2, env, history, ctm_args=cfg.ctm_args):
    if not history:
        history = []
    old = []
    if (len(history) > 0):
        old = history[:8*env.chi+4]
    new = []
    u, s, v = torch.svd(env.C[((0, 0), (-1, -1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.svd(env.C[((0, 0), (1, -1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.svd(env.C[((0, 0), (1, -1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.svd(env.C[((0, 0), (1, 1))])
    for i in range(env.chi):
        new.append(s[i].item())

    u, s, v = torch.svd(
        env.T[((0, 0), (0, -1))].reshape(env.chi, env.chi*args.bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.svd(
        env.T[((0, 0), (0, 1))].permute(1, 0, 2).reshape(env.chi, env.chi*args.bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.svd(
        env.T[((0, 0), (-1, 0))].permute(0, 2, 1).reshape(env.chi, env.chi*args.bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.svd(
        env.T[((0, 0), (1, 0))].reshape(env.chi, env.chi*args.bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    # from hosvd import sthosvd as hosvd
    # core, _, _ = hosvd(env.T[((0, 0), (0, -1))], [env.chi]*3)
    # new.append(core)
    # core, _, _ = hosvd(env.T[((0, 0), (0, 1))], [env.chi]*3)
    # new.append(core)
    # core, _, _ = hosvd(env.T[((0, 0), (-1, 0))], [env.chi]*3)
    # new.append(core)
    # core, _, _ = hosvd(env.T[((0, 0), (1, 0))], [env.chi]*3)
    # new.append(core)
    # print("core.shape: ", core.shape)
    # print("core: ", core)

    new.append(env.T[((0, 0), (0, -1))])
    new.append(env.T[((0, 0), (0, 1))])
    new.append(env.T[((0, 0), (-1, 0))])
    new.append(env.T[((0, 0), (1, 0))])
    # new.append(env.C[((0, 0), (-1, -1))])
    # new.append(env.C[((0, 0), (-1, 1))])
    # new.append(env.C[((0, 0), (1, -1))])
    # new.append(env.C[((0, 0), (1, 1))])

    diff = 0.
    if (len(history) > 0):
        for i in range(8*env.chi):
            history[i] = new[i]
            if (abs(old[i]-new[i]) > diff):
                diff = abs(old[i]-new[i])
        for i in range(4):
            history[8*env.chi+i] = new[8*env.chi+i]
            if ((old[8*env.chi+i]-new[8*env.chi+i]).norm() > diff):
                diff = (old[8*env.chi+i]-new[8*env.chi+i]).norm()
            # print(torch.div(old[4*env.chi+i], new[4*env.chi+i]))
            if i == 0:
                difftograph.append((old[8*env.chi+i]-new[8*env.chi+i]).norm())

    else:
        for i in range(8*env.chi+4):
            history.append(new[i])
    history.append(diff)
    print("diff={0:<50}".format(diff), end="\r")
    # print("diff={0:<50}".format(diff))
    # print(ctm_args.ctm_conv_tol)
    if (len(history[8*env.chi+4:]) > 1 and diff < ctm_args.ctm_conv_tol)\
            or len(history[8*env.chi+4:]) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(
            history[8*env.chi+4:]), "history": history[8*env.chi+4:]})
        print("")
        print("modified CTMRG length: "+str(len(history[8*env.chi+4:])))
        # import matplotlib.pyplot as plt
        # plt.plot(difftograph)
        # plt.show()
        return True, history
    return False, history


# def ctmrg_conv_energy(state2, env, history, ctm_args=cfg.ctm_args):
#     if not history:
#         history = []
#     old = []
#     if (len(history) > 0):
#         old = history[:4*env.chi+4]
#     new = []
#     u, s, v = torch.svd(env.C[((0, 0), (-1, -1))])
#     for i in range(env.chi):
#         new.append(s[i].item())
#     u, s, v = torch.svd(env.C[((0, 0), (1, -1))])
#     for i in range(env.chi):
#         new.append(s[i].item())
#     u, s, v = torch.svd(env.C[((0, 0), (1, -1))])
#     for i in range(env.chi):
#         new.append(s[i].item())
#     u, s, v = torch.svd(env.C[((0, 0), (1, 1))])
#     for i in range(env.chi):
#         new.append(s[i].item())

#     new.append(env.T[((0, 0), (0, -1))])
#     new.append(env.T[((0, 0), (0, 1))])
#     new.append(env.T[((0, 0), (-1, 0))])
#     new.append(env.T[((0, 0), (1, 0))])
#     # new.append(env.C[((0, 0), (-1, -1))])
#     # new.append(env.C[((0, 0), (-1, 1))])
#     # new.append(env.C[((0, 0), (1, -1))])
#     # new.append(env.C[((0, 0), (1, 1))])

#     diff = 0.
#     if (len(history) > 0):
#         for i in range(4*env.chi):
#             history[i] = new[i]
#             if (abs(old[i]-new[i]) > diff):
#                 diff = abs(old[i]-new[i])
#         for i in range(4):
#             history[4*env.chi+i] = new[4*env.chi+i]
#             if ((old[4*env.chi+i]-new[4*env.chi+i]).norm() > diff):
#                 diff = (old[4*env.chi+i]-new[4*env.chi+i]).norm()
#             # print(torch.div(old[4*env.chi+i], new[4*env.chi+i]))
#             if i == 0:
#                 difftograph.append((old[4*env.chi+i]-new[4*env.chi+i]).norm())

#     else:
#         for i in range(4*env.chi+4):
#             history.append(new[i])
#     history.append(diff)
#     # print("diff={0:<50}".format(diff), end="\r")
#     print("diff={0:<50}".format(diff))
#     # print(ctm_args.ctm_conv_tol)
#     if (len(history[4*env.chi+4:]) > 1 and diff < ctm_args.ctm_conv_tol)\
#             or len(history[4*env.chi+4:]) >= ctm_args.ctm_max_iter:
#         log.info({"history_length": len(
#             history[4*env.chi+4:]), "history": history[4*env.chi+4:]})
#         print("")
#         print("modified CTMRG length: "+str(len(history[4*env.chi+4:])))
#         # import matplotlib.pyplot as plt
#         # plt.plot(difftograph)
#         # plt.show()
#         return True, history
#     return False, history


# def ctmrg_conv_energy(state2, env, history, ctm_args=cfg.ctm_args):
#     if not history:
#         history = []
#     old = []
#     if (len(history) > 0):
#         old = history[:4*env.chi]
#     new = []
#     u, s, v = torch.svd(env.C[((0, 0), (-1, -1))])
#     for i in range(env.chi):
#         new.append(s[i].item())
#     u, s, v = torch.svd(env.C[((0, 0), (1, -1))])
#     for i in range(env.chi):
#         new.append(s[i].item())
#     u, s, v = torch.svd(env.C[((0, 0), (1, -1))])
#     for i in range(env.chi):
#         new.append(s[i].item())
#     u, s, v = torch.svd(env.C[((0, 0), (1, 1))])
#     for i in range(env.chi):
#         new.append(s[i].item())

#     diff = 0.
#     if (len(history) > 0):
#         for i in range(4*env.chi):
#             history[i] = new[i]
#             if (abs(old[i]-new[i]) > diff):
#                 diff = abs(old[i]-new[i])
#     else:
#         for i in range(4*env.chi):
#             history.append(new[i])
#     history.append(diff)
#     print("diff={0:<50}".format(diff), end="\r")
#     # print(ctm_args.ctm_conv_tol)
#     if (len(history[4*env.chi:]) > 1 and diff < ctm_args.ctm_conv_tol)\
#             or len(history[4*env.chi:]) >= ctm_args.ctm_max_iter:
#         log.info({"history_length": len(
#             history[4*env.chi:]), "history": history[4*env.chi:]})
#         print("")
#         print("CTMRG length: "+str(len(history[4*env.chi:])))
#         return True, history
#     return False, history


env = ENV(args.chi, state)
init_env(state, env)
if args.removeCTMRGenv and (os.path.exists(ENVfilenameC) or os.path.exists(ENVfilenameT)):
    print("Removing CTMRG ENV")
    os.remove(ENVfilenameC)
    os.remove(ENVfilenameT)
if args.reuseCTMRGenv and (os.path.exists(ENVfilenameC) and os.path.exists(ENVfilenameT)):
    print("Loading CTMRG ENV: "+ENVfilenameC+" "+ENVfilenameT)
    env.C = torch.load(ENVfilenameC)
    env.T = torch.load(ENVfilenameT)
    for k, c in env.C.items():
        env.C[k] = c.to(cfg.global_args.device)
    for k, t in env.T.items():
        env.T[k] = t.to(cfg.global_args.device)
else:
    env, _, *ctm_log = ctmrg.run(state, env, conv_check=ctmrg_conv_energy)

if args.UseVUMPSansazAC:
    import sys
    sys.path.append('Fixpoints')
    from Fixpoints import vumpsfixedpoints
    _verbose = True
    _steps = 1000
    _tol = 1e-6
    dimsA = state.site((0, 0)).shape
    site = torch.randn((env.chi, dimsA[1]**2, env.chi), dtype=cfg.global_args.torch_dtype,
                       device=cfg.global_args.device).cpu()
    siteDL = stateDL.site((0, 0)).to(cfg.global_args.device).cpu()
    lambd, AL, C, AR, FL, FR = vumpsfixedpoints.vumpsfixedpts(
        site.numpy(), siteDL.numpy(), verbose=_verbose, steps=_steps, tol=_tol)
    _l, _g = vumpsfixedpoints.toLambdaGamma(AL)
    ACUP = vumpsfixedpoints.contractLambdaGamma(_l, _g)
    ACUP = torch.from_numpy(ACUP).to(
        cfg.global_args.device).to(cfg.global_args.torch_dtype)
    # turn site counter clockwise
    siteDL = stateDL.site((0, 0)).cpu()
    siteDL = contiguous(permute(siteDL, (3, 0, 1, 2)))
    lambd, AL, C, AR, FL, FR = vumpsfixedpoints.vumpsfixedpts(
        site.numpy(), siteDL.numpy(), verbose=_verbose, steps=_steps, tol=_tol)
    ACRIGHT = vumpsfixedpoints.contractLambdaGamma(_l, _g)
    ACRIGHT = torch.from_numpy(ACRIGHT).to(
        cfg.global_args.device).to(cfg.global_args.torch_dtype)
    # ACRIGHT = contiguous(permute(ACRIGHT, (0, 2, 1)))
    # ACLEFT = torch.einsum('ijk,kl->ilj', AL, C)

    siteDL = stateDL.site((0, 0)).cpu()
    siteDL = contiguous(permute(siteDL, (3, 0, 1, 2)))
    lambd, AL, C, AR, FL, FR = vumpsfixedpoints.vumpsfixedpts(
        site.numpy(), siteDL.numpy(), verbose=_verbose, steps=_steps, tol=_tol)
    ACDOWN = vumpsfixedpoints.contractLambdaGamma(_l, _g)
    ACDOWN = torch.from_numpy(ACDOWN).to(
        cfg.global_args.device).to(cfg.global_args.torch_dtype)
    ACDOWN = contiguous(permute(ACDOWN, (1, 0, 2)))
    # ACDOWN = torch.einsum('ijk,kl->jil', AL, C)

    siteDL = stateDL.site((0, 0)).cpu()
    siteDL = contiguous(permute(siteDL, (3, 0, 1, 2)))
    lambd, AL, C, AR, FL, FR = vumpsfixedpoints.vumpsfixedpts(
        site.numpy(), siteDL.numpy(), verbose=_verbose, steps=_steps, tol=_tol)
    ACLEFT = vumpsfixedpoints.contractLambdaGamma(_l, _g)
    ACLEFT = torch.from_numpy(ACLEFT).to(
        cfg.global_args.device).to(cfg.global_args.torch_dtype)
    ACLEFT = contiguous(permute(ACLEFT, (0, 2, 1)))
    # ACRIGHT = contiguous(permute(ACRIGHT, (0, 1, 2)))
    # ACRIGHT = torch.einsum('ijk,kl->ijl', AL, C)
    print("Calculated VUMPS")
    env.T[((0, 0), (0, -1))] = ACUP
    env.T[((0, 0), (-1, 0))] = ACLEFT
    env.T[((0, 0), (0, 1))] = ACDOWN
    env.T[((0, 0), (1, 0))] = ACRIGHT
    print("Use VUMPS env as initial ENV.T")
    env, _, *ctm_log = ctmrg.run(state, env, conv_check=ctmrg_conv_energy)
if args.reuseCTMRGenv and not (os.path.exists(ENVfilenameC) or os.path.exists(ENVfilenameT)):
    print("Saving CTMRG ENV: "+ENVfilenameC+" "+ENVfilenameT)
    # new_env.C= { k: c.clone() for k,c in self.C.items() }
    # new_env.T= { k: t.clone() for k,t in self.T.items() }
    torch.save(env.C, ENVfilenameC)
    torch.save(env.T, ENVfilenameT)

# I tried this different conv criterion, because using conv_check=ctmrg_conv_energy, it appears that it will not converge
# env, _, *ctm_log = ctmrg.run(state, env, conv_check=ctmrg_conv_rdm2x1)

# env, P, Pt, *ctm_log = ctmrg_ex.run(state, env, conv_check=ctmrg_conv_energy)
# print ("E_per_site=", model.energy_2x2_1site_BP(state, env).item())

# trans_m = torch.einsum('ijk,jlmn,mab->ilaknb',env.T[((0,0),(0,-1))],stateDL.sites[(0,0)],env.T[((0,0),(0,1))])
# trans_m = trans_m.reshape(((args.chi*args.bond_dim)**2,(args.chi*args.bond_dim)**2))
# trans_m2 = trans_m.detach().cpu().numpy()
# e, v = np.linalg.eig(trans_m2)
# idx = np.argsort(e.real)
# e = e[idx]
# v = v[:,idx]
# print ("correlation_length=", -1/np.log(e[(args.chi*args.bond_dim)**2-2]/e[(args.chi*args.bond_dim)**2-1]).item().real)
################ Hamiltonian################
torch.pi = torch.tensor(
    np.pi, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
kx_int = args.kx
ky_int = args.ky
kx = kx_int*torch.pi/(2*args.size+2)
ky = ky_int*torch.pi/(2*args.size+2)
print("kx=", kx/torch.pi*(2*args.size+2))
print("ky=", ky/torch.pi*(2*args.size+2))
################ Static structure factor###############
s2 = su2.SU2(2, dtype=cfg.global_args.torch_dtype,
             device=cfg.global_args.device)
Id = torch.eye(2, dtype=cfg.global_args.torch_dtype,
               device=cfg.global_args.device)
Sz = 2*s2.SZ()
Sx = s2.SP()+s2.SM()
Sy = -(s2.SP()-s2.SM())*1j

II = torch.einsum('ij,ab->iajb', Id, Id).reshape(4, 4)
ZZ = torch.einsum('ij,ab->iajb', Sz, Sz).reshape(4, 4)
IY = torch.einsum('ij,ab->iajb', Id, Sy).reshape(4, 4)
YI = torch.einsum('ij,ab->iajb', Sy, Id).reshape(4, 4)
IX = torch.einsum('ij,ab->iajb', Id, Sx).reshape(4, 4)
XI = torch.einsum('ij,ab->iajb', Sx, Id).reshape(4, 4)
IZ = torch.einsum('ij,ab->iajb', Id, Sz).reshape(4, 4)
ZI = torch.einsum('ij,ab->iajb', Sz, Id).reshape(4, 4)
SI = torch.einsum('ij,ab->iajb', Sx+Sy+Sz, Id)
IS = torch.einsum('ij,ab->iajb', Id, Sx+Sy+Sz)
HH = 0.5*(SI+IS)
HHII = torch.einsum('iajb,cd,ef->iacejbdf', HH,
                    Id, Id).reshape(4, 4, 4, 4)
IIHH = torch.einsum('mn,kl,iajb->mkianljb', Id,
                    Id, HH).reshape(4, 4, 4, 4)
HHII_permute = torch.einsum('iajb,cd,ef->jbceiadf', HH,
                            Id, Id).reshape(4, 4, 4, 4)
IIHH_permute = torch.einsum('mn,kl,iajb->mkjbnlia', Id,
                            Id, HH).reshape(4, 4, 4, 4)
ZZII = torch.einsum('ij,ab,cd,ef->iacejbdf', Sz,
                    Sz, Id, Id).reshape(4, 4, 4, 4)
IIZZ = torch.einsum('ij,ab,cd,ef->iacejbdf', Id,
                    Id, Sz, Sz).reshape(4, 4, 4, 4)
XXII = torch.einsum('ij,ab,cd,ef->iacejbdf', Sx,
                    Sx, Id, Id).reshape(4, 4, 4, 4)
IIXX = torch.einsum('ij,ab,cd,ef->iacejbdf', Id,
                    Id, Sx, Sx).reshape(4, 4, 4, 4)
XXII_permute = torch.einsum('ij,ab,cd,ef->jbceiadf', Sx,
                            Sx, Id, Id).reshape(4, 4, 4, 4)
IIXX_permute = torch.einsum('ij,ab,cd,ef->iadfjbce', Id,
                            Id, Sx, Sx).reshape(4, 4, 4, 4)
IXXI = torch.einsum('ij,ab,cd,ef->iacejbdf', Id,
                    Sx, Sx, Id).reshape(4, 4, 4, 4)
IZZI = torch.einsum('ij,ab,cd,ef->iacejbdf', Id,
                    Sz, Sz, Id).reshape(4, 4, 4, 4)
IZZI_permute = permute(IZZI, (2, 3, 0, 1))
IYYI = torch.einsum('ij,ab,cd,ef->iacejbdf', Id,
                    Sy, Sy, Id).reshape(4, 4, 4, 4)
YIIY = torch.einsum('ij,ab,cd,ef->iacejbdf', Sy,
                    Id, Id, Sy).reshape(4, 4, 4, 4)
YIIY_permute = permute(YIIY, (2, 3, 0, 1))
IIII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id,
                    Id, Id, Id).reshape(4, 4, 4, 4)

# rdm2x1= rdm2x1((0,0),state,env)
# energy_per_site= torch.einsum('ijkl,ijkl',rdm2x1,args.Jy*IYYI + args.hx*(IX+XI)/4)
energy_per_site = 0
# if args.statefile == "KitaevLG.json":
#     print("KitaevLG shortcut energy_per_site")
#     energy_per_site = torch.tensor(-0.16346756553739336,
#                                    dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
# elif args.statefile == "KitaevLGDG.json":
#     print("KitaevLGDG shortcut energy_per_site")
#     energy_per_site = torch.tensor(-0.19628756414493875,
#                                    dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
# else:
energy_per_site = energy_f(
    state, env)
print("E_per_bond=", energy_per_site.item())
# IIII = torch.einsum('ij,ab->iajb', II, II).reshape(4,4,4,4)
# XX = torch.einsum('ij,ab->iajb', Sx, Sx).reshape(4,4)
# YYII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sy, Sy, Id).reshape(4,4,4,4)
# ZZII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sz, Sz, Id).reshape(4,4,4,4)

# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(False)
# now the rdm is related to hamiltonian as follows
#        s1 s2              x is on-site
#        s3 s4              y is on vertical
#                           z is on horizontal  all of them are on adjacent local sites
# i.e.
# XX = torch.einsum('ij,ab->iajb', Sx, Sx)
# YYII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sy, Sy, Id)
# ZZII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sz, Sz, Id)

# B_grad = torch.zeros((len(state.sites), model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
#                      dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
B_grad = torch.zeros((len(state.sites), model.phys_dim**2, bond_dim, bond_dim, bond_dim, bond_dim),
                     dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
if len(state.sites) == 1:
    B_grad[0].requires_grad_(True)
    sitesB = {(0, 0): B_grad[0]}
    stateB = IPEPS(sitesB, state.vertexToSite)
with torch.no_grad():
    P, Pt = Create_Projectors(state, stateDL, env, args)
if len(state.sites) == 1:
    ### new P Pt?###

    B_grad = B_grad[0].requires_grad_(True)

    lam = torch.tensor(1.0, dtype=cfg.global_args.torch_dtype,
                       device=cfg.global_args.device)
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Norm_Env(state, stateDL, stateB, env, P, Pt, lam, kx, ky, args)
    history = []
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = dict(
    ), dict(), dict(), dict(), dict(), dict(), dict(), dict()
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Norm_Env(state, stateDL, B_grad, env, lam, kx, ky,\
    #         C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, firsttime=True, lasttime=False)
    # for cnt in range(1000):
    #     C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Norm_Env(state, stateDL, B_grad, env, lam, kx, ky,\
    #         C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, firsttime=False, lasttime=False)
    #     isconv, history = ctmrg_conv_energy(None, env, history, ctm_args=cfg.ctm_args)
    #     print ("cnt=", cnt)
    #     if isconv:
    #         break
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Norm_Env(state, stateDL, B_grad, env, lam, kx, ky,\
    #         C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, firsttime=False, lasttime=True)
    # Norm = Create_Norm(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args)

    # # Create ACtual stateDL with e^ikr
    # base = state.site((0, 0))
    # ACsites_ = dict()
    # # prepare more sites for boundary conditions
    # for i in range(-args.size-3, args.size+2+3):
    #     for j in range(-args.size-3, args.size+2+3):
    #         ACsites_[(i, j)] = base + lam * torch.exp(-1j*(kx*i+ky*j)) * B_grad

    # def lattice_to_site(coord):
    #     return coord
    # ACstate = IPEPS(ACsites_, vertexToSite=lattice_to_site)
    # ACsitesDL = dict()
    # for coord, A in ACstate.sites.items():
    #     dimsA = A.size()
    #     a = contiguous(einsum('mefgh,mabcd->eafbgchd', A, conj(base)))
    #     a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    #     ACsitesDL[coord] = a
    # ACstateDL = IPEPS(sitesDL, state.vertexToSite)
    if (args.NormMat == True):
        # This is the use of original Wei-Lin way, with projectors calc on the fly
        C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Norm_Env(
            state, stateDL, B_grad, env, P, Pt, lam, kx, ky, args)
        Norm = Create_Norm(state, env, C_up, T_up, C_left,
                           T_left, C_down, T_down, C_right, T_right, args)
        # make_dot(Norm[(0, 0)], params={"B": B_grad}).render(
        #     args.datadir+"Norm_B.png", format="png")
        norm_factor = contract(Norm[(0, 0)].detach(), conj(
            state.site((0, 0))), ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])).detach()
        print("norm_factor=", norm_factor)
        # norm_factor = torch.norm(norm_factor)
        # print("norm_factor after norm=", norm_factor)

        # Norm[(0, 0)] = (Norm[(0, 0)])/norm_factor
        # print("divided norm_factor")

        shp = list(Norm[(0, 0)].size())
        newshp = shp.copy()
        for ii in shp:
            newshp.append(ii)
        NormMat = torch.zeros(
            newshp, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        elemsize = 1
        accu = [1 for ii in range(len(shp))]
        for ii in shp:
            elemsize = elemsize*ii
        for ii in range(1, len(shp)):
            accu[len(shp)-1-ii] = accu[len(shp)-1-ii+1]*shp[len(shp)-1-ii+1]
        print("Start caclulating NormMat...")
        t1 = time.time()
        # streams = [torch.cuda.Stream() for i in range(elemsize)]
        # @torch.compile
        # def innergrad(ii):
        #     loc = [0 for jj in range(len(shp))]
        #     n = ii
        #     for jj in range(len(shp)):
        #         loc[jj] = n//accu[jj]
        #         n = n%accu[jj]
        #     with torch.cuda.stream(streams[ii]):
        #         # print(ii)
        #         NormMat[(...,)+tuple(loc)] = conj(torch.autograd.grad(Norm[(0,0)][tuple(loc)].real, conj(B_grad), create_graph=False, retain_graph=True)[0])
        #         NormMat[(...,)+tuple(loc)] += conj(torch.autograd.grad(Norm[(0,0)][tuple(loc)].imag, conj(B_grad), create_graph=False, retain_graph=True)[0])
        #         # print("end", ii)
        # for ii in range(elemsize):
        #     streams[ii].wait_stream(torch.cuda.current_stream())
        for ii in range(elemsize):
            loc = [0 for jj in range(len(shp))]
            n = ii
            for jj in range(len(shp)):
                loc[jj] = n//accu[jj]
                n = n % accu[jj]
            # with torch.cuda.stream(streams[ii]):
            # print(ii)
            NormMat[(...,)+tuple(loc)] = 0.5*conj(torch.autograd.grad(Norm[(0, 0)]
                                                                      [tuple(loc)].real, B_grad, create_graph=False, retain_graph=True)[0])
            NormMat[(...,)+tuple(loc)] += 0.5*1j*conj(torch.autograd.grad(Norm[(0, 0)]
                                                                          [tuple(loc)].imag, B_grad, create_graph=False, retain_graph=True)[0])
            # print("end", ii)
            # innergrad(ii)
        t2 = time.time()
        print("NormMat caclulated, time=", t2-t1)
        # tmp_rdm= rdm.rdm1x1((0,0),state,env)
        # sxsx_exp = torch.trace(tmp_rdm@IXIX)
        # sysy_exp = torch.trace(tmp_rdm@IYIY)
        # szsz_exp = torch.trace(tmp_rdm@IZIZ)

        # NormMat2 = view(NormMat, (1024, 1024))
        # temp = contract(NormMat2, conj(
        #     view(state.site((0, 0)), (1024))), ([1], [0]))
        # print("<Norm>=", contract(temp, view(state.site((0, 0)), (1024)),
        #                           ([0], [0])).item().real/(2*args.size+2)**2)

        NormMat = NormMat.detach().cpu().numpy()
        np.save(args.datadir+"kx{}ky{}NormMat.npy".format(args.kx, args.ky), NormMat)
        print("NormMat saved to "+args.datadir +
              "kx{}ky{}NormMat.npy".format(args.kx, args.ky))

    B_grad = torch.zeros((len(state.sites), model.phys_dim**2, bond_dim, bond_dim, bond_dim, bond_dim),
                         dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
    B_grad = B_grad[0].requires_grad_(True)

    lam = torch.tensor(1.0, dtype=cfg.global_args.torch_dtype,
                       device=cfg.global_args.device)
    mu = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
                      device=cfg.global_args.device).requires_grad_(True)
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(state, stateDL, stateB, env, P, Pt, lam, mu, II, IXIX, IYIY, 1.0, kx, ky, args)
    Honsite = II + mu * args.Jz*ZZ  # might not have use
    # Hx = -args.hx*Sx

    # Hx = IIII + mu * (args.Jz*IZZI + args.Jx*(IIXX+XXII) /
    #                   4 - args.h*(HHII+IIHH)/4)
    # Hy = IIII + mu * (args.Jy*YIIY + args.Jx*(IIXX+XXII) /
    #                   4 - args.h*(HHII+IIHH)/4)

    # Hx = IIII + mu * (args.Jz*IZZI_permute + args.Jx*(IIXX_permute+XXII_permute) /
    #                   4 - args.h*(HHII_permute+IIHH_permute)/4)
    # Hy = IIII + mu * (args.Jy*YIIY_permute + args.Jx*(IIXX_permute+XXII_permute) /
    #                   4 - args.h*(HHII_permute+IIHH_permute)/4)
    # print("mu=", mu)
    Hx = IIII + mu * (args.Jz*IZZI + args.Jx*(IIXX+XXII) /
                      4 - args.h*(HHII+IIHH)/4)
    Hy = IIII + mu * (args.Jy*YIIY + args.Jx*(IIXX+XXII) /
                      4 - args.h*(HHII+IIHH)/4)
    # print("Hx=", Hx)
    # print("Hy=", Hy)

    # Hx = IIII + mu * (args.Jz*IZZI + args.Jx*(IIXX+XXII) /
    #                   4)
    # Hy = IIII + mu * (args.Jy*YIIY + args.Jx*(IIXX+XXII) /
    #                   4)

    # Hy = II - mu * -(ZZ + args.hx*(IX+XI)/4)
    # Hz = II - mu * -(ZZ + args.hx*(IX+XI)/4)
    # Hy = II - mu * ZZ
    # Hz = II - mu * ZZ
    history = []
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = dict(
    ), dict(), dict(), dict(), dict(), dict(), dict(), dict()
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Localsite_Hami_Env(state, stateDL, B_grad, env, lam, \
    #     Hz, Hy, Hx, II, kx, ky, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, firsttime=True, lasttime=False)
    # for cnt in range(1000):
    #     C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Localsite_Hami_Env(state, stateDL, B_grad, env, lam, \
    #     Hz, Hy, Hx, II, kx, ky, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, firsttime=False, lasttime=False)
    #     isconv, history = ctmrg_conv_energy(None, env, history, ctm_args=cfg.ctm_args)
    #     print ("hami cnt=", cnt)
    #     if isconv:
    #         break
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Localsite_Hami_Env(state, stateDL, B_grad, env, lam, \
    #     Hz, Hy, Hx, II, kx, ky, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, firsttime=False, lasttime=True)
    # Hami = Create_Localsite_Hami(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, Hz, Hy, Hx, II, args)

    # # Create ACtual stateDL with e^ikr
    # base = state.site((0, 0))
    # ACsites_ = dict()
    # # prepare more sites for boundary conditions
    # for i in range(-args.size-3, args.size+2+3):
    #     for j in range(-args.size-3, args.size+2+3):
    #         ACsites_[(i, j)] = base + lam * torch.exp(-1j*(kx*i+ky*j)) * B_grad
    # ACstate = IPEPS(ACsites_, vertexToSite=lattice_to_site)
    # ACsitesDL = dict()
    # for coord, A in ACstate.sites.items():
    #     dimsA = A.size()
    #     a = einsum('mabcd,em->eabcd', A, II + mu * (args.Jy*IY + args.Jz*ZZ/4))
    #     a = einsum('mabcd,em->eabcd', A, II + mu * (args.Jy*YI + args.Jz*ZZ/4))
    #     a = einsum('mabcd,em->eabcd', A, II + mu * (args.Jx*IX + args.Jz*ZZ/4))
    #     a = einsum('mabcd,em->eabcd', A, II + mu * (args.Jx*XI + args.Jz*ZZ/4))
    #     a = contiguous(einsum('mefgh,mabcd->eafbgchd', a, conj(base)))
    #     a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    #     ACsitesDL[coord] = a
    # ACstateDL = IPEPS(sitesDL, state.vertexToSite)

    if (args.HamiMat == True):
        # This is the use of original Wei-Lin way, with projectors calc on the fly
        isOnsiteWorking = False
        # with torch.autograd.graph.save_on_cpu(True):
        C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Localsite_Hami_Env(state, stateDL, B_grad, env, lam,
                                                                                                 Hx, Hy, Honsite, IIII, kx, ky, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, True, True, P, Pt, isOnsiteWorking=isOnsiteWorking, MultiGPU=args.MultiGPU)
        # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Hami_Env(
        #     state, stateDL, B_grad, env, P, Pt, lam, Hx, Hy, kx, ky, args)
        # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = checkpoint(Create_Localsite_Hami_Env, state, stateDL, B_grad, env, lam,
        #                                                                           Hx, Hy, Honsite, II, kx, ky, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, True, True, P, Pt, isOnsiteWorking, use_reentrant=False)
        # Hami = Create_Hami(state, env, C_up, T_up, C_left, T_left,
        #                    C_down, T_down, C_right, T_right, Hx, Hy, args)
        Hami = Create_Localsite_Hami(state, env, C_up, T_up, C_left, T_left, C_down,
                                     T_down, C_right, T_right, Hx, Hy, Honsite, IIII, args, isOnsiteWorking=isOnsiteWorking)
        # Hami = checkpoint(Create_Localsite_Hami, state, env, C_up, T_up, C_left, T_left, C_down,
        #                   T_down, C_right, T_right, Hx, Hy, Honsite, II, args, isOnsiteWorking, use_reentrant=False)

        # Hami[(0, 0)] = Hami[(0, 0)]/norm_factor
        # print("norm_factor divided")

        print("G(H)_dot_state=", contract(Hami[(0, 0)], conj(
            state.site((0, 0))), ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])).item())
        shp = list(Hami[(0, 0)].size())
        newshp = shp.copy()
        for ii in shp:
            newshp.append(ii)
        HamiMat0 = torch.zeros(
            shp, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        HamiMat = torch.zeros(
            newshp, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        elemsize = 1
        accu = [1 for ii in range(len(shp))]
        for ii in shp:
            elemsize = elemsize*ii
        for ii in range(1, len(shp)):
            accu[len(shp)-1-ii] = accu[len(shp)-1-ii+1] * \
                shp[len(shp)-1-ii+1]
        print("Start caclulating HamiMat...")
        t1 = time.time()
        # streams = [torch.cuda.Stream() for i in range(elemsize)]
        # @torch.compile
        # def innergrad(ii):
        #     loc = [0 for jj in range(len(shp))]
        #     n = ii
        #     for jj in range(len(shp)):
        #         loc[jj] = n//accu[jj]
        #         n = n%accu[jj]
        #     with torch.cuda.stream(streams[ii]):
        #         # print(loc)
        #         HamiMat0[tuple(loc)] = conj(torch.autograd.grad(Hami[(0,0)][tuple(loc)].real, mu, create_graph=True, retain_graph=True)[0])
        #         HamiMat0[tuple(loc)] += conj(torch.autograd.grad(Hami[(0,0)][tuple(loc)].imag, mu, create_graph=True, retain_graph=True)[0])
        #         HamiMat[(...,)+tuple(loc)] = conj(torch.autograd.grad(HamiMat0[tuple(loc)].real, B_grad, create_graph=False, retain_graph=True)[0])
        #         HamiMat[(...,)+tuple(loc)] += conj(torch.autograd.grad(HamiMat0[tuple(loc)].imag, B_grad, create_graph=False, retain_graph=True)[0])
        #         HamiMat0.detach_()

        # for ii in range(elemsize):
        #     streams[ii].wait_stream(torch.cuda.current_stream())
        for ii in range(elemsize):
            loc = [0 for jj in range(len(shp))]
            n = ii
            for jj in range(len(shp)):
                loc[jj] = n//accu[jj]
                n = n % accu[jj]
            # with torch.cuda.stream(streams[ii]):
            # print(loc)
            HamiMat0[tuple(loc)] = 0.5*conj(torch.autograd.grad(
                Hami[(0, 0)][tuple(loc)].real, mu, create_graph=True, retain_graph=True)[0])
            HamiMat0[tuple(loc)] += 0.5*1j*conj(torch.autograd.grad(Hami[(0, 0)]
                                                                    [tuple(loc)].imag, mu, create_graph=True, retain_graph=True)[0])
            HamiMat[(...,)+tuple(loc)] = 0.5*conj(torch.autograd.grad(HamiMat0[tuple(loc)
                                                                               ].real, B_grad, create_graph=False, retain_graph=True)[0])
            HamiMat[(...,)+tuple(loc)] += 0.5*1j*conj(torch.autograd.grad(
                HamiMat0[tuple(loc)].imag, B_grad, create_graph=False, retain_graph=True)[0])
            # HamiMat0.detach_()
            # HamiMat.detach_()

            HamiMat0.detach_()
            HamiMat.detach_()
            # innergrad(ii)
        t2 = time.time()
        print("HamiMat caclulated, time=", t2-t1)

        # HamiMat2 = view(HamiMat, (1024, 1024))
        # temp = contract(HamiMat2, conj(
        #     view(state.site((0, 0)), (1024))), ([1], [0]))
        # print("<Hami>=", 2*contract(temp, view(state.site((0, 0)), (1024)),
        #                             ([0], [0])).item().real/(2*args.size+2)**3/(2*args.size+1))

        HamiMat = HamiMat.detach().cpu().numpy()
        np.save(args.datadir +
                "kx{}ky{}HamiMat.npy".format(args.kx, args.ky), HamiMat)
        print("HamiMat saved to "+args.datadir +
              "kx{}ky{}HamiMat.npy".format(args.kx, args.ky))

    # # Testing Area
    # N_para = state.site((0,0)).size()[0]*state.site((0,0)).size()[1]*state.site((0,0)).size()[2]*state.site((0,0)).size()[3]*state.site((0,0)).size()[4]
    # H_final = torch.zeros((N_para,N_para),\
    #                       dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    # Hami_d = view(Hami[(0,0)],(N_para))
    # dev_accu = torch.zeros((N_para), \
    #                        dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    # for i in range(N_para):
    #     #Hami_d[i].backward(torch.ones_like(Hami_d[i]), retain_graph=True)
    #     #temp_d = B_grad.grad.view(N_para).clone()
    #     g = torch.autograd.grad(Hami_d[i].real,mu,create_graph=True)
    #     g[0].real.backward(retain_graph=True)
    #     temp_d = B_grad.grad.view(N_para).clone()
    #     dev = temp_d.clone() - dev_accu
    #     dev_accu = temp_d.clone()
    #     H_final[i] = dev
    # HamiMat = H_final

    # if args.NormMat == True:
    #     NormMat = NormMat.detach().cpu().numpy()
    #     np.save(args.datadir+"kx{}ky{}NormMat.npy".format(args.kx, args.ky), NormMat)
    # if args.HamiMat == True:
    #     HamiMat = HamiMat.detach().cpu().numpy()
    #     np.save(args.datadir+"kx{}ky{}HamiMat.npy".format(args.kx, args.ky), HamiMat)

    # NormMat = NormMat.reshape(np.prod(NormMat.shape[:5]), np.prod(NormMat.shape[5:]))
    # HamiMat = HamiMat.reshape(np.prod(HamiMat.shape[:5]), np.prod(HamiMat.shape[5:]))
    # print(NormMat.shape)
    # print(HamiMat.shape)
    # Es, Bs = scipy.linalg.eig(HamiMat, NormMat)

    # np.save("Es_P.npy", Es)
    # np.save("Bs_P.npy", Bs)

    # Save data
    # tmp = [h, ((stat_x)/norm_factor/2).item().real + ((stat_y)/norm_factor/2).item().real + ((stat_z)/norm_factor/2).item().real]
    # tmp = np.asarray(tmp)
    # with open(SSFfn, "a") as f:
    #     np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
    # f.close()


tEnd = time.time()
print("time_ener=", tEnd - tStart)
