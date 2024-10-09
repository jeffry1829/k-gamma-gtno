# python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
# python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --seed 123 --j2 0. --Kz 1.5 --num_h 10
# python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --Kz 1.5 --num_h 10
# python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim 2 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python SMA_stored_mat_withP.py --hx 2.5 --GLOBALARGS_dtype complex128 --bond_dim 2 --size 12 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python SMA_stored_mat.py --hx 2.5 --GLOBALARGS_dtype complex128 --bond_dim 2 --size 12 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
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
from Norm_ori_withP import *
# from Hami_ori import *
from Localsite_Hami_ori import *
# from Test import *
from models import ising
# from models import ising
# from models import aniso_k_HsuKe as aniso_k
# from models import kitaev as aniso_k
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
log = logging.getLogger(__name__)

tStart = time.time()

# from models import aniso_k
parser = cfg.get_args_parser()
parser.add_argument("--kx", type=float, default=0.,
                    help="kx of TFIM")
parser.add_argument("--ky", type=float, default=0.,
                    help="ky of TFIM")
parser.add_argument("--hx", type=float, default=0.,
                    help="nearest-neighbour coupling")
parser.add_argument("--q", type=float, default=0.,
                    help="next nearest-neighbour coupling")
parser.add_argument("--size", type=int, default=10, help="effective size")
parser.add_argument("--statefile", type=str, default="TFIM_output_state.json",
                    help="filename for TFIM input state")
parser.add_argument("--datadir", type=str, default="data/KitaevJx-1Jy-1Jz-1h0chi8/",
                    help="datadir")
parser.add_argument("--MultiGPU", type=str, default="False", help="MultiGPU")
parser.add_argument("--reuseCTMRGenv", type=str, default="True", help="Whether to reuse the ENV after CTMRG optimization\
    or not. If True, the ENV after CTMRG optimization will be saved in the same directory, and named as statefile+ENVC or ENVT+chi+size.pt")
parser.add_argument("--removeCTMRGenv", type=str, default="False", help="Whether to remove the ENV after CTMRG optimization\
    or not. If True, the ENV after CTMRG optimization will be removed.")
parser.add_argument("--SSF", type=str, default="True",
                    help="Whether to calculate Static Structure Factor")
parser.add_argument("--eig_size", type=int, default=1,
                    help="Number of eigenvalues to be calculated")
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
if args.SSF == "True":
    args.SSF = True
else:
    args.SSF = False

# cfg.print_config()
torch.set_num_threads(64)
torch.set_num_interop_threads(64)  # Inter-op parallelism
torch.set_num_threads(64)  # Intra-op parallelism
model = ising.ISING(hx=args.hx, q=args.q)
energy_f = model.energy_1x1
bond_dim = args.bond_dim


def _cast_to_real(t):
    return t.real


state = read_ipeps(args.datadir+args.statefile, global_args=cfg.global_args)


def symmetrize(state):
    A = state.site((0, 0))
    if A.is_complex():
        A_symm = make_c4v_symm(A.real) + \
            make_c4v_symm(A.imag, irreps=["A2"]) * 1.0j
    else:
        A_symm = make_c4v_symm(A)
    # A_symm = make_c4v_symm_A1(A)
    symm_state = IPEPS({(0, 0): A_symm}, vertexToSite=state.vertexToSite)
    return symm_state


state = symmetrize(state)
ENVfilenameC = args.datadir+Path(args.statefile).stem+"ENVC"+"chi" + \
    str(args.chi)+"size"+str(args.size)+".pt"
ENVfilenameT = args.datadir+Path(args.statefile).stem+"ENVT"+"chi" + \
    str(args.chi)+"size"+str(args.size)+".pt"

if str(state.sites[(0, 0)].dtype) == str(cfg.global_args.torch_dtype) and str(state.sites[(0, 0)].device) == str(cfg.global_args.device):
    print("state.dtype and state.device are already correct")
else:
    for key, site in state.sites.items():
        state.sites[key] = site.type(
            cfg.global_args.torch_dtype).to(cfg.global_args.device)
    state.dtype = cfg.global_args.torch_dtype
    state.device = cfg.global_args.device
phys_dim = state.site((0, 0)).size()[0]
sitesDL = dict()
for coord, A in state.sites.items():
    dimsA = A.size()
    a = contiguous(einsum('mefgh,mabcd->eafbgchd', A, conj(A)))
    a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    sitesDL[coord] = a
stateDL = IPEPS(sitesDL, state.vertexToSite)

################ Hamiltonian################
torch.pi = torch.tensor(
    np.pi, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
kx_int = args.kx
ky_int = args.ky
kx = kx_int*torch.pi/(2*args.size+2)
ky = ky_int*torch.pi/(2*args.size+2)
print("kx=", kx/torch.pi*(2*args.size+2))
print("ky=", ky/torch.pi*(2*args.size+2))


def ctmrg_conv_energy(state2, env, history, ctm_args=cfg.ctm_args):
    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200)
    torch.set_printoptions(precision=8)
    if not history:
        history = []
    old = []
    if (len(history) > 0):
        old = history[:8*env.chi+8]
    new = []
    u, s, v = torch.linalg.svd(env.C[((0, 0), (-1, -1))])
    # print("C singular matrix:", s)
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(env.C[((0, 0), (1, -1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(env.C[((0, 0), (1, -1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(env.C[((0, 0), (1, 1))])
    for i in range(env.chi):
        new.append(s[i].item())

    u, s, v = torch.linalg.svd(
        env.T[((0, 0), (0, -1))].reshape(env.chi, env.chi*args.bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(
        env.T[((0, 0), (0, 1))].permute(1, 0, 2).reshape(env.chi, env.chi*args.bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(
        env.T[((0, 0), (-1, 0))].permute(0, 2, 1).reshape(env.chi, env.chi*args.bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(
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
    new.append(env.C[((0, 0), (-1, -1))])
    new.append(env.C[((0, 0), (-1, 1))])
    new.append(env.C[((0, 0), (1, -1))])
    new.append(env.C[((0, 0), (1, 1))])

    diff = 0.
    if (len(history) > 0):
        for i in range(8*env.chi):
            history[i] = new[i]
            if (abs(old[i]-new[i]) > diff):
                diff = abs(old[i]-new[i])
        for i in range(8):
            history[8*env.chi+i] = new[8*env.chi+i]
            if ((old[8*env.chi+i]-new[8*env.chi+i]).abs().max() > diff):
                diff = (old[8*env.chi+i]-new[8*env.chi+i]).abs().max()
            # print(torch.div(old[4*env.chi+i], new[4*env.chi+i]))
            # if i == 0:
            #     difftograph.append((old[4*env.chi+i]-new[4*env.chi+i]).norm())

    else:
        for i in range(8*env.chi+8):
            history.append(new[i])
    history.append(diff)
    print("diff={0:<50}".format(diff), end="\r")
    # print("diff={0:<50}".format(diff))
    # print(ctm_args.ctm_conv_tol)
    if (len(history[8*env.chi+8:]) > 1 and diff < ctm_args.ctm_conv_tol)\
            or len(history[8*env.chi+8:]) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(
            history[8*env.chi+8:]), "history": history[8*env.chi+8:]})
        print("")
        print("modified CTMRG length: "+str(len(history[8*env.chi+8:])))
        # import matplotlib.pyplot as plt
        # plt.plot(difftograph)
        # plt.show()
        return True, history
    return False, history


env = ENV(args.chi, state)
init_env(state, env)
# if args.removeCTMRGenv and (os.path.exists(ENVfilenameC) or os.path.exists(ENVfilenameT)):
#     print("Removing CTMRG ENV")
#     os.remove(ENVfilenameC)
#     os.remove(ENVfilenameT)
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
if args.reuseCTMRGenv and not (os.path.exists(ENVfilenameC) or os.path.exists(ENVfilenameT)):
    print("Saving CTMRG ENV: "+ENVfilenameC+" "+ENVfilenameT)
    # new_env.C= { k: c.clone() for k,c in self.C.items() }
    # new_env.T= { k: t.clone() for k,t in self.T.items() }
    torch.save(env.C, ENVfilenameC)
    torch.save(env.T, ENVfilenameT)

# env, _, *ctm_log = ctmrg.run(state, env, conv_check=ctmrg_conv_energy)

SzSz = model.h2
Sx = model.h1
iden = torch.eye(2, dtype=cfg.global_args.torch_dtype,
                 device=cfg.global_args.device).contiguous()
H_temp = -SzSz - args.hx * \
    (torch.einsum('ij,kl->ikjl', Sx, iden)+torch.einsum('ij,kl->ikjl', iden, Sx))/4
lam = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
                   device=cfg.global_args.device).requires_grad_(True)
lamb = torch.tensor(1.0, dtype=cfg.global_args.torch_dtype,
                    device=cfg.global_args.device)
iden2 = torch.einsum('ij,kl->ikjl', iden, iden)
H = iden2 + lam * H_temp
# H2 = iden2

# calculate the energy per bond
rdm2x1 = rdm2x1((0, 0), state, env)
energy_per_site = torch.einsum('ijkl,ijkl', rdm2x1, H_temp)
print("E_per_bond=", energy_per_site.item().real)

NormMat = np.load(args.datadir+"kx{}ky{}NormMat.npy".format(args.kx, args.ky))
HamiMat = np.load(args.datadir+"kx{}ky{}HamiMat.npy".format(args.kx, args.ky))
NormMat = NormMat.reshape(
    np.prod(NormMat.shape[:5]), np.prod(NormMat.shape[5:]))
HamiMat = HamiMat.reshape(
    np.prod(HamiMat.shape[:5]), np.prod(HamiMat.shape[5:]))

# I dunno whether this is necessary
NormMat = (NormMat + np.conj(np.transpose(NormMat)))/2.
HamiMat = (HamiMat + np.conj(np.transpose(HamiMat)))/2.
print("symmetrized")
print("NormMat diag sum=", np.trace(NormMat))

state_t = view(state.site((0, 0)), (NormMat.shape[0]))
temp = contract(torch.from_numpy(NormMat).to(
    device=cfg.global_args.device), conj(state_t), ([1], [0]))
norm_factor____ = contract(temp, state_t,
                           ([0], [0])).item()
print("norm_factor____=", norm_factor____)
print("<Norm>=", contract(temp, state_t,
      ([0], [0])).item())
# NormMat = NormMat/norm_factor____
# HamiMat = HamiMat/norm_factor____
# print("norm_factor____ divided")

state_t = view(state.site((0, 0)), (HamiMat.shape[0]))
temp = contract(torch.from_numpy(HamiMat).to(
    device=cfg.global_args.device), conj(state_t), ([1], [0]))
print("<Hami>=", contract(temp, state_t, ([0], [0])).item(
))

# NormMat = NormMat/((2*args.size+2))**2
# HamiMat = 2*HamiMat/((2*args.size+2))**3/((2*args.size+1))

# NormMat = NormMat/norm_factor____
# HamiMat = HamiMat/norm_factor____/((2*args.size+2))/((2*args.size+1))

# NormMat = NormMat - norm_factor____/2
# HamiMat = HamiMat - norm_factor____*((2*args.size+2))*((2*args.size+1))/2

# NormMat = NormMat/(2*args.size+2)**2
# HamiMat = HamiMat/(2*args.size+2)**3/(2*args.size+1)/2
HamiMat = HamiMat/(2*args.size+2)/(2*args.size+1)/2

np.set_printoptions(threshold=np.inf)
# print("NormMat", NormMat)
# print("HamiMat", HamiMat)

# This is for Testing Only
# e, v = np.linalg.eig(NormMat)
# idx = np.argsort(-e.real)
# e = e[idx]
# v = v[:,idx]
# print (e)
# e, v2 = np.linalg.eig(HamiMat)
# idx = np.argsort(-e.real)
# e = e[idx]
# v2 = v2[:,idx]
# print (e)

# # Find projector for NormMat
# Nes, Nvs = np.linalg.eig(NormMat)
# # print(Nes)
# for i,e in reversed(list(enumerate(Nes))):
#     if np.abs(e)<1e-3:
#         Nvs = np.delete(Nvs, i, axis=1)
#         print("delete: ", i)
# Proj_ = Nvs
# ProjDag_ = np.conj(np.transpose(Nvs))
# print("Proj.shape: ", Proj_.shape)
# HamiMat = ProjDag_@HamiMat@Proj_
# NormMat = ProjDag_@NormMat@Proj_
# print("HamiMat.shape: ", HamiMat.shape)
# print("NormMat.shape: ", NormMat.shape)

# lam = 3e-4
# HamiMat = HamiMat + np.identity(HamiMat.shape[0])*lam
# NormMat = NormMat + np.identity(NormMat.shape[0])*lam

# # Check if Hami and Norm are Normal
# if not np.allclose(HamiMat@np.conj(np.transpose(HamiMat)),np.conj(np.transpose(HamiMat))@HamiMat, atol=1e-3):
#     raise ValueError("HamiMat is not Normal")
# if not np.allclose(NormMat@np.conj(np.transpose(NormMat)),np.conj(np.transpose(NormMat))@NormMat, atol=1e-3):
#     raise ValueError("NormMat is not Normal")

# # Check if Hami and Norm Hermitian
# if not np.allclose(HamiMat,np.conj(np.transpose(HamiMat)), atol=1e-5):
#     raise ValueError("HamiMat is not Hermitian")
# if not np.allclose(NormMat,np.conj(np.transpose(NormMat)), atol=1e-5):
#     raise ValueError("NormMat is not Hermitian")

e, v = np.linalg.eig(NormMat)
# print(e)
# idx = np.argsort(-e.real)
idx = np.argsort(-np.abs(e))
e = e[idx]
v = v[:, idx]
################ Projector###############
eig_size = args.eig_size
if kx == 0 and ky == 0:
    eig_size = eig_size + 1
eig_truncate_up = 0
vt = np.zeros((NormMat.shape[0], eig_size -
              eig_truncate_up), dtype=cfg.global_args.dtype)
with open(args.datadir+"eigN.txt", "a") as f:
    f.write("#kx={}, ky={}, hx={}, q={}\n".format(
        args.kx, args.ky, args.hx, args.q))
    f.write(" ".join(str(np.abs(_)) for _ in e))
    f.write("\n")
for i in range(eig_size):
    if i >= eig_truncate_up:
        vt[:, i-eig_truncate_up] = v[:, i]
Proj = vt
# ProjDag = np.conj(np.transpose(vt))
ProjDag = scipy.linalg.pinv(vt, rtol=1e-25, atol=1e-25, check_finite=True)
print("Proj@ProjDag: ", np.allclose(Proj@ProjDag, np.identity(Proj.shape[0])))
print("ProjDag@Proj: ", np.allclose(ProjDag @
      Proj, np.identity(ProjDag.shape[0])))
print("Proj@ProjDag@Proj: ", np.allclose(Proj@ProjDag@Proj, Proj))
print("ProjDag@Proj@ProjDag: ", np.allclose(ProjDag@Proj@ProjDag, ProjDag))
HamiMat_Ori = HamiMat
NormMat_Ori = NormMat
HamiMat = ProjDag@HamiMat@Proj
NormMat = ProjDag@NormMat@Proj

# , cond=0.000001, rcond=0.000001)
# NormMat_inv = linalg.pinv(NormMat, check_finite=True)
# Es, Bs = np.linalg.eig(np.matmul(NormMat_inv, HamiMat))
Es, Bs = scipy.linalg.eig(HamiMat, NormMat)
idx = np.argsort(Es)
Es = Es[idx]
Bs = Bs[:, idx]
Bs_Ori = Proj@Bs
# print (ef)

# shift back
# Es = Es - lam
# print("E_lowest_ex=", (2*args.size+2)*(2*args.size+1)
#       * (Es[0]-4*energy_per_site.item().real))
# print("ALL_E_lowest_ex=", (2*args.size+2) *
#       (2*args.size+1)*(Es-4*energy_per_site.item().real))

# print("E_lowest_ex=", (Es[0]-energy_per_site.item()))
# print("ALL_E_lowest_ex=", (Es-energy_per_site.item()))
print("E_lowest_ex=", Es[0])
print("ALL_E_lowest_ex=", Es)

# print("HamiMat.shape: ", HamiMat.shape)
# print("NormMat.shape: ", NormMat.shape)
# Es, Bs = scipy.linalg.eig(HamiMat, NormMat)

# np.save(args.datadir+"kx{}ky{}Es.npy".format(args.kx, args.ky), Es)
# np.save(args.datadir+"kx{}ky{}Bs.npy".format(args.kx, args.ky), Bs)

# Es = np.load("Es.npy")
# Bs = np.load("Bs.npy")
# Es = Es.reshape(4, 4, 4, 4, 4)
# print("Es.shape", Es.shape)
# print("Bs.shape", Bs.shape)
# print(Es)
# print(Bs)

# Constructing S @ A
s2 = su2.SU2(2, dtype=cfg.global_args.torch_dtype,
             device=cfg.global_args.device)
Id = torch.eye(2, dtype=cfg.global_args.torch_dtype,
               device=cfg.global_args.device)
Sz = 2*s2.SZ()
Sx = s2.SP()+s2.SM()
Sy = -(s2.SP()-s2.SM())*1j

OpX = Sx
OpY = Sy
OpZ = Sz
A_Ori = state.site((0, 0)).flatten().detach().cpu().numpy()
cA_Ori = conj(state.site((0, 0))).resolve_conj(
).flatten().detach().cpu().numpy()
SxA = torch.einsum('ij,jabcd->iabcd', OpX, state.site((0, 0))
                   ).reshape(state.site((0, 0)).shape).flatten().detach().cpu().numpy()
SyA = torch.einsum('ij,jabcd->iabcd', OpY, state.site((0, 0))
                   ).reshape(state.site((0, 0)).shape).flatten().detach().cpu().numpy()
SzA = torch.einsum('ij,jabcd->iabcd', OpZ, state.site((0, 0))
                   ).reshape(state.site((0, 0)).shape).flatten().detach().cpu().numpy()
SxAconj = torch.einsum('ij,jabcd->iabcd', OpX, conj(state.site((0, 0)))
                       ).reshape(state.site((0, 0)).shape).flatten().detach().cpu().numpy()
SyAconj = torch.einsum('ij,jabcd->iabcd', OpY, conj(state.site((0, 0)))
                       ).reshape(state.site((0, 0)).shape).flatten().detach().cpu().numpy()
SzAconj = torch.einsum('ij,jabcd->iabcd', OpZ, conj(state.site((0, 0)))
                       ).reshape(state.site((0, 0)).shape).flatten().detach().cpu().numpy()
# Project SxA, SyA, SzA to the subspace
SxA_Ori = SxA
SyA_Ori = SyA
SzA_Ori = SzA
SxAconj_Ori = SxAconj
SyAconj_Ori = SyAconj
SzAconj_Ori = SzAconj
SxA = ProjDag@SxA
SyA = ProjDag@SyA
SzA = ProjDag@SzA
SxAconj = ProjDag@SxAconj
SyAconj = ProjDag@SyAconj
SzAconj = ProjDag@SzAconj

# Plot spectral weight
# Es = Es.real
Es = (2*args.size+2)*(2*args.size+1)*2*(Es.real-energy_per_site.item().real)
with open(args.datadir+"excitedE.txt", "a") as f:
    f.write("#kx={}, ky={}, hx={}, q={}\n".format(
        args.kx, args.ky, args.hx, args.q))
    f.write(" ".join(str(e) for e in Es))
    f.write("\n")

# x is (kx,ky)
# y is w
grid_x, grid_y = np.mgrid[0:1, np.amin(Es):
                          np.amax(Es):100j]
points = []
values = []

# Spectral weight
SW = []
for i in range(Es.shape[0]):
    ans = np.linalg.norm(((Bs_Ori[:, i]))@NormMat_Ori@(SxAconj_Ori))**2 +\
        np.linalg.norm(((Bs_Ori[:, i]))@NormMat_Ori@(SyAconj_Ori))**2 +\
        np.linalg.norm(((Bs_Ori[:, i]))@NormMat_Ori@(SzAconj_Ori))**2
    SW.append((i, Es[i], ans))
    points.append((0, Es[i]))
    values.append(ans)

with open(args.datadir+"SW.txt", "a") as f:
    f.write("#kx={}, ky={}, hx={}, q={}\n".format(
        args.kx, args.ky, args.hx, args.q))
    for i in range(Es.shape[0]):
        f.write("{} {}\n".format(Es[i], SW[i][2]))

# grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

# plt.subplot(221)
# plt.imshow(grid_z2.T, extent=(0,1,np.amin(Es),np.amax(Es)), origin='lower')
# plt.title('Cubic')
# plt.gcf().set_size_inches(6, 6)
# plt.show()

# SSF
if (args.SSF == True):
    with torch.no_grad():
        P, Pt = Create_Projectors(state, stateDL, env, args)
    B_grad = torch.zeros((len(state.sites), model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),
                         dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
    if len(state.sites) == 1:
        B_grad[0].requires_grad_(True)
        sitesB = {(0, 0): B_grad[0]}
        stateB = IPEPS(sitesB, state.vertexToSite)
    B_grad = B_grad[0].requires_grad_(True)

    tmp_rdm = rdm.rdm1x1((0, 0), state, env)
    sxsx_exp = torch.trace(tmp_rdm@Sx)
    # sx_rot_exp = torch.trace(tmp_rdm@Sx_rot)
    sysy_exp = torch.trace(tmp_rdm@Sy)
    # sy_rot_exp = torch.trace(tmp_rdm@Sy_rot)
    szsz_exp = torch.trace(tmp_rdm@Sz)
    # sz_rot_exp = torch.trace(tmp_rdm@Sz_rot)
    # print (sx_exp,sx_rot_exp,sy_exp,sy_rot_exp,sz_exp,sz_rot_exp)

    lam = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
                       device=cfg.global_args.device)
    lamb = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
                        device=cfg.global_args.device).requires_grad_(True)
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(state, stateDL, stateB, env, P, Pt, lam, lamb, iden, Sx, Sx_rot, Sx, sx_exp, sx_rot_exp, 1.0, kx, ky, args)
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(
        state, stateDL, B_grad, env, P, Pt, lam, lamb, Id, Sx, Sx, Sx, sxsx_exp, sxsx_exp, 1.0, kx, ky, args)
    Hami, Hami2 = Create_Stat(state, env, C_up, T_up, C_left,
                              T_left, C_down, T_down, C_right, T_right, args)
    Ham = contract(Hami[(0, 0)], conj(state.site((0, 0))),
                   ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))
    # sx_exp2 = Ham.detach()
    # print (sx_exp2/norm_factor)
    # print (sx_exp*iden)
    stat_x = conj(torch.autograd.grad(
        Ham.real, lamb, retain_graph=True, create_graph=False)[0])
    stat_x = stat_x + \
        1j*conj(torch.autograd.grad(Ham.imag, lamb, retain_graph=True,
                                    create_graph=False)[0])
    # Ham.backward()
    # stat_x = lamb.real.grad + lamb.imag.grad

    lam = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
                       device=cfg.global_args.device)
    lamb = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
                        device=cfg.global_args.device).requires_grad_(True)
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(
        state, stateDL, B_grad, env, P, Pt, lam, lamb, Id, Sy, Sy, Sy, sysy_exp, sysy_exp, 1.0, kx, ky, args)
    Hami, Hami2 = Create_Stat(state, env, C_up, T_up, C_left,
                              T_left, C_down, T_down, C_right, T_right, args)
    Ham = contract(Hami[(0, 0)], conj(state.site((0, 0))),
                   ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))

    stat_y = conj(torch.autograd.grad(
        Ham.real, lamb, retain_graph=True, create_graph=False)[0])
    stat_y = stat_y + \
        1j*conj(torch.autograd.grad(Ham.imag, lamb, retain_graph=True,
                                    create_graph=False)[0])
    # Ham.backward()
    # stat_y = lamb.real.grad + lamb.imag.grad

    lam = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
                       device=cfg.global_args.device)
    lamb = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
                        device=cfg.global_args.device).requires_grad_(True)
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(
        state, stateDL, B_grad, env, P, Pt, lam, lamb, Id, Sz, Sz, Sz, szsz_exp, szsz_exp, 1.0, kx, ky, args)
    Hami, Hami2 = Create_Stat(state, env, C_up, T_up, C_left,
                              T_left, C_down, T_down, C_right, T_right, args)
    Ham = contract(Hami[(0, 0)], conj(state.site((0, 0))),
                   ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))

    stat_z = conj(torch.autograd.grad(Ham.real, lamb,
                                      retain_graph=True, create_graph=False)[0])
    stat_z = stat_z + 1j*conj(torch.autograd.grad(Ham.imag,
                                                  lamb, retain_graph=True, create_graph=False)[0])
    # Ham.backward()
    # stat_z = lamb.real.grad + lamb.imag.grad

    norm_factor = 1.0

    SSF = ((stat_x)/norm_factor/2).item().real + ((stat_y)/norm_factor /
                                                  2).item().real + ((stat_z)/norm_factor/2).item().real
    print("Static_structure_factor=", SSF)
    print(((stat_x)/norm_factor/2).item().real, ((stat_y)/norm_factor /
                                                 2).item().real, ((stat_z)/norm_factor/2).item().real)

    with open(args.datadir+"SSF.txt", "a") as f:
        f.write("#kx={}, ky={}, hx={}, q={}\n".format(
            args.kx, args.ky, args.hx, args.q))
        f.write("{}\n".format(SSF))

    # rot_op = s2.BP_rot()
    # iden = torch.eye(2, dtype=cfg.global_args.torch_dtype,
    #                  device=cfg.global_args.device).contiguous()
    # S = s2.SP()
    # S_rot = torch.einsum('ki,kc,ca->ia', rot_op, S, rot_op)
    # Sm = s2.SM()
    # Sm_rot = torch.einsum('ki,kc,ca->ia', rot_op, Sm, rot_op)
    # Sx = (S+Sm)/2.
    # Sx_rot = torch.einsum('ki,kc,ca->ia', rot_op, Sx, rot_op)
    # Sy = (S-Sm)/2./1j
    # Sy_rot = torch.einsum('ki,kc,ca->ia', rot_op, Sy, rot_op)
    # Sz = s2.SZ()
    # Sz_rot = torch.einsum('ki,kc,ca->ia', rot_op, Sz, rot_op)
    # with torch.no_grad():
    #     P, Pt = Create_Projectors(state, stateDL, env, args)
    # B_grad = torch.zeros((len(state.sites), model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),
    #                      dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
    # if len(state.sites) == 1:
    #     B_grad[0].requires_grad_(True)
    #     sitesB = {(0, 0): B_grad[0]}
    #     stateB = IPEPS(sitesB, state.vertexToSite)
    # B_grad = B_grad[0].requires_grad_(True)

    # tmp_rdm = rdm.rdm1x1((0, 0), state, env)
    # sxsx_exp = torch.trace(tmp_rdm@Sx)
    # sx_rot_exp = torch.trace(tmp_rdm@Sx_rot)
    # sysy_exp = torch.trace(tmp_rdm@Sy)
    # sy_rot_exp = torch.trace(tmp_rdm@Sy_rot)
    # szsz_exp = torch.trace(tmp_rdm@Sz)
    # sz_rot_exp = torch.trace(tmp_rdm@Sz_rot)
    # # print (sx_exp,sx_rot_exp,sy_exp,sy_rot_exp,sz_exp,sz_rot_exp)

    # lam = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
    #                    device=cfg.global_args.device)
    # lamb = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
    #                     device=cfg.global_args.device).requires_grad_(True)
    # # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(state, stateDL, stateB, env, P, Pt, lam, lamb, iden, Sx, Sx_rot, Sx, sx_exp, sx_rot_exp, 1.0, kx, ky, args)
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(
    #     state, stateDL, B_grad, env, P, Pt, lam, lamb, iden, Sx, Sx_rot, Sx, sxsx_exp, sxsx_exp, 1.0, kx, ky, args)
    # Hami, Hami2 = Create_Stat(
    #     state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args)
    # Ham = contract(Hami[(0, 0)], conj(state.site((0, 0))),
    #                ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))
    # # sx_exp2 = Ham.detach()
    # # print (sx_exp2/norm_factor)
    # # print (sx_exp*iden)
    # stat_x = conj(torch.autograd.grad(
    #     Ham.real, lamb, retain_graph=True, create_graph=False)[0])
    # stat_x = stat_x + \
    #     1j*conj(torch.autograd.grad(
    #         Ham.imag, lamb, retain_graph=True, create_graph=False)[0])
    # # Ham.backward()
    # # stat_x = lamb.grad

    # lam = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
    #                    device=cfg.global_args.device)
    # lamb = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
    #                     device=cfg.global_args.device).requires_grad_(True)
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(
    #     state, stateDL, B_grad, env, P, Pt, lam, lamb, iden, Sy, Sy_rot, Sy, sysy_exp, sysy_exp, 1.0, kx, ky, args)
    # Hami, Hami2 = Create_Stat(
    #     state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args)
    # Ham = contract(Hami[(0, 0)], conj(state.site((0, 0))),
    #                ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))

    # stat_y = conj(torch.autograd.grad(
    #     Ham.real, lamb, retain_graph=True, create_graph=False)[0])
    # stat_y = stat_y + \
    #     1j*conj(torch.autograd.grad(
    #         Ham.imag, lamb, retain_graph=True, create_graph=False)[0])
    # # Ham.backward()
    # # stat_y = lamb.grad

    # lam = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
    #                    device=cfg.global_args.device)
    # lamb = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
    #                     device=cfg.global_args.device).requires_grad_(True)
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(
    #     state, stateDL, B_grad, env, P, Pt, lam, lamb, iden, Sz, Sz_rot, Sz, szsz_exp, szsz_exp, 1.0, kx, ky, args)
    # Hami, Hami2 = Create_Stat(
    #     state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args)
    # Ham = contract(Hami[(0, 0)], conj(state.site((0, 0))),
    #                ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))

    # stat_z = conj(torch.autograd.grad(
    #     Ham.real, lamb, retain_graph=True, create_graph=False)[0])
    # stat_z = stat_z + \
    #     1j*conj(torch.autograd.grad(
    #         Ham.imag, lamb, retain_graph=True, create_graph=False)[0])
    # # Ham.backward()
    # # stat_z = lamb.grad

    # norm_factor = 1.0

    # SSF = ((stat_x)/norm_factor/2).item().real + ((stat_y)/norm_factor /
    #                                               2).item().real + ((stat_z)/norm_factor/2).item().real
    # print("Static_structure_factor=", SSF)
    # print(((stat_x)/norm_factor/2).item().real, ((stat_y)/norm_factor /
    #       2).item().real, ((stat_z)/norm_factor/2).item().real)

    # with open(args.datadir+"SSF.txt", "a") as f:
    #     f.write("#kx={}, ky={}, hx={}, q={}\n".format(
    #         args.kx, args.ky, args.hx, args.q))
    #     f.write("{}\n".format(SSF))
