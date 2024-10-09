# python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
# python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --seed 123 --j2 0. --Kz 1.5 --num_h 10
# python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --Kz 1.5 --num_h 10
# python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim 2 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python SMA_stored_mat_withP.py --hx 2.5 --GLOBALARGS_dtype complex128 --bond_dim 2 --size 12 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from GTNOs import *
from models import aniso_k
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
# from Hami_ori import *
from Localsite_Hami_ori_withP import *
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

parser = cfg.get_args_parser()
parser.add_argument("--kx", type=float, default=0.,
                    help="kx of TFIM")
parser.add_argument("--ky", type=float, default=0.,
                    help="ky of TFIM")
parser.add_argument("--hx", type=float, default=1.,
                    help="hx of TFIM")
parser.add_argument("--q", type=float, default=0.,
                    help="q of TFIM")
parser.add_argument("--num_h", type=int, default=0,
                    help="index of h list")
parser.add_argument("--size", type=int, default=10, help="effective size")
parser.add_argument("--statefile", type=str, default="TFIM_output_state.json",
                    help="filename for TFIM input state")
parser.add_argument("--datadir", type=str, default="data/h3.0chi8/",
                    help="datadir")
args, unknown_args = parser.parse_known_args()

cfg.configure(args)
# cfg.print_config()
torch.set_num_threads(64)
torch.set_num_interop_threads(64)  # Inter-op parallelism
torch.set_num_threads(64)  # Intra-op parallelism
model = ising.ISING(hx=args.hx, q=args.q)
energy_f = model.energy_1x1
bond_dim = args.bond_dim


def _cast_to_real(t):
    return t.real


state = read_ipeps(args.datadir+args.statefile)
# ##########################################################################################
# state.sites[(0,0)] = torch.from_numpy(np.load(args.datadir+"test.npy"))
# ##########################################################################################

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
torch.pi = torch.tensor(np.pi, dtype=torch.complex128)
kx_int = args.kx
ky_int = args.ky
kx = kx_int*torch.pi/24.
ky = ky_int*torch.pi/24.
print("kx=", kx/torch.pi*(2*args.size+2))
print("ky=", ky/torch.pi*(2*args.size+2))


def ctmrg_conv_energy(state2, env, history, ctm_args=cfg.ctm_args):
    if not history:
        history = []
    old = []
    if (len(history) > 0):
        old = history[:4*env.chi]
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

    diff = 0.
    if (len(history) > 0):
        for i in range(4*env.chi):
            history[i] = new[i]
            if (abs(old[i]-new[i]) > diff):
                diff = abs(old[i]-new[i])
    else:
        for i in range(4*env.chi):
            history.append(new[i])
    history.append(diff)
    # print("diff=", diff)
    print("diff={0:<50}".format(diff), end="\r")
    # print(ctm_args.ctm_conv_tol)
    if (len(history[4*env.chi:]) > 1 and diff < ctm_args.ctm_conv_tol)\
            or len(history[4*env.chi:]) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(
            history[4*env.chi:]), "history": history[4*env.chi:]})
        print("")
        print("CTMRG length: "+str(len(history[4*env.chi:])))
        return True, history
    return False, history

    if (len(history[4*env.chi:]) > 1 and diff < ctm_args.ctm_conv_tol)\
            or len(history[4*env.chi:]) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(
            history[4*env.chi:]), "history": history[4*env.chi:]})
        # print (len(history[4*env.chi:]))
        return True, history
    return False, history


env = ENV(args.chi, state)
init_env(state, env)

env, _, *ctm_log = ctmrg.run(state, env, conv_check=ctmrg_conv_energy)

s2 = su2.SU2(2, dtype=cfg.global_args.torch_dtype,
             device=cfg.global_args.device)
Id = torch.eye(2, dtype=cfg.global_args.torch_dtype,
               device=cfg.global_args.device)
Sz = 2*s2.SZ()
Sx = s2.SP()+s2.SM()
Sy = -(s2.SP()-s2.SM())*1j
IX = torch.einsum('ij,ab->iajb', Id, Sx).reshape(2, 2, 2, 2)
XI = torch.einsum('ij,ab->iajb', Sx, Id).reshape(2, 2, 2, 2)
ZZ = torch.einsum('ij,ab->iajb', Sz, Sz).reshape(2, 2, 2, 2)
YY = torch.einsum('ij,ab->iajb', Sy, Sy).reshape(2, 2, 2, 2)
rdm2x1 = rdm2x1((0, 0), state, env)
energy_per_site = torch.einsum(
    'ijkl,ijkl', rdm2x1, -(ZZ + YY + args.hx*(IX+XI)/4))
print("E_per_bond=", 2*energy_per_site.item().real)
print("E_per_bond2=", energy_per_site.item())

NormMat = np.load(args.datadir+"kx{}ky{}NormMat.npy".format(args.kx, args.ky))
HamiMat = np.load(args.datadir+"kx{}ky{}HamiMat.npy".format(args.kx, args.ky))
NormMat = NormMat.reshape(
    np.prod(NormMat.shape[:5]), np.prod(NormMat.shape[5:]))
HamiMat = HamiMat.reshape(
    np.prod(HamiMat.shape[:5]), np.prod(HamiMat.shape[5:]))

state_t = view(state.site((0, 0)), (NormMat.shape[0]))
temp = contract(torch.from_numpy(NormMat).to(
    cfg.global_args.device), conj(state_t), ([1], [0]))
print("<Norm>=", contract(temp, state_t,
      ([0], [0])).item().real/(2*args.size+2)**2)

state_t = view(state.site((0, 0)), (HamiMat.shape[0]))
temp = contract(torch.from_numpy(HamiMat).to(
    cfg.global_args.device), conj(state_t), ([1], [0]))
print("<Hami>=", 2*contract(temp, state_t,
      ([0], [0])).item().real/(2*args.size+2)**3/(2*args.size+1))

# NormMat = NormMat/(2*args.size+2)**2
# HamiMat = 2*HamiMat/(2*args.size+2)**3/(2*args.size+1)

# I dunno whether this is necessary
# NormMat = NormMat + np.conj(np.transpose(NormMat))
# HamiMat = HamiMat + np.conj(np.transpose(HamiMat))

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
#     if np.abs(e)<1e-8:
#         Nvs = np.delete(Nvs, i, axis=1)
#         print("delete: ", i)
# Proj = Nvs
# ProjDag = np.conj(np.transpose(Nvs))
# print("Proj.shape: ", Proj.shape)
# HamiMat = ProjDag@HamiMat@Proj
# NormMat = ProjDag@NormMat@Proj

e, v = np.linalg.eig(NormMat)
idx = np.argsort(-e.real)
e = e[idx]
v = v[:, idx]
################ Projector###############
eig_size = 3
vt = np.zeros((NormMat.shape[0], eig_size), dtype=np.complex128)
with open(args.datadir+"eigN.txt", "a") as f:
    f.write("#kx={}, ky={}, hx={}, q={}\n".format(
        args.kx, args.ky, args.hx, args.q))
    f.write(" ".join(str(_.real) for _ in e))
    f.write("\n")
for i in range(eig_size):
    vt[:, i] = v[:, i]
Proj = vt
ProjDag = np.conj(np.transpose(vt))
# NormMat = np.matmul(NormMat, vt)
# NormMat = np.matmul(np.transpose(np.conj(vt)), NormMat)
# HamiMat = np.matmul(HamiMat, vt)
# HamiMat = np.matmul(np.transpose(np.conj(vt)), HamiMat)
HamiMat_Ori = HamiMat
HamiMat = ProjDag@HamiMat@Proj
NormMat = ProjDag@NormMat@Proj

NormMat_inv = linalg.pinvh(NormMat)  # , cond=0.000001, rcond=0.000001)
Es, Bs = np.linalg.eig(np.matmul(NormMat_inv, HamiMat))
idx = np.argsort(Es)
Es = Es[idx]
Bs = Bs[:, idx]
# print (ef)
print("E_lowest_ex=", (2*args.size+2)*(2*args.size+1)
      * (Es[0]-2*energy_per_site.item().real))
print("ALL_E_lowest_ex=", (2*args.size+2) *
      (2*args.size+1)*(Es-2*energy_per_site.item().real))


# print("HamiMat.shape: ", HamiMat.shape)
# print("NormMat.shape: ", NormMat.shape)
# Es, Bs = scipy.linalg.eig(HamiMat, NormMat)
# np.save("kx{}ky{}Es.npy".format(args.kx, args.ky), Es)
# np.save("kx{}ky{}Bs.npy".format(args.kx, args.ky), Bs)
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
# Project SxA, SyA, SzA to the subspace
SxA = ProjDag@SxA
SyA = ProjDag@SyA
SzA = ProjDag@SzA


# Plot spectral weight

Es = Es.real
Es = (2*args.size+2)*(2*args.size+1)*(Es-2*energy_per_site.item().real)
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
    ans = np.linalg.norm(np.conj(np.transpose(Bs))[:, i]@NormMat@SxA)**2 +\
        np.linalg.norm(np.conj(np.transpose(Bs))[:, i]@NormMat@SyA)**2 +\
        np.linalg.norm(np.conj(np.transpose(Bs))[:, i]@NormMat@SzA)**2
    SW.append((i, Es[i], ans))
    points.append((0, Es[i]))
    values.append(ans)
    # points.append((1, Es[i]))
    # values.append(ans)
    # print(Es[i])
    # print(ans)

with open(args.datadir+"SW.txt", "a") as f:
    f.write("#kx={}, ky={}, hx={}, q={}\n".format(
        args.kx, args.ky, args.hx, args.q))
    for i in range(Es.shape[0]):
        f.write("{} {}\n".format(Es[i], SW[i][2]))

# Ground State Energy
print("NormHami:", np.linalg.norm(HamiMat_Ori))
print("TraceHami:", np.trace(HamiMat_Ori))
print("A_Ori Norm: ", np.linalg.norm(A_Ori))
print("cA_Ori Norm: ", np.linalg.norm(cA_Ori))
# A_Ori = A_Ori/np.linalg.norm(A_Ori)
# cA_Ori = cA_Ori/np.linalg.norm(cA_Ori)
# HamiMat_Ori = HamiMat_Ori/np.linalg.norm(HamiMat_Ori)
# HamiMat_Ori = HamiMat_Ori/np.trace(HamiMat_Ori)
ans = ((A_Ori))@HamiMat_Ori@cA_Ori
# print("GS Energy: ", ans/4/((2*args.size+2))/((2*args.size+1)))
# below vv       sigma_x = 2*Sz and 4*3*2 bonds
print("GS Energy: ", ans/4/3/2/4/4)

# grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

# plt.subplot(221)
# plt.imshow(grid_z2.T, extent=(0,1,np.amin(Es),np.amax(Es)), origin='lower')
# plt.title('Cubic')
# plt.gcf().set_size_inches(6, 6)
# plt.show()
