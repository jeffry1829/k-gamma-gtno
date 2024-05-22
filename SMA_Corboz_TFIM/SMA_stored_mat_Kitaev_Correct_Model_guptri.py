# python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
# python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --seed 123 --j2 0. --Kz 1.5 --num_h 10
# python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --Kz 1.5 --num_h 10
# python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim 2 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python SMA_stored_mat_withP.py --hx 2.5 --GLOBALARGS_dtype complex128 --bond_dim 2 --size 12 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python SMA_stored_mat.py --hx 2.5 --GLOBALARGS_dtype complex128 --bond_dim 2 --size 12 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# import torch
# import config as cfg
# from ipeps.ipeps import *
# from ctm.generic.env import *
# from ctm.generic.rdm import *
# from ctm.generic import ctmrg_ex
# from ctm.generic import ctmrg
# from ctm.generic.ctm_projectors import *
# from Stat_ori import *
# from Norm_ori_withP import *
# from Hami_ori import *
# from Localsite_Hami_ori_withP import *
# from Test import *
# from models import j1j2
# from models import ising
# from models import aniso_k_HsuKe as aniso_k
# from models import kitaev as aniso_k
# from groups.pg import *
# import groups.su2 as su2
# from optim.ad_optim_lbfgs_mod import optimize_state
# from tn_interface import contract, einsum
# from tn_interface import conj
# from tn_interface import contiguous, view, permute
import numpy as np


# from models import aniso_k
# parser = cfg.get_args_parser()
# parser.add_argument("--kx", type=float, default=0.,
#                     help="kx of TFIM")
# parser.add_argument("--ky", type=float, default=0.,
#                     help="ky of TFIM")
# parser.add_argument("--Jx", type=float, default=1.,
#                     help="Jx of Kitave")
# parser.add_argument("--Jy", type=float, default=1.,
#                     help="Jy of Kitave")
# parser.add_argument("--Jz", type=float, default=1.,
#                     help="Jz of Kitave")
# parser.add_argument("--h", type=float, default=0.,
#                     help="h of Kitave")
# parser.add_argument("--num_h", type=int, default=0,
#                     help="index of h list")
# parser.add_argument("--size", type=int, default=10, help="effective size")
# parser.add_argument("--statefile", type=str, default="TFIM_output_state.json",
#                     help="filename for TFIM input state")
# parser.add_argument("--datadir", type=str, default="data/KitaevJx-1Jy-1Jz-1h0chi8/",
#                     help="datadir")
# parser.add_argument("--MultiGPU", type=str, default="False", help="MultiGPU")
# parser.add_argument("--reuseCTMRGenv", type=str, default="True", help="Whether to reuse the ENV after CTMRG optimization\
#     or not. If True, the ENV after CTMRG optimization will be saved in the same directory, and named as statefile+ENVC or ENVT+chi+size.pt")
# parser.add_argument("--removeCTMRGenv", type=str, default="False", help="Whether to remove the ENV after CTMRG optimization\
#     or not. If True, the ENV after CTMRG optimization will be removed.")
# parser.add_argument("--SSF", type=str, default="True",
#                     help="Whether to calculate Static Structure Factor")
# args, unknown_args = parser.parse_known_args()

# cfg.configure(args)

# if MultiGPU == "True":
#     MultiGPU = True
# else:
#     MultiGPU = False
# if reuseCTMRGenv == "True":
#     reuseCTMRGenv = True
# else:
#     reuseCTMRGenv = False
# if removeCTMRGenv == "True":
#     removeCTMRGenv = True
# else:
#     removeCTMRGenv = False
# if SSF == "True":
#     SSF = True
# else:
#     SSF = False

# bond_dim = bond_dim


def _cast_to_real(t):
    return t.real
args = dict()
datadir="data/HsuKe/h_00_to_015_kz_25_free/KitaevAnyon_withP_Jx1.0Jy1.0Jz2.5h0.00chi16size3bonddim4dtypecomplex128/"
kx = 0.0
ky = 0.0
size=4

NormMat = np.load(datadir+"kx{}ky{}NormMat.npy".format(kx, ky))
HamiMat = np.load(datadir+"kx{}ky{}HamiMat.npy".format(kx, ky))
NormMat = NormMat.reshape(
    np.prod(NormMat.shape[:5]), np.prod(NormMat.shape[5:]))
HamiMat = HamiMat.reshape(
    np.prod(HamiMat.shape[:5]), np.prod(HamiMat.shape[5:]))

# I dunno whether this is necessary
NormMat = (NormMat + np.conj(np.transpose(NormMat)))/2.
HamiMat = (HamiMat + np.conj(np.transpose(HamiMat)))/2.
print("symmetrized")
print("NormMat diag sum=", np.trace(NormMat))

# state_t = view(state.site((0, 0)), (NormMat.shape[0]))
# temp = contract(torch.from_numpy(NormMat).to(
#     device=cfg.global_device), conj(state_t), ([1], [0]))
# norm_factor____ = contract(temp, state_t,
#                            ([0], [0])).item()
# print("norm_factor____=", norm_factor____)
# print("<Norm>=", contract(temp, state_t,
#       ([0], [0])).item())
# NormMat = NormMat/norm_factor____
# HamiMat = HamiMat/norm_factor____
# print("norm_factor____ divided")

# state_t = view(state.site((0, 0)), (HamiMat.shape[0]))
# temp = contract(torch.from_numpy(HamiMat).to(
#     device=cfg.global_device), conj(state_t), ([1], [0]))
# print("<Hami>=", contract(temp, state_t, ([0], [0])).item(
# ))

# NormMat = NormMat/((2*size+2))**2
# HamiMat = 2*HamiMat/((2*size+2))**3/((2*size+1))

# NormMat = NormMat/norm_factor____
# HamiMat = HamiMat/norm_factor____/((2*size+2))/((2*size+1))

# NormMat = NormMat - norm_factor____/2
# HamiMat = HamiMat - norm_factor____*((2*size+2))*((2*size+1))/2

# NormMat = NormMat/(2*size+2)**2
# HamiMat = HamiMat/4/(2*size+2)**3/(2*size+1)/2
HamiMat = HamiMat/4/(2*size+2)/(2*size+1)/2

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

# Plot spectral weight

Es = []
from guptri_py import guptri, kcf_blocks
print("a")
NormMat = np.ascontiguousarray(NormMat)
HamiMat = np.ascontiguousarray(HamiMat)
S, T, P, Q, kstr = guptri(HamiMat, NormMat)
print("b")
kcfBs = kcf_blocks(kstr)
print("kcfBs=", kcfBs)
nrow = kcfBs[0,2]
ncol = kcfBs[1,2]
accurow = np.sum(kcfBs[0,:2])
accucol = np.sum(kcfBs[1,:2])
for i in range(nrow):
    for j in range(ncol):
        Es.append(S[accurow+i,accucol+j] / T[accurow+i,accucol+j])
        
print("Es=", Es)
Es = Es.real