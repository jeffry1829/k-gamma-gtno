#python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
#python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --seed 123 --j2 0. --Kz 1.5 --num_h 10
#python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --Kz 1.5 --num_h 10
#python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 1 --Kz 1.5 --num_h 10 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
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
from models import aniso_k
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
parser.add_argument("--DSFw", type=float, default="3.14",
                    help="w for dynamical spectral function")
parser.add_argument("--DSFkx", type=float, default="0",
                    help="kx for dynamical spectral function")
parser.add_argument("--DSFky", type=float, default="12",
                    help="ky for dynamical spectral function")
args, unknown_args = parser.parse_known_args()

SSFfn = args.SSFfn
os.makedirs(os.path.dirname(SSFfn), exist_ok=True)

cfg.configure(args)
cfg.print_config()
torch.set_num_threads(64)
torch.set_num_interop_threads(64)  # Inter-op parallelism
torch.set_num_threads(64)  # Intra-op parallelism
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

css = np.loadtxt(csinput)
css = np.asarray(css)
num_params = 13+1+4
# for i in range(css.shape[0]):
i_ = args.num_h
h = css[i_, 0]
praw = css[i_, 1:]
for j in range(len(praw)-1):
    if j < 13:
        praw[j+1] = praw[j+1]/praw[0]
params = []
for j in range(num_params):
    praw[j+1] = _cast_to_real(praw[j+1])
    params.append(torch.tensor(
        praw[j+1], dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device))
Q = LG(params[13])
A1 = state_111()
A1[0, 0, 0, 0] = torch.cos(params[14]/2)
A1[1, 0, 0, 0] = torch.exp(-1j*params[15])*torch.sin(params[14]/2)
A2 = state_m1m1m1()
A2[0, 0, 0, 0] = torch.cos(params[16]/2)
A2[1, 0, 0, 0] = torch.exp(-1j*params[17])*torch.sin(params[16]/2)

G1, G2, cs = G_kitaev_ani(params[:13])
G1A = torch.einsum('ijklm,jabc->ikalbmc', G1,
                    A1).reshape(model.phys_dim, 2, 2, 2)
G1A = torch.einsum('ijklm,jabc->ikalbmc', Q,
                    G1A).reshape(model.phys_dim, 4, 4, 4)
G2A = torch.einsum('ijklm,jabc->ikalbmc', G2,
                    A2).reshape(model.phys_dim, 2, 2, 2)
G2A = torch.einsum('ijklm,jabc->ikalbmc', Q,
                    G2A).reshape(model.phys_dim, 4, 4, 4)
A_ = torch.einsum('iklm,akde->ialmde', G1A,
                    G2A).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)
A_ = A_/A_.norm()
A_ = A_.to(device=cfg.global_args.device)
sites_={(0,0): A_}
def lattice_to_site(coord):
    return (0,0)
state = IPEPS(sites_,vertexToSite=lattice_to_site)
phys_dim = state.site((0,0)).size()[0]
sitesDL=dict()
for coord,A in state.sites.items():
    dimsA = A.size()
    a = contiguous(einsum('mefgh,mabcd->eafbgchd',A,conj(A)))
    a = view(a, (dimsA[1]**2,dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    sitesDL[coord]=a
stateDL = IPEPS(sitesDL,state.vertexToSite)

NormMat = np.load("NormMat.npy")
HamiMat = np.load("HamiMat.npy")
NormMat = NormMat.reshape(np.prod(NormMat.shape[:5]), np.prod(NormMat.shape[5:]))
HamiMat = HamiMat.reshape(np.prod(HamiMat.shape[:5]), np.prod(HamiMat.shape[5:]))

# Project NormMat
u, s, vh = np.linalg.svd(NormMat, full_matrices=False)
M, N = u.shape[0], vh.shape[1]
rcond = np.finfo(s.dtype).eps * max(M, N)
tol = np.amax(s) * rcond
num = np.sum(s > tol, dtype=int)
u = u[:, :num]
s = s[:num]
vh = vh[:num, :]
NormMat = u @ np.diag(s) @ vh
# Project HamiMat
u, s, vh = np.linalg.svd(HamiMat, full_matrices=False)
# M, N = u.shape[0], vh.shape[1]
# rcond = np.finfo(s.dtype).eps * max(M, N)
# tol = np.amax(s) * rcond
# num = np.sum(s > tol, dtype=int)
u = u[:, :num]
s = s[:num]
vh = vh[:num, :]
HamiMat = u @ np.diag(s) @ vh
print("num: ", num)

Es, Bs = scipy.linalg.eig(HamiMat, NormMat)
np.save("Es.npy", Es)
np.save("Bs.npy", Bs)
# Es = np.load("Es.npy")
# Bs = np.load("Bs.npy")
# Es = Es.reshape(4, 4, 4, 4, 4)
print("Es.shape", Es.shape)
print("Bs.shape", Bs.shape)
# print(Es)
# print(Bs)

# Constructing S @ A
s2 = su2.SU2(2, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
Id = torch.eye(2, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
Sz = 2*s2.SZ()
Sx = s2.SP()+s2.SM()
Sy = -(s2.SP()-s2.SM())*1j
II = torch.einsum('ij,ab->iajb', Id, Id).reshape(4,4)
XX = torch.einsum('ij,ab->iajb', Sx, Sx).reshape(4,4)
IX = torch.einsum('ij,ab->iajb', Id, Sx).reshape(4,4)
XI = torch.einsum('ij,ab->iajb', Sx, Id).reshape(4,4)
YY = torch.einsum('ij,ab->iajb', Sy, Sy).reshape(4,4)
IY = torch.einsum('ij,ab->iajb', Id, Sy).reshape(4,4)
YI = torch.einsum('ij,ab->iajb', Sy, Id).reshape(4,4)
ZZ = torch.einsum('ij,ab->iajb', Sz, Sz).reshape(4,4)
IZ = torch.einsum('ij,ab->iajb', Id, Sz).reshape(4,4)
ZI = torch.einsum('ij,ab->iajb', Sz, Id).reshape(4,4)

OpX = IX+XI
OpY = IY+YI
OpZ = IZ+ZI
SxA = torch.einsum('ij,jabcd->iabcd', OpX, A_).reshape(model.phys_dim*model.phys_dim,4,4,4,4).flatten().detach().cpu().numpy()
SyA = torch.einsum('ij,jabcd->iabcd', OpY, A_).reshape(model.phys_dim*model.phys_dim,4,4,4,4).flatten().detach().cpu().numpy()
SzA = torch.einsum('ij,jabcd->iabcd', OpZ, A_).reshape(model.phys_dim*model.phys_dim,4,4,4,4).flatten().detach().cpu().numpy()
# Start calculating dynamical spectral function
# Plot
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# x is (kx,ky)
# y is w
grid_x, grid_y = np.mgrid[0:1, np.amin(Es):np.amax(Es):100j]
points = []
values = []

print("np.amin(Es): ", np.amin(np.abs(Es)))
print("np.amax(Es): ", np.amax(np.abs(Es)))

# Spectral weight
SW = []
for i in range(Es.shape[0]):
    ans = np.linalg.norm(conj(Bs[:,i])@NormMat@SxA)**2+np.linalg.norm(conj(Bs[:,i])@NormMat@SyA)**2+np.linalg.norm(conj(Bs[:,i])@NormMat@SzA)**2
    SW.append((i, Es[i], ans))
    points.append((0, Es[i]))
    values.append(ans)
    points.append((1, Es[i]))
    values.append(ans)
    # print(Es[i])

grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

plt.subplot(221)
plt.imshow(grid_z2.T, extent=(0,1,np.amin(Es),np.amax(Es)), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(6, 6)
plt.show()