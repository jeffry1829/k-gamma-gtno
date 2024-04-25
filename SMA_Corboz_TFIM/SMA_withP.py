# python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
# python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --seed 123 --j2 0. --Kz 1.5 --num_h 10
# python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 1 --hx 3.0 --chi 8 --statefile D=2TFIM_output_state.json --size 11 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python SMA_withP.py --GLOBALARGS_dtype complex128 --bond_dim 2 --hx 3.0 --chi 8 --statefile D=2TFIM_output_state.json --size 11 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
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
parser.add_argument("--kx", type=float, default=1.,
                    help="kx")
parser.add_argument("--ky", type=float, default=1.,
                    help="ky")
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
# torch.set_num_threads(args.omp_cores)
torch.set_num_threads(64)
torch.set_num_interop_threads(64)  # Inter-op parallelism
torch.set_num_threads(64)  # Intra-op parallelism

# torch.backends.cuda.matmul.allow_tf32 = True
# print(torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

model = ising.ISING(hx=args.hx, q=args.q)
energy_f = model.energy_1x1
bond_dim = args.bond_dim


def _cast_to_real(t):
    return t.real


state = read_ipeps(args.datadir+args.statefile)
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
kx = kx_int*torch.pi/24.
ky = ky_int*torch.pi/24.
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

II = torch.einsum('ij,ab->iajb', Id, Id).reshape(2, 2, 2, 2)
XX = torch.einsum('ij,ab->iajb', Sx, Sx).reshape(2, 2, 2, 2)
IX = torch.einsum('ij,ab->iajb', Id, Sx).reshape(2, 2, 2, 2)
XI = torch.einsum('ij,ab->iajb', Sx, Id).reshape(2, 2, 2, 2)
YY = torch.einsum('ij,ab->iajb', Sy, Sy).reshape(2, 2, 2, 2)
ZZ = torch.einsum('ij,ab->iajb', Sz, Sz).reshape(2, 2, 2, 2)

rdm2x1 = rdm2x1((0, 0), state, env)
energy_per_site = torch.einsum('ijkl,ijkl', rdm2x1, -(ZZ + args.hx*(IX+XI)/4))
print("E_per_bond=", 2*energy_per_site.item().real)
# IIII = torch.einsum('ij,ab->iajb', II, II).reshape(4,4,4,4)
# XX = torch.einsum('ij,ab->iajb', Sx, Sx).reshape(4,4)
# YYII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sy, Sy, Id).reshape(4,4,4,4)
# ZZII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sz, Sz, Id).reshape(4,4,4,4)

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
B_grad = torch.zeros((len(state.sites), model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),
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
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
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

    # This is the use of original Wei-Lin way, with projectors calc on the fly
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Norm_Env(state, stateDL, B_grad, env, P, Pt, lam, kx, ky,
                                                                                   args)
    Norm = Create_Norm(state, env, C_up, T_up, C_left, T_left,
                       C_down, T_down, C_right, T_right, args)
    norm_factor = contract(Norm[(0, 0)].detach(), conj(
        state.site((0, 0))), ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))
    print("norm_factor=", norm_factor.item())
    Norm[(0, 0)] = (Norm[(0, 0)])/norm_factor
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
    for ii in range(elemsize):
        loc = [0 for jj in range(len(shp))]
        n = ii
        for jj in range(len(shp)):
            loc[jj] = n//accu[jj]
            n = n % accu[jj]
        NormMat[(...,)+tuple(loc)] = torch.autograd.grad(Norm[(0, 0)]
                                                         [tuple(loc)].real, B_grad, create_graph=False, retain_graph=True)[0]
    t2 = time.time()
    print("NormMat caclulated, time=", t2-t1)
    # tmp_rdm= rdm.rdm1x1((0,0),state,env)
    # sxsx_exp = torch.trace(tmp_rdm@IXIX)
    # sysy_exp = torch.trace(tmp_rdm@IYIY)
    # szsz_exp = torch.trace(tmp_rdm@IZIZ)

    B_grad = torch.zeros((len(state.sites), model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),
                         dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
    B_grad = B_grad[0].requires_grad_(True)

    lam = torch.tensor(1.0, dtype=cfg.global_args.torch_dtype,
                       device=cfg.global_args.device)
    mu = torch.tensor(0.0, dtype=cfg.global_args.torch_dtype,
                      device=cfg.global_args.device).requires_grad_(True)
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(state, stateDL, stateB, env, P, Pt, lam, mu, II, IXIX, IYIY, 1.0, kx, ky, args)
    Hx = Id - mu * args.hx*Sx  # might not have use
    # Hx = -args.hx*Sx
    Hy = II - mu * (ZZ + args.hx*(IX+XI)/4)
    Hz = II - mu * (ZZ + args.hx*(IX+XI)/4)
    # Hy = II - mu * ZZ
    # Hz = II - mu * ZZ
    history = []
    # C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
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

    # This is the use of original Wei-Lin way, with projectors calc on the fly
    isOnsiteWorking = False
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Localsite_Hami_Env(state, stateDL, B_grad, env, lam,
                                                                                             Hz, Hy, Hx, Id, kx, ky, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, firsttime=True, lasttime=True, P=P, Pt=Pt, isOnsiteWorking=isOnsiteWorking)
    Hami = Create_Localsite_Hami(state, env, C_up, T_up, C_left, T_left, C_down,
                                 T_down, C_right, T_right, Hz, Hy, Hx, Id, args, isOnsiteWorking=isOnsiteWorking)
    Hami[(0, 0)] = Hami[(0, 0)]/norm_factor
    print("G(H)_dot_state=", contract(Hami[(0, 0)], conj(
        state.site((0, 0))), ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])).item().real)
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
        accu[len(shp)-1-ii] = accu[len(shp)-1-ii+1]*shp[len(shp)-1-ii+1]
    print("Start caclulating HamiMat...")
    t1 = time.time()
    for ii in range(elemsize):
        loc = [0 for jj in range(len(shp))]
        n = ii
        for jj in range(len(shp)):
            loc[jj] = n//accu[jj]
            n = n % accu[jj]
        print(loc)
        HamiMat0[tuple(loc)] = torch.autograd.grad(
            Hami[(0, 0)][tuple(loc)].real, mu, create_graph=True, retain_graph=True)[0]
        HamiMat[(...,)+tuple(loc)] = torch.autograd.grad(HamiMat0[tuple(loc)].real,
                                                         B_grad, create_graph=False, retain_graph=True)[0]
        HamiMat0.detach_()
    t2 = time.time()
    print("HamiMat caclulated, time=", t2-t1)

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

    NormMat = NormMat.detach().cpu().numpy()
    HamiMat = HamiMat.detach().cpu().numpy()
    np.save(args.datadir+"kx{}ky{}NormMat.npy".format(args.kx, args.ky), NormMat)
    np.save(args.datadir+"kx{}ky{}HamiMat.npy".format(args.kx, args.ky), HamiMat)
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
