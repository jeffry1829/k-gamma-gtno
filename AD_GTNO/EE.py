# python optim_ising_c4v.py --bond_dim 1 --chi 16 --seed 1234 --hx 3.1 --CTMARGS_fwd_checkpoint_move --OPTARGS_tolerance_grad 1.0e-8 --out_prefix ex-hx31D2 --params_out paramsh31D2 --opt_max_iter 1000 --instate ex-hx31D2_state.json --params_in paramsh31D2
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

# from optim.ad_optim_lbfgs_mod import optimize_state
from ctm.generic import transferops
from optim.ad_optim_sgd_mod import optimize_state

from GTNOs import *
import unittest
import logging
import json
import os

# 引入 time 模組
import time
import numpy as np

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--h", type=float, default=0.,
                    help="On site field in <1,1,1> direction.")
parser.add_argument("--Kx", type=float, default=0,
                    help="Kitaev coupling on x bond.")
parser.add_argument("--Ky", type=float, default=0,
                    help="Kitaev coupling on y bond.")
parser.add_argument("--Kz", type=float, default=0,
                    help="Kitaev coupling on z bond.")
parser.add_argument("--top_freq", type=int, default=-1,
                    help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues" +
                    "of transfer operator to compute")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice",
                    choices=["2SITE"])
parser.add_argument("--EEfn", type=str, default="datas/anik/EE_h0_0.6toCSL.txt",
                    help="filename for entanglement entropy")
parser.add_argument("--topEEfn", type=str, default="datas/anik/topEE_h0_0.6toCSL.txt",
                    help="filename for topological entanglement entropy")
args, unknown_args = parser.parse_known_args()


def _cast_to_real(t):
    return t.real


# -------- filenames --------
# Efn = f"datas/appLG/anik_E3_z4_appLG.txt"
# magfn = f"datas/appLG/anik_mag3_z4_appLG.txt"
# csfn = f"datas/appLG/anik_cs3_z4_appLG.txt"
# csinput =  f"datas/appLG/anik_cs3_z4_appLG.txt"
csinput = "datas/aniksdg_noLG_sweepLG/cs_h0_0.6toCSL_K{}.txt".format(args.Kz)

EEfn = args.EEfn
os.makedirs(os.path.dirname(EEfn), exist_ok=True)
topEEfn = args.topEEfn
os.makedirs(os.path.dirname(topEEfn), exist_ok=True)
# -------- filenames --------


def to_bloch(A_):
    A = A_[:, 0, 0, 0]
    a = A[0].real
    b = A[0].imag
    c = A[1].real
    d = A[1].imag
    N = A.conj().dot(A)
    theta = 2*np.arccos((a))
    phi = np.arccos((c)/(np.sin(theta/2)))
    return [theta, phi]


def get_XYZ(A, B):
    Id, X, Y, Z = paulis()
    print("Xa =", torch.einsum("aijk,ba,bijk", A, X, A.conj()) /
          torch.einsum("aijk,aijk", A, A.conj()))
    print("Ya =", torch.einsum("aijk,ba,bijk", A, Y, A.conj()) /
          torch.einsum("aijk,aijk", A, A.conj()))
    print("Za =", torch.einsum("aijk,ba,bijk", A, Z, A.conj()) /
          torch.einsum("aijk,aijk", A, A.conj()))
    print("Xb =", torch.einsum("aijk,ba,bijk", B, X, B.conj()) /
          torch.einsum("aijk,aijk", B, B.conj()))
    print("Yb =", torch.einsum("aijk,ba,bijk", B, Y, B.conj()) /
          torch.einsum("aijk,aijk", B, B.conj()))
    print("Zb =", torch.einsum("aijk,ba,bijk", B, Z, B.conj()) /
          torch.einsum("aijk,aijk", B, B.conj()))


def main():

    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.set_num_interop_threads(4)  # Inter-op parallelism
    torch.set_num_threads(4)  # Intra-op parallelism

    num_params = 13+1+4
    model = aniso_k.ANISO_K(Kx=args.Kx, Ky=args.Ky, Kz=args.Kz, h=args.h)
    # if args.tiling == "2SITE":
    #     energy_f = model.energy_2x2_2site
    # else:
    #     energy_f = model.energy_2x2
    bond_dim = args.bond_dim
    kx, ky, kz, h = args.Kx, args.Ky, args.Kz, args.h

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

    css = np.loadtxt(csinput)
    css = np.asarray(css)

    EEs = []
    EEss = []
    hs = []
    # for i in np.linspace(0, 90, num=30, endpoint=False, retstep=False, dtype=None, axis=0):
    # for i in [0]:
    for i in range(css.shape[0]):
        # for i in [80]:
        h = css[i, 0]
        praw = css[i, 1:]
        for j in range(len(praw)-1):
            if j < 13:
                praw[j+1] = praw[j+1]/praw[0]

        params = []
        for j in range(num_params):
            praw[j+1] = _cast_to_real(praw[j+1])
            params.append(torch.tensor(
                praw[j+1], dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device))

        # Q = LG(params[14])
        # A1 = state_001()
        # A1[0, 0, 0, 0] = torch.cos(params[15]/2)
        # A1[1, 0, 0, 0] = torch.exp(-1j*params[16])*torch.sin(params[15]/2)
        # A2 = state_001()
        # A2[0, 0, 0, 0] = torch.cos(params[17]/2)
        # A2[1, 0, 0, 0] = torch.exp(-1j*params[18])*torch.sin(params[17]/2)

        # G1, G2, cs = G_kitaev_ani_c0(params[:14])

        # G1A = torch.einsum('ijklm,jabc->ikalbmc', G1,
        #                    A1).reshape(model.phys_dim, 2, 2, 2)
        # G1A = torch.einsum('ijklm,jabc->ikalbmc', Q,
        #                    G1A).reshape(model.phys_dim, 4, 4, 4)
        # G2A = torch.einsum('ijklm,jabc->ikalbmc', G2,
        #                    A2).reshape(model.phys_dim, 2, 2, 2)
        # G2A = torch.einsum('ijklm,jabc->ikalbmc', Q,
        #                    G2A).reshape(model.phys_dim, 4, 4, 4)
        # A_ = torch.einsum('iklm,akde->ialmde', G1A,
        #                   G2A).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)
        # A_ = A_/A_.norm()
        # A_ = A_.cuda(cfg.global_args.device)
        # sites = {(0, 0): A_}
        # if args.tiling in ["2SITE"]:
        #     print("2SITE!!!!")

        #     def lattice_to_site(coord):
        #         vx = (coord[0] + abs(coord[0]) * 2) % 2
        #         vy = (coord[1] + abs(coord[1]) * 1) % 1
        #         return (vx, vy)
        #     sites[(1, 0)] = A_
        # else:
        #     def lattice_to_site(coord):
        #         return (0, 0)
        # state_sym = IPEPS(sites, vertexToSite=lattice_to_site)
        # ctm_env = ENV(args.chi, state_sym)
        # init_env(state_sym, ctm_env)
        # ctm_env, *ctm_log = ctmrg.run(state_sym,
        #                               ctm_env, conv_check=ctmrg_conv_rdm2x1)
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
        A_ = A_.cuda(cfg.global_args.device)
        sites = {(0, 0): A_}

        if args.tiling in ["2SITE"]:
            def lattice_to_site(coord):  # This is stripe wrt SITEs
                vx = (coord[0] + abs(coord[0]) * 2) % 2
                vy = (coord[1] + abs(coord[1]) * 1) % 1
                return (vx, vy)
            B1 = A1
            B2 = A2
            G1_, G2_, cs = G_kitaev_ani(params[:13])
            G1B = torch.einsum('ijklm,jabc->ikalbmc', G1_,
                               B1).reshape(model.phys_dim, 2, 2, 2)
            G1B = torch.einsum('ijklm,jabc->ikalbmc', Q,
                               G1B).reshape(model.phys_dim, 4, 4, 4)
            G2B = torch.einsum('ijklm,jabc->ikalbmc', G2_,
                               B2).reshape(model.phys_dim, 2, 2, 2)
            G2B = torch.einsum('ijklm,jabc->ikalbmc', Q,
                               G2B).reshape(model.phys_dim, 4, 4, 4)
            B_ = torch.einsum(
                'iklm,akde->ialmde', G1B, G2B).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)
            B_ = B_/B_.norm()
            B_ = B_.cuda(cfg.global_args.device)
            sites[(1, 0)] = B_
        else:
            def lattice_to_site(coord):
                return (0, 0)

        state_sym = IPEPS(sites, vertexToSite=lattice_to_site)
        ctm_env = ENV(args.chi, state_sym)
        init_env(state_sym, ctm_env)
        ctm_env_out, * \
            ctm_log = ctmrg.run(state_sym, ctm_env,
                                conv_check=ctmrg_conv_rdm2x1)
        Ls = [1, 2, 3, 4, 5]
        tmp_EEs = []
        for L_ in Ls:
            print("L = ", L_)
            spec = transferops.get_full_EH_spec_Ttensor(L=L_, coord=(
                0, 0), direction=(0, 1), state=state_sym, env=ctm_env, verbosity=0)
            print(spec)
            spec /= spec.norm()
            EE = 0
            for ele in spec:
                # EE += -(ele.item())*np.log(ele.item())
                EE += -np.exp(ele.item())*(ele.item())
            tmp_EEs.append(EE)
        print(np.real(tmp_EEs))  # entanglement entropy
        EEs.append(np.real(tmp_EEs))

        from scipy.optimize import curve_fit

        def func(x, a, b):
            return a*x+b
        popt, pcov = curve_fit(func, np.asarray(Ls),  np.asarray(tmp_EEs))
        print(popt[1])

        EEss.append(popt[1])  # topological entanglement entropy
        # plt.plot(np.abs(datas13[:, 0]), func_d2(np.abs(datas13[:, 0]), *popt), color = 'r', lw = 0, marker = 'o')#label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        # popt, pcov = curve_fit(func, np.abs(datas12[:, 0]),  datas12[:, -1])
        # plt.plot(np.abs(datas12[:, 0]), func_d2(np.abs(datas12[:, 0]), *popt), color = 'g', lw = 0, marker = 'o')#label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        tmp = np.asarray(np.real(tmp_EEs))
        tmp = np.insert(tmp, 0, h)
        with open(EEfn, "a") as f:
            np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
        f.close()
        tmp = np.asarray(popt[1])
        tmp = np.insert(tmp, 0, h)
        with open(topEEfn, "a") as f:
            np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
        f.close()
    print("hs")
    print(hs)
    print("EEs")
    print(EEs)
    print("EEss")
    print(EEss)
    return 0


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
