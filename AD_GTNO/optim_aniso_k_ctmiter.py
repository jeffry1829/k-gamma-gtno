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
parser.add_argument(
    "--Efn", type=str, default="datas/anik/anik_E1_h0_25toCSL.txt", help="filename for energy")
parser.add_argument("--magfn", type=str, default="datas/anik/anik_mag1_h0_25toCSL.txt",
                    help="filename for magnetization")
parser.add_argument("--csfn", type=str, default="datas/anik/anik_cs1_h0_25toCSL.txt",
                    help="filename for correlation function")
parser.add_argument("--Wfn", type=str, default="datas/anik/anik_W1_h0_25toCSL.txt",
                    help="filename for correlation function")
parser.add_argument("--IterEfn", type=str, default="datas/anik/anik_W1_h0_25toCSL.txt",
                    help="filename for correlation function")
args, unknown_args = parser.parse_known_args()


def _cast_to_real(t):
    return t.real


# -------- filenames --------
# Efn = f"datas/anik/anik_E1_h0_25toCSL_Kz1.2.txt"
# magfn = f"datas/anik/anik_mag1_h0_25toCSL_Kz1.2.txt"
# csfn = f"datas/anik/anik_cs1_h0_25toCSL_Kz1.2.txt"
Efn = args.Efn
magfn = args.magfn
csfn = args.csfn
Wfn = args.Wfn
IterEfn = args.IterEfn
os.makedirs(os.path.dirname(Efn), exist_ok=True)
os.makedirs(os.path.dirname(magfn), exist_ok=True)
os.makedirs(os.path.dirname(csfn), exist_ok=True)
os.makedirs(os.path.dirname(Wfn), exist_ok=True)

folder = "datas/aniksgdvarLG/"
csinput = folder+"cs_h0_0.6toCSL_K{}.txt".format(args.Kz)

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
    return theta, phi


def main():

    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.set_num_interop_threads(4)  # Inter-op parallelism
    torch.set_num_threads(4)  # Intra-op parallelism
    # A = state_m1m1m1()
    # print(to_bloch(A))
    # exit()
    num_params = 13+0+4
    model = aniso_k.ANISO_K(Kx=args.Kx, Ky=args.Ky, Kz=args.Kz, h=args.h)
    if args.tiling == "2SITE":
        energy_f = model.energy_2x2_2site
    else:
        energy_f = model.energy_2x2
    bond_dim = args.bond_dim
    # kx,ky,kz,h = 1,1,1.5,1
    kx, ky, kz, h = args.Kx, args.Ky, args.Kz, args.h
    # praw=[1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,  0, 5.3280 ,2.3564, 5.3280 ,2.3564] # |1,1,1> h->inf limit
    # praw=[1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4, 0, 1e-4,1e-4,3.14,1e-4] # |0,0,1> |0,0,-1> kz->inf limit
    # praw=[1e-4,1e-1,1e-4,1e-4,1e-1,1e-4,1e-1,1e-4,1e-1,1e-4,1e-1,1e-4,1e-4, 0.99, 1e-4,1e-4,2,1e-4] # |0,0,1> |0,0,-1> kz->inf limit
    # praw=[0,1,0,1,0,1,0,0,0,0,0,0,0,  1, 5.3280 ,2.3564, 5.3280 ,2.3564,] # FM kitaev
    # praw=[0,-0.9,0,-0.9,0,-0.9,0,0,0,0,0,0,0,  1, 5.3280 ,2.3564, 4.0969 , -0.7854] # AFM kitaev

    praw = [0, -0.9, 0, -0.9, 0, -0.9, 0, 0, 0, 0, 0, 0, 0, 5.3280,
            2.3564, 4.0969, -0.7854]  # AFM kitaev no LG no params

    params = []
    for i in range(num_params):
        praw[i] = _cast_to_real(praw[i])
        params.append(torch.tensor(
            praw[i], dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device))

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

    Q = LG(torch.tensor(0, dtype=cfg.global_args.torch_dtype,
           device=cfg.global_args.device))
    # Q = LG(params[13])
    A1 = state_001()
    A1[0, 0, 0, 0] = torch.cos(params[13]/2)
    A1[1, 0, 0, 0] = torch.exp(-1j*params[14])*torch.sin(params[13]/2)
    A2 = state_001()
    A2[0, 0, 0, 0] = torch.cos(params[15]/2)
    A2[1, 0, 0, 0] = torch.exp(-1j*params[16])*torch.sin(params[15]/2)
    G1, G2, cs = G_kitaev_ani(params[:13])
    # G1, G2, cs = G_kitaev_ani(praw[:13])
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
        def lattice_to_site(coord):
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
        B_ = torch.einsum('iklm,akde->ialmde', G1B,
                          G2B).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)
        B_ = B_/B_.norm()
        B_ = B_.cuda(cfg.global_args.device)
        sites[(1, 0)] = B_
    else:
        def lattice_to_site(coord):
            return (0, 0)
    state_sym = IPEPS(sites, vertexToSite=lattice_to_site)
    ctm_env = ENV(args.chi, state_sym)
    init_env(state_sym, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state_sym, ctm_env,
                                  conv_check=ctmrg_conv_rdm2x1)

    # # hs = np.linspace(2,0,40)
    # # hs = np.linspace(0,-1,40)
    # # hs = np.linspace(-0.25, 0, 40)
    # hs = np.linspace(0, -0.6, 120)
    # # zs = [1]
    # for h in hs:
    # # for kz in zs:
    #     print("h = ", h)
    #     # print("z = ", kz)
    css = np.loadtxt(csinput)
    # css = [[0.000000000000000000e+00,1.247583434455966422e+00,4.314883583498645386e-04,-5.110121832191995006e-01,-1.426260985942023494e-03,-4.801889387355199323e-01,-4.181675450647031611e-01,-5.130897004226329106e-01,-1.011528429267444544e-03,1.000000000000000048e-04,8.546385031974417998e-03,1.000000000000000048e-04,2.567400052651009484e-03,1.000000000000000048e-04,2.048893400961039526e-03,0.000000000000000000e+00,6.276061356540131975e+00,2.683293995338117188e+00,3.125123284355635800e+00,-9.193463199351623594e-01]]
    css = np.asarray(css)
    num_params = 13+1+4
    # for i in range(css.shape[0]):
    for i in range(68, 69, 1):
        h = css[i, 0]
        praw = css[i, 1:]
        # print(praw)
        praw = np.insert(praw, 13, 0)
        # print(praw)
        # praw = css[i,1:13]
        # praw.append(0)
        # praw.append(css[i,13:])

        for j in range(len(praw)-1):
            if j < 13:
                praw[j+1] = praw[j+1]/praw[0]

        params = []
        for j in range(num_params):
            praw[j+1] = _cast_to_real(praw[j+1])
            params.append(torch.tensor(
                praw[j+1], dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device))

        def loss_fn(state, ctm_env_in, params, opt_context):
            if opt_context != []:
                ctm_args = opt_context["ctm_args"]
                opt_args = opt_context["opt_args"]

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
            if cfg.opt_args.opt_ctm_reinit:
                init_env(state_sym, ctm_env_in)
            ctm_env_out, * \
                ctm_log = ctmrg.run(state_sym, ctm_env_in,
                                    conv_check=ctmrg_conv_rdm2x1)
            loss = energy_f(state_sym, ctm_env_out, kx, ky, kz, h)
            model.get_m()
            print("Energy = ", loss.item())
            for p in cs:
                print(p.item().real, end=" ,")
            print("\n==============")
            for p in params[13:]:
                print(p.item().real, end=" ,")
            print(" ")

            state = state_sym
            return (loss, ctm_env_out, *ctm_log)

        @torch.no_grad()
        def save(state, ctm_env_in, params):
            print("Start saving ...")
            Es = model.get_E()  # [ExK EyK EzK ExG EyG EzG, total]
            ms = model.get_m()
            Qxx = model.get_Qxx()
            W = model.get_Wp()
            W = [W]

            print("Mags = ", ms)
            print("E_total = ", Es[-1])
            print("W = ", W)

            plist = []
            G1, G2, cs = G_kitaev_ani(params[:13])
            for p in cs:
                plist.append(p.clone().detach().item().real)
            for p in params[13:]:
                plist.append(p.clone().detach().item().real)
            print("params : ", plist)

            tmp = np.asarray(Es)
            tmp = np.insert(tmp, 0, h)
            with open(Efn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()
            tmp = np.asarray(ms)
            tmp = np.insert(tmp, -1, Qxx)
            tmp = np.insert(tmp, 0, h)
            with open(magfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()
            tmp = np.asarray(plist)
            tmp = np.insert(tmp, 0, h)
            with open(csfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()
            tmp = np.asarray(W)
            tmp = np.insert(tmp, 0, h)
            with open(Wfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()
            return 0
        cfg.opt_args.lr = 0.5
        cfg.opt_args.momentum = 0.9
        cfg.opt_args.tolerance_change = 1e-3
        # cfg.opt_args.dampening = 0.1

        # optimize_state(state_sym, ctm_env, loss_fn, params)
        # save(state_sym, ctm_env, params)

        LGps = np.linspace(0, 1, 20)
        bestE = 999999
        bestparams = []
        Es = []
        for p in LGps:
            print("p = ", p)
            params[13] = torch.tensor(
                p, dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
            E = loss_fn(state_sym, ctm_env, params, [])[0]
            if E < bestE:
                bestE = E.detach().cpu().clone().item()
                bestparams = []
                for param in params:
                    bestparams.append(param.detach().cpu().clone())
            Es.append(E.detach().cpu().clone().item())
        print(Es)
        E = loss_fn(state_sym, ctm_env, bestparams, [])[0]
        save(state_sym, ctm_env, bestparams)
    return 0


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
