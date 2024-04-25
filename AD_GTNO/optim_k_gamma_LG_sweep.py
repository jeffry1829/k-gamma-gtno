#python optim_ising_c4v.py --bond_dim 1 --chi 16 --seed 1234 --hx 3.1 --CTMARGS_fwd_checkpoint_move --OPTARGS_tolerance_grad 1.0e-8 --out_prefix ex-hx31D2 --params_out paramsh31D2 --opt_max_iter 1000 --instate ex-hx31D2_state.json --params_in paramsh31D2
import context
import torch
import argparse
import config as cfg

from ctm.generic.env import *
from ctm.generic import ctmrg
from ipeps.ipeps import *
from ctm.generic import rdm
from models import gamma, kitaev, k_gamma
# from ipeps.ipeps_c4v import *
from groups.pg import make_c4v_symm

# from ctm.one_site_c4v.env_c4v import *
# from ctm.one_site_c4v import ctmrg_c4v
# from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
# from ctm.one_site_c4v import transferops_c4v
# from models import ising
# from optim.ad_optim_sgd_mod import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
# from optim.ad_optim import optimize_state



from GTNOs import *
import unittest
import logging
import json

# 引入 time 模組
import time
import numpy as np

log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--hx", type=float, default=0., help="transverse field")
parser.add_argument("--q", type=float, default=0, help="next nearest-neighbour coupling")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("--tiling", default="BIPARTITE", help="tiling of the lattice", \
    choices=["2SITE"])
args, unknown_args = parser.parse_known_args()

def _cast_to_real(t):
    return t.real

# -------- filenames --------
Efn = f"datas/kg_LG_E1.txt"
magfn = f"datas/kg_LG_mag1.txt"
csfn = f"datas/kg_LG_cs1.txt"
# -------- filenames --------

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)

    torch.set_num_interop_threads(4) # Inter-op parallelism
    torch.set_num_threads(4) # Intra-op parallelism

    num_params = 7 # 7 for GTNO 8=2*4 for states

    model = k_gamma.K_GAMMA(hx=args.hx, q=args.q)

    if args.tiling == "2SITE":
        energy_f = model.energy_2x2_2site
    else:
        energy_f = model.energy_2x2

    bond_dim = args.bond_dim
    params = []
    #praw = [np.cos(0.146), np.sin(0.146), 0, 0, 0, 0, 0, 2.3564, 5.3280, 2.3564, 5.3280, 2.3564, 5.3280, 2.3564, 5.3280]
    #phi = 3.205e-02; praw = [1.244239416018405597e+00, 4.310500845003572001e-01, 3.066188784427117259e-02, -1.632992975845787842e-01, -2.043957760259060286e-01, 1.888908251710834096e-01, 2.654031654801884410e-01]
    phi = 6.411413578754679432e-02; praw=[ 1.315388238727896564e+00, 5.172812050531362393e-01 ,1.744151331413451578e-01, -3.065793474053498113e-01, -3.046676615781068187e-01, 1.889160874238953458e-01, 3.056285642148779402e-01]      
    ## ENSURE GTNO params are real
    for i in range(7):
        praw[i] = _cast_to_real(praw[i])
    for i in range(num_params):
        params.append(torch.tensor(praw[i],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device))

    print(params)


    @torch.no_grad()
    def ctmrg_conv_rdm2x1(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})

        rdm2x1 = rdm.rdm2x1((0,0), state, env)

        dist= float('inf')
        if len(history["log"]) > 0:
            dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
        history["rdm"]=rdm2x1
        history["log"].append(dist)
        if dist<ctm_args.ctm_conv_tol or len(history["log"]) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log']})
            return True, history
        return False, history
    
    bond_dim = args.bond_dim

    css = np.loadtxt("datas/kg_sym_cs1.txt")
    for i in range(css.shape[0]):
        phi = css[i,0]
        praw = css[i,1:]

        params = []
        for j in range(num_params):
            params.append(torch.tensor(praw[j],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device))
        zs = torch.tensor(np.linspace(1.0, 0, 5), device=cfg.global_args.device)

        BestE = 99999999
        for z in zs:
            print(" z  =", z)
            Q = LG(z)
            A1 = state_111()
            A2 = state_111()
            G1, G2, cs = G_gamma_systematic(params[:7])
            G1A = torch.einsum('ijklm,jabc->ikalbmc', G1, A1).reshape(model.phys_dim, 4, 4, 4)
            QG1A = torch.einsum('ijklm,jabc->ikalbmc', Q, G1A).reshape(model.phys_dim, 8, 8, 8)
            G2A = torch.einsum('ijklm,jabc->ikalbmc', G2, A2).reshape(model.phys_dim, 4, 4, 4)
            QG2A = torch.einsum('ijklm,jabc->ikalbmc', Q, G2A).reshape(model.phys_dim, 8, 8, 8)
            A_= torch.einsum('iklm,akde->ialmde', QG1A, QG2A).reshape(model.phys_dim*model.phys_dim, 8, 8, 8, 8)
            A_= A_/A_.norm()
            A_ = A_.cuda(cfg.global_args.device)
            sites={(0,0): A_}

            if args.tiling in ["2SITE"]:
                def lattice_to_site(coord):
                    vx = (coord[0] + abs(coord[0]) * 2) % 2
                    vy = (coord[1] + abs(coord[1]) * 1) % 1
                    return (vx, vy)
                Q = LG(z)
                B1 = state_111()
                B2 = state_111()
                G1_, G2_, cs = G_gamma_systematic(params[:7])
                G1B = torch.einsum('ijklm,jabc->ikalbmc', G1_, B1).reshape(model.phys_dim, 4, 4, 4)
                QG1B = torch.einsum('ijklm,jabc->ikalbmc', Q, G1B).reshape(model.phys_dim, 8, 8, 8)
                G2B = torch.einsum('ijklm,jabc->ikalbmc', G2_, B2).reshape(model.phys_dim, 4, 4, 4)
                QG2B = torch.einsum('ijklm,jabc->ikalbmc', Q, G2B).reshape(model.phys_dim, 8, 8, 8)
                B_= torch.einsum('iklm,akde->ialmde', QG1B, QG2B).reshape(model.phys_dim*model.phys_dim, 8, 8, 8, 8)
                B_= B_/B_.norm()
                B_ = B_.cuda(cfg.global_args.device)
                sites[(1,0)]= B_
            else:
                def lattice_to_site(coord):
                    return (0,0)

            state_sym = IPEPS(sites,vertexToSite=lattice_to_site)
            ctm_env = ENV(args.chi, state_sym)
            init_env(state_sym, ctm_env)

            st = time.time()  
            ctm_env, *ctm_log = ctmrg.run(state_sym, ctm_env, conv_check=ctmrg_conv_rdm2x1)
            end = time.time()
            print("ctmrg run time : ", end-st)

            st = time.time() 
            loss = energy_f(state_sym, ctm_env, phi=phi)
            end = time.time()
            print("energy eval time : ", end-st) 

            print("loss = ", loss.item())

            if loss.item()<BestE:
                Bestz = z.detach().clone().item()
                BestE = loss.detach().clone().item()
                Es = model.get_E() # [ExK EyK EzK ExG EyG EzG, total]
                ms = model.get_m()

        tmp = np.asarray(Es);  tmp = np.insert(tmp, 0, phi); tmp = np.append(tmp, BestE)
        with open(Efn, "a") as f:
            np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
        f.close()
        tmp = np.asarray(ms); tmp = np.insert(tmp, 0, phi)
        with open(magfn, "a") as f:
            np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
        f.close()
        plist = []
        for p in params:
            plist.append(p.clone().detach().cpu().item().real)
        tmp = np.asarray(plist); tmp = np.insert(tmp, 0, phi); tmp = np.append(tmp, Bestz)
        with open(csfn, "a") as f:
            np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
        f.close()


    exit()

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()