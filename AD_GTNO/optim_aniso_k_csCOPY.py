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
 
# from optim.ad_optim_lbfgs_mod import optimize_state
from optim.ad_optim_sgd_mod import optimize_state
 
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
 
def _cast_to_real(t):
    return t.real
 
#-------- filenames --------
# Efn = f"datas/anik/anik_E2_TCtoH_Z2.txt"
# magfn = f"datas/anik/anik_mag2_TCtoH_Z2.txt"
# csfn = f"datas/anik/anik_cs2_TCtoH_Z2.txt"
# Efn = f"datas/anik/anik_E3_h0z3.txt"
# magfn = f"datas/anik/anik_mag3_h0z3.txt"
# csfn = f"datas/anik/anik_cs3_h0z3.txt"
 
# Efn = f"datas/anik/anik_E1_h0_25oTC.txt"
# magfn = f"datas/anik/anik_mag1_h0_25toTC.txt"
# csfn = f"datas/anik/anik_cs1_h0_25toTC.txt"
 
Efn = f"datas/appLG/anik_E1_z4_appLG.txt"
magfn = f"datas/appLG/anik_mag1_z4_appLG.txt"
csfn = f"datas/appLG/anik_cs1_z4_appLG.txt"
 
csinput = f"datas/anik/anik_cs1_test_z4.txt"
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
    return [theta, phi]
 
def get_XYZ(A, B):
    Id, X, Y, Z = paulis()
    print("Xa =", torch.einsum("aijk,ba,bijk",A,X,A.conj())/torch.einsum("aijk,aijk",A,A.conj()))
    print("Ya =", torch.einsum("aijk,ba,bijk",A,Y,A.conj())/torch.einsum("aijk,aijk",A,A.conj()))
    print("Za =", torch.einsum("aijk,ba,bijk",A,Z,A.conj())/torch.einsum("aijk,aijk",A,A.conj()))
    print("Xb =", torch.einsum("aijk,ba,bijk",B,X,B.conj())/torch.einsum("aijk,aijk",B,B.conj()))
    print("Yb =", torch.einsum("aijk,ba,bijk",B,Y,B.conj())/torch.einsum("aijk,aijk",B,B.conj()))
    print("Zb =", torch.einsum("aijk,ba,bijk",B,Z,B.conj())/torch.einsum("aijk,aijk",B,B.conj()))
 
def main():
 
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.set_num_interop_threads(4) # Inter-op parallelism
    torch.set_num_threads(4) # Intra-op parallelism
 
    num_params = 14+1+4
    model = aniso_k.ANISO_K(Kx = args.Kx, Ky = args.Ky, Kz = args.Kz, h = args.h)
    if args.tiling == "2SITE":
        energy_f = model.energy_2x2_2site
    else:
        energy_f = model.energy_2x2
    bond_dim = args.bond_dim
    kx,ky,kz,h = 1,1,2.5,0
 
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
    
    #praw = [1.326642420064569894e+00, -1.521939744844448827e-01 ,-1.353651465022003009e-01, -1.293233640300963175e-02 ,-4.788250733257096159e-01, 3.684791906177163900e-01 ,-1.304175402834510111e-01 ,-3.465694053428229615e-02, 2.114624029016122508e-02 ,-5.535360261303069046e-02 ,-1.520857413308695019e-02, 9.742034418953970584e-03 ,3.782134450264275360e-02 ,-5.616176221425803194e-02, 1.000000000000000056e-01, 5.589523064884118320e+00 ,2.713186969175815921e+00, 2.140537338998368622e+00, -9.587407273654786621e-01]
    praw = [1.197570493393642188e+00,3.242570437960407698e-02,-4.921373061441208230e-01,-9.714270847895687799e-04,-4.832182650557964032e-01,-3.921395047320763161e-01,-4.916403309423041379e-01,-5.513749212592774195e-03,-6.693363907490857551e-03,-2.335929664888970313e-03,-3.219589828606695542e-03,-2.665429355429387286e-04,4.071761527555063653e-02,-1.543644016458786856e-03,0.000000000000000000e+00,6.195156756129924780e+00,2.680334884510975346e+00,2.993892797613240564e+00,-9.199935218170242068e-01]
    params = []
    for i in range(num_params):
        praw[i] = _cast_to_real(praw[i])
        params.append(torch.tensor(praw[i],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device))
 
    Q = LG(params[14])
    A1 = state_001()
    A1[0,0,0,0] = torch.cos(params[15]/2)
    A1[1,0,0,0] = torch.exp(-1j*params[16])*torch.sin(params[15]/2)
    A2 = state_001()
    A2[0,0,0,0] = torch.cos(params[17]/2)
    A2[1,0,0,0] = torch.exp(-1j*params[18])*torch.sin(params[17]/2)
 
    G1, G2, cs = G_kitaev_ani_c0(params[:14])
 
    G1A = torch.einsum('ijklm,jabc->ikalbmc', G1, A1).reshape(model.phys_dim, 2,2,2)
    G1A = torch.einsum('ijklm,jabc->ikalbmc', Q, G1A).reshape(model.phys_dim, 4,4,4)
    G2A = torch.einsum('ijklm,jabc->ikalbmc', G2, A2).reshape(model.phys_dim, 2,2,2)
    G2A = torch.einsum('ijklm,jabc->ikalbmc', Q, G2A).reshape(model.phys_dim, 4,4,4)
    A_= torch.einsum('iklm,akde->ialmde', G1A, G2A).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)
    A_= A_/A_.norm()
    A_ = A_.cuda(cfg.global_args.device)
    sites={(0,0): A_}
    if args.tiling in ["2SITE"]:
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, vy)
        # B1 = A1
        # B2 = A2
        # G1_, G2_, cs = G_kitaev_ani_c0(params[:14])
        # G1B = torch.einsum('ijklm,jabc->ikalbmc', G1_, B1).reshape(model.phys_dim, 2,2,2)
        # G1B = torch.einsum('ijklm,jabc->ikalbmc', Q, G1B).reshape(model.phys_dim, 4,4,4)
        # G2B = torch.einsum('ijklm,jabc->ikalbmc', G2_, B2).reshape(model.phys_dim, 2,2,2)
        # G2B = torch.einsum('ijklm,jabc->ikalbmc', Q, G2B).reshape(model.phys_dim, 4,4,4)
        # B_= torch.einsum('iklm,akde->ialmde', G1B, G2B).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)
        # B_= B_/B_.norm()
        # B_ = B_.cuda(cfg.global_args.device)
        sites[(1,0)]= A_
    else:
        def lattice_to_site(coord):
            return (0,0)
    state_sym = IPEPS(sites,vertexToSite=lattice_to_site)
    ctm_env = ENV(args.chi, state_sym)
    init_env(state_sym, ctm_env)
    ctm_env, *ctm_log = ctmrg.run(state_sym, ctm_env, conv_check=ctmrg_conv_rdm2x1)
 
    css = np.loadtxt(csinput)
    # css = [[0.000000000000000000e+00,1.247583434455966422e+00,4.314883583498645386e-04,-5.110121832191995006e-01,-1.426260985942023494e-03,-4.801889387355199323e-01,-4.181675450647031611e-01,-5.130897004226329106e-01,-1.011528429267444544e-03,1.000000000000000048e-04,8.546385031974417998e-03,1.000000000000000048e-04,2.567400052651009484e-03,1.000000000000000048e-04,2.048893400961039526e-03,0.000000000000000000e+00,6.276061356540131975e+00,2.683293995338117188e+00,3.125123284355635800e+00,-9.193463199351623594e-01]]
    css = np.asarray(css)
    for i in range(css.shape[0]):
        h = css[i,0]
        praw = css[i,1:]
 
        params = []
        for j in range(num_params):
            praw[j] = _cast_to_real(praw[j])
            params.append(torch.tensor(praw[j],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device))
 
        def loss_fn(state, ctm_env_in, params):
            # ctm_args= opt_context["ctm_args"]
            # opt_args= opt_context["opt_args"]
            # for i in range(13):
            #     params[i].requires_grad_(False)
            Q = LG(params[14])
            A1 = state_111()
            A1[0,0,0,0] = torch.cos(params[15]/2)
            A1[1,0,0,0] = torch.exp(-1j*params[16])*torch.sin(params[15]/2)
            A2 = state_m1m1m1()
            A2[0,0,0,0] = torch.cos(params[17]/2)
            A2[1,0,0,0] = torch.exp(-1j*params[18])*torch.sin(params[17]/2)
 
            G1, G2, cs = G_kitaev_ani_c0(params[:14])
            G1A = torch.einsum('ijklm,jabc->ikalbmc', G1, A1).reshape(model.phys_dim, 2,2,2)
            G1A = torch.einsum('ijklm,jabc->ikalbmc', Q, G1A).reshape(model.phys_dim, 4,4,4)
            G2A = torch.einsum('ijklm,jabc->ikalbmc', G2, A2).reshape(model.phys_dim, 2,2,2)
            G2A = torch.einsum('ijklm,jabc->ikalbmc', Q, G2A).reshape(model.phys_dim, 4,4,4)
            A_= torch.einsum('iklm,akde->ialmde', G1A, G2A).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)
            A_= A_/A_.norm()
            A_ = A_.cuda(cfg.global_args.device)
            sites={(0,0): A_}
 
            if args.tiling in ["2SITE"]:
                def lattice_to_site(coord): # This is stripe wrt SITEs
                    vx = (coord[0] + abs(coord[0]) * 2) % 2
                    vy = (coord[1] + abs(coord[1]) * 1) % 1
                    return (vx, vy)
                # B1 = A1
                # B2 = A2
                # G1_, G2_, cs = G_kitaev_ani_c0(params[:14])
                # G1B = torch.einsum('ijklm,jabc->ikalbmc', G1_, B1).reshape(model.phys_dim, 2,2,2)
                # G1B = torch.einsum('ijklm,jabc->ikalbmc', Q, G1B).reshape(model.phys_dim, 4,4,4)
                # G2B = torch.einsum('ijklm,jabc->ikalbmc', G2_, B2).reshape(model.phys_dim, 2,2,2)
                # G2B = torch.einsum('ijklm,jabc->ikalbmc', Q, G2B).reshape(model.phys_dim, 4,4,4)
                # B_= torch.einsum('iklm,akde->ialmde', G1B, G2B).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)
                # B_= B_/B_.norm()
                # B_ = B_.cuda(cfg.global_args.device)
                sites[(1,0)]= A_
            else:
                def lattice_to_site(coord):
                    return (0,0)
 
            state_sym = IPEPS(sites,vertexToSite=lattice_to_site)
            ctm_env = ENV(args.chi, state_sym)
            if cfg.opt_args.opt_ctm_reinit:
                init_env(state_sym, ctm_env_in)
            ctm_env_out, *ctm_log = ctmrg.run(state_sym, ctm_env_in, conv_check=ctmrg_conv_rdm2x1)
            loss = energy_f(state_sym, ctm_env_out, kx, ky, kz, h)
            model.get_m()
            model.get_Wp()
            print("Energy = ", loss.item())
            # for p in cs:
            #     print(p.item().real,end= " ,")
            # print("\n==============")
            # for p in params[13:]:
            #     print(p.item().real,end= " ,")
            # get_XYZ(A1, A2)
            # print(" ")
            return loss
        
        @torch.no_grad()
        def save(state, ctm_env_in, params):
            print("Start saving ...")
            Es = model.get_E() # [ExK EyK EzK ExG EyG EzG, total]
            ms = model.get_m()
            Qxx = model.get_Qxx()
            W = model.get_Wp()
            print("Qxx = ", Qxx)
            print("Mags = ", ms)
            print("E_total = ", Es[-1])
 
            plist = []
            G1, G2, cs = G_kitaev_ani_c0(params[:14])
            for p in cs:
                plist.append(p.clone().detach().item().real)
            for p in params[14:]:
                plist.append(p.clone().detach().item().real)
            print("params : ", plist)
 
            tmp = np.asarray(Es);  tmp = np.insert(tmp, 0, h); 
            with open(Efn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()
            tmp = np.asarray(ms); tmp = np.insert(tmp, -1, Qxx); tmp = np.insert(tmp, 0, W);  tmp = np.insert(tmp, 0, h)
            with open(magfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()
            tmp = np.asarray(plist); tmp = np.insert(tmp, 0, h)
            with open(csfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close() 
            return 0
 
        # print(params)
        # optimize_state(state_sym, ctm_env, loss_fn, params)
 
 
        LGps = np.linspace(0,1,20)
        bestE = 999999
        bestparams = []
        Es = []
        for p in LGps:
            print("p = ", p)
            params[14] = torch.tensor(p,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
            E = loss_fn(state_sym, ctm_env, params)
            if E<bestE:
                bestE = E.detach().cpu().clone().item()
                bestparams = []
                for param in params:
                    bestparams.append(param.detach().cpu().clone())
            Es.append(E.detach().cpu().clone().item())
        print(Es)
        E = loss_fn(state_sym, ctm_env, bestparams)
        save(state_sym, ctm_env, bestparams)
 
 
    return 0
 
if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()