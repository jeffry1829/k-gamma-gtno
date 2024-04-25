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
Efn = f"datas/kg_E.txt"
magfn = f"datas/kg_mag.txt"
csfn = f"datas/kg_cs.txt"
# -------- filenames --------

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # model = ising.ISING_C4V(hx=args.hx, q=args.q)
    num_params = 15 # 7 for GTNO 8=2*4 for states
    # num_params = 7 # 7 for GTNO 8=2*4 for states
    # num_params = 7 # 7 for GTNO 8=2*4 for states
    # model = gamma.GAMMA(hx=args.hx, q=args.q)
    model = k_gamma.K_GAMMA(hx=args.hx, q=args.q)
    # energy_f= model.energy_1x1_nn if 俺!args.q==0 else model.energy_1x1_plaqette
    if args.tiling == "2SITE":
        energy_f = model.energy_2x2_2site
        # eval_obs_f= model.eval_obs
    else:
        energy_f = model.energy_2x2
    # energy_f= model.energy_1x1

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_c4v(args.instate)
        #state.sites[(0,0)][0,0,0,0,0]+= torch.tensor(3.14159,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)#
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
        
        filename1 = args.params_in+'.txt'
        with open(filename1, 'r') as file_to_read2:
            params = []
            lines2 = file_to_read2.readline()
            q_tmp = [float(k) for k in lines2.replace('i', 'j').split()]
            for i in range(num_params):
                params.append(torch.as_tensor(q_tmp[i],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device))
##        noise =0.1
##        for i in range(9):
##            rand_t = torch.rand(1, dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
##            params[i]+= noise * rand_t[0]
    elif args.opt_resume is not None:
        state= IPEPS_C4V()
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        # pat = torch.tensor([1,1,0,0,0,0,0],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        # for i in range(num_params):
        #     params.append(torch.tensor(pat[i],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device))

            # params.append(torch.rand(219,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)[0]+
            # 1j*torch.rand(11,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)[0])
        # print(params)
            # params.append(torch.rand(1,dtype=torch.float,device=cfg.global_args.device)[0].type(cfg.global_args.torch_dtype))

        params = []
        # Energy =  -0.33972333943809935
        # (0.38768404226788467+1.4626739821320736j),(-0.609870765707362+0.8736707234718116j),(-0.2763050900726205+0.05330711383988729j),(0.34612571801730496+1.0342902910048568j),(0.06761978664033545+0.5845124335220583j),(0.03501584416507501+1.6083114275812709j),(-0.1470634370572521+0.6049450450822822j),(0.5822032253421052+0.6059836590922572j),(0.37023310203692184+1.454780176072319j),(-0.16652324658468304+0.3818705934641633j),(-1.2873468978781009+0.8443546145961454j),(0.4318071796127747+0.877608339626112j),(-0.17520863020533226+1.6365678001682875j),(0.04736081165259422+0.3360242783847406j),(-0.5351397173024228+1.2456911108620745j), 
        # praw = [0.3298+0.0057j, -0.7930-0.0089j, -0.0833-0.0040j, 0.3409+0.0049j, 0.0113+0.0014j, -0.1157-0.0006j, -0.0110-0.0003j, -0.0180-0.0805j, -0.0246+0.0409j, -0.5349+0.8605j, 0.7144-0.1709j, -0.3380+0.4929j, -1.0177+0.1390j, -0.2417+0.7808j, -0.9978+0.5322j]
        # praw = [np.cos(0.146), np.sin(0.146), 0, 0, 0, 0, 0, -0.8881+0.0000j,  -0.3251-0.3251j, -0.8881+0.0000j,  -0.3251-0.3251j, -0.8881+0.0000j,  -0.3251-0.3251j, -0.8881+0.0000j,  -0.3251-0.3251j]
        praw = [np.cos(0.146), np.sin(0.146), 0.1, 0.1, 0.1, 0.1, 0.1, 2.3564, 5.3280, 2.3564, 5.3280, 2.3564, 5.3280, 2.3564, 5.3280]
        
        ## ENSURE GTNO params are real
        for i in range(num_params):
            praw[i] = _cast_to_real(praw[i])
            params.append(torch.tensor(praw[i],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).real)

        print(params)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    # print(state)
    
    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = energy_f(state, env)
        history.append(e_curr.item())
        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history), "history": history})
            return True, history
        return False, history

    @torch.no_grad()
    def ctmrg_conv_rdm2x1(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        # rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
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

    # G1,cs = G1_gamma(params) #(s,s,vX,vY,vZ) contain all params
    # G2 = G2_gamma() #(s,s,vX,vY,vZ) constant, params on G1
    # kcs = [torch.cos(params[0]),torch.sin(params[0]),0,0,0,0,0]
    # G1, G2, cs = G_gamma_systematic(kcs[:7])

    G1, G2, cs = G_gamma_systematic(params[:7])

    # Q = LG([])
    # print("fjewoidew : ", (G-G_).norm())
    # checkC6U6(G)
    # exit()
    
    A1 = state_111()
    A1[0,0,0,0] = torch.cos(params[-1]/2)
    A1[1,0,0,0] = torch.exp(-1j*params[-2])*torch.sin(params[-1]/2)
    A2 = state_111()
    A2[0,0,0,0] = torch.cos(params[-3]/2)
    A2[1,0,0,0] = torch.exp(-1j*params[-4])*torch.sin(params[-3]/2)

    G1A = torch.einsum('ijklm,jabc->ikalbmc', G1, A1).reshape(model.phys_dim, 4, 4, 4)
    G2A = torch.einsum('ijklm,jabc->ikalbmc', G2, A2).reshape(model.phys_dim, 4, 4, 4)
    # A_= torch.einsum('iklm,akde->ialmde', G1A, G2A).reshape(model.phys_dim*model.phys_dim, 3, 3, 3, 3)
    A_= torch.einsum('iklm,akde->ialmde', G1A, G2A).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)

    A_= A_/A_.norm()
    sites={(0,0): A_}

    if args.tiling in ["2SITE"]:
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, vy)

        # kcs = [torch.cos(params[0]),torch.sin(params[0]),0,0,0,0,0]
        # G1_, G2_, cs = G_gamma_systematic(kcs[:7])
        G1_, G2_, cs = G_gamma_systematic(params[:7])

        # B1 = state_111()
        # B2 = state_111()

        B1 = state_111()
        B1[0,0,0,0] = torch.cos(params[-5]/2)
        B1[1,0,0,0] = torch.exp(-1j*params[-6])*torch.sin(params[-5]/2)
        B2 = state_111()
        B2[0,0,0,0] = torch.cos(params[-7]/2)
        B2[1,0,0,0] = torch.exp(-1j*params[-8])*torch.sin(params[-7]/2)

        G1B = torch.einsum('ijklm,jabc->ikalbmc', G1_, B1).reshape(model.phys_dim, 4, 4, 4)
        G2B = torch.einsum('ijklm,jabc->ikalbmc', G2_, B2).reshape(model.phys_dim, 4, 4, 4)
        # A_= torch.einsum('iklm,akde->ialmde', G1A, G2A).reshape(model.phys_dim*model.phys_dim, 3, 3, 3, 3)
        B_= torch.einsum('iklm,akde->ialmde', G1B, G2B).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)

        B_= B_/B_.norm()
        
        sites[(1,0)]= B_
    else:
        def lattice_to_site(coord):
            return (0,0)

    st = time.time()
    state_sym = IPEPS(sites,vertexToSite=lattice_to_site)
    end = time.time()
    print("ipeps init time : ", end-st)
    st = time.time()
    ctm_env = ENV(args.chi, state_sym)
    end = time.time()
    print("env init time : ", end-st)
    st = time.time()
    init_env(state_sym, ctm_env)
    end = time.time()    
    print("env init2 time : ", end-st)  
    st = time.time()  
    ctm_env, *ctm_log = ctmrg.run(state_sym, ctm_env, conv_check=ctmrg_conv_rdm2x1)
    end = time.time()
    print("ctmrg run time : ", end-st) 
    st = time.time() 
    loss = energy_f(state_sym, ctm_env, phi=0)
    print(model.get_m())
    
        # np.cos(phi) * modelK.energy_2x2_2site + np.sin(phi)*modelG.energy_2x2_2site
    end = time.time()
    print("energy eval time : ", end-st) 
    print("loss = ", loss.item())

    # exit()

    phis = np.pi*np.linspace(0.025, 0.5, 20).flip()

    for phi in phis:
        print("===================")
        print("phi = ", phi)

        def loss_fn(state, ctm_env_in, params, opt_context):
            ctm_args= opt_context["ctm_args"]
            opt_args= opt_context["opt_args"]

            A1 = state_111()
            A1[0,0,0,0] = torch.cos(params[-1]/2)
            A1[1,0,0,0] = torch.exp(-1j*params[-2])*torch.sin(params[-1]/2)
            A2 = state_111()
            A2[0,0,0,0] = torch.cos(params[-3]/2)
            A2[1,0,0,0] = torch.exp(-1j*params[-4])*torch.sin(params[-3]/2)

            G1, G2, cs = G_gamma_systematic(params[:7])

            G1A = torch.einsum('ijklm,jabc->ikalbmc', G1, A1).reshape(model.phys_dim, 4, 4, 4)
            G2A = torch.einsum('ijklm,jabc->ikalbmc', G2, A2).reshape(model.phys_dim, 4, 4, 4)
            A_= torch.einsum('iklm,akde->ialmde', G1A, G2A).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)
            A_= A_/A_.norm()
            sites={(0,0): A_}

            if args.tiling in ["2SITE"]:
                def lattice_to_site(coord): # This is stripe wrt SITEs
                    vx = (coord[0] + abs(coord[0]) * 2) % 2
                    vy = (coord[1] + abs(coord[1]) * 1) % 1
                    return (vx, vy)
                
        
                G1_, G2_, cs = G_gamma_systematic(params[:7])

                B1 = state_111()
                B1[0,0,0,0] = torch.cos(params[-5]/2)
                B1[1,0,0,0] = torch.exp(-1j*params[-6])*torch.sin(params[-5]/2)
                B2 = state_111()
                B2[0,0,0,0] = torch.cos(params[-7]/2)
                B2[1,0,0,0] = torch.exp(-1j*params[-8])*torch.sin(params[-7]/2)

                G1B = torch.einsum('ijklm,jabc->ikalbmc', G1_, B1).reshape(model.phys_dim, 4, 4, 4)
                G2B = torch.einsum('ijklm,jabc->ikalbmc', G2_, B2).reshape(model.phys_dim, 4, 4, 4)
                # A_= torch.einsum('iklm,akde->ialmde', G1A, G2A).reshape(model.phys_dim*model.phys_dim, 3, 3, 3, 3)
                B_= torch.einsum('iklm,akde->ialmde', G1B, G2B).reshape(model.phys_dim*model.phys_dim, 4, 4, 4, 4)

                B_= B_/B_.norm()
                
                sites[(1,0)]= B_
            else:
                def lattice_to_site(coord):
                    return (0,0)

            state_sym = IPEPS(sites,vertexToSite=lattice_to_site)
            ctm_env = ENV(args.chi, state_sym)
            # init_env(state_sym, ctm_env)
            # ctm_env, *ctm_log = ctmrg.run(state_sym, ctm_env, conv_check=ctmrg_conv_energy)\
            # possibly re-initialize the environment
            if cfg.opt_args.opt_ctm_reinit:
                init_env(state_sym, ctm_env_in)

            # # 1) compute environment by CTMRG
            ctm_env_out, *ctm_log = ctmrg.run(state_sym, ctm_env_in, conv_check=ctmrg_conv_rdm2x1)
            loss = energy_f(state_sym, ctm_env_out, phi = phi)
            model.get_m()
            print("Energy = ", loss.item())
            for p in params:
                print(p.item(),end= ",")
            print(" ")
            return (loss, ctm_env_out, *ctm_log)

        @torch.no_grad()
        def save(state, ctm_env_in, params):
            print("Start saving ...")
            # ctm_args= opt_context["ctm_args"]
            # opt_args= opt_context["opt_args"]
            Es = model.get_E() # [ExK EyK EzK ExG EyG EzG, total]
            ms = model.get_m()

            print("Mags = ", ms)
            print("E_total = ", Es[-1])

            plist = []
            for p in params:
                plist.append(p.clone().detach().item().real)
            print("params : ", plist)

            tmp = np.asarray(Es);  tmp = np.insert(tmp, 0, phi); 
            with open(Efn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()
            tmp = np.asarray(ms); tmp = np.insert(tmp, 0, phi);  
            with open(magfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close()
            tmp = np.asarray(plist); tmp = np.insert(tmp, 0, phi)
            with open(csfn, "a") as f:
                np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
            f.close() 
            return 0

        optimize_state(state_sym, ctm_env, loss_fn, params)
        save(state_sym, ctm_env, params)
        print("===================")
if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()