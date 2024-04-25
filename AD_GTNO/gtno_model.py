#python optim_ising_c4v.py --bond_dim 1 --chi 16 --seed 1234 --hx 3.1 --CTMARGS_fwd_checkpoint_move --OPTARGS_tolerance_grad 1.0e-8 --out_prefix ex-hx31D2 --self.params_out self.paramsh31D2 --opt_max_iter 1000 --instate ex-hx31D2_state.json --self.params_in self.paramsh31D2
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
from optim.ad_optim_lbfgs_mod import optimize_state

from GTNOs import *
import unittest
import logging
import json

# 引入 time 模組
import time
import numpy as np


def _cast_to_real(t):
    return t.real
        
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

class gtno_model:
    def __init__(self, model, chi):
        self.model = model
        self.ctm_env = None
        self.state = None
        self.chi = chi
        self.reqgrad = []

        
    def set_params(self, praw):
        params = []
        for i in range(len(praw)):
            praw[i] = _cast_to_real(praw[i])
            params.append(torch.tensor(praw[i],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device))
        self.params = params
        
    def set_reqgrad(self, reqgrad):
        self.reqgrad = reqgrad


    def initialize(self): 
        Q = LG(self.params[13])
        A1 = state_001()
        A1[0,0,0,0] = torch.cos(self.params[14]/2)
        A1[1,0,0,0] = torch.exp(-1j*self.params[15])*torch.sin(self.params[14]/2)
        A2 = state_001()
        A2[0,0,0,0] = torch.cos(self.params[16]/2)
        A2[1,0,0,0] = torch.exp(-1j*self.params[17])*torch.sin(self.params[16]/2)
        G1, G2, cs = G_kitaev_ani(self.params[:13])
        G1A = torch.einsum('ijklm,jabc->ikalbmc', G1, A1).reshape(2, 2,2,2)
        G1A = torch.einsum('ijklm,jabc->ikalbmc', Q, G1A).reshape(2, 4,4,4)
        G2A = torch.einsum('ijklm,jabc->ikalbmc', G2, A2).reshape(2, 2,2,2)
        G2A = torch.einsum('ijklm,jabc->ikalbmc', Q, G2A).reshape(2, 4,4,4)
        A_= torch.einsum('iklm,akde->ialmde', G1A, G2A).reshape(2*2, 4, 4, 4, 4)
        A_= A_/A_.norm()
        A_ = A_.cuda(cfg.global_args.device)
        sites={(0,0): A_}
        # if args.tiling in ["2SITE"]:
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, vy)
        B1 = A1
        B2 = A2
        G1_, G2_ = G1 , G2
        G1B = torch.einsum('ijklm,jabc->ikalbmc', G1_, B1).reshape(2, 2,2,2)
        G1B = torch.einsum('ijklm,jabc->ikalbmc', Q, G1B).reshape(2, 4,4,4)
        G2B = torch.einsum('ijklm,jabc->ikalbmc', G2_, B2).reshape(2, 2,2,2)
        G2B = torch.einsum('ijklm,jabc->ikalbmc', Q, G2B).reshape(2, 4,4,4)
        B_= torch.einsum('iklm,akde->ialmde', G1B, G2B).reshape(2*2, 4, 4, 4, 4)
        B_= B_/B_.norm()
        B_ = B_.cuda(cfg.global_args.device)
        sites[(1,0)]= B_
        # else:
            # def lattice_to_site(coord):
            #     return (0,0)
        state_sym = IPEPS(sites,vertexToSite=lattice_to_site)
        ctm_env = ENV(self.chi, state_sym)
        init_env(state_sym, ctm_env)
        ctm_env, *ctm_log = ctmrg.run(state_sym, ctm_env, conv_check=ctmrg_conv_rdm2x1)
        self.ctm_env = ctm_env
        self.state = state_sym


    def loss_fn(state, ctm_env_in, params, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        for i in range(len(self.reqgrad)):
            if self.reqgrad[i] == 0:
                params[i].require_grad_(False)
            else:
                params[i].require_grad_(True)
        
        Q = LG(params[13])
        A1 = state_111()
        A1[0,0,0,0] = torch.cos(params[14]/2)
        A1[1,0,0,0] = torch.exp(-1j*params[15])*torch.sin(params[14]/2)
        A2 = state_m1m1m1()
        A2[0,0,0,0] = torch.cos(params[16]/2)
        A2[1,0,0,0] = torch.exp(-1j*params[17])*torch.sin(params[16]/2)

        G1, G2, cs = G_kitaev_ani(params[:13])
        G1A = torch.einsum('ijklm,jabc->ikalbmc', G1, A1).reshape(2, 2,2,2)
        G1A = torch.einsum('ijklm,jabc->ikalbmc', Q, G1A).reshape(2, 4,4,4)
        G2A = torch.einsum('ijklm,jabc->ikalbmc', G2, A2).reshape(2, 2,2,2)
        G2A = torch.einsum('ijklm,jabc->ikalbmc', Q, G2A).reshape(2, 4,4,4)
        A_= torch.einsum('iklm,akde->ialmde', G1A, G2A).reshape(2*2, 4, 4, 4, 4)
        A_= A_/A_.norm()
        A_ = A_.cuda(cfg.global_args.device)
        sites={(0,0): A_}

        # if args.tiling in ["2SITE"]:
        def lattice_to_site(coord): # This is stripe wrt SITEs
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = (coord[1] + abs(coord[1]) * 1) % 1
            return (vx, vy)
        B1 = A1
        B2 = A2
        G1_, G2_ = G1 , G2
        # G1_, G2_, cs = G_kitaev_ani(params[:13])
        G1B = torch.einsum('ijklm,jabc->ikalbmc', G1_, B1).reshape(2, 2,2,2)
        G1B = torch.einsum('ijklm,jabc->ikalbmc', Q, G1B).reshape(2, 4,4,4)
        G2B = torch.einsum('ijklm,jabc->ikalbmc', G2_, B2).reshape(2, 2,2,2)
        G2B = torch.einsum('ijklm,jabc->ikalbmc', Q, G2B).reshape(2, 4,4,4)
        B_= torch.einsum('iklm,akde->ialmde', G1B, G2B).reshape(2*2, 4, 4, 4, 4)
        B_= B_/B_.norm()
        B_ = B_.cuda(cfg.global_args.device)
        sites[(1,0)]= B_
        # else:
        #     def lattice_to_site(coord):
        #         return (0,0)

        state_sym = IPEPS(sites,vertexToSite=lattice_to_site)
        ctm_env = ENV(self.chi, state_sym)
        if cfg.opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env_in)
        ctm_env_out, *ctm_log = ctmrg.run(state_sym, ctm_env_in, conv_check=ctmrg_conv_rdm2x1)
        loss = energy_f(state_sym, ctm_env_out, kx, ky, kz, h)
        model.get_m()
        print("Energy = ", loss.item())
        for p in cs:
            print(p.item().real,end= " ,")
        print("\n==============")
        for p in params[13:]:
            print(p.item().real,end= " ,")
        print(" ")
        return (loss, ctm_env_out, *ctm_log)    

    @torch.no_grad()
    def save_obs(state, ctm_env_in, params):
        print("Start saving ...")
        Es = model.get_E() # [ExK EyK EzK ExG EyG EzG, total]
        ms = model.get_m()
        Qxx = model.get_Qxx()

        print("Mags = ", ms)
        print("E_total = ", Es[-1])

        plist = []
        G1, G2, cs = G_kitaev_ani(self.params[:13])
        for p in cs:
            plist.append(p.clone().detach().item().real)
        for p in self.params[13:]:
            plist.append(p.clone().detach().item().real)
        print("self.params : ", plist)

        tmp = np.asarray(Es);  tmp = np.insert(tmp, 0, h); 
        with open(Efn, "a") as f:
            np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
        f.close()
        tmp = np.asarray(ms); tmp = np.insert(tmp, -1, Qxx);  tmp = np.insert(tmp, 0, h);  
        with open(magfn, "a") as f:
            np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
        f.close()
        tmp = np.asarray(plist); tmp = np.insert(tmp, 0, h)
        with open(csfn, "a") as f:
            np.savetxt(f, tmp.reshape(1, tmp.shape[0]))
        f.close() 
        return 0

    def evaluate(self):
        loss_fn(self.state, self.ctm_env)

    def optimize(self):
        optimize_state(self.state, self.ctm_env, self.loss_fn, self.params)

    def save(self):
        save_obs(self.state, self.ctm_env, self.params)