#python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
import context
import time
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic.rdm import *
from ctm.generic import ctmrg_ex
from ctm.generic.ctm_projectors import *
from Norm_ori import *
from Hami_ori import *
from Stat_ori import *
from Test import *
from models import j1j2
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
log = logging.getLogger(__name__)

tStart = time.time()

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--size", type=int, default=10, help="effective size")
args, unknown_args = parser.parse_known_args()

cfg.configure(args)
#cfg.print_config()
torch.set_num_threads(args.omp_cores)
torch.manual_seed(args.seed)

model = j1j2.J1J2(j1=args.j1, j2=args.j2)

state = read_ipeps(args.instate, vertexToSite=None)
def symmetrize(state):
    A= state.site((0,0))
    A_symm= make_c4v_symm_A1(A)
    symm_state= IPEPS({(0,0): A_symm}, vertexToSite=state.vertexToSite)
    return symm_state
state= symmetrize(state)
sitesDL=dict()
for coord,A in state.sites.items():
    dimsA = A.size()
    a = contiguous(einsum('mefgh,mabcd->eafbgchd',A,conj(A)))
    a = view(a, (dimsA[1]**2,dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    sitesDL[coord]=a
stateDL = IPEPS(sitesDL,state.vertexToSite)

def ctmrg_conv_energy(state2, env, history, ctm_args=cfg.ctm_args):
    if not history:
        history=[]

    if (len(history)>0):
        old = history[:4*env.chi]
    new = []
    u,s,v = torch.svd(env.C[((0,0),(-1,-1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u,s,v = torch.svd(env.C[((0,0),(1,-1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u,s,v = torch.svd(env.C[((0,0),(1,-1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u,s,v = torch.svd(env.C[((0,0),(1,1))])
    for i in range(env.chi):
        new.append(s[i].item())

    diff = 0.
    if (len(history)>0):
        for i in range(4*env.chi):
            history[i] = new[i]
            if (abs(old[i]-new[i])>diff):
                diff = abs(old[i]-new[i])
    else:
        for i in range(4*env.chi):
            history.append(new[i])
    history.append(diff)

    if (len(history[4*env.chi:]) > 1 and diff < ctm_args.ctm_conv_tol)\
        or len(history[4*env.chi:]) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(history[4*env.chi:]), "history": history[4*env.chi:]})
        #print (len(history[4*env.chi:]))
        return True, history
    return False, history

env = ENV(args.chi, state)
init_env(state, env)

env, P, Pt, *ctm_log = ctmrg_ex.run(state, env, conv_check=ctmrg_conv_energy)
print ("E_per_site=", model.energy_2x2_1site_BP(state, env).item())

##trans_m = torch.einsum('ijk,jlmn,mab->ilaknb',env.T[((0,0),(0,-1))],stateDL.sites[(0,0)],env.T[((0,0),(0,1))])
##trans_m = trans_m.reshape(((args.chi*args.bond_dim)**2,(args.chi*args.bond_dim)**2))
##trans_m2 = trans_m.detach().cpu().numpy()
##e, v = np.linalg.eig(trans_m2)
##idx = np.argsort(e.real)   
##e = e[idx]
##v = v[:,idx]
##print ("correlation_length=", -1/np.log(e[(args.chi*args.bond_dim)**2-2]/e[(args.chi*args.bond_dim)**2-1]).item().real)
################Hamiltonian################
torch.pi = torch.tensor(np.pi, dtype=torch.complex128)
kx_int = 24
ky_int = 0
kx = kx_int*torch.pi/24.
ky = ky_int*torch.pi/24.
print ("kx=", kx/torch.pi*(2*args.size+2))
print ("ky=", ky/torch.pi*(2*args.size+2))
SS = model.SS_rot
iden= torch.eye(2,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).contiguous()
H_temp = args.j1 * SS
lam = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).requires_grad_(True)
lamb = torch.tensor(1.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
iden2 = torch.einsum('ij,kl->ikjl',iden,iden)
H = iden2 + lam * H_temp
H2 = iden2
rdm2x1= rdm2x1((0,0),state,env)
energy_per_site= torch.einsum('ijkl,ijkl',rdm2x1,H_temp)
print ("E_per_bond=", 2*energy_per_site.item().real)
bond_dim = args.bond_dim

################Static structure factor###############
s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
rot_op = s2.BP_rot()
iden= torch.eye(2,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).contiguous()
S = s2.SP()
S_rot = torch.einsum('ki,kc,ca->ia',rot_op,S,rot_op)
Sm = s2.SM()
Sm_rot = torch.einsum('ki,kc,ca->ia',rot_op,Sm,rot_op)
Sx = (S+Sm)/2.
Sx_rot = torch.einsum('ki,kc,ca->ia',rot_op,Sx,rot_op)
Sy = (S-Sm)/2./1j
Sy_rot = torch.einsum('ki,kc,ca->ia',rot_op,Sy,rot_op)
Sz = s2.SZ()
Sz_rot = torch.einsum('ki,kc,ca->ia',rot_op,Sz,rot_op)
torch.autograd.set_detect_anomaly(True)

B_grad = torch.zeros((len(state.sites), model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
                     dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
if len(state.sites)==1:
    sitesB = {(0,0): B_grad[0]}
    stateB = IPEPS(sitesB, state.vertexToSite)

with torch.no_grad():
    P, Pt = Create_Projectors(state, stateDL, env, args)
    
if len(state.sites)==1:
    ###new P Pt?###
    
    lamb = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Norm_Env(state, stateDL, stateB, env, P, Pt, lamb, kx, ky, args)
    Norm, Norm2 = Create_Norm(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args)
    norm_factor = contract(Norm2[(0,0)],conj(state.site((0,0))),([0,1,2,3,4],[0,1,2,3,4]))

    tmp_rdm= rdm.rdm1x1((0,0),state,env)
    sx_exp = torch.trace(tmp_rdm@Sx)
    sx_rot_exp = torch.trace(tmp_rdm@Sx_rot)
    sy_exp = torch.trace(tmp_rdm@Sy)
    sy_rot_exp = torch.trace(tmp_rdm@Sy_rot)
    sz_exp = torch.trace(tmp_rdm@Sz)
    sz_rot_exp = torch.trace(tmp_rdm@Sz_rot)
    #print (sx_exp,sx_rot_exp,sy_exp,sy_rot_exp,sz_exp,sz_rot_exp)

    lam = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    lamb = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).requires_grad_(True)
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(state, stateDL, stateB, env, P, Pt, lam, lamb, iden, Sx, Sx_rot, Sx, sx_exp, sx_rot_exp, 1.0, kx, ky, args)
    Hami, Hami2 = Create_Stat(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args)
    Ham = contract(Hami[(0,0)],conj(state.site((0,0))),([0,1,2,3,4],[0,1,2,3,4]))
    #sx_exp2 = Ham.detach()
    #print (sx_exp2/norm_factor)
    #print (sx_exp*iden)
    Ham.backward()
    stat_x = lamb.grad

    lam = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    lamb = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).requires_grad_(True)
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(state, stateDL, stateB, env, P, Pt, lam, lamb, iden, Sy, Sy_rot, Sy, sy_exp, sy_rot_exp, 1.0, kx, ky, args)
    Hami, Hami2 = Create_Stat(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args)
    Ham = contract(Hami[(0,0)],conj(state.site((0,0))),([0,1,2,3,4],[0,1,2,3,4]))
    
    Ham.backward()
    stat_y = lamb.grad

    lam = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    lamb = torch.tensor(0.0,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device).requires_grad_(True)
    C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right = Create_Stat_Env(state, stateDL, stateB, env, P, Pt, lam, lamb, iden, Sz, Sz_rot, Sz, sz_exp, sz_rot_exp, 1.0, kx, ky, args)
    Hami, Hami2 = Create_Stat(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args)
    Ham = contract(Hami[(0,0)],conj(state.site((0,0))),([0,1,2,3,4],[0,1,2,3,4]))
    
    Ham.backward()
    stat_z = lamb.grad

##    stat_x_r = ((stat_x)/(2*args.size+2)/(2*args.size+2)).item().real
##    stat_x_i = ((stat_x)/(2*args.size+2)/(2*args.size+2)).item().imag
##    stat_y_r = ((stat_y)/(2*args.size+2)/(2*args.size+2)).item().real
##    stat_y_i = ((stat_y)/(2*args.size+2)/(2*args.size+2)).item().imag
##    stat_z_r = ((stat_z)/(2*args.size+2)/(2*args.size+2)).item().real
##    stat_z_i = ((stat_z)/(2*args.size+2)/(2*args.size+2)).item().imag


    print ("Static_structure_factor=",
           ((stat_x)/norm_factor/2).item().real + ((stat_y)/norm_factor/2).item().real + ((stat_z)/norm_factor/2).item().real)
    print (((stat_x)/norm_factor/2).item().real, ((stat_y)/norm_factor/2).item().real, ((stat_z)/norm_factor/2).item().real)
##    print ("Static_structure_factor=",
##           np.sqrt(stat_x_r**2 + stat_x_i**2) + np.sqrt(stat_y_r**2 + stat_y_i**2) + np.sqrt(stat_z_r**2 + stat_z_i**2))



tEnd = time.time()
print ("time_ener=", tEnd - tStart)
