#python optim_ising_c4v.py --bond_dim 1 --chi 16 --seed 1234 --hx 3.1 --CTMARGS_fwd_checkpoint_move --OPTARGS_tolerance_grad 1.0e-8 --out_prefix ex-hx31D2 --params_out paramsh31D2 --opt_max_iter 1000 --instate ex-hx31D2_state.json --params_in paramsh31D2
import context
import torch
import argparse
import config as cfg

from ctm.generic.env import *
from ctm.generic import ctmrg
from ipeps.ipeps import *
from ctm.generic import rdm
from models import kitaev
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
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--hx", type=float, default=0., help="transverse field")
parser.add_argument("--q", type=float, default=0, help="next nearest-neighbour coupling")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    # model = ising.ISING_C4V(hx=args.hx, q=args.q)
    num_params = 1
    model = kitaev.KITAEV(hx=args.hx, q=args.q)

    # energy_f= model.energy_1x1_nn if 俺!args.q==0 else model.energy_1x1_plaqette
    energy_f= model.energy_2x2
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
        # A = torch.rand((1, bond_dim, bond_dim, bond_dim, bond_dim),\
        #     dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
        # # state = IPEPS_C4V(A)

        # sites={(0,0): A}
        # def vertexToSite(coord):
        #     return (0,0)
        # state = IPEPS(sites,vertexToSite)

        # params = [torch.tensor(0.728968627421,dtype=torch.float,device=cfg.global_args.device).type(cfg.global_args.torch_dtype),
        #          torch.tensor(0.684547105929,dtype=torch.float,device=cfg.global_args.device).type(cfg.global_args.torch_dtype)]

        params = [torch.tensor(0.24*3.14,dtype=torch.float,device=cfg.global_args.device).type(cfg.global_args.torch_dtype)]

        # params = []
        # for i in range(num_params):
            # params.append(torch.rand(1,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)[0])
            # params.append(torch.rand(1,dtype=torch.float,device=cfg.global_args.device)[0].type(cfg.global_args.torch_dtype))
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


    def ctmrg_conv_rdm2x2(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        # rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
        rdm2x2 = rdm.rdm2x2((0,0), state, env)

        dist= float('inf')
        if len(history["log"]) > 0:
            dist= torch.dist(rdm2x2, history["rdm"], p=2).item()
        history["rdm"]=rdm2x2
        history["log"].append(dist)
        if dist<ctm_args.ctm_conv_tol or len(history["log"]) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log']})
            return True, history
        return False, history
    
    bond_dim = args.bond_dim
    GTNO, cs = G_kitaev(params) #(s,s,vX,vY,vZ)
    Q = LG([])

    # double_A = torch.rand((model.phys_dim*model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
    #             dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    # single_A = torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim),\
    #             dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    # single_A[0,0,0,0] = torch.tensor((2/(2**0.5)+1))
    # single_A[1,0,0,0] = torch.tensor((1+1j)/(2**0.5))
    # print("Norm : ", (Q-create_loop_gas_operator("1/2")).norm())

    single_A = prepare_magnetized_state()
    AQ = torch.einsum('ijklm,jabc->ikalbmc', Q, single_A).reshape(model.phys_dim, 2, 2, 2)
    A = torch.einsum('iabc,jmnc->ijabmn', AQ, AQ).reshape(model.phys_dim**2, 2, 2, 2, 2)

    # from itertools import permutations 
    # perm = permutations([1, 2, 3, 4]) 
    # for i in list(perm): 
    #     print((0,)+i)
    #     A_ = torch.permute(A, (0,)+i)
    #     A_= A_/A_.norm()
    #     # state_sym = IPEPS_C4V(A)
    #     sites={(0,0): A_}
    #     def vertexToSite(coord):
    #         return (0,0)
    #     state_sym = IPEPS(sites,vertexToSite)
    #     # print(state_sym)
    #     # state_sym = IPEPS(A)
    #     # ctm_env = ENV_C4V(args.chi, state_sym)
    #     ctm_env = ENV(args.chi, state_sym)
    #     init_env(state_sym, ctm_env)

    #     ctm_env, *ctm_log = ctmrg_c4v.run(state_sym, ctm_env, conv_check=ctmrg_conv_rdm2x1)
    #     # ctm_env, *ctm_log = ctmrg.run(state_sym, ctm_env, conv_check=ctmrg_conv_energy)

    #     loss = energy_f(state_sym, ctm_env)
    
    #     # obs_values, obs_labels= model.eval_obs(state_sym,ctm_env)
    # # obs_labels = []
    # # obs_values = []
    # print(", ".join(["epoch","energy"]+obs_labels))
    # print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))
    # exit()

    phis = torch.linspace(0, 0.5*np.pi, 20)
    print(phis)
    A = torch.permute(A, (0,2,3,4,1))
    energys = []
    from torch import tensor
    # energys_ = [tensor(-1.3080, dtype=torch.float64), tensor(-1.3263, dtype=torch.float64), tensor(-1.3741, dtype=torch.float64), tensor(-1.4338, dtype=torch.float64), tensor(-1.4880, dtype=torch.float64), tensor(-1.5271, dtype=torch.float64), tensor(-1.5508, dtype=torch.float64), tensor(-1.5634, dtype=torch.float64), tensor(-1.5695, dtype=torch.float64), tensor(-1.5715, dtype=torch.float64), tensor(-1.5693, dtype=torch.float64), tensor(-1.5612, dtype=torch.float64), tensor(-1.5448, dtype=torch.float64), tensor(-1.5180, dtype=torch.float64), tensor(-1.4808, dtype=torch.float64), tensor(-1.4354, dtype=torch.float64), tensor(-1.3864, dtype=torch.float64), tensor(-1.3430, dtype=torch.float64), tensor(-1.3136, dtype=torch.float64), tensor(-1.3030, dtype=torch.float64)]
    # for e in energys_:
    #     energys.append(e.item())
    # print(energys)
    # exit()
    for phi in phis:
        G, cs = G_kitaev([phi]) #(s,s,vX,vY,vZ)
        single_A = prepare_magnetized_state()

        # QA = torch.einsum('ijklm,jabc->ikalbmc', Q, single_A).reshape(model.phys_dim, 2, 2, 2)
        # GQA = torch.einsum('ijklm,jabc->ikalbmc', G, QA).reshape(model.phys_dim, 4, 4, 4)
        # AA = torch.einsum('iabc,jmnc->ijabmn', GQA, GQA).reshape(model.phys_dim**2, 4, 4, 4, 4)


        GA = torch.einsum('ijklm,jabc->ikalbmc', G, single_A).reshape(model.phys_dim, 2, 2, 2)
        QGA = torch.einsum('ijklm,jabc->ikalbmc', Q, GA).reshape(model.phys_dim, 4, 4, 4)
        AA = torch.einsum('iabc,jmnc->ijabmn', QGA, QGA).reshape(model.phys_dim**2, 4, 4, 4, 4)

        # GA = torch.einsum('ijklm,jabc->ikalbmc', G, single_A).reshape(model.phys_dim, 2, 2, 2)
        # AA = torch.einsum('iabc,jmnc->ijabmn', GA, GA).reshape(model.phys_dim**2, 2, 2, 2, 2)

        AA = torch.permute(AA, (0,2,3,4,1))
        AA= AA/AA.norm()
        sites={(0,0): AA}
        def vertexToSite(coord):
            return (0,0)
        state_sym = IPEPS(sites,vertexToSite)
        ctm_env = ENV(args.chi, state_sym)
        init_env(state_sym, ctm_env)

        ctm_env, *ctm_log = ctmrg.run(state_sym, ctm_env, conv_check=ctmrg_conv_rdm2x1)
        loss = energy_f(state_sym, ctm_env)
        energys.append(loss.item())
        # obs_values, obs_labels= model.eval_obs(state_sym,ctm_env)
    # obs_labels = []
    # obs_values = []
    print(energys)
    # print(", ".join(["epoch","energy"]+obs_labels))
    # print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))
    exit()


    # print("theta=",state.sites[(0,0)][0,0,0,0,0].item())
    # print(cs[0].item(), cs[1].item(), cs[2].item(), sum(i*i for i in cs).item())
    
    # s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    # X = (s2.SP()+s2.SM())/2.
    # Z = s2.SZ()
    # A2= torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
    #                 dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    # A2[0,0,0,0,0] = torch.cos(state.sites[(0,0)][0,0,0,0,0])
    # A2[1,0,0,0,0] = torch.sin(state.sites[(0,0)][0,0,0,0,0])
    # A2= torch.einsum('ijklmn,jabcd->ikalbmcnd',GTNO,A2).reshape(model.phys_dim,2*bond_dim,2*bond_dim,2*bond_dim,2*bond_dim)
    # XA = torch.einsum('ij,jabcd->iabcd',X,A2)
    # XA= XA/XA.norm()
    
    # A= torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
    #                 dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
    # A[0,0,0,0,0] = torch.cos(state.sites[(0,0)][0,0,0,0,0])
    # A[1,0,0,0,0] = torch.sin(state.sites[(0,0)][0,0,0,0,0])
    # GTNOZ4 = torch.einsum('ijklmn,ka,lb,mc,nd->ijabcd',GTNO,Z,Z,Z,Z)
    # AZ4= torch.einsum('ijklmn,jabcd->ikalbmcnd',GTNOZ4,A).reshape(model.phys_dim,2*bond_dim,2*bond_dim,2*bond_dim,2*bond_dim)
    # AZ4= AZ4/AZ4.norm()
    # VOP = (XA-AZ4).norm()
    # print ("VOP=",VOP.item())

#     def loss_fn(state, ctm_env_in, params, opt_context):
#         ctm_args= opt_context["ctm_args"]
#         opt_args= opt_context["opt_args"]
#         # create a copy of state, symmetrize and normalize making all operations
#         # tracked. This does not "overwrite" the parameters tensors, living outside
#         # the scope of loss_fn

#         GTNO, cs = G_kitaev(params) #(s,s,vX,vY,vZ)
#         single_A = torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim),\
#                     dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
#         single_A[0,0,0,0] = torch.tensor(2/(2**0.5)+1)
#         single_A[1,0,0,0] = torch.tensor((1+1j)/(2**0.5))
#         double_A = torch.einsum('iabc,jdec->ijabde', single_A, single_A).reshape(model.phys_dim**2, bond_dim, bond_dim, bond_dim, bond_dim)
#         double_GTNO = torch.einsum('ijklm,abcdm->iajbklcd', GTNO, GTNO).reshape(model.phys_dim**2, model.phys_dim**2, 2*bond_dim, 2*bond_dim, 2*bond_dim, 2*bond_dim)
#         double_Q =  torch.einsum('ijklm,abcdm->iajbklcd', Q, Q).reshape(model.phys_dim**2, model.phys_dim**2, 2, 2, 2, 2)
#         A = torch.einsum('ijklmn,jabcd->ikalbmcnd',double_Q, double_A).reshape(model.phys_dim**2, 2*bond_dim,2*bond_dim, 2*bond_dim, 2*bond_dim)
#         A = torch.einsum('ijklmn,jabcd->ikalbmcnd',double_GTNO, A).reshape(model.phys_dim**2, 2*2*bond_dim, 2*2*bond_dim, 2*2*bond_dim, 2*2*bond_dim)
#         A = A/A.norm()

#         sites={(0,0): A}
#         def vertexToSite(coord):
#             return (0,0)
#         state_sym = IPEPS(sites,vertexToSite)
#         # print(state_sym)
#         # state_sym = IPEPS_C4V(A)

#         # possibly re-initialize the environment
#         if cfg.opt_args.opt_ctm_reinit:
#             init_env(state_sym, ctm_env_in)

#         # # 1) compute environment by CTMRG
#         # ctm_env_out, *ctm_log= ctmrg_c4v.run(state_sym, ctm_env_in, 
#         #     conv_check=ctmrg_conv_rdm2x1, ctm_args=ctm_args)
#         ctm_env_out, *ctm_log = ctmrg.run(state_sym, ctm_env_in, conv_check=ctmrg_conv_rdm2x1)
#         loss = energy_f(state_sym, ctm_env_out)
#         print("Energy : ", loss.item())
#         print(cs)
#         # print("theta=",state.sites[(0,0)][0,0,0,0,0].item())
        
#         return (loss, ctm_env_out, *ctm_log)

#     def _to_json(l):
#         re=[l[i,0].item() for i in range(l.size()[0])]
#         im=[l[i,1].item() for i in range(l.size()[0])]
#         return dict({"re": re, "im": im})

#     @torch.no_grad()
#     def obs_fn(state, ctm_env, params, opt_context):
#         return 0
#         # if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
#         #     or not "line_search" in opt_context.keys():
#         #     GTNO, cs = G_Ising(params)
#         #     A= torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
#         #                     dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
#         #     A[0,0,0,0,0] = torch.cos(state.sites[(0,0)][0,0,0,0,0])
#         #     A[1,0,0,0,0] = torch.sin(state.sites[(0,0)][0,0,0,0,0])
#         #     A= torch.einsum('ijklmn,jabcd->ikalbmcnd',GTNO,A).reshape(model.phys_dim,2*bond_dim,2*bond_dim,2*bond_dim,2*bond_dim)
#         #     A= A/A.norm()
#         #     state_sym = IPEPS_C4V(A)
            
#         #     epoch= len(opt_context["loss_history"]["loss"]) 
#         #     loss= opt_context["loss_history"]["loss"][-1]
#         #     obs_values, obs_labels = model.eval_obs(state_sym,ctm_env)
#         #     print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]\
#         #         + [f"{state.site().norm()}"]))
#         #     print(cs[0].item(), cs[1].item(), cs[2].item(), cs[3].item(), cs[4].item(),\
#         #           cs[5].item(), cs[6].item(), cs[7].item(), cs[8].item(), sum(i*i for i in cs).item())
#         #     print(torch.cos(state.sites[(0,0)][0,0,0,0,0]).item(),torch.sin(state.sites[(0,0)][0,0,0,0,0]).item())

#         #     if args.top_freq>0 and epoch%args.top_freq==0:
#         #         coord_dir_pairs=[((0,0), (1,0))]
#         #         for c,d in coord_dir_pairs:
#         #             # transfer operator spectrum
#         #             print(f"TOP spectrum(T)[{c},{d}] ",end="")
#         #             l= transferops_c4v.get_Top_spec_c4v(args.top_n, state_sym, ctm_env)
#         #             print("TOP "+json.dumps(_to_json(l)))

#     # optimize
#     # optimize_state(state, ctm_env, loss_fn, params, obs_fn=obs_fn)
#     optimize_state(state, ctm_env, loss_fn, params)

#     # compute final observables for the best variational state
#     # outputstatefile= args.out_prefix+"_state.json"
#     # state= read_ipeps_c4v(outputstatefile)
#     # filename1 = args.params_out+'.txt'
#     # with open(filename1, 'r') as file_to_read2:
#     #     params = []
#     #     lines2 = file_to_read2.readline()
#     #     q_tmp = [float(k) for k in lines2.replace('i', 'j').split()]
#     #     for i in range(9):
#     #         params.append(torch.as_tensor(q_tmp[i],dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device))
#     GTNO, cs = G_kitaev(params) #(s,s,vX,vY,vZ)

#     single_A = torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
#                     dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
#     single_A[0,0,0,0,0] = torch.tensor(2/(2**0.5)+1)
#     single_A[1,0,0,0,0] = torch.tensor((1+1j)/(2**0.5))
#     double_A = torch.einsum('ijklm,abcdm->iajbklcd', single_A, single_A).reshape(model.phys_dim**2, bond_dim, bond_dim, bond_dim, bond_dim)

#     double_GTNO = torch.einsum('ijklm,abcdm->iajbklcd', GTNO, GTNO).reshape(model.phys_dim**2, model.phys_dim**2, 2*bond_dim, 2*bond_dim, 2*bond_dim, 2*bond_dim)
#     double_Q =  torch.einsum('ijklm,abcdm->iajbklcd', Q, Q).reshape(model.phys_dim**2, model.phys_dim**2, 2, 2, 2, 2)
#     A = torch.einsum('ijklmn,jabcd->ikalbmcnd',double_Q, double_A).reshape(model.phys_dim**2, 2*bond_dim,2*bond_dim, 2*bond_dim, 2*bond_dim)
#     A = torch.einsum('ijklmn,jabcd->ikalbmcnd',double_GTNO, A).reshape(model.phys_dim**2, 2*2*bond_dim, 2*2*bond_dim, 2*2*bond_dim, 2*2*bond_dim)
#     A= A/A.norm()

#     sites={(0,0): A}
#     def vertexToSite(coord):
#         return (0,0)
#     state2 = IPEPS(sites,vertexToSite)
    
#     ctm_env = ENV(args.chi, state2)
#     init_env(state2, ctm_env)
#     ctm_env, *ctm_log = ctmrg.run(state2, ctm_env, conv_check=ctmrg_conv_rdm2x1)

#     opt_energy = energy_f(state2,ctm_env)
#     print("Final energy : ", opt_energy.item())
#     print(cs)


#     # obs_values, obs_labels = model.eval_obs(state2,ctm_env)
#     # print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))
#     # print("theta=",state.sites[(0,0)][0,0,0,0,0].item())
#     # print(cs[0].item(), cs[1].item(), cs[2].item(), cs[3].item(), cs[4].item(),\
#     #       cs[5].item(), cs[6].item(), cs[7].item(), cs[8].item(), sum(i*i for i in cs).item())

#     # s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
#     # X = (s2.SP()+s2.SM())/2.
#     # Z = s2.SZ()
#     # A2= torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
#     #                 dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
#     # A2[0,0,0,0,0] = torch.cos(state.sites[(0,0)][0,0,0,0,0])
#     # A2[1,0,0,0,0] = torch.sin(state.sites[(0,0)][0,0,0,0,0])
#     # A2= torch.einsum('ijklmn,jabcd->ikalbmcnd',GTNO,A2).reshape(model.phys_dim,2*bond_dim,2*bond_dim,2*bond_dim,2*bond_dim)
#     # XA = torch.einsum('ij,jabcd->iabcd',X,A2)
#     # XA= XA/XA.norm()
    
#     # A= torch.zeros((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
#     #                 dtype=cfg.global_args.torch_dtype,device=cfg.global_args.device)
#     # A[0,0,0,0,0] = torch.cos(state.sites[(0,0)][0,0,0,0,0])
#     # A[1,0,0,0,0] = torch.sin(state.sites[(0,0)][0,0,0,0,0])
#     # GTNOZ4 = torch.einsum('ijklmn,ka,lb,mc,nd->ijabcd',GTNO,Z,Z,Z,Z)
#     # AZ4= torch.einsum('ijklmn,jabcd->ikalbmcnd',GTNOZ4,A).reshape(model.phys_dim,2*bond_dim,2*bond_dim,2*bond_dim,2*bond_dim)
#     # AZ4= AZ4/AZ4.norm()
#     # VOP = (XA-AZ4).norm()
#     # print ("VOP=",VOP.item())

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

# class TestOpt(unittest.TestCase):
#     def setUp(self):
#         args.hx=0.0
#         args.bond_dim=2
#         args.chi=16
#         args.opt_max_iter=3

#     # basic tests
#     def test_opt_SYMEIG(self):
#         args.CTMARGS_projector_svd_method="SYMEIG"
#         main()

#     @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
#     def test_opt_SYMEIG_gpu(self):
#         args.GLOBALARGS_device="cuda:0"
#         args.CTMARGS_projector_svd_method="SYMEIG"
#         main()
