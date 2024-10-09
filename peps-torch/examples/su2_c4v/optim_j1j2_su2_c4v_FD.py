import context
import copy
import torch
import argparse
import config as cfg
from ipeps.ipeps_lc import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from ctm.one_site_c4v import transferops_c4v
from models import j1j2
from optim.fd_optim_lbfgs_mod import optimize_state
import su2sym.sym_ten_parser as tenSU2
import time
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--force_cpu", action="store_true", help="force energy and observable evaluation on CPU")
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

@torch.no_grad()
def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
    if not history:
        history=dict({"log": []})
    rdm= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu, \
        verbosity=cfg.ctm_args.verbosity_rdm)
    dist= float('inf')
    if len(history["log"]) > 1:
        dist= torch.dist(rdm, history["rdm"], p=2).item()
    history["rdm"]=rdm
    history["log"].append(dist)
    if dist<ctm_args.ctm_conv_tol:
        log.info({"history_length": len(history['log']), "history": history['log'],
            "final_multiplets": compute_multiplets(env)})
        return True, history
    elif len(history['log']) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(history['log']), "history": history['log'],
            "final_multiplets": compute_multiplets(env)})
        return False, history
    return False, history

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)
    energy_f= model.energy_1x1_lowmem

    # initialize an ipeps
    if args.instate!=None:
        state = read_ipeps_lc_1site_pg(args.instate)
        assert len(state.coeffs)==1, "Not a 1-site ipeps"

        abd= args.bond_dim
        cbd= max(state.get_aux_bond_dims())
        if abd > cbd and abd in [3,5,7,9]:
            su2sym_t= tenSU2.import_sym_tensors_FIX(2,abd ,"A_1",\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)

            A= torch.zeros(len(su2sym_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)

            # get uuid lists
            uuid_orig=[t[0]["meta"]["name"].replace("T_","T_"+((abd-cbd)//2)*"0") for t in state.su2_tensors]
            uuid_new=[t[0]["meta"]["name"] for t in su2sym_t]
            print(f"{uuid_orig}")
            print(f"{uuid_new}")
            coeffs_orig=next(iter(state.coeffs.values()))
            for i,uuid in enumerate(uuid_orig):
                A[uuid_new.index(uuid)]=coeffs_orig[i]

            coeffs= {(0,0): A}
            state= IPEPS_LC_1SITE_PG(elem_tensors=su2sym_t, coeffs=coeffs)
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        if args.bond_dim in [3,5,7,9]:
            su2sym_t= tenSU2.import_sym_tensors_FIX(2,args.bond_dim,"A_1",\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported -bond_dim= "+str(args.bond_dim))
        A= torch.zeros(len(su2sym_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        coeffs = {(0,0): A}
        state= IPEPS_LC_1SITE_PG(su2sym_t, coeffs)
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        if args.bond_dim in [3,5,7,9]:
            su2sym_t= tenSU2.import_sym_tensors_FIX(2,args.bond_dim,"A_1",\
                dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        else:
            raise ValueError("Unsupported -bond_dim= "+str(args.bond_dim))

        A= torch.rand(len(su2sym_t), dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        A= A/torch.max(torch.abs(A))
        coeffs = {(0,0): A}
        state = IPEPS_LC_1SITE_PG(su2sym_t, coeffs)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)

    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)

    loss0 = energy_f(state, ctm_env, force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss0}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]

        # build on-site tensors from su2sym components
        state.sites= state.build_onsite_tensors()

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state, ctm_env_in)

        # 1) compute environment by CTMRG
        t0_ctm_main= time.perf_counter() 
        ctm_env_out, history, t_ctm, t_obs= ctmrg_c4v.run(state, ctm_env_in, \
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        t1_ctm_main= time.perf_counter()
        t0_energy= time.perf_counter()
        loss0 = energy_f(state, ctm_env_out, force_cpu=args.force_cpu)
        t1_energy= time.perf_counter()
        
        loc_ctm_args= copy.deepcopy(ctm_args)
        loc_ctm_args.ctm_max_iter= 1
        ctm_env_out, history1, t_ctm1, t_obs1= ctmrg_c4v.run(state, ctm_env_out, \
            ctm_args=loc_ctm_args)
        t2_energy= time.perf_counter()
        loss1 = energy_f(state, ctm_env_out, force_cpu=args.force_cpu)
        t3_energy= time.perf_counter()

        timings= dict({ "t_ctm_main": t1_ctm_main-t0_ctm_main, "t_ctm": t_ctm, \
            "t_obs": t_obs, "t_energy": (t1_energy-t0_energy)+(t3_energy-t2_energy)})
        #loss=(loss0+loss1)/2
        loss= torch.max(loss0,loss1)

        return loss, ctm_env_out, history, timings

    def _to_json(l):
        re=[l[i,0].item() for i in range(l.size()[0])]
        im=[l[i,1].item() for i in range(l.size()[0])]
        return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if opt_context["line_search"]:
            epoch= len(opt_context["loss_history"]["loss_ls"])
            loss= opt_context["loss_history"]["loss_ls"][-1]
            print("LS",end=" ")
        else:
            epoch= len(opt_context["loss_history"]["loss"]) 
            loss= opt_context["loss_history"]["loss"][-1] 
        obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
        print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]))

        if (not opt_context["line_search"]) and args.top_freq>0 and epoch%args.top_freq==0:
            coord_dir_pairs=[((0,0), (1,0))]
            for c,d in coord_dir_pairs:
                # transfer operator spectrum
                print(f"TOP spectrum(T)[{c},{d}] ",end="")
                l= transferops_c4v.get_Top_spec_c4v(args.top_n, state_sym, ctm_env)
                print("TOP "+json.dumps(_to_json(l)))

    # optimize
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_lc_1site_pg(outputstatefile)
    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state,ctm_env,force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs(state,ctm_env,force_cpu=args.force_cpu)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=3
        args.chi=18
        args.opt_max_iter=3
        try:
            import scipy.sparse.linalg
            self.SCIPY= True
        except:
            print("Warning: Missing scipy. ARNOLDISVD is not available.")
            self.SCIPY= False

    # basic tests
    def test_opt_SYMEIG_LS(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        main()

    def test_opt_SYMARP_LS_SYMARP(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method="SYMARP"
        args.OPTARGS_line_search="backtracking"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        args.OPTARGS_line_search="backtracking"
        main()
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMARP_gpu(self):
        if not self.SCIPY: self.skipTest("test skipped: missing scipy")
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMARP"
        args.OPTARGS_line_search="backtracking"
        main()