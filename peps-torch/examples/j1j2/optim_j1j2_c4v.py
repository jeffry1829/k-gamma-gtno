# python optim_j1j2_c4v.py --GLOBALARGS_dtype complex128 --bond_dim 3 --chi 16 --opt_max_iter 40 --CTMARGS_ctm_max_iter 100000 --CTMARGS_projector_eps_multiplet 1e-12 --CTMARGS_projector_svd_reltol 1e-12 --CTMARGS_projector_method 4X4
import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import *
from groups.pg import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from ctm.one_site_c4v import transferops_c4v
from models import j1j2
# from optim.ad_optim_sgd_mod import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
# from optim.ad_optim import optimize_state
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--j1", type=float, default=1.,
                    help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0.,
                    help="next nearest-neighbour coupling")
parser.add_argument("--j3", type=float, default=0.,
                    help="next-to-next nearest-neighbour coupling")
parser.add_argument("--hz_stag", type=float, default=0.,
                    help="staggered mag. field")
parser.add_argument("--delta_zz", type=float, default=1.,
                    help="easy-axis (nearest-neighbour) anisotropy")
parser.add_argument("--top_freq", type=int, default=-1,
                    help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues" +
                    "of transfer operator to compute")
parser.add_argument("--force_cpu", action='store_true',
                    help="evaluate energy on cpu")
args, unknown_args = parser.parse_known_args()


def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model = j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2, j3=args.j3,
                                    hz_stag=args.hz_stag, delta_zz=args.delta_zz)
    # energy_f= model.energy_1x1
    energy_f = model.energy_1x1_lowmem

    # initialize the ipeps
    if args.instate != None:
        state = read_ipeps_c4v(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state = extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
        # state.sites[(0,0)]= state.sites[(0,0)]/torch.max(torch.abs(state.sites[(0,0)]))
        state.sites[(0, 0)] = state.site()/state.site().norm()
    elif args.opt_resume is not None:
        state = IPEPS_C4V()
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type == 'RANDOM':
        bond_dim = args.bond_dim
        A = torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),
                       dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
        # A= make_c4v_symm(A)
        # A= A/torch.max(torch.abs(A))
        A = A/A.norm()
        state = IPEPS_C4V(A)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "
                         + str(args.ipeps_init_type)+" is not supported")

    print(state)

    # @torch.no_grad()
    # def ctmrg_conv_f(state, ctm_env, history, ctm_args=cfg.ctm_args):
    #     print("ctm:", ctm_env.C.keys())
    #     print("ctm:", ctm_env.T.keys())
    #     if not history:
    #         history = dict({"log": []})
    #     rdm2x1 = rdm2x1_sl(state, ctm_env, force_cpu=ctm_args.conv_check_cpu)
    #     dist = float('inf')
    #     if len(history["log"]) > 0:
    #         dist = torch.dist(rdm2x1, history["rdm"], p=2).item()
    #     history["rdm"] = rdm2x1
    #     history["log"].append(dist)
    #     if dist < ctm_args.ctm_conv_tol or len(history["log"]) >= ctm_args.ctm_max_iter:
    #         log.info({"history_length": len(
    #             history['log']), "history": history['log']})
    #         return True, history
    #     return False, history
    @torch.no_grad()
    def ctmrg_conv_f(state2, env, history, ctm_args=cfg.ctm_args):
        torch.set_printoptions(profile="full")
        torch.set_printoptions(linewidth=200)
        torch.set_printoptions(precision=8)
        if not history:
            history = []
        old = []
        if (len(history) > 0):
            old = history[:2*env.chi+2]
        new = []
        u, s, v = torch.linalg.svd(env.get_C())
        # print("C singular matrix:", s)
        for i in range(env.chi):
            new.append(s[i].item())

        u, s, v = torch.linalg.svd(
            env.get_T().reshape(env.chi, env.chi*args.bond_dim**2))
        for i in range(env.chi):
            new.append(s[i].item())

        # from hosvd import sthosvd as hosvd
        # core, _, _ = hosvd(env.T[((0, 0), (0, -1))], [env.chi]*3)
        # new.append(core)
        # core, _, _ = hosvd(env.T[((0, 0), (0, 1))], [env.chi]*3)
        # new.append(core)
        # core, _, _ = hosvd(env.T[((0, 0), (-1, 0))], [env.chi]*3)
        # new.append(core)
        # core, _, _ = hosvd(env.T[((0, 0), (1, 0))], [env.chi]*3)
        # new.append(core)
        # print("core.shape: ", core.shape)
        # print("core: ", core)

        new.append(env.get_T())
        new.append(env.get_C())

        diff = 0.
        if (len(history) > 0):
            for i in range(2*env.chi):
                history[i] = new[i]
                if (abs(old[i]-new[i]) > diff):
                    diff = abs(old[i]-new[i])
            for i in range(2):
                history[2*env.chi+i] = new[2*env.chi+i]
                if ((old[2*env.chi+i]-new[2*env.chi+i]).abs().max() > diff):
                    diff = (old[2*env.chi+i]-new[2*env.chi+i]).abs().max()
                # print(torch.div(old[4*env.chi+i], new[4*env.chi+i]))
                # if i == 0:
                #     difftograph.append((old[4*env.chi+i]-new[4*env.chi+i]).norm())

        else:
            for i in range(2*env.chi+2):
                history.append(new[i])
        history.append(diff)
        print("diff={0:<50}".format(diff), end="\r")
        # print("diff={0:<50}".format(diff))
        # print(ctm_args.ctm_conv_tol)
        if (len(history[2*env.chi+2:]) > 1 and diff < ctm_args.ctm_conv_tol)\
                or len(history[2*env.chi+2:]) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(
                history[2*env.chi+8:]), "history": history[2*env.chi+2:]})
            print("")
            print("modified CTMRG length: "+str(len(history[2*env.chi+2:])))
            # import matplotlib.pyplot as plt
            # plt.plot(difftograph)
            # plt.show()
            return True, history
        return False, history

    state_sym = to_ipeps_c4v(state)
    ctm_env = ENV_C4V(args.chi, state_sym)
    init_env(state_sym, ctm_env)

    ctm_env, *ctm_log = ctmrg_c4v.run(state_sym,
                                      ctm_env, conv_check=ctmrg_conv_f)

    loss = energy_f(state_sym, ctm_env, force_cpu=args.force_cpu)
    obs_values, obs_labels = model.eval_obs(state_sym, ctm_env)
    print(", ".join(["epoch", "energy"]+obs_labels))
    print(", ".join([f"{-1}", f"{loss}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env, opt_context):
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]
        # 0) preprocess
        # create a copy of state, symmetrize and normalize making all operations
        # tracked. This does not "overwrite" the parameters tensors, living outside
        # the scope of loss_fn
        state_sym = to_ipeps_c4v(state, normalize=True)

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env)

        # 1) compute environment by CTMRG
        ctm_env, *ctm_log = ctmrg_c4v.run(state_sym, ctm_env,
                                          conv_check=ctmrg_conv_f, ctm_args=ctm_args)

        # 2) evaluate loss with converged environment
        loss = energy_f(state_sym, ctm_env, force_cpu=args.force_cpu)

        return (loss, ctm_env, *ctm_log)

    def _to_json(l):
        re = [l[i, 0].item() for i in range(l.size()[0])]
        im = [l[i, 1].item() for i in range(l.size()[0])]
        return dict({"re": re, "im": im})

    @ torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if opt_context["line_search"]:
            epoch = len(opt_context["loss_history"]["loss_ls"])
            loss = opt_context["loss_history"]["loss_ls"][-1]
            print("LS", end=" ")
        else:
            epoch = len(opt_context["loss_history"]["loss"])
            loss = opt_context["loss_history"]["loss"][-1]
        state_sym = to_ipeps_c4v(state, normalize=True)
        obs_values, obs_labels = model.eval_obs(state_sym, ctm_env)
        print(", ".join([f"{epoch}", f"{loss}"]+[f"{v}" for v in obs_values] +
                        [f"{torch.max(torch.abs(state.site((0,0))))}"]))

        if (not opt_context["line_search"]) and args.top_freq > 0 and epoch % args.top_freq == 0:
            coord_dir_pairs = [((0, 0), (1, 0))]
            for c, d in coord_dir_pairs:
                # transfer operator spectrum
                print(f"TOP spectrum(T)[{c},{d}] ", end="")
                l = transferops_c4v.get_Top_spec_c4v(
                    args.top_n, state_sym, ctm_env)
                print("TOP "+json.dumps(_to_json(l)))

    def post_proc(state, ctm_env, opt_context):
        symm, max_err = verify_c4v_symm_A1(state.site())
        # print(f"post_proc {symm} {max_err}")
        if not symm:
            # force symmetrization outside of autograd
            with torch.no_grad():
                symm_site = make_c4v_symm(state.site())
                # we **cannot** simply normalize the on-site tensors, as the LBFGS
                # takes into account the scale
                # symm_site= symm_site/torch.max(torch.abs(symm_site))
                state.sites[(0, 0)].copy_(symm_site)

    # optimize
    # optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn, post_proc=post_proc)
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile = args.out_prefix+"_state.json"
    state = read_ipeps_c4v(outputstatefile)
    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state, ctm_env)
    obs_values, obs_labels = model.eval_obs(state, ctm_env)
    print(", ".join([f"{args.opt_max_iter}",
          f"{opt_energy}"]+[f"{v}" for v in obs_values]))


if __name__ == '__main__':
    if len(unknown_args) > 0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2 = 0.0
        args.bond_dim = 2
        args.chi = 16
        args.opt_max_iter = 3
        try:
            import scipy.sparse.linalg
            self.SCIPY = True
        except:
            print("Warning: Missing scipy. Arnoldi methods not available.")
            self.SCIPY = False

    # basic tests
    def test_opt_COMPLEX(self):
        args.GLOBALARGS_dtype = "complex128"
        main()
        args.GLOBALARGS_dtype = "float64"

    def test_opt_SYMEIG(self):
        args.CTMARGS_projector_svd_method = "SYMEIG"
        main()

    def test_opt_SYMEIG_LS_strong_wolfe(self):
        args.CTMARGS_projector_svd_method = "SYMEIG"
        args.OPTARGS_line_search = "strong_wolfe"
        main()

    def test_opt_SYMEIG_LS_backtracking(self):
        args.CTMARGS_projector_svd_method = "SYMEIG"
        args.OPTARGS_line_search = "backtracking"
        main()

    def test_opt_SYMEIG_LS_backtracking_SYMARP(self):
        if not self.SCIPY:
            self.skipTest("test skipped: missing scipy")
        args.CTMARGS_projector_svd_method = "SYMEIG"
        args.OPTARGS_line_search = "backtracking"
        args.OPTARGS_line_search_svd_method = "SYMARP"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_COMPLEX_gpu(self):
        args.GLOBALARGS_device = "cuda:0"
        args.GLOBALARGS_dtype = "complex128"
        main()
        args.GLOBALARGS_dtype = "float64"

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_gpu(self):
        args.GLOBALARGS_device = "cuda:0"
        args.CTMARGS_projector_svd_method = "SYMEIG"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_LS_backtracking_SYMARP_gpu(self):
        if not self.SCIPY:
            self.skipTest("test skipped: missing scipy")
        args.GLOBALARGS_device = "cuda:0"
        args.CTMARGS_projector_svd_method = "SYMEIG"
        args.OPTARGS_line_search = "backtracking"
        args.OPTARGS_line_search_svd_method = "SYMARP"
        main()
