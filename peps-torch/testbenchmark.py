# import context
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic import ctmrg

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
parser.add_argument("--tensor", default="TFIM.cytnx", help="Input building block tensor for iPEPS.")
parser.add_argument("--bondim", type=int, default=2)
args, unknown_args = parser.parse_known_args()

# def test(tmp):
#     a = contiguous(cytnx.ncon([tmp,conj(tmp)],[[1,2,-1,-3,3],[1,2,-2,-4,3]]))
#     a = view(a, [args.bondim**2,args.bondim**2] )
# def truncated_svd(M, chi):
#     # return cytnx.linalg.Svd_truncate(M,chi,0,True,0)
#     return cytnx.linalg.Gesvd_truncate(M,chi,0,True,True,0)

def main():
    # M = cytnx.UniTensor.Load("Svd_M.cytnx")
    # S, U, V = truncated_svd(M, args.chi)  # M = USV^{T}
    # exit()
    
    cfg.configure(args)
    print("device arg = ", cfg.global_args.device)
    tmp = torch.rand([2,args.bondim,args.bondim,args.bondim,args.bondim], dtype = torch.complex128, device = cfg.global_args.device)
    print("device = ", tmp.device)
    tmp = tmp/tmp.abs().max().item()
    # tmp= tmp/tmp.get_block().Abs().Max().item()
    
    sites = {(0,0): tmp}
    state = IPEPS(sites)
    # # print(state.sites[(0,0)])
    # # exit()
    # # def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
    # #     return False, history
    ctm_env_init = ENV(args.chi, state)
    # tmp.to_(-1)
    # tmp.to_(cfg.global_args.device)
    init_env(state, ctm_env_init)
    # print(state.sites[(0,0)])
    # init_env(state, ctm_env_init)
    # print(state.sites[(0,0)])
    # exit()
    # print(state.sites[(0,0)])
    # exit()
    # print(", ".join(["epoch","energy"]+obs_labels))
    # print(", ".join([f"{-1}",f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    
    def ctmrg_conv_C(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        old = []
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
        # print("diff={0:<50}".format(diff), end="\r")
        # print(ctm_args.ctm_conv_tol)
        if (len(history[4*env.chi:]) > 1 and diff < ctm_args.ctm_conv_tol)\
            or len(history[4*env.chi:]) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history[4*env.chi:]), "history": history[4*env.chi:]})
            print("")
            # print("CTMRG length: "+str(len(history[4*env.chi:])))
            return True, history
        return False, history

    env, history, t_ctm, t_obs = ctmrg.run(state, ctm_env_init, conv_check= ctmrg_conv_C)

    #################################################
    # import cProfile
    # import pstats
    # prof = cProfile.Profile()
    # prof.runctx('ctmrg.run(state, ctm_env_init, conv_check= ctmrg_conv_C)', globals(), locals())
    # prof.dump_stats('output.pstats')
    #################################################

    # stream = open('output.txt', 'w')
    # stats = pstats.Stats('output.pstats', stream=stream)
    # stats.sort_stats('cumtime')
    # stats.print_stats()

    print("t_ctm = ", t_ctm)
    print("t_svd = ", cfg.global_args.svd_time)
    print("t_net = ", cfg.global_args.net_time)

    # print("t_obs = ", t_obs)
    # # 6) compute final observables
    # e_curr0 = energy_f(state, ctm_env_init)
    # obs_values0, obs_labels = eval_obs_f(state,ctm_env_init)
    # history, t_ctm, t_obs= ctm_log
    # print("\n")
    # print(", ".join(["epoch","energy"]+obs_labels))
    # print("FINAL "+", ".join([f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    # print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # path = args.txt
    # with open(path, 'a') as f:
    #     f.write("  ".join([f"{args.h}"]+[f"{e_curr0}"]+[f"{v}" for v in obs_values0]))
    #     f.write("\n")
    #         print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    # if len(unknown_args)>0:
    #     print("args not recognized: "+str(unknown_args))
    #     raise Exception("Unknown command line arguments")
    main()
