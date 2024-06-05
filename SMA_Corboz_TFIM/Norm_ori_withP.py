import context
import time
import torch
import argparse
import config as cfg
from ctm.generic.ctm_projectors import *
from ipeps.ipeps import *
from tn_interface import contract, einsum
from tn_interface import conj
from tn_interface import contiguous, view, permute
from math import sqrt

# from ctm.generic.ctmrg import *
from ctm.generic import ctmrg

bond_dim = 2


def ctmrg_conv_energy(state2, env, history, ctm_args=cfg.ctm_args):
    if not history:
        history = []
    old = []
    if (len(history) > 0):
        old = history[:8*env.chi+8]
    new = []
    u, s, v = torch.linalg.svd(env.C[((0, 0), (-1, -1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(env.C[((0, 0), (1, -1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(env.C[((0, 0), (1, -1))])
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(env.C[((0, 0), (1, 1))])
    for i in range(env.chi):
        new.append(s[i].item())

    u, s, v = torch.linalg.svd(
        env.T[((0, 0), (0, -1))].reshape(env.chi, env.chi*bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(
        env.T[((0, 0), (0, 1))].permute(1, 0, 2).reshape(env.chi, env.chi*bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(
        env.T[((0, 0), (-1, 0))].permute(0, 2, 1).reshape(env.chi, env.chi*bond_dim**2))
    for i in range(env.chi):
        new.append(s[i].item())
    u, s, v = torch.linalg.svd(
        env.T[((0, 0), (1, 0))].reshape(env.chi, env.chi*bond_dim**2))
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

    new.append(env.T[((0, 0), (0, -1))])
    new.append(env.T[((0, 0), (0, 1))])
    new.append(env.T[((0, 0), (-1, 0))])
    new.append(env.T[((0, 0), (1, 0))])
    new.append(env.C[((0, 0), (-1, -1))])
    new.append(env.C[((0, 0), (-1, 1))])
    new.append(env.C[((0, 0), (1, -1))])
    new.append(env.C[((0, 0), (1, 1))])

    diff = 0.
    if (len(history) > 0):
        for i in range(8*env.chi):
            history[i] = new[i]
            if (abs(old[i]-new[i]) > diff):
                diff = abs(old[i]-new[i])
        for i in range(8):
            history[8*env.chi+i] = new[8*env.chi+i]
            if ((old[8*env.chi+i]-new[8*env.chi+i]).abs().max() > diff):
                diff = (old[8*env.chi+i]-new[8*env.chi+i]).abs().max()
            # print(torch.div(old[4*env.chi+i], new[4*env.chi+i]))
            # if i == 0:
            #     difftograph.append((old[4*env.chi+i]-new[4*env.chi+i]).norm())

    else:
        for i in range(8*env.chi+8):
            history.append(new[i])
    history.append(diff)
    print("diff={0:<50}".format(diff), end="\r")
    # print("diff={0:<50}".format(diff))
    # print(ctm_args.ctm_conv_tol)
    if (len(history[8*env.chi+8:]) > 1 and diff < ctm_args.ctm_conv_tol)\
            or len(history[8*env.chi+8:]) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(
            history[8*env.chi+8:]), "history": history[8*env.chi+8:]})
        print("")
        print("modified CTMRG length: "+str(len(history[8*env.chi+8:])))
        # import matplotlib.pyplot as plt
        # plt.plot(difftograph)
        # plt.show()
        return True, history
    return False, history


def Create_Projectors(state, stateDL, env, args):
    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200)
    torch.set_printoptions(precision=8)

    def move_normalize_c(nC1, nC2, nT, norm_type=ctm_args.ctm_absorb_normalization,
                         verbosity=ctm_args.verbosity_ctm_move):
        _ord = 2
        if norm_type == 'inf':
            _ord = float('inf')

        with torch.no_grad():
            scale_nC1 = torch.linalg.vector_norm(nC1, ord=_ord)
            scale_nC2 = torch.linalg.vector_norm(nC2, ord=_ord)
            scale_nT = torch.linalg.vector_norm(nT, ord=_ord)
        if verbosity > 0:
            print(f"nC1 {scale_nC1} nC2 {scale_nC2} nT {scale_nT}")
        nC1 = nC1/scale_nC1
        nC2 = nC2/scale_nC2
        nT = nT/scale_nT
        return nC1, nC2, nT
    # AX - XB = C
    # def sylvester(A, B, C, X=None):
    #     m = B.shape[-1]
    #     n = A.shape[-1]
    #     R, U = torch.linalg.eig(A)
    #     S, V = torch.linalg.eig(B)
    #     F = torch.linalg.solve(U, (C + 0j) @ V)
    #     W = R[..., :, None] - S[..., None, :]
    #     Y = F / W
    #     X = U[..., :n, :n] @ Y[..., :n,
    #                             :m] @ torch.linalg.inv(V)[..., :m, :m]
    #     return X.real if all(torch.isreal(x.flatten()[0])
    #                             for x in [A, B, C]) else X

    # solve AX-XB=C
    # def sylvester(A, B, C):
    #     # use scipy to solve
    #     import scipy
    #     from scipy.linalg import solve_sylvester
    #     X = solve_sylvester(A.cpu().numpy(), -B.cpu().numpy(), C.cpu().numpy())
    #     return torch.from_numpy(X).to(A.device).to(A.dtype)

    # solve AX=XB C is zero
    # assume A,B square
    def sylvester(A, B, C):
        if not torch.allclose(C, torch.zeros_like(C)):
            raise ValueError("C must be zero")
        Ichi = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        A = A.contiguous()
        B = B.contiguous()
        AI = torch.kron(A, Ichi)
        IBt = torch.kron(Ichi, B.permute(1, 0).contiguous())
        AI = AI.cpu().numpy()
        IBt = IBt.cpu().numpy()
        import scipy
        Xvec = scipy.linalg.null_space(AI-IBt, rcond=1e-6)
        print("Xvec shape: ", Xvec.shape)
        return torch.from_numpy(Xvec[:, 0].reshape(A.shape[0], A.shape[1])).to(A.device).to(A.dtype)

    def chk_distinct_eigs(A, B):
        eo, _ = torch.linalg.eig(A)
        ep, _ = torch.linalg.eig(B)
        # print("eigenvalues of ToM: ", eo)
        # print("eigenvalues of TpM: ", ep)
        for i in range(len(eo)):
            for j in range(len(ep)):
                if torch.allclose(eo[i], ep[j], atol=1e-3):
                    print(
                        "Warning: eigenvalues of ToM and TpM are not distinct")
                    return False
        return True

    def _get_gauge_T(To, Tp, D2idx=0, check=True, direction=None):
        if direction == (-1, 0):
            # To.shape = Tp.shape = (chi,chi,D^2)
            ToM = To[:, :, D2idx]  # (chi,chi)
            TpM = Tp[:, :, D2idx]  # (chi,chi)
            ToM, TpM = TpM, ToM
        elif direction == (1, 0):
            # To.shape = Tp.shape = (chi,D^2,chi)
            ToM = To[:, D2idx, :]  # (chi,chi)
            TpM = Tp[:, D2idx, :]  # (chi,chi)
        elif direction == (0, 1):
            # To.shape = Tp.shape = (D^2,chi,chi)
            ToM = To[D2idx, :, :]  # (chi,chi)
            TpM = Tp[D2idx, :, :]  # (chi,chi)
            ToM, TpM = TpM, ToM
        elif direction == (0, -1):
            # To.shape = Tp.shape = (chi,D^2,chi)
            ToM = To[:, D2idx, :]  # (chi,chi)
            TpM = Tp[:, D2idx, :]  # (chi,chi)
        else:
            raise ValueError(
                "Invalid direction in _get_gauge_T: "+str(direction))

        # if check:
        #     chk_distinct_eigs(ToM, TpM)
        return sylvester(TpM, ToM, torch.zeros_like(ToM))

    # find the subspace intersection of matrices, where each matrix are column vectors to span subspace
    def _zassenhaus(*Matrixs):
        # Matrixs = [A,B,C,D,E,...]
        if len(Matrixs) < 2:
            raise ValueError("_zassenhaus: At least two matrices are needed")
        At = Matrixs[0].t()
        for i in range(1, len(Matrixs)):
            Bt = Matrixs[i].t()
            H = torch.cat((At, At), 1)
            _tmp = torch.cat((Bt, torch.zeros_like(Bt)), 1)
            H = torch.cat((H, _tmp), 0)
            p, l, u = torch.lu(H)
            firstAllzero = At.shape[0]+Bt.shape[0]
            for j in range(firstAllzero-1, 0, -1):
                if torch.allclose(u[j, :At.shape[1]], torch.zeros_like(u[j, :At.shape[1]])):
                    firstAllzero = j
            print("firstAllzero:", firstAllzero)
            At = u[firstAllzero:, At.shape[1]:]
            # return u[firstAllzero:,At.shape[1]:]
        return At
    if cfg.ctm_args.projector_method == '4X4':
        ctm_get_projectors = ctm_get_projectors_4x4
    elif cfg.ctm_args.projector_method == '4X2':
        ctm_get_projectors = ctm_get_projectors_4x2
    else:
        raise ValueError("Invalid Projector method: " +
                         str(cfg.ctm_args.projector_method))
    P = dict()
    Pt = dict()
    # prevT = env.clone().T

    ctm_env_ex2 = env.clone()
    for i in range(args.size+1):
        for coord in stateDL.sites.keys():
            P[(i, coord, (-1, 0))], Pt[(i, coord, (-1, 0))] = ctm_get_projectors((-1,
                                                                                  0), coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)
            P[(i, coord, (1, 0))], Pt[(i, coord, (1, 0))] = ctm_get_projectors(
                (1, 0), coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)
            P[(i, coord, (0, 1))], Pt[(i, coord, (0, 1))] = ctm_get_projectors(
                (0, 1), coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)
            P[(i, coord, (0, -1))], Pt[(i, coord, (0, -1))] = ctm_get_projectors((0, -1),
                                                                                 coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)

            # prevT = ctm_env_ex2.clone().T

            # dimsA = state.site(coord).size()
            # Aket = state.site(coord)
            # DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(coord))),
            #                     (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
            # # upper is right of matrix, down is left of matrix
            # _Pt = Pt[(i, coord, (0, -1))].reshape([env.chi,dimsA[2]**2,env.chi])
            # _P = P[(i, coord, (0, -1))].reshape([env.chi,dimsA[2]**2,env.chi])
            # PtAcAP_up = torch.einsum('abc,dbfg,hgi->cfiadh', _Pt, DL, _P).reshape([dimsA[1]**2*env.chi**2,dimsA[3]**2*env.chi**2])
            # e, v = torch.linalg.eig(PtAcAP_up)
            # # print("eigenvalues of PtAcAP_up: ", e)
            # idx = np.argsort(-np.abs(e))
            # e = e[idx]
            # v = v[:, idx]
            # # use dominant eigenvector as new T
            # ctm_env_ex2.T[(coord,(0, -1))] = v[:, 0].reshape([env.chi,dimsA[1]**2,env.chi])/torch.linalg.norm(v[:, 0], ord=torch.inf)
            # _, sprev, _ = torch.linalg.svd(prevT[(coord, (0, -1))].reshape(env.chi, env.chi*args.bond_dim**2))
            # _, snow, _ = torch.linalg.svd(ctm_env_ex2.T[(coord,(0, -1))].reshape(env.chi, env.chi*args.bond_dim**2))
            # print("prevT:", sprev)
            # print("ctm_env_ex2.T:", snow)
            # if torch.allclose(prevT[(coord, (0, -1))], ctm_env_ex2.T[(coord, (0, -1))]):
            #     print("T Tensors are the same")
            # else:
            #     print("T Tensors are different")

            # prevT = ctm_env_ex2.clone().T

            old_itercnt = cfg.ctm_args.ctm_max_iter
            cfg.ctm_args.ctm_max_iter = 1
            ctm_env_ex2, _, *ctm_log = ctmrg.run(state, ctm_env_ex2, conv_check=ctmrg_conv_energy,
                                                 ctm_args=cfg.ctm_args, global_args=cfg.global_args)
            cfg.ctm_args.ctm_max_iter = old_itercnt

            # for direction in cfg.ctm_args.ctm_move_sequence:
            #     # if direction == (0, -1):
            #     #     rel_CandT_vecs = {"nC1": (1, -1), "nC2": (-1, -1), "nT": direction}
            #     # elif direction == (-1, 0):
            #     #     rel_CandT_vecs = {"nC1": (-1, -1), "nC2": (-1, 1), "nT": direction}
            #     # elif direction == (0, 1):
            #     #     rel_CandT_vecs = {"nC1": (-1, 1), "nC2": (1, 1), "nT": direction}
            #     # elif direction == (1, 0):
            #     #     rel_CandT_vecs = {"nC1": (1, 1), "nC2": (1, -1), "nT": direction}
            #     _P = dict()
            #     _Pt = dict()
            #     if direction == (0, -1):
            #         _P[(0, 0)] = P[(i, coord, direction)]
            #         _Pt[(0, 0)] = Pt[(i, coord, direction)]
            #         nC1, nC2, nT = absorb_truncate_CTM_MOVE_UP(
            #             coord, stateDL, ctm_env_ex2, _P, _Pt, ctm_args)
            #         nC1, nC2, nT = move_normalize_c(nC1, nC2, nT)
            #         ctm_env_ex2.C[(coord, (1, -1))] = nC1
            #         ctm_env_ex2.C[(coord, (-1, -1))] = nC2
            #         ctm_env_ex2.T[(coord, direction)] = nT
            #     elif direction == (-1, 0):
            #         _P[(0, 0)] = P[(i, coord, direction)]
            #         _Pt[(0, 0)] = Pt[(i, coord, direction)]
            #         nC1, nC2, nT = absorb_truncate_CTM_MOVE_LEFT(
            #             coord, stateDL, ctm_env_ex2, _P, _Pt, ctm_args)
            #         nC1, nC2, nT = move_normalize_c(nC1, nC2, nT)
            #         ctm_env_ex2.C[(coord, (-1, -1))] = nC1
            #         ctm_env_ex2.C[(coord, (-1, 1))] = nC2
            #         ctm_env_ex2.T[(coord, direction)] = nT
            #     elif direction == (0, 1):
            #         _P[(0, 0)] = P[(i, coord, direction)]
            #         _Pt[(0, 0)] = Pt[(i, coord, direction)]
            #         nC1, nC2, nT = absorb_truncate_CTM_MOVE_DOWN(
            #             coord, stateDL, ctm_env_ex2, _P, _Pt, ctm_args)
            #         nC1, nC2, nT = move_normalize_c(nC1, nC2, nT)
            #         ctm_env_ex2.C[(coord, (-1, 1))] = nC1
            #         ctm_env_ex2.C[(coord, (1, 1))] = nC2
            #         ctm_env_ex2.T[(coord, direction)] = nT
            #     elif direction == (1, 0):
            #         _P[(0, 0)] = P[(i, coord, direction)]
            #         _Pt[(0, 0)] = Pt[(i, coord, direction)]
            #         nC1, nC2, nT = absorb_truncate_CTM_MOVE_RIGHT(
            #             coord, stateDL, ctm_env_ex2, _P, _Pt, ctm_args)
            #         nC1, nC2, nT = move_normalize_c(nC1, nC2, nT)
            #         ctm_env_ex2.C[(coord, (1, 1))] = nC1
            #         ctm_env_ex2.C[(coord, (1, -1))] = nC2
            #         ctm_env_ex2.T[(coord, direction)] = nT
            #     else:
            #         raise ValueError(
            #             "Invalid direction: "+str(direction))

            # if torch.allclose(prevT[(coord, (0, -1))], ctm_env_ex2.T[(coord, (0, -1))]):
            #     print("T Tensors are the same before")
            # else:
            #     print("T Tensors are different before")

            # X = torch.randn(env.chi, env.chi).to(
            #     cfg.global_args.device).to(cfg.global_args.torch_dtype)
            # test = _get_gauge_T(nT, torch.einsum(
            #     'ab,bcd,de->ace', X, nT, torch.linalg.inv(X)), D2idx=0, check=True, direction=(0, -1))
            # print("test:", test)

            # XL = _get_gauge_T(prevT[(coord, (-1, 0))], ctm_env_ex2.T[(
            #     coord, (-1, 0))], D2idx=0, check=True, direction=(-1, 0))
            # P[(i, coord, (-1, 0))] = P[(i, coord, (-1, 0))] @ XL
            # Pt[(i, coord, (-1, 0))] = Pt[(i, coord, (-1, 0))
            #                              ] @ torch.linalg.inv(XL).t()
            # XL = _get_gauge_T(prevT[(coord, (1, 0))], ctm_env_ex2.T[(
            #     coord, (1, 0))], D2idx=0, check=True, direction=(1, 0))
            # P[(i, coord, (1, 0))] = P[(i, coord, (1, 0))] @ XL
            # Pt[(i, coord, (1, 0))] = Pt[(i, coord, (1, 0))
            #                             ] @ torch.linalg.inv(XL).t()
            # XL = _get_gauge_T(prevT[(coord, (0, 1))], ctm_env_ex2.T[(
            #     coord, (0, 1))], D2idx=0, check=True, direction=(0, 1))
            # P[(i, coord, (0, 1))] = P[(i, coord, (0, 1))] @ XL
            # Pt[(i, coord, (0, 1))] = Pt[(i, coord, (0, 1))
            #                             ] @ torch.linalg.inv(XL).t()
            # XL = _get_gauge_T(prevT[(coord, (0, -1))], ctm_env_ex2.T[(
            #     coord, (0, -1))], D2idx=0, check=True, direction=(0, -1))
            # P[(i, coord, (0, -1))] = P[(i, coord, (0, -1))] @ XL
            # Pt[(i, coord, (0, -1))] = Pt[(i, coord, (0, -1))
            #                              ] @ torch.linalg.inv(XL).t()

            # for direction in cfg.ctm_args.ctm_move_sequence:
            #     if direction == (-1, 0):
            #         ################ edge tensor################
            #         vec_coord = (-args.size+i, 0)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
            #         vec = (0, -1)

            #         coord_shift_up = stateDL.vertexToSite(
            #             (new_coord[0]+vec[0], new_coord[1]+vec[1]))
            #         coord_shift_down = stateDL.vertexToSite(
            #             (new_coord[0]-vec[0], new_coord[1]-vec[1]))
            #         P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
            #             coord_shift_down).size()[0], env.chi))
            #         Pt2 = view(Pt[(i, new_coord, direction)], (env.chi,
            #                    stateDL.site(new_coord).size()[2], env.chi))
            #         P1 = view(P[(i, coord_shift_up, direction)],
            #                   (env.chi, stateDL.site(new_coord).size()[0], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
            #             coord_shift_up).size()[2], env.chi))

            #         nT = contract(
            #             P1, ctm_env_ex2.T[(new_coord, direction)], ([0], [0]))
            #         dimsA = state.site(new_coord).size()
            #         Aket = state.site(new_coord)
            #         DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord))),
            #                            (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
            #         nT = contract(nT, DL, ([0, 3], [0, 1]))
            #         nT = contract(nT, Pt2, ([1, 2], [0, 1]))
            #         tempT2 = contiguous(permute(nT, (0, 2, 1)))
            #         # print(tempT2.abs().max())
            #         tempT2 = tempT2/tempT2.abs().max()

            #         vec_coord = (vec_coord[0]+1, vec_coord[1])
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.T[(new_coord, direction)] = tempT2.clone()

            #         ################ corner tensor################
            #         vec_coord_u = (-args.size+i, -args.size)
            #         new_coord_u = state.vertexToSite(
            #             (coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
            #         coord_shift_up = stateDL.vertexToSite(
            #             (new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
            #         coord_shift_down = stateDL.vertexToSite(
            #             (new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
            #         P2 = view(P[(i, new_coord_u, direction)], (env.chi, stateDL.site(
            #             coord_shift_down).size()[0], env.chi))
            #         Pt2 = view(Pt[(i, new_coord_u, direction)], (env.chi, stateDL.site(
            #             new_coord_u).size()[2], env.chi))
            #         P1 = view(P[(i, coord_shift_up, direction)], (env.chi,
            #                   stateDL.site(new_coord_u).size()[0], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
            #             coord_shift_up).size()[2], env.chi))
            #         nC1 = contract(ctm_env_ex2.C[(
            #             new_coord_u, (-1, -1))], ctm_env_ex2.T[(new_coord_u, (0, -1))], ([1], [0]))
            #         nC1 = contract(Pt1, nC1, ([0, 1], [0, 1]))
            #         # print(nC1.abs().max())
            #         tempT2 = nC1/nC1.abs().max()

            #         vec_coord = (vec_coord_u[0]+1, vec_coord_u[1])
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.C[(new_coord, (-1, -1))] = tempT2.clone()

            #         vec_coord_d = (-args.size+i, args.size+1)
            #         new_coord_d = state.vertexToSite(
            #             (coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
            #         coord_shift_up = stateDL.vertexToSite(
            #             (new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
            #         coord_shift_down = stateDL.vertexToSite(
            #             (new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
            #         P2 = view(P[(i, new_coord_d, direction)], (env.chi, stateDL.site(
            #             coord_shift_down).size()[0], env.chi))
            #         Pt2 = view(Pt[(i, new_coord_d, direction)], (env.chi, stateDL.site(
            #             new_coord_d).size()[2], env.chi))
            #         P1 = view(P[(i, coord_shift_up, direction)], (env.chi,
            #                   stateDL.site(new_coord_d).size()[0], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
            #             coord_shift_up).size()[2], env.chi))
            #         nC2 = contract(ctm_env_ex2.C[(
            #             new_coord_d, (-1, 1))], ctm_env_ex2.T[(new_coord_d, (0, 1))], ([1], [1]))
            #         nC2 = contract(P2, nC2, ([0, 1], [0, 1]))
            #         # print(nC2.abs().max())
            #         tempT2 = nC2/nC2.abs().max()

            #         vec_coord = (vec_coord_d[0]+1, vec_coord_d[1])
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.C[(new_coord, (-1, 1))] = tempT2.clone()

            #     elif direction == (1, 0):
            #         ################ edge tensor################
            #         vec_coord = (args.size-i, 0)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
            #         vec = (0, 1)

            #         coord_shift_down = stateDL.vertexToSite(
            #             (new_coord[0]+vec[0], new_coord[1]+vec[1]))
            #         coord_shift_up = stateDL.vertexToSite(
            #             (new_coord[0]-vec[0], new_coord[1]-vec[1]))
            #         P2 = view(P[(i, new_coord, direction)], (env.chi,
            #                   stateDL.site(coord_shift_up).size()[2], env.chi))
            #         Pt2 = view(Pt[(i, new_coord, direction)], (env.chi,
            #                    stateDL.site(new_coord).size()[0], env.chi))
            #         P1 = view(P[(i, coord_shift_down, direction)],
            #                   (env.chi, stateDL.site(new_coord).size()[2], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
            #             coord_shift_down).size()[0], env.chi))

            #         nT = contract(
            #             Pt2, ctm_env_ex2.T[(new_coord, direction)], ([0], [0]))
            #         dimsA = state.site(new_coord).size()
            #         Aket = state.site(new_coord)
            #         DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord))),
            #                            (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
            #         nT = contract(nT, DL, ([0, 2], [0, 3]))
            #         nT = contract(nT, P1, ([1, 3], [0, 1]))
            #         tempT2 = contiguous(nT)
            #         # print(tempT2.abs().max())
            #         tempT2 = tempT2/tempT2.abs().max()

            #         vec_coord = (vec_coord[0]-1, vec_coord[1])
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.T[(new_coord, direction)] = tempT2.clone()

            #         ################ corner tensor################
            #         vec_coord_u = (args.size-i+1, -args.size)
            #         new_coord_u = state.vertexToSite(
            #             (coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
            #         coord_shift_down = stateDL.vertexToSite(
            #             (new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
            #         coord_shift_up = stateDL.vertexToSite(
            #             (new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
            #         P2 = view(P[(i, new_coord_u, direction)], (env.chi,
            #                   stateDL.site(coord_shift_up).size()[2], env.chi))
            #         Pt2 = view(Pt[(i, new_coord_u, direction)], (env.chi, stateDL.site(
            #             new_coord_u).size()[0], env.chi))
            #         P1 = view(P[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
            #             new_coord_u).size()[2], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
            #             coord_shift_down).size()[0], env.chi))
            #         nC2 = contract(ctm_env_ex2.C[(
            #             new_coord_u, (1, -1))], ctm_env_ex2.T[(new_coord_u, (0, -1))], ([0], [2]))
            #         nC2 = contract(nC2, P2, ([0, 2], [0, 1]))
            #         # print(nC2.abs().max())
            #         tempT2 = nC2/nC2.abs().max()

            #         vec_coord = (vec_coord_u[0]-1, vec_coord_u[1])
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.C[(new_coord, (1, -1))] = tempT2.clone()

            #         vec_coord_d = (args.size-i+1, args.size+1)
            #         new_coord_d = state.vertexToSite(
            #             (coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
            #         coord_shift_down = stateDL.vertexToSite(
            #             (new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
            #         coord_shift_up = stateDL.vertexToSite(
            #             (new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
            #         P2 = view(P[(i, new_coord_d, direction)], (env.chi,
            #                   stateDL.site(coord_shift_up).size()[2], env.chi))
            #         Pt2 = view(Pt[(i, new_coord_d, direction)], (env.chi, stateDL.site(
            #             new_coord_d).size()[0], env.chi))
            #         P1 = view(P[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
            #             new_coord_d).size()[2], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
            #             coord_shift_down).size()[0], env.chi))
            #         nC1 = contract(ctm_env_ex2.C[(new_coord_d, (1, 1))], ctm_env_ex2.T[(
            #             new_coord_d, (0, 1))], ([1], [2]))
            #         nC1 = contract(Pt1, nC1, ([0, 1], [0, 1]))
            #         # print(nC1.abs().max())
            #         tempT2 = nC1/nC1.abs().max()

            #         vec_coord = (vec_coord_d[0]-1, vec_coord_d[1])
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.C[(new_coord, (1, 1))] = tempT2.clone()
            #     elif direction == (0, -1):
            #         ################ edge tensor################
            #         vec_coord = (0, -args.size+i)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
            #         vec = (1, 0)

            #         coord_shift_left = stateDL.vertexToSite(
            #             (new_coord[0]-vec[0], new_coord[1]-vec[1]))
            #         coord_shift_right = stateDL.vertexToSite(
            #             (new_coord[0]+vec[0], new_coord[1]+vec[1]))
            #         P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
            #             coord_shift_left).size()[3], env.chi))
            #         Pt2 = view(Pt[(i, new_coord, direction)], (env.chi,
            #                    stateDL.site(new_coord).size()[1], env.chi))
            #         P1 = view(P[(i, coord_shift_right, direction)],
            #                   (env.chi, stateDL.site(new_coord).size()[3], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
            #             coord_shift_right).size()[1], env.chi))

            #         nT = contract(
            #             Pt2, ctm_env_ex2.T[(new_coord, direction)], ([0], [0]))
            #         dimsA = state.site(new_coord).size()
            #         Aket = state.site(new_coord)
            #         DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord))),
            #                            (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
            #         nT = contract(nT, DL, ([0, 2], [1, 0]))
            #         nT = contract(nT, P1, ([1, 3], [0, 1]))
            #         tempT2 = contiguous(nT)
            #         # print(tempT2.abs().max())
            #         tempT2 = tempT2/tempT2.abs().max()

            #         vec_coord = (vec_coord[0], vec_coord[1]+1)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.T[(new_coord, direction)] = tempT2.clone()

            #         ################ corner tensor################
            #         vec_coord_l = (-args.size, -args.size+i)
            #         new_coord_l = state.vertexToSite(
            #             (coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
            #         coord_shift_left = stateDL.vertexToSite(
            #             (new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
            #         coord_shift_right = stateDL.vertexToSite(
            #             (new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
            #         P2 = view(P[(i, new_coord_l, direction)], (env.chi, stateDL.site(
            #             coord_shift_left).size()[3], env.chi))
            #         Pt2 = view(Pt[(i, new_coord_l, direction)], (env.chi, stateDL.site(
            #             new_coord_l).size()[1], env.chi))
            #         P1 = view(P[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
            #             new_coord_l).size()[3], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
            #             coord_shift_right).size()[1], env.chi))
            #         nC2 = contract(ctm_env_ex2.C[(
            #             new_coord_l, (-1, -1))], ctm_env_ex2.T[(new_coord_l, (-1, 0))], ([0], [0]))
            #         nC2 = contract(nC2, P2, ([0, 2], [0, 1]))
            #         # print(nC2.abs().max())
            #         tempT2 = nC2/nC2.abs().max()

            #         vec_coord = (vec_coord_l[0], vec_coord_l[1]+1)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.C[(new_coord, (-1, -1))] = tempT2.clone()

            #         vec_coord_r = (args.size+1, -args.size+i)
            #         new_coord_r = state.vertexToSite(
            #             (coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
            #         coord_shift_left = stateDL.vertexToSite(
            #             (new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
            #         coord_shift_right = stateDL.vertexToSite(
            #             (new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
            #         P2 = view(P[(i, new_coord_r, direction)], (env.chi, stateDL.site(
            #             coord_shift_left).size()[3], env.chi))
            #         Pt2 = view(Pt[(i, new_coord_r, direction)], (env.chi, stateDL.site(
            #             new_coord_r).size()[1], env.chi))
            #         P1 = view(P[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
            #             new_coord_r).size()[3], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
            #             coord_shift_right).size()[1], env.chi))
            #         nC1 = contract(ctm_env_ex2.C[(
            #             new_coord_r, (1, -1))], ctm_env_ex2.T[(new_coord_r, (1, 0))], ([1], [0]))
            #         nC1 = contract(Pt1, nC1, ([0, 1], [0, 1]))
            #         # print(nC1.abs().max())
            #         tempT2 = nC1/nC1.abs().max()

            #         vec_coord = (vec_coord_r[0], vec_coord_r[1]+1)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.C[(new_coord, (1, -1))] = tempT2.clone()

            #     elif direction == (0, 1):
            #         ################ edge tensor################
            #         vec_coord = (0, args.size-i)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
            #         vec = (-1, 0)

            #         coord_shift_right = stateDL.vertexToSite(
            #             (new_coord[0]-vec[0], new_coord[1]-vec[1]))
            #         coord_shift_left = stateDL.vertexToSite(
            #             (new_coord[0]+vec[0], new_coord[1]+vec[1]))
            #         P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
            #             coord_shift_right).size()[1], env.chi))
            #         Pt2 = view(Pt[(i, new_coord, direction)], (env.chi,
            #                    stateDL.site(new_coord).size()[3], env.chi))
            #         P1 = view(P[(i, coord_shift_left, direction)],
            #                   (env.chi, stateDL.site(new_coord).size()[1], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
            #             coord_shift_left).size()[3], env.chi))

            #         nT = contract(
            #             P1, ctm_env_ex2.T[(new_coord, direction)], ([0], [1]))
            #         dimsA = state.site(new_coord).size()
            #         Aket = state.site(new_coord)
            #         DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord))),
            #                            (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
            #         nT = contract(nT, DL, ([0, 2], [1, 2]))
            #         nT = contract(nT, Pt2, ([1, 3], [0, 1]))
            #         tempT2 = contiguous(permute(nT, (1, 0, 2)))
            #         # print(tempT2.abs().max())
            #         tempT2 = tempT2/tempT2.abs().max()

            #         vec_coord = (vec_coord[0], vec_coord[1]-1)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.T[(new_coord, direction)] = tempT2.clone()

            #         ################ corner tensor################
            #         vec_coord_l = (-args.size, args.size-i+1)
            #         new_coord_l = state.vertexToSite(
            #             (coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
            #         coord_shift_right = stateDL.vertexToSite(
            #             (new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
            #         coord_shift_left = stateDL.vertexToSite(
            #             (new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
            #         P2 = view(P[(i, new_coord_l, direction)], (env.chi, stateDL.site(
            #             coord_shift_right).size()[1], env.chi))
            #         Pt2 = view(Pt[(i, new_coord_l, direction)], (env.chi, stateDL.site(
            #             new_coord_l).size()[3], env.chi))
            #         P1 = view(P[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
            #             new_coord_l).size()[1], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
            #             coord_shift_left).size()[3], env.chi))
            #         nC1 = contract(ctm_env_ex2.C[(
            #             new_coord_l, (-1, 1))], ctm_env_ex2.T[(new_coord_l, (-1, 0))], ([0], [1]))
            #         nC1 = contract(nC1, Pt1, ([0, 2], [0, 1]))
            #         # print(nC1.abs().max())
            #         tempT2 = nC1/nC1.abs().max()

            #         vec_coord = (vec_coord_l[0], vec_coord_l[1]-1)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.C[(new_coord, (-1, 1))] = tempT2.clone()

            #         vec_coord_r = (args.size+1, args.size-i+1)
            #         new_coord_r = state.vertexToSite(
            #             (coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
            #         coord_shift_right = stateDL.vertexToSite(
            #             (new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
            #         coord_shift_left = stateDL.vertexToSite(
            #             (new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
            #         P2 = view(P[(i, new_coord_r, direction)], (env.chi, stateDL.site(
            #             coord_shift_right).size()[1], env.chi))
            #         Pt2 = view(Pt[(i, new_coord_r, direction)], (env.chi, stateDL.site(
            #             new_coord_r).size()[3], env.chi))
            #         P1 = view(P[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
            #             new_coord_r).size()[1], env.chi))
            #         Pt1 = view(Pt[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
            #             coord_shift_left).size()[3], env.chi))
            #         nC2 = contract(ctm_env_ex2.C[(new_coord_r, (1, 1))], ctm_env_ex2.T[(
            #             new_coord_r, (1, 0))], ([0], [2]))
            #         nC2 = contract(nC2, P2, ([0, 2], [0, 1]))
            #         # print(nC2.abs().max())
            #         tempT2 = nC2/nC2.abs().max()

            #         vec_coord = (vec_coord_r[0], vec_coord_r[1]-1)
            #         new_coord = state.vertexToSite(
            #             (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

            #         ctm_env_ex2.C[(new_coord, (1, 1))] = tempT2.clone()

    # ctm_env_ex2 = env.clone()
    # for i in range(args.size+1):
    #     for coord in stateDL.sites.keys():
    #         P[(i, coord, (0, 1))], Pt[(i, coord, (0, 1))] = ctm_get_projectors(
    #             (0, 1), coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)
    #         P[(i, coord, (0, -1))], Pt[(i, coord, (0, -1))] = ctm_get_projectors((0, -1),
    #                                                                              coord, stateDL, ctm_env_ex2, cfg.ctm_args, cfg.global_args)
    #         for direction in cfg.ctm_args.ctm_move_sequence:
    #             if direction == (-1, 0):
    #                 ################ edge tensor################
    #                 vec_coord = (-args.size+i, 0)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
    #                 vec = (0, -1)

    #                 coord_shift_up = stateDL.vertexToSite(
    #                     (new_coord[0]+vec[0], new_coord[1]+vec[1]))
    #                 coord_shift_down = stateDL.vertexToSite(
    #                     (new_coord[0]-vec[0], new_coord[1]-vec[1]))
    #                 P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
    #                     coord_shift_down).size()[0], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord, direction)], (env.chi,
    #                            stateDL.site(new_coord).size()[2], env.chi))
    #                 P1 = view(P[(i, coord_shift_up, direction)],
    #                           (env.chi, stateDL.site(new_coord).size()[0], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
    #                     coord_shift_up).size()[2], env.chi))

    #                 nT = contract(
    #                     P1, ctm_env_ex2.T[(new_coord, direction)], ([0], [0]))
    #                 dimsA = state.site(new_coord).size()
    #                 Aket = state.site(new_coord)
    #                 DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord))),
    #                                    (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    #                 nT = contract(nT, DL, ([0, 3], [0, 1]))
    #                 nT = contract(nT, Pt2, ([1, 2], [0, 1]))
    #                 tempT2 = contiguous(permute(nT, (0, 2, 1)))
    #                 # print(tempT2.abs().max())
    #                 tempT2 = tempT2

    #                 vec_coord = (vec_coord[0]+1, vec_coord[1])
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.T[(new_coord, direction)] = tempT2.clone()

    #                 ################ corner tensor################
    #                 vec_coord_u = (-args.size+i, -args.size)
    #                 new_coord_u = state.vertexToSite(
    #                     (coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
    #                 coord_shift_up = stateDL.vertexToSite(
    #                     (new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
    #                 coord_shift_down = stateDL.vertexToSite(
    #                     (new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
    #                 P2 = view(P[(i, new_coord_u, direction)], (env.chi, stateDL.site(
    #                     coord_shift_down).size()[0], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord_u, direction)], (env.chi, stateDL.site(
    #                     new_coord_u).size()[2], env.chi))
    #                 P1 = view(P[(i, coord_shift_up, direction)], (env.chi,
    #                           stateDL.site(new_coord_u).size()[0], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
    #                     coord_shift_up).size()[2], env.chi))
    #                 nC1 = contract(ctm_env_ex2.C[(
    #                     new_coord_u, (-1, -1))], ctm_env_ex2.T[(new_coord_u, (0, -1))], ([1], [0]))
    #                 nC1 = contract(Pt1, nC1, ([0, 1], [0, 1]))
    #                 # print(nC1.abs().max())
    #                 tempT2 = nC1

    #                 vec_coord = (vec_coord_u[0]+1, vec_coord_u[1])
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.C[(new_coord, (-1, -1))] = tempT2.clone()

    #                 vec_coord_d = (-args.size+i, args.size+1)
    #                 new_coord_d = state.vertexToSite(
    #                     (coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
    #                 coord_shift_up = stateDL.vertexToSite(
    #                     (new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
    #                 coord_shift_down = stateDL.vertexToSite(
    #                     (new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
    #                 P2 = view(P[(i, new_coord_d, direction)], (env.chi, stateDL.site(
    #                     coord_shift_down).size()[0], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord_d, direction)], (env.chi, stateDL.site(
    #                     new_coord_d).size()[2], env.chi))
    #                 P1 = view(P[(i, coord_shift_up, direction)], (env.chi,
    #                           stateDL.site(new_coord_d).size()[0], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
    #                     coord_shift_up).size()[2], env.chi))
    #                 nC2 = contract(ctm_env_ex2.C[(
    #                     new_coord_d, (-1, 1))], ctm_env_ex2.T[(new_coord_d, (0, 1))], ([1], [1]))
    #                 nC2 = contract(P2, nC2, ([0, 1], [0, 1]))
    #                 # print(nC2.abs().max())
    #                 tempT2 = nC2

    #                 vec_coord = (vec_coord_d[0]+1, vec_coord_d[1])
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.C[(new_coord, (-1, 1))] = tempT2.clone()

    #             elif direction == (1, 0):
    #                 ################ edge tensor################
    #                 vec_coord = (args.size-i, 0)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
    #                 vec = (0, 1)

    #                 coord_shift_down = stateDL.vertexToSite(
    #                     (new_coord[0]+vec[0], new_coord[1]+vec[1]))
    #                 coord_shift_up = stateDL.vertexToSite(
    #                     (new_coord[0]-vec[0], new_coord[1]-vec[1]))
    #                 P2 = view(P[(i, new_coord, direction)], (env.chi,
    #                           stateDL.site(coord_shift_up).size()[2], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord, direction)], (env.chi,
    #                            stateDL.site(new_coord).size()[0], env.chi))
    #                 P1 = view(P[(i, coord_shift_down, direction)],
    #                           (env.chi, stateDL.site(new_coord).size()[2], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
    #                     coord_shift_down).size()[0], env.chi))

    #                 nT = contract(
    #                     Pt2, ctm_env_ex2.T[(new_coord, direction)], ([0], [0]))
    #                 dimsA = state.site(new_coord).size()
    #                 Aket = state.site(new_coord)
    #                 DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord))),
    #                                    (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    #                 nT = contract(nT, DL, ([0, 2], [0, 3]))
    #                 nT = contract(nT, P1, ([1, 3], [0, 1]))
    #                 tempT2 = contiguous(nT)
    #                 # print(tempT2.abs().max())
    #                 tempT2 = tempT2

    #                 vec_coord = (vec_coord[0]-1, vec_coord[1])
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.T[(new_coord, direction)] = tempT2.clone()

    #                 ################ corner tensor################
    #                 vec_coord_u = (args.size-i+1, -args.size)
    #                 new_coord_u = state.vertexToSite(
    #                     (coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
    #                 coord_shift_down = stateDL.vertexToSite(
    #                     (new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
    #                 coord_shift_up = stateDL.vertexToSite(
    #                     (new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
    #                 P2 = view(P[(i, new_coord_u, direction)], (env.chi,
    #                           stateDL.site(coord_shift_up).size()[2], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord_u, direction)], (env.chi, stateDL.site(
    #                     new_coord_u).size()[0], env.chi))
    #                 P1 = view(P[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
    #                     new_coord_u).size()[2], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
    #                     coord_shift_down).size()[0], env.chi))
    #                 nC2 = contract(ctm_env_ex2.C[(
    #                     new_coord_u, (1, -1))], ctm_env_ex2.T[(new_coord_u, (0, -1))], ([0], [2]))
    #                 nC2 = contract(nC2, P2, ([0, 2], [0, 1]))
    #                 # print(nC2.abs().max())
    #                 tempT2 = nC2

    #                 vec_coord = (vec_coord_u[0]-1, vec_coord_u[1])
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.C[(new_coord, (1, -1))] = tempT2.clone()

    #                 vec_coord_d = (args.size-i+1, args.size+1)
    #                 new_coord_d = state.vertexToSite(
    #                     (coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
    #                 coord_shift_down = stateDL.vertexToSite(
    #                     (new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
    #                 coord_shift_up = stateDL.vertexToSite(
    #                     (new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
    #                 P2 = view(P[(i, new_coord_d, direction)], (env.chi,
    #                           stateDL.site(coord_shift_up).size()[2], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord_d, direction)], (env.chi, stateDL.site(
    #                     new_coord_d).size()[0], env.chi))
    #                 P1 = view(P[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
    #                     new_coord_d).size()[2], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
    #                     coord_shift_down).size()[0], env.chi))
    #                 nC1 = contract(ctm_env_ex2.C[(new_coord_d, (1, 1))], ctm_env_ex2.T[(
    #                     new_coord_d, (0, 1))], ([1], [2]))
    #                 nC1 = contract(Pt1, nC1, ([0, 1], [0, 1]))
    #                 # print(nC1.abs().max())
    #                 tempT2 = nC1

    #                 vec_coord = (vec_coord_d[0]-1, vec_coord_d[1])
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.C[(new_coord, (1, 1))] = tempT2.clone()
    #             elif direction == (0, -1):
    #                 ################ edge tensor################
    #                 vec_coord = (0, -args.size+i)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
    #                 vec = (1, 0)

    #                 coord_shift_left = stateDL.vertexToSite(
    #                     (new_coord[0]-vec[0], new_coord[1]-vec[1]))
    #                 coord_shift_right = stateDL.vertexToSite(
    #                     (new_coord[0]+vec[0], new_coord[1]+vec[1]))
    #                 P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
    #                     coord_shift_left).size()[3], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord, direction)], (env.chi,
    #                            stateDL.site(new_coord).size()[1], env.chi))
    #                 P1 = view(P[(i, coord_shift_right, direction)],
    #                           (env.chi, stateDL.site(new_coord).size()[3], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
    #                     coord_shift_right).size()[1], env.chi))

    #                 nT = contract(
    #                     Pt2, ctm_env_ex2.T[(new_coord, direction)], ([0], [0]))
    #                 dimsA = state.site(new_coord).size()
    #                 Aket = state.site(new_coord)
    #                 DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord))),
    #                                    (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    #                 nT = contract(nT, DL, ([0, 2], [1, 0]))
    #                 nT = contract(nT, P1, ([1, 3], [0, 1]))
    #                 tempT2 = contiguous(nT)
    #                 # print(tempT2.abs().max())
    #                 tempT2 = tempT2

    #                 vec_coord = (vec_coord[0], vec_coord[1]+1)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.T[(new_coord, direction)] = tempT2.clone()

    #                 ################ corner tensor################
    #                 vec_coord_l = (-args.size, -args.size+i)
    #                 new_coord_l = state.vertexToSite(
    #                     (coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
    #                 coord_shift_left = stateDL.vertexToSite(
    #                     (new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
    #                 coord_shift_right = stateDL.vertexToSite(
    #                     (new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
    #                 P2 = view(P[(i, new_coord_l, direction)], (env.chi, stateDL.site(
    #                     coord_shift_left).size()[3], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord_l, direction)], (env.chi, stateDL.site(
    #                     new_coord_l).size()[1], env.chi))
    #                 P1 = view(P[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
    #                     new_coord_l).size()[3], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
    #                     coord_shift_right).size()[1], env.chi))
    #                 nC2 = contract(ctm_env_ex2.C[(
    #                     new_coord_l, (-1, -1))], ctm_env_ex2.T[(new_coord_l, (-1, 0))], ([0], [0]))
    #                 nC2 = contract(nC2, P2, ([0, 2], [0, 1]))
    #                 # print(nC2.abs().max())
    #                 tempT2 = nC2

    #                 vec_coord = (vec_coord_l[0], vec_coord_l[1]+1)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.C[(new_coord, (-1, -1))] = tempT2.clone()

    #                 vec_coord_r = (args.size+1, -args.size+i)
    #                 new_coord_r = state.vertexToSite(
    #                     (coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
    #                 coord_shift_left = stateDL.vertexToSite(
    #                     (new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
    #                 coord_shift_right = stateDL.vertexToSite(
    #                     (new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
    #                 P2 = view(P[(i, new_coord_r, direction)], (env.chi, stateDL.site(
    #                     coord_shift_left).size()[3], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord_r, direction)], (env.chi, stateDL.site(
    #                     new_coord_r).size()[1], env.chi))
    #                 P1 = view(P[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
    #                     new_coord_r).size()[3], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
    #                     coord_shift_right).size()[1], env.chi))
    #                 nC1 = contract(ctm_env_ex2.C[(
    #                     new_coord_r, (1, -1))], ctm_env_ex2.T[(new_coord_r, (1, 0))], ([1], [0]))
    #                 nC1 = contract(Pt1, nC1, ([0, 1], [0, 1]))
    #                 # print(nC1.abs().max())
    #                 tempT2 = nC1

    #                 vec_coord = (vec_coord_r[0], vec_coord_r[1]+1)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.C[(new_coord, (1, -1))] = tempT2.clone()

    #             elif direction == (0, 1):
    #                 ################ edge tensor################
    #                 vec_coord = (0, args.size-i)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
    #                 vec = (-1, 0)

    #                 coord_shift_right = stateDL.vertexToSite(
    #                     (new_coord[0]-vec[0], new_coord[1]-vec[1]))
    #                 coord_shift_left = stateDL.vertexToSite(
    #                     (new_coord[0]+vec[0], new_coord[1]+vec[1]))
    #                 P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
    #                     coord_shift_right).size()[1], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord, direction)], (env.chi,
    #                            stateDL.site(new_coord).size()[3], env.chi))
    #                 P1 = view(P[(i, coord_shift_left, direction)],
    #                           (env.chi, stateDL.site(new_coord).size()[1], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
    #                     coord_shift_left).size()[3], env.chi))

    #                 nT = contract(
    #                     P1, ctm_env_ex2.T[(new_coord, direction)], ([0], [1]))
    #                 dimsA = state.site(new_coord).size()
    #                 Aket = state.site(new_coord)
    #                 DL = torch.reshape(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord))),
    #                                    (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
    #                 nT = contract(nT, DL, ([0, 2], [1, 2]))
    #                 nT = contract(nT, Pt2, ([1, 3], [0, 1]))
    #                 tempT2 = contiguous(permute(nT, (1, 0, 2)))
    #                 # print(tempT2.abs().max())
    #                 tempT2 = tempT2

    #                 vec_coord = (vec_coord[0], vec_coord[1]-1)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.T[(new_coord, direction)] = tempT2.clone()

    #                 ################ corner tensor################
    #                 vec_coord_l = (-args.size, args.size-i+1)
    #                 new_coord_l = state.vertexToSite(
    #                     (coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
    #                 coord_shift_right = stateDL.vertexToSite(
    #                     (new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
    #                 coord_shift_left = stateDL.vertexToSite(
    #                     (new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
    #                 P2 = view(P[(i, new_coord_l, direction)], (env.chi, stateDL.site(
    #                     coord_shift_right).size()[1], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord_l, direction)], (env.chi, stateDL.site(
    #                     new_coord_l).size()[3], env.chi))
    #                 P1 = view(P[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
    #                     new_coord_l).size()[1], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
    #                     coord_shift_left).size()[3], env.chi))
    #                 nC1 = contract(ctm_env_ex2.C[(
    #                     new_coord_l, (-1, 1))], ctm_env_ex2.T[(new_coord_l, (-1, 0))], ([0], [1]))
    #                 nC1 = contract(nC1, Pt1, ([0, 2], [0, 1]))
    #                 # print(nC1.abs().max())
    #                 tempT2 = nC1

    #                 vec_coord = (vec_coord_l[0], vec_coord_l[1]-1)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.C[(new_coord, (-1, 1))] = tempT2.clone()

    #                 vec_coord_r = (args.size+1, args.size-i+1)
    #                 new_coord_r = state.vertexToSite(
    #                     (coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
    #                 coord_shift_right = stateDL.vertexToSite(
    #                     (new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
    #                 coord_shift_left = stateDL.vertexToSite(
    #                     (new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
    #                 P2 = view(P[(i, new_coord_r, direction)], (env.chi, stateDL.site(
    #                     coord_shift_right).size()[1], env.chi))
    #                 Pt2 = view(Pt[(i, new_coord_r, direction)], (env.chi, stateDL.site(
    #                     new_coord_r).size()[3], env.chi))
    #                 P1 = view(P[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
    #                     new_coord_r).size()[1], env.chi))
    #                 Pt1 = view(Pt[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
    #                     coord_shift_left).size()[3], env.chi))
    #                 nC2 = contract(ctm_env_ex2.C[(new_coord_r, (1, 1))], ctm_env_ex2.T[(
    #                     new_coord_r, (1, 0))], ([0], [2]))
    #                 nC2 = contract(nC2, P2, ([0, 2], [0, 1]))
    #                 # print(nC2.abs().max())
    #                 tempT2 = nC2

    #                 vec_coord = (vec_coord_r[0], vec_coord_r[1]-1)
    #                 new_coord = state.vertexToSite(
    #                     (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))

    #                 ctm_env_ex2.C[(new_coord, (1, 1))] = tempT2.clone()

    return P, Pt


def Create_Norm_Env(state, stateDL, B_grad, env, P, Pt, lam, kx, ky, args):
    C_up = dict()
    T_up = dict()
    T_up2 = dict()
    C_left = dict()
    T_left = dict()
    T_left2 = dict()
    C_down = dict()
    T_down = dict()
    T_down2 = dict()
    C_right = dict()
    T_right = dict()
    T_right2 = dict()
    for coord in stateDL.sites.keys():
        for direction in cfg.ctm_args.ctm_move_sequence:
            if direction == (0, -1):
                vec = (1, 0)
                vec_coord = (-args.size, -args.size)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_up["left"] = env.C[(new_coord, (-1, -1))].clone()
                vec_coord = (args.size+1, -args.size)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_up["right"] = env.C[(new_coord, (1, -1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size+j, -args.size)
                    new_coord = state.vertexToSite(
                        (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    T_up[(j)] = env.T[(new_coord, direction)].clone()

                for i in range(args.size+1):
                    vec_coord_l = (-args.size, -args.size+i)
                    new_coord_l = state.vertexToSite(
                        (coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
                    coord_shift_left = stateDL.vertexToSite(
                        (new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
                    coord_shift_right = stateDL.vertexToSite(
                        (new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
                    P2 = view(P[(i, new_coord_l, direction)], (env.chi, stateDL.site(
                        coord_shift_left).size()[3], env.chi))
                    Pt2 = view(Pt[(i, new_coord_l, direction)], (env.chi, stateDL.site(
                        new_coord_l).size()[1], env.chi))
                    P1 = view(P[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
                        new_coord_l).size()[3], env.chi))
                    Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
                        coord_shift_right).size()[1], env.chi))
                    nC2 = contract(C_up["left"], env.T[(
                        new_coord_l, (-1, 0))], ([0], [0]))
                    nC2 = contract(nC2, P2, ([0, 2], [0, 1]))
                    # print(nC2.abs().max())
                    C_up["left"] = nC2/nC2.abs().max()
                    # env.C[(new_coord_l,(-1,-1))] = C_up["left"]

                    vec_coord_r = (args.size+1, -args.size+i)
                    new_coord_r = state.vertexToSite(
                        (coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
                    coord_shift_left = stateDL.vertexToSite(
                        (new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
                    coord_shift_right = stateDL.vertexToSite(
                        (new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
                    P2 = view(P[(i, new_coord_r, direction)], (env.chi, stateDL.site(
                        coord_shift_left).size()[3], env.chi))
                    Pt2 = view(Pt[(i, new_coord_r, direction)], (env.chi, stateDL.site(
                        new_coord_r).size()[1], env.chi))
                    P1 = view(P[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
                        new_coord_r).size()[3], env.chi))
                    Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
                        coord_shift_right).size()[1], env.chi))
                    nC1 = contract(C_up["right"], env.T[(
                        new_coord_r, (1, 0))], ([1], [0]))
                    nC1 = contract(Pt1, nC1, ([0, 1], [0, 1]))
                    # print(nC1.abs().max())
                    C_up["right"] = nC1/nC1.abs().max()
                    # env.C[(new_coord_r,(1,-1))] = C_up["right"]

                    for j in range(2*args.size+2):
                        vec_coord = (-args.size+j, -args.size+i)
                        new_coord = state.vertexToSite(
                            (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_left = stateDL.vertexToSite(
                            (new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        coord_shift_right = stateDL.vertexToSite(
                            (new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
                            coord_shift_left).size()[3], env.chi))
                        Pt2 = view(Pt[(i, new_coord, direction)], (env.chi, stateDL.site(
                            new_coord).size()[1], env.chi))
                        P1 = view(P[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
                            new_coord).size()[3], env.chi))
                        Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
                            coord_shift_right).size()[1], env.chi))
                        if i == args.size and j == args.size:
                            nT = contract(Pt2, T_up[(j)], ([0], [0]))
                            dimsA = state.site(new_coord).size()
                            nT = view(
                                nT, (dimsA[2], dimsA[2], env.chi, dimsA[1], dimsA[1], env.chi))
                            Aket = state.site(
                                new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            nT = contract(nT, Aket, ([0, 3], [2, 1]))
                            nT = contract(
                                nT, view(P1, (env.chi, dimsA[4], dimsA[4], env.chi)), ([3, 6], [0, 1]))
                            tempT = contiguous(
                                permute(nT, (1, 3, 2, 0, 4, 5, 6)))

                            tempT2 = tempT.detach()
                            # print(tempT2.abs().max())
                            T_up[(j)] = tempT/tempT2.abs().max()
                            # env.T[(new_coord,direction)] = T_up[(j)]
                        else:
                            nT = contract(Pt2, T_up[(j)], ([0], [0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(
                                new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord)))),
                                      (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL, ([0, 2], [1, 0]))
                            nT = contract(nT, P1, ([1, 3], [0, 1]))
                            tempT = contiguous(nT)

                            tempT2 = tempT.detach()
                            # print(tempT2.abs().max())
                            T_up[(j)] = tempT/tempT2.abs().max()
                            # env.T[(new_coord,direction)] = T_up[(j)]

            elif direction == (0, 1):
                vec = (-1, 0)
                vec_coord = (-args.size, args.size+1)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_down["left"] = env.C[(new_coord, (-1, 1))].clone()
                vec_coord = (args.size+1, args.size+1)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_down["right"] = env.C[(new_coord, (1, 1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size+j, args.size+1)
                    new_coord = state.vertexToSite(
                        (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    T_down[(j)] = env.T[(new_coord, direction)].clone()

                for i in range(args.size+1):
                    vec_coord_l = (-args.size, args.size-i+1)
                    new_coord_l = state.vertexToSite(
                        (coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
                    coord_shift_right = stateDL.vertexToSite(
                        (new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
                    coord_shift_left = stateDL.vertexToSite(
                        (new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
                    P2 = view(P[(i, new_coord_l, direction)], (env.chi, stateDL.site(
                        coord_shift_right).size()[1], env.chi))
                    Pt2 = view(Pt[(i, new_coord_l, direction)], (env.chi, stateDL.site(
                        new_coord_l).size()[3], env.chi))
                    P1 = view(P[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
                        new_coord_l).size()[1], env.chi))
                    Pt1 = view(Pt[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
                        coord_shift_left).size()[3], env.chi))
                    nC1 = contract(C_down["left"], env.T[(
                        new_coord_l, (-1, 0))], ([0], [1]))
                    nC1 = contract(nC1, Pt1, ([0, 2], [0, 1]))
                    # print(nC1.abs().max())
                    C_down["left"] = nC1/nC1.abs().max()
                    # env.C[(new_coord_l,(-1,1))] = C_down["left"]

                    vec_coord_r = (args.size+1, args.size-i+1)
                    new_coord_r = state.vertexToSite(
                        (coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
                    coord_shift_right = stateDL.vertexToSite(
                        (new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
                    coord_shift_left = stateDL.vertexToSite(
                        (new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
                    P2 = view(P[(i, new_coord_r, direction)], (env.chi, stateDL.site(
                        coord_shift_right).size()[1], env.chi))
                    Pt2 = view(Pt[(i, new_coord_r, direction)], (env.chi, stateDL.site(
                        new_coord_r).size()[3], env.chi))
                    P1 = view(P[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
                        new_coord_r).size()[1], env.chi))
                    Pt1 = view(Pt[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
                        coord_shift_left).size()[3], env.chi))
                    nC2 = contract(C_down["right"], env.T[(
                        new_coord_r, (1, 0))], ([0], [2]))
                    nC2 = contract(nC2, P2, ([0, 2], [0, 1]))
                    # print(nC2.abs().max())
                    C_down["right"] = nC2/nC2.abs().max()
                    # env.C[(new_coord_r,(1,1))] = C_down["right"]

                    for j in range(2*args.size+2):
                        vec_coord = (-args.size+j, args.size-i+1)
                        new_coord = state.vertexToSite(
                            (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_right = stateDL.vertexToSite(
                            (new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        coord_shift_left = stateDL.vertexToSite(
                            (new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
                            coord_shift_right).size()[1], env.chi))
                        Pt2 = view(Pt[(i, new_coord, direction)], (env.chi, stateDL.site(
                            new_coord).size()[3], env.chi))
                        P1 = view(P[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
                            new_coord).size()[1], env.chi))
                        Pt1 = view(Pt[(i, coord_shift_left, direction)], (env.chi, stateDL.site(
                            coord_shift_left).size()[3], env.chi))
                        nT = contract(P1, T_down[(j)], ([0], [1]))
                        dimsA = state.site(new_coord).size()
                        Aket = state.site(
                            new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord)))),
                                  (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        nT = contract(nT, DL, ([0, 2], [1, 2]))
                        nT = contract(nT, Pt2, ([1, 3], [0, 1]))
                        tempT = contiguous(permute(nT, (1, 0, 2)))

                        tempT2 = tempT.detach()
                        # print(tempT2.abs().max())
                        T_down[(j)] = tempT/tempT2.abs().max()
                        # env.T[(new_coord,direction)] = T_down[(j)]

            elif direction == (-1, 0):
                vec = (0, -1)
                vec_coord = (-args.size, -args.size)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_left["up"] = env.C[(new_coord, (-1, -1))].clone()
                vec_coord = (-args.size, args.size+1)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_left["down"] = env.C[(new_coord, (-1, 1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size, -args.size+j)
                    new_coord = state.vertexToSite(
                        (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    T_left[(j)] = env.T[(new_coord, direction)].clone()

                for i in range(args.size+1):
                    vec_coord_u = (-args.size+i, -args.size)
                    new_coord_u = state.vertexToSite(
                        (coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
                    coord_shift_up = stateDL.vertexToSite(
                        (new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
                    coord_shift_down = stateDL.vertexToSite(
                        (new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
                    P2 = view(P[(i, new_coord_u, direction)], (env.chi, stateDL.site(
                        coord_shift_down).size()[0], env.chi))
                    Pt2 = view(Pt[(i, new_coord_u, direction)], (env.chi, stateDL.site(
                        new_coord_u).size()[2], env.chi))
                    P1 = view(P[(i, coord_shift_up, direction)], (env.chi,
                              stateDL.site(new_coord_u).size()[0], env.chi))
                    Pt1 = view(Pt[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
                        coord_shift_up).size()[2], env.chi))
                    nC1 = contract(C_left["up"], env.T[(
                        new_coord_u, (0, -1))], ([1], [0]))
                    nC1 = contract(Pt1, nC1, ([0, 1], [0, 1]))
                    # print(nC1.abs().max())
                    C_left["up"] = nC1/nC1.abs().max()
                    # env.C[(new_coord_u,(-1,-1))] = C_left["up"]

                    vec_coord_d = (-args.size+i, args.size+1)
                    new_coord_d = state.vertexToSite(
                        (coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
                    coord_shift_up = stateDL.vertexToSite(
                        (new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
                    coord_shift_down = stateDL.vertexToSite(
                        (new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
                    P2 = view(P[(i, new_coord_d, direction)], (env.chi, stateDL.site(
                        coord_shift_down).size()[0], env.chi))
                    Pt2 = view(Pt[(i, new_coord_d, direction)], (env.chi, stateDL.site(
                        new_coord_d).size()[2], env.chi))
                    P1 = view(P[(i, coord_shift_up, direction)], (env.chi,
                              stateDL.site(new_coord_d).size()[0], env.chi))
                    Pt1 = view(Pt[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
                        coord_shift_up).size()[2], env.chi))
                    nC2 = contract(C_left["down"], env.T[(
                        new_coord_d, (0, 1))], ([1], [1]))
                    nC2 = contract(P2, nC2, ([0, 1], [0, 1]))
                    # print(nC2.abs().max())
                    C_left["down"] = nC2/nC2.abs().max()
                    # env.C[(new_coord_d,(-1,1))] = C_left["down"]

                    for j in range(2*args.size+2):
                        vec_coord = (-args.size+i, -args.size+j)
                        new_coord = state.vertexToSite(
                            (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_up = stateDL.vertexToSite(
                            (new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        coord_shift_down = stateDL.vertexToSite(
                            (new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
                            coord_shift_down).size()[0], env.chi))
                        Pt2 = view(Pt[(i, new_coord, direction)], (env.chi, stateDL.site(
                            new_coord).size()[2], env.chi))
                        P1 = view(P[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
                            new_coord).size()[0], env.chi))
                        Pt1 = view(Pt[(i, coord_shift_up, direction)], (env.chi, stateDL.site(
                            coord_shift_up).size()[2], env.chi))
                        if i == args.size and j == args.size:
                            nT = contract(P1, T_left[(j)], ([0], [0]))
                            dimsA = state.site(new_coord).size()
                            nT = view(
                                nT, (dimsA[1], dimsA[1], env.chi, env.chi, dimsA[2], dimsA[2]))
                            Aket = state.site(
                                new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            nT = contract(nT, Aket, ([0, 4], [1, 2]))
                            nT = contract(
                                nT, view(Pt2, (env.chi, dimsA[3], dimsA[3], env.chi)), ([2, 5], [0, 1]))
                            tempT = contiguous(
                                permute(nT, (1, 3, 0, 2, 5, 4, 6)))

                            tempT2 = tempT.detach()
                            # print(tempT2.abs().max())
                            T_left[(j)] = tempT/tempT2.abs().max()
                            # env.T[(new_coord,direction)] = T_left[(j)]
                        else:
                            nT = contract(P1, T_left[(j)], ([0], [0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(
                                new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord)))),
                                      (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL, ([0, 3], [0, 1]))
                            nT = contract(nT, Pt2, ([1, 2], [0, 1]))
                            tempT = contiguous(permute(nT, (0, 2, 1)))

                            tempT2 = tempT.detach()
                            # print(tempT2.abs().max())
                            T_left[(j)] = tempT/tempT2.abs().max()
                            # env.T[(new_coord,direction)] = T_left[(j)]

            elif direction == (1, 0):
                vec = (0, 1)
                vec_coord = (args.size+1, -args.size)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_right["up"] = env.C[(new_coord, (1, -1))].clone()
                vec_coord = (args.size+1, args.size+1)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                C_right["down"] = env.C[(new_coord, (1, 1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (args.size+1, -args.size+j)
                    new_coord = state.vertexToSite(
                        (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    T_right[(j)] = env.T[(new_coord, direction)].clone()

                for i in range(args.size+1):
                    vec_coord_u = (args.size-i+1, -args.size)
                    new_coord_u = state.vertexToSite(
                        (coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
                    coord_shift_down = stateDL.vertexToSite(
                        (new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
                    coord_shift_up = stateDL.vertexToSite(
                        (new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
                    P2 = view(P[(i, new_coord_u, direction)], (env.chi,
                              stateDL.site(coord_shift_up).size()[2], env.chi))
                    Pt2 = view(Pt[(i, new_coord_u, direction)], (env.chi, stateDL.site(
                        new_coord_u).size()[0], env.chi))
                    P1 = view(P[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
                        new_coord_u).size()[2], env.chi))
                    Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
                        coord_shift_down).size()[0], env.chi))
                    nC2 = contract(C_right["up"], env.T[(
                        new_coord_u, (0, -1))], ([0], [2]))
                    nC2 = contract(nC2, P2, ([0, 2], [0, 1]))
                    # print(nC2.abs().max())
                    C_right["up"] = nC2/nC2.abs().max()
                    # env.C[(new_coord_u,(1,-1))] = C_right["up"]

                    vec_coord_d = (args.size-i+1, args.size+1)
                    new_coord_d = state.vertexToSite(
                        (coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
                    coord_shift_down = stateDL.vertexToSite(
                        (new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
                    coord_shift_up = stateDL.vertexToSite(
                        (new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
                    P2 = view(P[(i, new_coord_d, direction)], (env.chi,
                              stateDL.site(coord_shift_up).size()[2], env.chi))
                    Pt2 = view(Pt[(i, new_coord_d, direction)], (env.chi, stateDL.site(
                        new_coord_d).size()[0], env.chi))
                    P1 = view(P[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
                        new_coord_d).size()[2], env.chi))
                    Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
                        coord_shift_down).size()[0], env.chi))
                    nC1 = contract(C_right["down"], env.T[(
                        new_coord_d, (0, 1))], ([1], [2]))
                    nC1 = contract(Pt1, nC1, ([0, 1], [0, 1]))
                    # print(nC1.abs().max())
                    C_right["down"] = nC1/nC1.abs().max()
                    # env.C[(new_coord_d,(1,1))] = C_right["down"]

                    for j in range(2*args.size+2):
                        vec_coord = (args.size-i+1, -args.size+j)
                        new_coord = state.vertexToSite(
                            (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_down = stateDL.vertexToSite(
                            (new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        coord_shift_up = stateDL.vertexToSite(
                            (new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        P2 = view(P[(i, new_coord, direction)], (env.chi, stateDL.site(
                            coord_shift_up).size()[2], env.chi))
                        Pt2 = view(Pt[(i, new_coord, direction)], (env.chi, stateDL.site(
                            new_coord).size()[0], env.chi))
                        P1 = view(P[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
                            new_coord).size()[2], env.chi))
                        Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
                            coord_shift_down).size()[0], env.chi))
                        nT = contract(Pt2, T_right[(j)], ([0], [0]))
                        dimsA = state.site(new_coord).size()
                        Aket = state.site(
                            new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd', Aket, conj(state.site(new_coord)))),
                                  (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        nT = contract(nT, DL, ([0, 2], [0, 3]))
                        nT = contract(nT, P1, ([1, 3], [0, 1]))
                        tempT = contiguous(nT)

                        tempT2 = tempT.detach()
                        # print(tempT2.abs().max())
                        T_right[(j)] = tempT/tempT2.abs().max()
                        # env.T[(new_coord,direction)] = T_right[(j)]
    return C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right


def Create_Norm(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args):
    Norm = dict()
    Norm2 = dict()
    for coord in state.sites.keys():
        with torch.no_grad():
            FL = contract(C_up["left"], C_down["left"], ([0], [0]))
            print("Create_Norm", FL.abs().max())
            FL = FL/FL.abs().max()
            FU = contract(C_left["up"], C_right["up"], ([1], [0]))
            print("Create_Norm", FU.abs().max())
            FU = FU/FU.abs().max()

        for i in range(args.size):
            temp = contract(FL, T_up[(i)], ([0], [0]))
            FL = contract(temp, T_down[(i)], ([0, 1], [1, 0]))

            FL2 = FL.detach()
            print("Create_Norm", FL2.abs().max())
            FL = FL/FL2.abs().max()

            temp = contract(FU, T_left[(i)], ([0], [0]))
            FU = contract(temp, T_right[(i)], ([0, 2], [0, 1]))

            FU2 = FU.detach()
            print("Create_Norm", FU2.abs().max())
            FU = FU/FU2.abs().max()

        with torch.no_grad():
            FR = contract(C_up["right"], C_down["right"], ([1], [0]))
            print("Create_Norm", FR.abs().max())
            FR = FR/FR.abs().max()
            FD = contract(C_left["down"], C_right["down"], ([1], [1]))
            print("Create_Norm", FD.abs().max())
            FD = FD/FD.abs().max()

        for i in range(args.size+1):
            temp = contract(FR, T_up[(2*args.size+1-i)], ([0], [2]))
            FR = contract(temp, T_down[(2*args.size+1-i)], ([0, 2], [2, 0]))

            FR2 = FR.detach()
            print("Create_Norm", FR2.abs().max())
            FR = FR/FR2.abs().max()

            temp = contract(FD, T_left[(2*args.size+1-i)], ([0], [1]))
            FD = contract(temp, T_right[(2*args.size+1-i)], ([0, 2], [2, 1]))

            FD2 = FD.detach()
            print("Create_Norm", FD2.abs().max())
            FD = FD/FD2.abs().max()

        dimsA = state.site(coord).size()

        H1 = contract(FL, T_up[(args.size)], ([0], [0]))
        H1 = contract(H1, view(
            T_down[(args.size)], (dimsA[3], dimsA[3], env.chi, env.chi)), ([0, 4], [2, 0]))
        H1 = contiguous(
            permute(contract(H1, FR, ([4, 6], [0, 1])), (0, 1, 2, 4, 3)))

        H12 = H1.detach()
        print("Create_Norm", H12.abs().max())
        H1 = H1/H12.abs().max()

        H2 = contract(FU, T_left[(args.size)], ([0], [0]))
        H2 = contract(H2, view(
            T_right[(args.size)], (env.chi, dimsA[4], dimsA[4], env.chi)), ([0, 5], [0, 1]))
        H2 = contract(H2, FD, ([4, 6], [0, 1]))

        H22 = H2.detach()
        print("Create_Norm", H22.abs().max())
        H2 = H2/H22.abs().max()

        Norm[coord] = H1/2. + H2/2.
        Norm2[coord] = H1.detach()/2. + H2.detach()/2.

    return Norm

# def Create_Norm(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args):
#     Norm = dict()
#     for coord in state.sites.keys():
#         FL = contract(C_up["left"],C_down["left"],([0],[0]))
#         FL = FL/FL.abs().max()

#         for i in range(args.size):
#             temp = contract(FL,T_up[(i)],([0],[0]))
#             FL = contract(temp,T_down[(i)],([0,1],[1,0]))

#             FL2 = FL.detach()

#             FL = FL/FL2.abs().max()

#         FR = contract(C_up["right"],C_down["right"],([1],[0]))
#         FR = FR/FR.abs().max()

#         for i in range(args.size+1):
#             temp = contract(FR,T_up[(2*args.size+1-i)],([0],[2]))
#             FR = contract(temp,T_down[(2*args.size+1-i)],([0,2],[2,0]))

#             FR2 = FR.detach()

#             FR = FR/FR2.abs().max()

#         dimsA = state.site(coord).size()

#         H1 = contract(FL,T_up[(args.size)],([0],[0]))
#         H1 = contract(H1,view(T_down[(args.size)],(dimsA[3],dimsA[3],env.chi,env.chi)),([0,4],[2,0]))
#         H1 = contiguous(permute(contract(H1,FR,([4,6],[0,1])),(0,1,2,4,3)))

#         H12 = H1.detach()

#         H1 = H1/H12.abs().max()

#         Norm[coord] = H1

#     return Norm
# def Create_Norm(state, env, B_grad, lam, kx, ky, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right):
#     Norm = dict()
#     tn = None

#     for coord in state.sites.keys():
#         # T[(0,-1)]
#         tmp = env.T[coord, (0,-1)]
#         tmp = T_up[(0)]
#         tmp = contract(tmp, T_up[(1)], ([2],[0]))
#         tnD = tmp.size()
#         tmp = view(contiguous(tmp), (tnD[0], tnD[1]*tnD[2], tnD[3]))
#         tmp = contract(tmp, T_up[(2)], ([2],[0]))
#         tnD = tmp.size()
#         tmp = view(contiguous(tmp), (tnD[0], tnD[1]*tnD[2], tnD[3]))
#         env.T[coord, (0,-1)] = tmp

#         # # T[(-1,0)]
#         # tmp = env.T[coord, (-1,0)]
#         # tmp = T_left[(0)]
#         # tmp = contract(tmp, T_left[(1)], ([1],[0]))
#         # tmp = permute(tmp, (0,2,1,3))
#         # tnD = tmp.size()
#         # tmp = view(contiguous(tmp), (tnD[0], tnD[1], tnD[2]*tnD[3]))
#         # tmp = contract(tmp, T_left[(2)], ([1],[0]))
#         # tmp = permute(tmp, (0,2,1,3))
#         # tnD = tmp.size()
#         # tmp = view(contiguous(tmp), (tnD[0], tnD[2], tnD[1]*tnD[3]))
#         # env.T[coord, (-1,0)] = tmp

#         # T[(0,1)]
#         tmp = env.T[coord, (0,1)]
#         tmp = T_down[(0)]
#         tmp = contract(tmp, T_down[(1)], ([2],[1]))
#         tmp = permute(tmp, (0,2,1,3))
#         tnD = tmp.size()
#         tmp = view(contiguous(tmp), (tnD[0]*tnD[1], tnD[2], tnD[3]))
#         tmp = contract(tmp, T_down[(2)], ([2],[1]))
#         tmp = permute(tmp, (0,2,1,3))
#         tnD = tmp.size()
#         tmp = view(contiguous(tmp), (tnD[0]*tnD[1], tnD[2], tnD[3]))
#         env.T[coord, (0,1)] = tmp

#         # # T[(1,0)]
#         # tmp = env.T[coord, (1,0)]
#         # tmp = T_right[(0)]
#         # tmp = contract(tmp, T_right[(1)], ([2],[0]))
#         # tnD = tmp.size()
#         # tmp = view(contiguous(tmp), (tnD[0], tnD[1]*tnD[2], tnD[3]))
#         # tmp = contract(tmp, T_right[(2)], ([2],[0]))
#         # tnD = tmp.size()
#         # tmp = view(contiguous(tmp), (tnD[0], tnD[1]*tnD[2], tnD[3]))

#         # Construct center tensor
#         dimsA = state.site(coord).size()
#         Aket = state.site(coord) + lam * torch.exp(-1j*(kx*coord[0]+ky*coord[1])) * B_grad
#         DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(coord)))),\
#             (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
#         #      0
#         #     1DL3
#         #      2
#         Aline = contract(DL, DL, ([3],[1]))
#         #now,
#         #          0  3
#         #          |  |
#         #        1-DL-DL-5
#         #          |  |
#         #          2  4
#         Aline = contract(Aline, DL, ([5],[1]))
#         #now,
#         #          0  3  5
#         #          |  |  |
#         #        1-DL-DL-DL-7
#         #          |  |  |
#         #          2  4  6

#         Aline = permute(Aline, (0,3,5,1,2,4,6,7))
#         AlineD = Aline.size()
#         #       012
#         #     3DLDLDL7
#         #       456
#         # Aline = view(contiguous(Aline), (AlineD[0]*AlineD[1]*AlineD[2], AlineD[3], AlineD[4]*AlineD[5]*AlineD[6], AlineD[7]))

#         # # construct Ts
#         # # up T
#         # env.T[coord, (0,-1)] = contract(env.T[coord, (0,-1)], Aline, ([1],[0]))
#         # env.T[coord, (0,-1)] = permute(env.T[coord, (0,-1)], (0,2,3,1,4))
#         # TD = env.T[coord, (0,-1)].size()
#         # env.T[coord, (0,-1)] = view(contiguous(env.T[coord, (0,-1)]), (TD[0]*TD[1], TD[2], TD[3]*TD[4]))
#         # # down T
#         # env.T[coord, (0,1)] = contract(Aline, env.T[coord, (0,1)], ([2],[0]))
#         # env.T[coord, (0,1)] = permute(env.T[coord, (0,1)], (0,1,3,2,4))
#         # TD = env.T[coord, (0,1)].size()
#         # env.T[coord, (0,1)] = view(contiguous(env.T[coord, (0,1)]), (TD[0], TD[1]*TD[2], TD[3]*TD[4]))

#         tn = contract(env.C[coord, (-1,-1)], env.T[coord, (0,-1)], ([1],[0]))
#         tn = contract(tn, env.C[coord, (1,-1)], ([2],[0]))
#         # now,
#         #       CTC
#         #       012
#         tn = contract(tn, T_left[(0)], ([0],[0]))
#         # now,
#         #
#         #       CTC
#         #       |01
#         #       T-3
#         #       2
#         Alineup = view(contiguous(Aline), (AlineD[0]*AlineD[1]*AlineD[2], AlineD[3], AlineD[4],AlineD[5],AlineD[6], AlineD[7]))
#         #         0
#         #     1Alineup5
#         #        234
#         tn = contract(tn, Alineup, ([0,3],[0,1]))
#         # now,
#         #       C----T-------C
#         #       T--DLDLDL-5  0
#         #       1   234
#         tn = contract(tn, T_right[(0)], ([0,5],[0,1]))
#         # now,
#         #       C----T-----C
#         #       T--DLDLDL--T
#         #       0   123    4
#         tn = contract(tn, T_left[(1)], ([0],[0]))
#         # now,
#         #       C----T-----C
#         #       T--DLDLDL--T
#         #       |   012    |
#         #       T-5        3
#         #       4
#         tn = contract(tn, DL, ([0,5],[0,1]))
#         # now,
#         #       C----T-----C
#         #       T--DLDLDL--T
#         #       |    01    |
#         #       T- DL5     2
#         #       3  4
#         centre = state.site(coord) + lam * torch.exp(-1j*(kx*coord[0]+ky*coord[1])) * B_grad
#         tnD = tn.size()
#         tn = view(contiguous(tn), (sqrt(tnD[0]),sqrt(tnD[0]),tnD[1],tnD[2],tnD[3],tnD[4],sqrt(tnD[5]),sqrt(tnD[5])))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       01 2     |
#         #       T-DL67           3
#         #       4  5
#         tn = contract(tn, centre, ([0,6],[1,2]))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       0 1      |  phys 6
#         #       T-DL5   C8       2
#         #       3  4    7
#         DLD = DL.size()
#         DLtmp = view(contiguous(DL), (DLD[0],sqrt(DLD[1]),sqrt(DLD[1]),DLD[2],DLD[3]))
#         # nowDLtmp,
#         #         0
#         #      12DLtmp4
#         #         3
#         tn = contract(tn, DLtmp, ([1,8],[0,1]))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       0        1  phys 5
#         #       T-DL4   C  7DL9
#         #       2  3    6    8
#         tn = contract(tn, T_right[(1)], ([1,9],[0,1]))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       0        |  phys 4
#         #       T-DL3   C    6DL-T
#         #       1  2    5      7 8
#         tn = contract(tn, T_left[(2)], ([1],[0]))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       0        |  phys 3
#         #       T-DL2   C    5DL-T
#         #       |  1    4      6 7
#         #       T9
#         #       8
#         Alinedn = view(contiguous(Aline), (AlineD[0],sqrt(AlineD[1]),sqrt(AlineD[1]),AlineD[2], AlineD[3], AlineD[4]*AlineD[5]*AlineD[6], AlineD[7]))
#         #       0 12 3
#         #     4Alinedn6
#         #         5
#         tn = contract(tn, Alinedn, ([1,4,6,9],[0,1,3,4]))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       0        |  phys 2
#         #       T-DL1   C    3DL-T
#         #       |       6        4
#         #       T-----DLDLDL-8
#         #       5       7
#         tn = contract(tn, T_right[(2)], ([4,8],[0,1]))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       0        |  phys 2
#         #       T-DL1   C    3DL-T
#         #       |       5        |
#         #       T-----DLDLDL-----T
#         #       4       6        7
#         tn = contract(tn, env.C[coord, (-1,1)], ([4],[0]))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       0        |  phys 2
#         #       T-DL1   C    3DL-T
#         #       |       4        |
#         #       T-----DLDLDL-----T
#         #       C7      5        6
#         tn = contract(tn, env.T[coord, (0,1)], ([7,5],[1,0]))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       0        |  phys 2
#         #       T-DL1   C    3DL-T
#         #       |       4        |
#         #       T-----DLDLDL-----T
#         #       C-------T-6      5
#         tn = contract(tn, env.C[coord, (1,1)], ([5,6],[0,1]))
#         # now,
#         #       C-------T--------C
#         #       T-----DLDLDL-----T
#         #       |       0        |  phys 2
#         #       T-DL1   C    3DL-T
#         #       |       4        |
#         #       T-----DLDLDL-----T
#         #       C-------T--------C
#         tn = contiguous(permute(2,0,1,4,3))
#         Norm[coord] = tn
#     return Norm
