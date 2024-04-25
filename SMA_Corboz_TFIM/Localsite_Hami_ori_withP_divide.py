import context
import time
import torch
import argparse
import config as cfg
from tn_interface import contract, einsum
from tn_interface import conj
from tn_interface import contiguous, view, permute
from math import sqrt
from ctm.generic.ctm_projectors import *

# You need to pass in H as iden+mu*H


def Create_Localsite_Hami_Env(state, stateDL, B_grad, env, lam, Hx, Hy, Honsite, Id, kx, ky,
                              C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args, firsttime, lasttime, P, Pt, isOnsiteWorking, MultiGPU=False):
    # if cfg.ctm_args.projector_method=='4X4':
    #     ctm_get_projectors=ctm_get_projectors_4x4
    # elif cfg.ctm_args.projector_method=='4X2':
    #     ctm_get_projectors=ctm_get_projectors_4x2
    # else:
    #     raise ValueError("Invalid Projector method: "+str(cfg.ctm_args.projector_method))

    # HyAndOnsiteFst = contiguous(contract(Honsite, Hy, ([1],[0])))
    # HyAndOnsiteLst = contiguous(contract(Hy, Honsite, ([3],[0])))
    OI = torch.einsum('ij,ab->iajb', Honsite, Id)
    IO = torch.einsum('ij,ab->iajb', Id, Honsite)

    HyAndOnsiteFst = Hy
    HyAndOnsiteLst = Hy
    if isOnsiteWorking:
        HyAndOnsiteFst = Hy + OI
        HyAndOnsiteLst = Hy + IO

    phys_dim = state.site((0, 0)).size()[0]
    for coord in stateDL.sites.keys():
        for direction in cfg.ctm_args.ctm_move_sequence:
            if direction == (0, -1):
                local_device = cfg.global_args.device
                if (MultiGPU):
                    local_device = torch.device('cuda:0')
                    lam = lam.to(local_device)
                    B_grad = B_grad.to(local_device)
                    kx = kx.to(local_device)
                    ky = ky.to(local_device)
                    HyAndOnsiteFst = HyAndOnsiteFst.to(local_device)
                    HyAndOnsiteLst = HyAndOnsiteLst.to(local_device)
                    Hx = Hx.to(local_device)
                    Hy = Hy.to(local_device)
                    for key, site in state.sites.items():
                        state.sites[key] = site.to(local_device)
                    state.device = local_device
                    for key, site in stateDL.sites.items():
                        stateDL.sites[key] = site.to(local_device)
                    stateDL.device = local_device
                    for key, site in env.C.items():
                        env.C[key] = site.to(local_device)
                    for key, site in env.T.items():
                        env.T[key] = site.to(local_device)
                    for key, site in P.items():
                        P[key] = site.to(local_device)
                    for key, site in Pt.items():
                        Pt[key] = site.to(local_device)
                vec = (1, 0)
                vec_coord = (-args.size, -args.size)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                # if "left" not in C_up:
                C_up["left"] = env.C[(new_coord, (-1, -1))].clone()
                vec_coord = (args.size+1, -args.size)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                # if "right" not in C_up:
                C_up["right"] = env.C[(new_coord, (1, -1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size+j, -args.size)
                    new_coord = state.vertexToSite(
                        (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    # if (j) not in T_up:
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
                    print(nC2.abs().max())
                    C_up["left"] = nC2
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
                    print(nC1.abs().max())
                    C_up["right"] = nC1
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
                        # Guess
                        #           |---0 chi
                        #  2 chi---Pt2
                        #           |---1
                        Pt2 = view(Pt[(i, new_coord, direction)], (env.chi, stateDL.site(
                            new_coord).size()[1], env.chi))
                        # Guess
                        # 0 chi-----|
                        #           P1--2 chi
                        # 1---------|
                        P1 = view(P[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
                            new_coord).size()[3], env.chi))
                        Pt1 = view(Pt[(i, coord_shift_right, direction)], (env.chi, stateDL.site(
                            coord_shift_right).size()[1], env.chi))
                        # if j % 2 == 0:
                        if i == args.size and j == args.size and lasttime:
                            nT = contract(Pt2, T_up[(j)], ([0], [0]))
                            dimsA = state.site(new_coord).size()
                            nT = view(
                                nT, (dimsA[2], dimsA[2], env.chi, phys_dim, phys_dim, dimsA[1], dimsA[1], env.chi))
                            Aket = state.site(
                                new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            nT = contract(nT, Aket, ([0, 5], [2, 1]))
                            nT = contract(
                                nT, view(P1, (env.chi, dimsA[4], dimsA[4], env.chi)), ([5, 8], [0, 1]))
                            # now nT=
                            #        |------T-----|        phy_in_T 2,3
                            # chi 1-Pt2     4     P1-8 chi phy_in_A 5
                            #        |---0  A  7--|
                            #               6
                            if i % 2 == 1:
                                nT = contract(nT, permute(
                                    HyAndOnsiteLst, (1, 0, 3, 2)), ([2, 3, 5], [0, 2, 1]))
                                tempT = contiguous(
                                    permute(nT, (1, 6, 2, 0, 3, 4, 5)))
                            else:
                                nT = contract(
                                    nT, HyAndOnsiteFst, ([2, 3], [0, 2]))
                                tempT = contiguous(
                                    permute(nT, (1, 3, 7, 8, 2, 0, 4, 5, 6)))

                            norm = tempT.detach()
                            # if i%2==1:
                            # tempT=
                            #        |------T-----|
                            # chi 0-Pt2     2     P1-6 chi phy_in_A 1
                            #        |---3  A  5--|
                            #               4
                            # else:
                            # tempT=
                            #        |------T-----|        phy_in_T 2,3
                            # chi 0-Pt2     4     P1-8 chi phy_in_A 1
                            #        |---5  A  7--|
                            #               6
                            print(norm.abs().max())
                            T_up[(j)] = tempT
                        else:
                            if i == 0 and firsttime:
                                # Guess nT=
                                #           |------------T3
                                #  1 chi---Pt2           2
                                #           |---0
                                nT = contract(Pt2, T_up[(j)], ([0], [0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(
                                    new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                                # Guess DL=
                                #           2
                                #   3   A conj(A)  5         phy 0,1
                                #           4
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL, ([0, 2], [3, 2]))
                                # Guess now
                                #           |-----T1
                                #  0 chi---Pt2    |               phy_in_DL 2,3
                                #           |-----DL5
                                #                 4
                                nT = contract(nT, P1, ([1, 5], [0, 1]))
                                # Guess now
                                #           |-----T-----|
                                #  0 chi---Pt2    |     P1---chi 4          phy_in_DL 1,2
                                #           |-----DL----|
                                #                 3

                                tempT = contiguous(nT)

                                norm = tempT.detach()
                                print(norm.abs().max())
                                T_up[(j)] = tempT
                                # Guess now
                                #    0---T--4
                                #        3      phy 1,2

                            else:
                                # Guess now
                                #    0--T--4          phy 1,2
                                #       3
                                nT = contract(Pt2, T_up[(j)], ([0], [0]))
                                # Guess now nT=
                                #           |-----T5
                                #  1 chi---Pt2    4    phy 2,3
                                #           |---0
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(
                                    new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                # Guess now DL=
                                #           2
                                #   3   A conj(A)  5         phy 0,1
                                #           4
                                nT = contract(nT, DL, ([0, 4], [3, 2]))
                                # Guess now
                                #           |-----T3
                                #  0 chi---Pt2    |               phy_in_T 1,2
                                #           |-----DL7             phy_in_DL 4,5
                                #                 6
                                nT = contract(nT, P1, ([3, 7], [0, 1]))
                                # Guess now
                                #           |-----T-----|
                                #  0 chi---Pt2    |     P1---chi 6    phy_in_T 1,2
                                #           |-----DL----|             phy_in_DL 3,4
                                #                 5

                                # Guess now H=
                                #         0  1
                                #         h1 h2
                                #         2  3
                                # now HyAndOnsite=
                                #         0  1
                                #         h1 h2
                                #         2  3
                                if i % 2 == 1:
                                    nT = contract(nT, permute(
                                        HyAndOnsiteLst, (1, 0, 3, 2)), ([1, 2, 3], [0, 2, 1]))
                                    # Guess now
                                    #           |-----T-----|
                                    #  0 chi---Pt2    |     P1---chi 3
                                    #           |-----DL----|             phy_in_DL 4,1
                                    #                 2
                                    tempT = contiguous(
                                        permute(nT, (0, 4, 1, 2, 3)))
                                    # Guess now
                                    #           |-----T-----|
                                    #  0 chi---Pt2    |     P1---chi 4
                                    #           |-----DL----|             phy_in_DL 1,2
                                    #                 3
                                else:
                                    nT = contract(
                                        nT, HyAndOnsiteFst, ([1, 2, 4], [0, 2, 3]))
                                    # Guess now
                                    #           |-----T-----|
                                    #  0 chi---Pt2    |     P1---chi 3
                                    #           |-----DL----|             phy_in_DL 1,4
                                    #                 2
                                    tempT = contiguous(
                                        permute(nT, (0, 1, 4, 2, 3)))
                                    # Guess now
                                    #           |-----T-----|
                                    #  0 chi---Pt2    |     P1---chi 4
                                    #           |-----DL----|             phy_in_DL 1,2
                                    #                 3

                                norm = tempT.detach()
                                print(norm.abs().max())
                                T_up[(j)] = tempT

                        # else:
                        #     if i == args.size and j == args.size and lasttime:
                        #         # Guess now
                        #         #    0--T--4          phy 1,2
                        #         #       3
                        #         nT = contract(Pt2, T_up[(j)], ([0], [0]))
                        #         # Guess now nT=
                        #         #           |-----T5
                        #         #  1 chi---Pt2    4    phy 2,3
                        #         #           |---0
                        #         dimsA = state.site(new_coord).size()
                        #         nT = view(
                        #             nT, (dimsA[2], dimsA[2], env.chi, phys_dim, phys_dim, dimsA[1], dimsA[1], env.chi))
                        #         # Guess now nT=
                        #         #           |-----T---7 chi
                        #         #  2 chi---Pt2    5,6    phy 3,4
                        #         #           |---0,1
                        #         Aket = state.site(
                        #             new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        #         # Guess now Aket=
                        #         #      1
                        #         #   2  A  4    phy 0
                        #         #      3
                        #         nT = contract(nT, Aket, ([0, 5], [2, 1]))
                        #         # Guess now nT=
                        #         #           |--------T---5 chi
                        #         #  1 chi---Pt2       4           phy_in_T 2,3
                        #         #           |---0   Aket-8       phy_in_Aket 6
                        #         #                    7
                        #         nT = contract(
                        #             nT, view(P1, (env.chi, dimsA[4], dimsA[4], env.chi)), ([5, 8], [0, 1]))
                        #         # Guess now nT=
                        #         #           |--------T--------|
                        #         #  1 chi---Pt2       4       P1-8 chi    phy_in_T 2,3
                        #         #           |---0   Aket  7---|          phy_in_Aket 5
                        #         #                    6
                        #         if i % 2 == 1:
                        #             nT = contract(
                        #                 nT, HyAndOnsiteFst, ([2, 3, 5], [0, 2, 1]))
                        #             tempT = contiguous(
                        #                 permute(nT, (1, 6, 2, 0, 3, 4, 5)))
                        #             # ok
                        #         else:
                        #             nT = contract(nT, permute(
                        #                 HyAndOnsiteLst, (1, 0, 3, 2)), ([2, 3], [0, 2]))
                        #             tempT = contiguous(
                        #                 permute(nT, (1, 3, 7, 8, 2, 0, 4, 5, 6)))

                        #         norm = tempT.detach()
                        #         print(norm.abs().max())
                        #         T_up[(j)] = tempT
                        #     else:
                        #         if i == 0 and firsttime:
                        #             # Guess now
                        #             #    0---T--2
                        #             #        1
                        #             nT = contract(Pt2, T_up[(j)], ([0], [0]))
                        #             # Guess nT=
                        #             #           |-----T3
                        #             #  1 chi---Pt2    2
                        #             #           |---0
                        #             dimsA = state.site(new_coord).size()
                        #             Aket = state.site(
                        #                 new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        #             DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                        #                       (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        #             # Guess now
                        #             #           2
                        #             #   3   A conj(A)  5         phy 0,1
                        #             #           4
                        #             nT = contract(nT, DL, ([0, 2], [3, 2]))
                        #             # Guess now
                        #             #           |-----T1
                        #             #  0 chi---Pt2    |               phy_in_DL 2,3
                        #             #           |-----DL5
                        #             #                 4
                        #             nT = contract(nT, P1, ([1, 5], [0, 1]))
                        #             # Guess now
                        #             #           |-----T-----|
                        #             #  0 chi---Pt2    |     P1---chi 4          phy_in_DL 1,2
                        #             #           |-----DL----|
                        #             #                 3
                        #             tempT = contiguous(nT)

                        #             norm = tempT.detach()
                        #             print(norm.abs().max())
                        #             T_up[(j)] = tempT
                        #             # Guess now
                        #             #    0--T--4          phy 1,2
                        #             #       3

                        #         else:
                        #             nT = contract(Pt2, T_up[(j)], ([0], [0]))
                        #             dimsA = state.site(new_coord).size()
                        #             Aket = state.site(
                        #                 new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        #             DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                        #                       (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        #             nT = contract(nT, DL, ([0, 4], [3, 2]))
                        #             nT = contract(nT, P1, ([3, 7], [0, 1]))
                        #             if i % 2 == 1:
                        #                 nT = contract(
                        #                     nT, HyAndOnsiteFst, ([1, 2, 3], [0, 2, 1]))
                        #                 tempT = contiguous(
                        #                     permute(nT, (0, 4, 1, 2, 3)))
                        #                 # checked this block
                        #             else:
                        #                 nT = contract(nT, permute(
                        #                     HyAndOnsiteLst, (1, 0, 3, 2)), ([1, 2, 4], [0, 2, 3]))
                        #                 tempT = contiguous(
                        #                     permute(nT, (0, 1, 4, 2, 3)))

                        #             norm = tempT.detach()
                        #             print(norm.abs().max())
                        #             T_up[(j)] = tempT

            elif direction == (0, 1):
                #     3
                #  0--T--4        phy 1,2
                local_device = cfg.global_args.device
                if (MultiGPU):
                    local_device = torch.device('cuda:1')
                    lam = lam.to(local_device)
                    B_grad = B_grad.to(local_device)
                    kx = kx.to(local_device)
                    ky = ky.to(local_device)
                    HyAndOnsiteFst = HyAndOnsiteFst.to(local_device)
                    HyAndOnsiteLst = HyAndOnsiteLst.to(local_device)
                    Hx = Hx.to(local_device)
                    Hy = Hy.to(local_device)
                    for key, site in state.sites.items():
                        state.sites[key] = site.to(local_device)
                    state.device = local_device
                    for key, site in stateDL.sites.items():
                        stateDL.sites[key] = site.to(local_device)
                    stateDL.device = local_device
                    for key, site in env.C.items():
                        env.C[key] = site.to(local_device)
                    for key, site in env.T.items():
                        env.T[key] = site.to(local_device)
                    for key, site in P.items():
                        P[key] = site.to(local_device)
                    for key, site in Pt.items():
                        Pt[key] = site.to(local_device)
                vec = (-1, 0)
                vec_coord = (-args.size, args.size+1)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                # if "left" not in C_down:
                C_down["left"] = env.C[(new_coord, (-1, 1))].clone()
                vec_coord = (args.size+1, args.size+1)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                # if "right" not in C_down:
                C_down["right"] = env.C[(new_coord, (1, 1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size+j, args.size+1)
                    new_coord = state.vertexToSite(
                        (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    # if (j) not in T_down:
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
                    print(nC1.abs().max())
                    C_down["left"] = nC1
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
                    print(nC2.abs().max())
                    C_down["right"] = nC2
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
                        # if j % 2 == 0:
                        if i == 0 and firsttime:
                            nT = contract(P1, T_down[(j)], ([0], [1]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(
                                new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                      (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL, ([0, 2], [3, 4]))
                            nT = contract(nT, Pt2, ([1, 5], [0, 1]))
                            # contiguous(permute(nT, (1,0,2)))
                            tempT = contiguous(nT)

                            norm = tempT.detach()
                            print(norm.abs().max())
                            T_down[(j)] = tempT
                            #         3
                            #      |--DL-|
                            #    0-|--T--|-4    phy 1,2
                        else:
                            nT = contract(P1, T_down[(j)], ([0], [0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(
                                new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                      (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL, ([0, 4], [3, 4]))
                            nT = contract(nT, Pt2, ([3, 7], [0, 1]))
                            if i % 2 == 1:
                                nT = contract(
                                    nT, HyAndOnsiteFst, ([1, 2, 3], [0, 2, 1]))
                                tempT = contiguous(
                                    permute(nT, (0, 4, 1, 2, 3)))
                                # checked this block
                            else:
                                nT = contract(nT, permute(
                                    HyAndOnsiteLst, (1, 0, 3, 2)), ([1, 2, 4], [0, 2, 3]))
                                tempT = contiguous(
                                    permute(nT, (0, 1, 4, 2, 3)))

                            norm = tempT.detach()
                            print(norm.abs().max())
                            T_down[(j)] = tempT

                        # else:
                        #     if i == 0 and firsttime:
                        #         nT = contract(P1, T_down[(j)], ([0], [1]))
                        #         dimsA = state.site(new_coord).size()
                        #         Aket = state.site(
                        #             new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        #         DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                        #                   (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        #         nT = contract(nT, DL, ([0, 2], [3, 4]))
                        #         nT = contract(nT, Pt2, ([1, 5], [0, 1]))
                        #         # contiguous(permute(nT, (1,0,2)))
                        #         tempT = contiguous(nT)

                        #         norm = tempT.detach()
                        #         print(norm.abs().max())
                        #         T_down[(j)] = tempT
                        #     else:
                        #         nT = contract(P1, T_down[(j)], ([0], [0]))
                        #         dimsA = state.site(new_coord).size()
                        #         Aket = state.site(
                        #             new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        #         DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                        #                   (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        #         nT = contract(nT, DL, ([0, 4], [3, 4]))
                        #         nT = contract(nT, Pt2, ([3, 7], [0, 1]))
                        #         if i % 2 == 1:
                        #             nT = contract(nT, permute(
                        #                 HyAndOnsiteLst, (1, 0, 3, 2)), ([1, 2, 3], [0, 2, 1]))
                        #             tempT = contiguous(
                        #                 permute(nT, (0, 4, 1, 2, 3)))
                        #         else:
                        #             nT = contract(
                        #                 nT, HyAndOnsiteFst, ([1, 2, 4], [0, 2, 3]))
                        #             tempT = contiguous(
                        #                 permute(nT, (0, 1, 4, 2, 3)))

                        #         norm = tempT.detach()
                        #         print(norm.abs().max())
                        #         T_down[(j)] = tempT
            elif direction == (-1, 0):
                local_device = cfg.global_args.device
                if (MultiGPU):
                    local_device = torch.device('cuda:2')
                    lam = lam.to(local_device)
                    B_grad = B_grad.to(local_device)
                    kx = kx.to(local_device)
                    ky = ky.to(local_device)
                    HyAndOnsiteFst = HyAndOnsiteFst.to(local_device)
                    HyAndOnsiteLst = HyAndOnsiteLst.to(local_device)
                    Hx = Hx.to(local_device)
                    Hy = Hy.to(local_device)
                    for key, site in state.sites.items():
                        state.sites[key] = site.to(local_device)
                    state.device = local_device
                    for key, site in stateDL.sites.items():
                        stateDL.sites[key] = site.to(local_device)
                    stateDL.device = local_device
                    for key, site in env.C.items():
                        env.C[key] = site.to(local_device)
                    for key, site in env.T.items():
                        env.T[key] = site.to(local_device)
                    for key, site in P.items():
                        P[key] = site.to(local_device)
                    for key, site in Pt.items():
                        Pt[key] = site.to(local_device)
                vec = (0, -1)
                vec_coord = (-args.size, -args.size)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                # if "up" not in C_left:
                C_left["up"] = env.C[(new_coord, (-1, -1))].clone()
                vec_coord = (-args.size, args.size+1)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                # if "down" not in C_left:
                C_left["down"] = env.C[(new_coord, (-1, 1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size, -args.size+j)
                    new_coord = state.vertexToSite(
                        (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    # if (j) not in T_left:
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
                    print(nC1.abs().max())
                    C_left["up"] = nC1
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
                    print(nC2.abs().max())
                    C_left["down"] = nC2
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
                        # if j % 2 == 0:
                        if i == args.size and j == args.size and lasttime:
                            nT = contract(P1, T_left[(j)], ([0], [0]))
                            dimsA = state.site(new_coord).size()
                            nT = view(
                                nT, (dimsA[1], dimsA[1], env.chi, phys_dim, phys_dim, dimsA[2], dimsA[2], env.chi))
                            Aket = state.site(
                                new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            nT = contract(nT, Aket, ([0, 5], [1, 2]))
                            nT = contract(
                                nT, view(Pt2, (env.chi, dimsA[3], dimsA[3], env.chi)), ([5, 7], [0, 1]))
                            if i % 2 == 1:
                                nT = contract(nT, permute(
                                    Hx, (1, 0, 3, 2)), ([2, 3, 5], [0, 2, 1]))
                                tempT = contiguous(
                                    permute(nT, (1, 6, 0, 2, 4, 3, 5)))
                            else:
                                nT = contract(nT, Hx, ([2, 3], [0, 2]))
                                tempT = contiguous(
                                    permute(nT, (1, 3, 7, 8, 0, 2, 5, 4, 6)))

                            norm = tempT.detach()
                            print(norm.abs().max())
                            T_left[(j)] = tempT
                        else:
                            if i == 0 and firsttime:
                                nT = contract(P1, T_left[(j)], ([0], [0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(
                                    new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL, ([0, 3], [2, 3]))
                                nT = contract(nT, Pt2, ([1, 4], [0, 1]))
                                # contiguous(permute(nT, (0,2,1)))
                                tempT = contiguous(nT)

                                norm = tempT.detach()
                                print(norm.abs().max())
                                T_left[(j)] = tempT
                                #      0
                                #    T  DL3   phy 1,2
                                #      4
                            else:
                                nT = contract(P1, T_left[(j)], ([0], [0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(
                                    new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL, ([0, 4], [2, 3]))
                                nT = contract(nT, Pt2, ([3, 6], [0, 1]))
                                if i % 2 == 1:
                                    nT = contract(nT, permute(
                                        Hx, (1, 0, 3, 2)), ([1, 2, 3], [0, 2, 1]))
                                    tempT = contiguous(
                                        permute(nT, (0, 4, 1, 2, 3)))
                                else:
                                    nT = contract(
                                        nT, Hx, ([1, 2, 4], [0, 2, 3]))
                                    tempT = contiguous(
                                        permute(nT, (0, 1, 4, 2, 3)))

                                norm = tempT.detach()
                                print(norm.abs().max())
                                T_left[(j)] = tempT

                        # else:
                        #     if i == args.size and j == args.size and lasttime:
                        #         nT = contract(P1, T_left[(j)], ([0], [0]))
                        #         dimsA = state.site(new_coord).size()
                        #         nT = view(
                        #             nT, (dimsA[1], dimsA[1], env.chi, phys_dim, phys_dim, dimsA[2], dimsA[2], env.chi))
                        #         Aket = state.site(
                        #             new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        #         nT = contract(nT, Aket, ([0, 5], [1, 2]))
                        #         nT = contract(
                        #             nT, view(Pt2, (env.chi, dimsA[3], dimsA[3], env.chi)), ([5, 7], [0, 1]))
                        #         if i % 2 == 1:
                        #             nT = contract(
                        #                 nT, Hx, ([2, 3, 5], [0, 2, 1]))
                        #             tempT = contiguous(
                        #                 permute(nT, (1, 6, 0, 2, 4, 3, 5)))
                        #         else:
                        #             nT = contract(nT, permute(
                        #                 Hx, (1, 0, 3, 2)), ([2, 3], [0, 2]))
                        #             tempT = contiguous(
                        #                 permute(nT, (1, 3, 7, 8, 0, 2, 5, 4, 6)))

                        #         norm = tempT.detach()
                        #         print(norm.abs().max())
                        #         T_left[(j)] = tempT
                        #     else:
                        #         if i == 0 and firsttime:
                        #             nT = contract(P1, T_left[(j)], ([0], [0]))
                        #             dimsA = state.site(new_coord).size()
                        #             Aket = state.site(
                        #                 new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        #             DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                        #                       (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        #             nT = contract(nT, DL, ([0, 3], [2, 3]))
                        #             nT = contract(nT, Pt2, ([1, 4], [0, 1]))
                        #             # contiguous(permute(nT, (0,2,1)))
                        #             tempT = contiguous(nT)

                        #             norm = tempT.detach()
                        #             print(norm.abs().max())
                        #             T_left[(j)] = tempT
                        #         else:
                        #             nT = contract(P1, T_left[(j)], ([0], [0]))
                        #             dimsA = state.site(new_coord).size()
                        #             Aket = state.site(
                        #                 new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        #             DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                        #                       (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        #             nT = contract(nT, DL, ([0, 4], [2, 3]))
                        #             nT = contract(nT, Pt2, ([3, 6], [0, 1]))
                        #             if i % 2 == 1:
                        #                 nT = contract(
                        #                     nT, Hx, ([1, 2, 3], [0, 2, 1]))
                        #                 tempT = contiguous(
                        #                     permute(nT, (0, 4, 1, 2, 3)))
                        #             else:
                        #                 nT = contract(nT, permute(
                        #                     Hx, (1, 0, 3, 2)), ([1, 2, 4], [0, 2, 3]))
                        #                 tempT = contiguous(
                        #                     permute(nT, (0, 1, 4, 2, 3)))

                        #             norm = tempT.detach()
                        #             print(norm.abs().max())
                        #             T_left[(j)] = tempT

            elif direction == (1, 0):
                if (MultiGPU):
                    local_device = torch.device('cuda:3')
                    lam = lam.to(local_device)
                    B_grad = B_grad.to(local_device)
                    kx = kx.to(local_device)
                    ky = ky.to(local_device)
                    HyAndOnsiteFst = HyAndOnsiteFst.to(local_device)
                    HyAndOnsiteLst = HyAndOnsiteLst.to(local_device)
                    Hx = Hx.to(local_device)
                    Hy = Hy.to(local_device)
                    for key, site in state.sites.items():
                        state.sites[key] = site.to(local_device)
                    state.device = local_device
                    for key, site in stateDL.sites.items():
                        stateDL.sites[key] = site.to(local_device)
                    stateDL.device = local_device
                    for key, site in env.C.items():
                        env.C[key] = site.to(local_device)
                    for key, site in env.T.items():
                        env.T[key] = site.to(local_device)
                    for key, site in P.items():
                        P[key] = site.to(local_device)
                    for key, site in Pt.items():
                        Pt[key] = site.to(local_device)
                vec = (0, 1)
                vec_coord = (args.size+1, -args.size)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                # if "up" not in C_right:
                C_right["up"] = env.C[(new_coord, (1, -1))].clone()
                vec_coord = (args.size+1, args.size+1)
                new_coord = state.vertexToSite(
                    (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                # if "down" not in C_right:
                C_right["down"] = env.C[(new_coord, (1, 1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (args.size+1, -args.size+j)
                    new_coord = state.vertexToSite(
                        (coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    # if (j) not in T_right:
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
                    print(nC2.abs().max())
                    C_right["up"] = nC2
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
                    print(nC1.abs().max())
                    C_right["down"] = nC1
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
                        # Guess, Pt2=
                        #       2 chi
                        #        |
                        #     |--Pt2--|
                        #     1        0 chi
                        P1 = view(P[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
                            new_coord).size()[2], env.chi))
                        Pt1 = view(Pt[(i, coord_shift_down, direction)], (env.chi, stateDL.site(
                            coord_shift_down).size()[0], env.chi))
                        if j % 2 == 0:
                            if i == 0 and firsttime:
                                #    0
                                # 1--T
                                #    2
                                nT = contract(Pt2, T_right[(j)], ([0], [0]))
                                # now,
                                #       1 chi
                                #        |
                                #     |--Pt2--|
                                #     0       |
                                #          2--T
                                #             3
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(
                                    new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                # now DL=
                                #           2
                                #   3   A conj(A)  5         phy 0,1
                                #           4
                                nT = contract(nT, DL, ([0, 2], [2, 5]))
                                # now,
                                #       0 chi
                                #        |
                                #     |--Pt2--|
                                #     |       |          phy 2,3
                                #   4-DL------T
                                #     5       1
                                nT = contract(nT, P1, ([1, 5], [0, 1]))
                                # now,
                                #       0 chi
                                #        |
                                #     |--Pt2--|
                                #     |       |          phy 1,2
                                #   3-DL------T
                                #     |       |
                                #     |---P1--|
                                #         4
                                tempT = contiguous(nT)

                                norm = tempT.detach()
                                print(norm.abs().max())
                                T_right[(j)] = tempT
                            else:
                                nT = contract(Pt2, T_right[(j)], ([0], [0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(
                                    new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL, ([0, 4], [2, 5]))
                                nT = contract(nT, P1, ([3, 7], [0, 1]))
                                if i % 2 == 1:
                                    nT = contract(
                                        nT, Hx, ([1, 2, 3], [0, 2, 1]))
                                    tempT = contiguous(
                                        permute(nT, (0, 4, 1, 2, 3)))
                                else:
                                    nT = contract(nT, permute(
                                        Hx, (1, 0, 3, 2)), ([1, 2, 4], [0, 2, 3]))
                                    tempT = contiguous(
                                        permute(nT, (0, 1, 4, 2, 3)))

                                norm = tempT.detach()
                                print(norm.abs().max())
                                T_right[(j)] = tempT

                        else:
                            if i == 0 and firsttime:
                                nT = contract(Pt2, T_right[(j)], ([0], [0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(
                                    new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL, ([0, 2], [2, 5]))
                                nT = contract(nT, P1, ([1, 5], [0, 1]))
                                tempT = contiguous(nT)

                                norm = tempT.detach()
                                print(norm.abs().max())
                                T_right[(j)] = tempT
                            else:
                                nT = contract(Pt2, T_right[(j)], ([0], [0]))
                                dimsA = state.site(new_coord).size()
                                Aket = state.site(
                                    new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                                DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd', Aket, conj(state.site(new_coord)))),
                                          (phys_dim, phys_dim, dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                                nT = contract(nT, DL, ([0, 4], [2, 5]))
                                nT = contract(nT, P1, ([3, 7], [0, 1]))
                                if i % 2 == 1:
                                    nT = contract(nT, permute(
                                        Hx, (1, 0, 3, 2)), ([1, 2, 3], [0, 2, 1]))
                                    tempT = contiguous(
                                        permute(nT, (0, 4, 1, 2, 3)))
                                else:
                                    nT = contract(
                                        nT, Hx, ([1, 2, 4], [0, 2, 3]))
                                    tempT = contiguous(
                                        permute(nT, (0, 1, 4, 2, 3)))

                                norm = tempT.detach()
                                print(norm.abs().max())
                                T_right[(j)] = tempT
                                # checked this block

    lam = lam.to(cfg.global_args.device)
    B_grad = B_grad.to(cfg.global_args.device)
    kx = kx.to(cfg.global_args.device)
    ky = ky.to(cfg.global_args.device)
    Hx = Hx.to(cfg.global_args.device)
    Hy = Hy.to(cfg.global_args.device)
    Honsite = Honsite.to(cfg.global_args.device)
    for key, site in state.sites.items():
        state.sites[key] = site.to(cfg.global_args.device)
    state.device = cfg.global_args.device
    for key, site in stateDL.sites.items():
        stateDL.sites[key] = site.to(cfg.global_args.device)
    stateDL.device = cfg.global_args.device
    for key, site in P.items():
        P[key] = site.to(cfg.global_args.device)
    for key, site in Pt.items():
        Pt[key] = site.to(cfg.global_args.device)
    for key, site in env.C.items():
        env.C[key] = site.to(cfg.global_args.device)
    for key, site in env.T.items():
        env.T[key] = site.to(cfg.global_args.device)
    for key, C in C_up.items():
        C_up[key] = C.to(cfg.global_args.device)
    for key, C in C_down.items():
        C_down[key] = C.to(cfg.global_args.device)
    for key, C in C_left.items():
        C_left[key] = C.to(cfg.global_args.device)
    for key, C in C_right.items():
        C_right[key] = C.to(cfg.global_args.device)
    for key, T in T_up.items():
        T_up[key] = T.to(cfg.global_args.device)
    for key, T in T_down.items():
        T_down[key] = T.to(cfg.global_args.device)
    for key, T in T_left.items():
        T_left[key] = T.to(cfg.global_args.device)
    for key, T in T_right.items():
        T_right[key] = T.to(cfg.global_args.device)

    return C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right


def Create_Localsite_Hami(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, Hx, Hy, Honsite, Id, args, isOnsiteWorking):
    phys_dim = state.site((0, 0)).size()[0]
    Hami = dict()
    OI = torch.einsum('ij,ab->iajb', Honsite, Id)
    IO = torch.einsum('ij,ab->iajb', Id, Honsite)

    HyAndOnsiteFst = Hy
    HyAndOnsiteLst = Hy
    HyAndOnsiteBoth = Hy
    HxAndOnsiteFst = Hx
    HxAndOnsiteLst = Hx
    HxAndOnsiteBoth = Hx
    if isOnsiteWorking:
        HyAndOnsiteFst = Hy + OI
        HyAndOnsiteLst = Hy + IO
        HyAndOnsiteBoth = Hy + OI + IO
        HxAndOnsiteFst = Hx + OI
        HxAndOnsiteLst = Hx + IO
        HxAndOnsiteBoth = Hx + OI + IO
    #    0--T--4
    #       3         phy 1,2

    #       0
    #    3--T         phy 1,2
    #       4

    #     3
    #  0--T--4        phy 1,2

    # Based on the calculation of T
    # I GUESSED that

    #  0
    #  T--3           phy 1,2
    #  4
    for coord in state.sites.keys():
        with torch.no_grad():
            FL = contract(C_up["left"], C_down["left"], ([0], [0]))
            print("CreateHami: ", FL.abs().max())
            FL = FL/FL.abs().max()
            FU = contract(C_left["up"], C_right["up"], ([1], [0]))
            print("CreateHami: ", FU.abs().max())
            FU = FU/FU.abs().max()

        for i in range(args.size):
            if i % 2 == 0:
                temp = contract(FL, T_up[(i)], ([0], [0]))
                temp = contract(temp, T_down[(i)], ([0, 3], [0, 3]))
                # now
                #   C--T--2
                #   |  |      phy_T1 = 0,1
                #   C--T--5   phy_T2 = 3,4
                # This is vertical H
                # FL = contract(temp,Hy,([0,1,3,4],[0,2,1,3]))
                # We put on-site term now
                # Note that in Create_Localsite_Hami_Env,
                # We put on-site term by vertical H
                # Thus, we need to put it now, and don't need to put
                # it in horizontal H later
                FL = contract(temp, HyAndOnsiteBoth,
                              ([0, 1, 3, 4], [0, 2, 1, 3]))

                FL2 = FL.detach()
                print("CreateHami: ", FL2.abs().max())
                FL = FL/FL2.abs().max()

                temp = contract(FU, T_left[(i)], ([0], [0]))
                temp = contract(temp, T_right[(i)], ([0, 3], [0, 3]))
                FU = contract(temp, Hx, ([0, 1, 3, 4], [0, 2, 1, 3]))

                FU2 = FU.detach()
                print("CreateHami: ", FU2.abs().max())
                FU = FU/FU2.abs().max()

            else:
                temp = contract(FL, T_up[(i)], ([0], [0]))
                temp = contract(temp, T_down[(i)], ([0, 3], [0, 3]))
                FL = contract(temp, permute(HyAndOnsiteBoth,
                                            (1, 0, 3, 2)), ([0, 1, 3, 4], [0, 2, 1, 3]))

                FL2 = FL.detach()
                print("CreateHami: ", FL2.abs().max())
                FL = FL/FL2.abs().max()

                temp = contract(FU, T_left[(i)], ([0], [0]))
                temp = contract(temp, T_right[(i)], ([0, 3], [0, 3]))
                FU = contract(temp, permute(Hx, (1, 0, 3, 2)),
                              ([0, 1, 3, 4], [0, 2, 1, 3]))

                FU2 = FU.detach()
                print("CreateHami: ", FU2.abs().max())
                FU = FU/FU2.abs().max()

        with torch.no_grad():
            FR = contract(C_up["right"], C_down["right"], ([1], [0]))
            print("CreateHami: ", FR.abs().max())
            FR = FR/FR.abs().max()
            FD = contract(C_left["down"], C_right["down"], ([1], [1]))
            print("CreateHami: ", FD.abs().max())
            FD = FD/FD.abs().max()

        for i in range(args.size+1):
            if i % 2 == 0:
                temp = contract(FR, T_up[(2*args.size+1-i)], ([0], [4]))
                temp = contract(
                    temp, T_down[(2*args.size+1-i)], ([0, 4], [4, 3]))
                FR = contract(temp, permute(HyAndOnsiteBoth,
                                            (1, 0, 3, 2)), ([1, 2, 4, 5], [0, 2, 1, 3]))

                FR2 = FR.detach()
                print("CreateHami: ", FR2.abs().max())
                FR = FR/FR2.abs().max()

                temp = contract(FD, T_left[(2*args.size+1-i)], ([0], [4]))
                temp = contract(
                    temp, T_right[(2*args.size+1-i)], ([0, 4], [4, 3]))
                FD = contract(temp, permute(Hx, (1, 0, 3, 2)),
                              ([1, 2, 4, 5], [0, 2, 1, 3]))

                FD2 = FD.detach()
                print("CreateHami: ", FD2.abs().max())
                FD = FD/FD2.abs().max()
            else:
                temp = contract(FR, T_up[(2*args.size+1-i)], ([0], [4]))
                temp = contract(
                    temp, T_down[(2*args.size+1-i)], ([0, 4], [4, 3]))
                FR = contract(temp, HyAndOnsiteBoth,
                              ([1, 2, 4, 5], [0, 2, 1, 3]))

                FR2 = FR.detach()
                print("CreateHami: ", FR2.abs().max())
                FR = FR/FR2.abs().max()

                temp = contract(FD, T_left[(2*args.size+1-i)], ([0], [4]))
                temp = contract(
                    temp, T_right[(2*args.size+1-i)], ([0, 4], [4, 3]))
                FD = contract(temp, Hx, ([1, 2, 4, 5], [0, 2, 1, 3]))

                FD2 = FD.detach()
                print("CreateHami: ", FD2.abs().max())
                FD = FD/FD2.abs().max()

        dimsA = state.site(coord).size()

        if args.size % 2 == 1:
            H1 = contract(FL, T_up[(args.size)], ([0], [0]))
            H1 = contract(H1, view(T_down[(
                args.size)], (env.chi, phys_dim, phys_dim, dimsA[3], dimsA[3], env.chi)), ([0, 4], [0, 3]))
            H1 = contract(H1, HyAndOnsiteBoth, ([0, 5, 6], [0, 1, 3]))
            H1 = contiguous(
                permute(contract(H1, FR, ([3, 5], [0, 1])), (4, 0, 1, 3, 2)))

            H12 = H1.detach()
            print("CreateHami: ", H12.abs().max())
            H1 = H1/H12.abs().max()

            H2 = contract(FU, T_left[(args.size)], ([0], [0]))
            H2 = contract(H2, view(T_right[(
                args.size)], (env.chi, phys_dim, phys_dim, dimsA[4], dimsA[4], env.chi)), ([0, 5], [0, 3]))
            H2 = contract(H2, Hx, ([0, 5, 6], [0, 1, 3]))
            H2 = contiguous(
                permute(contract(H2, FD, ([3, 5], [0, 1])), (4, 0, 1, 2, 3)))

            H22 = H2.detach()
            print("CreateHami: ", H22.abs().max())
            H2 = H2/H22.abs().max()

        else:
            H1 = contract(FL, T_up[(args.size)], ([0], [0]))
            H1 = contract(H1, view(T_down[(
                args.size)], (env.chi, phys_dim, phys_dim, dimsA[3], dimsA[3], env.chi)), ([0, 6], [0, 3]))
            H1 = contract(H1, HyAndOnsiteBoth, ([0, 1, 7, 8], [0, 2, 1, 3]))
            H1 = contiguous(
                permute(contract(H1, FR, ([4, 6], [0, 1])), (0, 1, 2, 4, 3)))

            H12 = H1.detach()
            print("CreateHami: ", H12.abs().max())
            H1 = H1/H12.abs().max()

            H2 = contract(FU, T_left[(args.size)], ([0], [0]))
            H2 = contract(H2, view(T_right[(
                args.size)], (env.chi, phys_dim, phys_dim, dimsA[4], dimsA[4], env.chi)), ([0, 7], [0, 3]))
            H2 = contract(H2, Hx, ([0, 1, 7, 8], [0, 2, 1, 3]))
            H2 = contract(H2, FD, ([4, 6], [0, 1]))

            H22 = H2.detach()
            print("CreateHami: ", H22.abs().max())
            H2 = H2/H22.abs().max()

        Hami[coord] = H1/2. + H2/2.
        # Hami[coord] = H1

    return Hami

# def Create_Localsite_Hami(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, Hx, Hy, Honsite, Id, args):
#     phys_dim = state.site((0,0)).size()[0]
#     Hami = dict()
#     OI = torch.einsum('ij,ab->iajb', Honsite, Id)
#     IO = torch.einsum('ij,ab->iajb', Id, Honsite)
#     HyAndOnsiteFst = Hy + OI
#     HyAndOnsiteLst = Hy + IO
#     HyAndOnsiteBoth = Hy + OI + IO
#     #    0--T--4
#     #       3         phy 1,2

#     #       0
#     #    3--T         phy 1,2
#     #       4

#     #     3
#     #  0--T--4        phy 1,2

#     # Based on the calculation of T
#     # I GUESSED that

#     #  0
#     #  T--3           phy 1,2
#     #  4
#     for coord in state.sites.keys():
#         FL = contract(C_up["left"],C_down["left"],([0],[0]))
#         FL = FL/FL.abs().max()

#         for i in range(args.size):
#             if i%2 == 0:
#                 temp = contract(FL,T_up[(i)],([0],[0]))
#                 temp = contract(temp,T_down[(i)],([0,3],[0,3]))
#                 # now
#                 #   C--T--2
#                 #   |  |      phy_T1 = 0,1
#                 #   C--T--5   phy_T2 = 3,4
#                 # This is vertical H
#                 # FL = contract(temp,Hy,([0,1,3,4],[0,2,1,3]))
#                 # We put on-site term now
#                 # Note that in Create_Localsite_Hami_Env,
#                 # We put on-site term by vertical H
#                 # Thus, we need to put it now, and don't need to put
#                 # it in horizontal H later
#                 FL = contract(temp,HyAndOnsiteBoth,([0,1,3,4],[0,2,1,3]))


#                 FL2 = FL.detach()

#                 FL = FL/FL2.abs().max()

#             else:
#                 temp = contract(FL,T_up[(i)],([0],[0]))
#                 temp = contract(temp,T_down[(i)],([0,3],[0,3]))
#                 FL = contract(temp,permute(HyAndOnsiteBoth, (1,0,3,2)),([0,1,3,4],[0,2,1,3]))

#                 FL2 = FL.detach()

#                 FL = FL/FL2.abs().max()

#         FR = contract(C_up["right"],C_down["right"],([1],[0]))
#         FR = FR/FR.abs().max()

#         for i in range(args.size+1):
#             if i%2 == 0:
#                 temp = contract(FR,T_up[(2*args.size+1-i)],([0],[4]))
#                 temp = contract(temp,T_down[(2*args.size+1-i)],([0,4],[4,3]))
#                 FR = contract(temp,permute(HyAndOnsiteBoth, (1,0,3,2)),([1,2,4,5],[0,2,1,3]))

#                 FR2 = FR.detach()

#                 FR = FR/FR2.abs().max()
#             else:
#                 temp = contract(FR,T_up[(2*args.size+1-i)],([0],[4]))
#                 temp = contract(temp,T_down[(2*args.size+1-i)],([0,4],[4,3]))
#                 FR = contract(temp,HyAndOnsiteBoth,([1,2,4,5],[0,2,1,3]))

#                 FR2 = FR.detach()

#                 FR = FR/FR2.abs().max()

#         dimsA = state.site(coord).size()

#         if args.size%2==1:
#             H1 = contract(FL,T_up[(args.size)],([0],[0]))
#             H1 = contract(H1,view(T_down[(args.size)],(env.chi,phys_dim,phys_dim,dimsA[3],dimsA[3],env.chi)),([0,4],[0,3]))
#             H1 = contract(H1,HyAndOnsiteBoth,([0,5,6],[0,1,3]))
#             H1 = contiguous(permute(contract(H1,FR,([3,5],[0,1])),(4,0,1,3,2)))

#             H12 = H1.detach()

#             H1 = H1/H12.abs().max()

#         else:
#             H1 = contract(FL,T_up[(args.size)],([0],[0]))
#             H1 = contract(H1,view(T_down[(args.size)],(env.chi,phys_dim,phys_dim,dimsA[3],dimsA[3],env.chi)),([0,6],[0,3]))
#             H1 = contract(H1,HyAndOnsiteBoth,([0,1,7,8],[0,2,1,3]))
#             H1 = contiguous(permute(contract(H1,FR,([4,6],[0,1])),(0,1,2,4,3)))

#             H12 = H1.detach()

#             H1 = H1/H12.abs().max()

#         Hami[coord] = H1

#     return Hami

# def Create_Localsite_Hami(state, env, B_grad, lam, kx, ky, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, Hx, Hy, Honsite, args):
#     Hami = dict()
#     tn = None
#     phys_dim = state.site((0,0)).size()[0]

#     for coord in state.sites.keys():
#         # Construct T line for up and down

#         #    0--T--4
#         #       3         phy 1,2

#         #       0
#         #    3--T         phy 1,2
#         #       4

#         #     3
#         #  0--T--4        phy 1,2

#         # Based on the calculation of T
#         # I GUESSED that

#         #  0
#         #  T--3           phy 1,2
#         #  4

#         # T[(0,-1)]
#         tmp = T_up[(0)]
#         tmp = contract(tmp, T_up[(1)], ([4],[0]))
#         # now
#         #    0--T--T--7      phy1 1,2
#         #       3  6         phy2 4,5
#         tmp = contract(tmp, Hx, ([1,4],[0,1]))
#         # now
#         #    0--T--T--5  phy1 6,1
#         #       2  4      phy2 7,3
#         tmp = contiguous(permute(tmp, (0,6,1,2,7,3,4,5)))
#         # now
#         #    0--T--T--7      phy1 1,2
#         #       3  6         phy2 4,5
#         tmp = contract(tmp, T_up[(2)], ([7],[0]))
#         # now
#         #    0--T--T--T--10      phy1 1,2
#         #       3  6  9          phy2 4,5
#         #                        phy3 7,8
#         tmp = contract(tmp, Hx, ([4,7],[0,1]))
#         # now
#         #    0--T--T--T--8       phy1 1,2
#         #       3  5  7          phy2 9,4
#         #                        phy3 10,6
#         tmp = contiguous(permute(tmp, (0,1,2,3,9,4,5,10,6,7,8)))
#         # now
#         #    0--T--T--T--10      phy1 1,2
#         #       3  6  9          phy2 4,5
#         #                        phy3 7,8
#         env.T[coord, (0,-1)] = tmp

#         # T[(0,1)]
#         tmp = T_down[(0)]
#         tmp = contract(tmp, T_down[(1)], ([4],[0]))
#         # now
#         #       3  6          phy1 1,2
#         #    0--T--T--7       phy2 4,5
#         tmp = contract(tmp, Hx, ([1,4],[0,1]))
#         # now
#         #       2  4      phy1 6,1
#         #    0--T--T--5   phy2 7,3
#         tmp = contiguous(permute(tmp, (0,6,1,2,7,3,4,5)))
#         # now
#         #       3  6          phy1 1,2
#         #    0--T--T--7       phy2 4,5
#         tmp = contract(tmp, T_down[(2)], ([7],[0]))
#         # now
#         #       3  6  9       phy1 1,2
#         #    0--T--T--T--10   phy2 4,5
#         #                     phy3 7,8
#         tmp = contract(tmp, Hx, ([4,7],[0,1]))
#         # now
#         #       3  5  7          phy1 1,2
#         #    0--T--T--T--8       phy2 9,4
#         #                        phy3 10,6
#         tmp = contiguous(permute(tmp, (0,1,2,3,9,4,5,10,6,7,8)))
#         # now
#         #       3  6  9       phy1 1,2
#         #    0--T--T--T--10   phy2 4,5
#         #                     phy3 7,8
#         env.T[coord, (0,1)] = tmp

#         # Construct DL line = DLL
#         dimsA = state.site(coord).size()
#         Aket = state.site(coord) + lam * torch.exp(-1j*(kx*coord[0]+ky*coord[1])) * B_grad
#         DL = view(contiguous(einsum('mefgh,nabcd->mneafbgchd',Aket,conj(state.site(coord)))),\
#             (phys_dim, phys_dim ,dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
#         #      2
#         #     3DL5      phy 0,1
#         #      4
#         DLL = contract(DL, DL, ([5],[3]))
#         #now,
#         #          2    7
#         #          |    |             phy_in_DL1 0,1
#         #        3-DL1--DL2-9         phy_in_DL2 5,6
#         #          |    |
#         #          4    8
#         DLL = contract(DLL, Hx, ([0,5],[0,1]))
#         #now,
#         #          1    5
#         #          |    |             phy_in_DL1 8,0
#         #        2-DL1--DL2-7         phy_in_DL2 9,4
#         #          |    |
#         #          3    6
#         DLL = contiguous(permute(DLL, (8,0,1,2,3,9,4,5,6,7)))
#         #now,
#         #          2    7
#         #          |    |             phy_in_DL1 0,1
#         #        3-DL1--DL2-9         phy_in_DL2 5,6
#         #          |    |
#         #          4    8
#         DLL = contract(DLL, DL, ([9],[3]))
#         #          2    7    11
#         #          |    |    |              phy_in_DL1 0,1
#         #        3-DL1--DL2--DL3-13         phy_in_DL2 5,6
#         #          |    |    |              phy_in_DL3 9,10
#         #          4    8    12
#         DLL = contract(DLL, Hx, ([5,9],[0,1]))
#         #          2    6    9
#         #          |    |    |              phy_in_DL1 0,1
#         #        3-DL1--DL2--DL3-11         phy_in_DL2 12,5
#         #          |    |    |              phy_in_DL3 13,8
#         #          4    7    10
#         DLL = contiguous(permute(DLL, (2,6,9,3,4,7,10,11,0,1,12,5,13,8)))
#         #          0    1    2
#         #          |    |    |             phy_in_DL1 8,9
#         #        3-DL1--DL2--DL3-7         phy_in_DL2 10,11
#         #          |    |    |             phy_in_DL3 12,13
#         #          4    5    6

#         #    0--T--T--T--10      phy1 1,2
#         #       3  6  9          phy2 4,5
#         #                        phy3 7,8
#         tn = contract(env.C[coord, (-1,-1)], env.T[coord, (0,-1)], ([1],[0]))
#         tn = contract(tn, env.C[coord, (1,-1)], ([10],[0]))
#         # now,
#         #    C--T--T--T--C       phy1 1,2
#         #    0  3  6  9  10      phy2 4,5
#         #                        phy3 7,8
#         tn = contract(tn, T_left[(0)], ([0],[0]))
#         # now,
#         #    C--T--T--T--C       phy1 0,1
#         #    |  2  5  8  9       phy2 3,4
#         #    T-12                phy3 6,7
#         #    13                  phy_L_T 10,11

#         # now DLL=
#         #          0    1    2
#         #          |    |    |             phy_in_DL1 8,9
#         #        3-DL1--DL2--DL3-7         phy_in_DL2 10,11
#         #          |    |    |             phy_in_DL3 12,13
#         #          4    5    6

#         # Now we contract DLL with Hy
#         # Hy=
#         #      0  1
#         #      h1 h2
#         #      2  3
#         DLL = contract(DLL, Hy, ([8],[1]))
#         # now DLL=
#         #       13,14
#         #         Hy
#         #          0    1    2
#         #          |    |    |             phy_in_DL1 15,8
#         #        3-DL1--DL2--DL3-7         phy_in_DL2 9,10
#         #          |    |    |             phy_in_DL3 11,12
#         #          4    5    6
#         DLL = contract(DLL, Hy, ([9],[1]))
#         # now DLL=
#         #       12,13 15,16
#         #         Hy   Hy
#         #          0    1    2
#         #          |    |    |             phy_in_DL1 14,8
#         #        3-DL1--DL2--DL3-7         phy_in_DL2 17,9
#         #          |    |    |             phy_in_DL3 10,11
#         #          4    5    6
#         DLL = contract(DLL, Hy, ([10],[1]))
#         # now DLL=
#         #       11,12 14,15 17,18
#         #         Hy   Hy   Hy
#         #          0    1    2
#         #          |    |    |             phy_in_DL1 13,8
#         #        3-DL1--DL2--DL3-7         phy_in_DL2 16,9
#         #          |    |    |             phy_in_DL3 19,10
#         #          4    5    6

#         tn = contract(tn, DLL, ([12,2,5,8],[3,0,1,2]))
#         # now,
#         #    C--T--T--T--C       phy1 0,1
#         #    |  |  |  |  6       phy2 2,3
#         #    T--DL-DL-DL-13      phy3 4,5
#         #    9  10 11 12         phy_L_T 7,8
#         #                        phy_in_DL1 14,15
#         #                        phy_in_DL2 16,17
#         #                        phy_in_DL3 18,19
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
#         Hami[coord] = tn
#     return Hami
