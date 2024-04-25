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

def Create_Norm_Env(state, stateDL, B_grad, env, lam, kx, ky, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args,\
    firsttime,lasttime):
    if cfg.ctm_args.projector_method=='4X4':
        ctm_get_projectors=ctm_get_projectors_4x4
    elif cfg.ctm_args.projector_method=='4X2':
        ctm_get_projectors=ctm_get_projectors_4x2
    else:
        raise ValueError("Invalid Projector method: "+str(cfg.ctm_args.projector_method))
    for coord in stateDL.sites.keys():
        for direction in cfg.ctm_args.ctm_move_sequence:
            if direction==(0,-1):
                vec = (1,0)
                vec_coord = (-args.size,-args.size)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                if "left" not in C_up:
                    C_up["left"] = env.C[(new_coord,(-1,-1))].clone()
                vec_coord = (args.size+1,-args.size)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                if "right" not in C_up:
                    C_up["right"] = env.C[(new_coord,(1,-1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size+j,-args.size)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    if (j) not in T_up:
                        print("T_up is empty")
                        T_up[(j)] = env.T[(new_coord,direction)].clone()
                    
                for i in range(args.size+1):
                    vec_coord_l = (-args.size,-args.size+i)
                    new_coord_l = state.vertexToSite((coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
                    coord_shift_left= stateDL.vertexToSite((new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
                    coord_shift_right = stateDL.vertexToSite((new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
                    P2, Pt2 = ctm_get_projectors(direction, new_coord_l, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P2 = view(P2, (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord_l).size()[1],env.chi))
                    P1, Pt1 = ctm_get_projectors(direction, coord_shift_right, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P1 = view(P1, (env.chi,stateDL.site(new_coord_l).size()[3],env.chi))
                    Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    nC2 = contract(C_up["left"], env.T[(new_coord_l,(-1,0))],([0],[0]))
                    nC2 = contract(nC2, P2,([0,2],[0,1]))
                    C_up["left"] = nC2/nC2.abs().max()
                    env.C[(new_coord_l,(-1,-1))] = C_up["left"]
                    
                    vec_coord_r = (args.size+1,-args.size+i)
                    new_coord_r = state.vertexToSite((coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
                    coord_shift_left= stateDL.vertexToSite((new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
                    coord_shift_right = stateDL.vertexToSite((new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
                    P2, Pt2 = ctm_get_projectors(direction, new_coord_r, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P2 = view(P2, (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord_r).size()[1],env.chi))
                    P1, Pt1 = ctm_get_projectors(direction, coord_shift_right, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P1 = view(P1, (env.chi,stateDL.site(new_coord_r).size()[3],env.chi))
                    Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    nC1 = contract(C_up["right"], env.T[(new_coord_r,(1,0))],([1],[0]))
                    nC1 = contract(Pt1, nC1,([0,1],[0,1]))
                    C_up["right"] = nC1/nC1.abs().max()
                    env.C[(new_coord_r,(1,-1))] = C_up["right"]

                    for j in range(2*args.size+2):
                        vec_coord = (-args.size+j,-args.size+i)
                        new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_left= stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        coord_shift_right = stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        P2, Pt2 = ctm_get_projectors(direction, new_coord, stateDL, env, cfg.ctm_args, cfg.global_args)
                        P2 = view(P2, (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                        Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord).size()[1],env.chi))
                        P1, Pt1 = ctm_get_projectors(direction, coord_shift_right, stateDL, env, cfg.ctm_args, cfg.global_args)
                        P1 = view(P1, (env.chi,stateDL.site(new_coord).size()[3],env.chi))
                        Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                        if i == args.size and j == args.size and lasttime:
                            nT = contract(Pt2, T_up[(j)], ([0],[0]))
                            dimsA = state.site(new_coord).size()
                            nT = view(nT,(dimsA[2],dimsA[2],env.chi,dimsA[1],dimsA[1],env.chi))
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            nT = contract(nT, Aket,([0,3],[2,1]))
                            nT = contract(nT, view(P1,(env.chi,dimsA[4],dimsA[4],env.chi)),([3,6],[0,1]))
                            tempT = contiguous(permute(nT, (1,3,2,0,4,5,6)))

                            tempT2 = tempT.detach()

                            T_up[(j)] = tempT/tempT2.abs().max()
                            # env.T[(new_coord,direction)] = T_up[(j)]
                        else:
                            nT = contract(Pt2, T_up[(j)], ([0],[0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord)))),\
                                      (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL,([0,2],[1,0]))
                            nT = contract(nT, P1,([1,3],[0,1]))
                            tempT = contiguous(nT)

                            tempT2 = tempT.detach()

                            T_up[(j)] = tempT/tempT2.abs().max()
                            # env.T[(new_coord,direction)] = T_up[(j)]

            elif direction==(0,1):
                vec = (-1,0)
                vec_coord = (-args.size,args.size+1)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                if "left" not in C_down:
                    C_down["left"] = env.C[(new_coord,(-1,1))].clone()
                vec_coord = (args.size+1,args.size+1)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                if "right" not in C_down:
                    C_down["right"] = env.C[(new_coord,(1,1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size+j,args.size+1)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    if (j) not in T_down:
                        print("T_down is empty")
                        T_down[(j)] = env.T[(new_coord,direction)].clone()
                    
                for i in range(args.size+1):
                    vec_coord_l = (-args.size,args.size-i+1)
                    new_coord_l = state.vertexToSite((coord[0]+vec_coord_l[0], coord[1]+vec_coord_l[1]))
                    coord_shift_right= stateDL.vertexToSite((new_coord_l[0]-vec[0], new_coord_l[1]-vec[1]))
                    coord_shift_left = stateDL.vertexToSite((new_coord_l[0]+vec[0], new_coord_l[1]+vec[1]))
                    P2, Pt2 = ctm_get_projectors(direction, new_coord_l, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P2 = view(P2, (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord_l).size()[3],env.chi))
                    P1, Pt1 = ctm_get_projectors(direction, coord_shift_left, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P1 = view(P1, (env.chi,stateDL.site(new_coord_l).size()[1],env.chi))
                    Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    nC1 = contract(C_down["left"], env.T[(new_coord_l,(-1,0))],([0],[1]))
                    nC1 = contract(nC1, Pt1, ([0,2],[0,1]))
                    C_down["left"] = nC1/nC1.abs().max()
                    env.C[(new_coord_l,(-1,1))] = C_down["left"]
                    
                    vec_coord_r = (args.size+1,args.size-i+1)
                    new_coord_r = state.vertexToSite((coord[0]+vec_coord_r[0], coord[1]+vec_coord_r[1]))
                    coord_shift_right= stateDL.vertexToSite((new_coord_r[0]-vec[0], new_coord_r[1]-vec[1]))
                    coord_shift_left = stateDL.vertexToSite((new_coord_r[0]+vec[0], new_coord_r[1]+vec[1]))
                    P2, Pt2 = ctm_get_projectors(direction, new_coord_r, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P2 = view(P2, (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                    Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord_r).size()[3],env.chi))
                    P1, Pt1 = ctm_get_projectors(direction, coord_shift_left, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P1 = view(P1, (env.chi,stateDL.site(new_coord_r).size()[1],env.chi))
                    Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                    nC2 = contract(C_down["right"], env.T[(new_coord_r,(1,0))],([0],[2]))
                    nC2 = contract(nC2, P2, ([0,2],[0,1]))
                    C_down["right"] = nC2/nC2.abs().max()
                    env.C[(new_coord_r,(1,1))] = C_down["right"]

                    for j in range(2*args.size+2):
                        vec_coord = (-args.size+j,args.size-i+1)
                        new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_right= stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        coord_shift_left = stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        P2, Pt2 = ctm_get_projectors(direction, new_coord, stateDL, env, cfg.ctm_args, cfg.global_args)
                        P2 = view(P2, (env.chi,stateDL.site(coord_shift_right).size()[1],env.chi))
                        Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord).size()[3],env.chi))
                        P1, Pt1 = ctm_get_projectors(direction, coord_shift_left, stateDL, env, cfg.ctm_args, cfg.global_args)
                        P1 = view(P1, (env.chi,stateDL.site(new_coord).size()[1],env.chi))
                        Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_left).size()[3],env.chi))
                        nT = contract(P1, T_down[(j)], ([0],[1]))
                        dimsA = state.site(new_coord).size()
                        Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord)))),\
                                  (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        nT = contract(nT, DL,([0,2],[1,2]))
                        nT = contract(nT, Pt2,([1,3],[0,1]))
                        tempT = contiguous(permute(nT, (1,0,2)))

                        tempT2 = tempT.detach()
                            
                        T_down[(j)] = tempT/tempT2.abs().max()
                        # env.T[(new_coord,direction)] = T_down[(j)]

            elif direction==(-1,0):
                vec = (0,-1)
                vec_coord = (-args.size,-args.size)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                if "up" not in C_left:
                    C_left["up"] = env.C[(new_coord,(-1,-1))].clone()
                vec_coord = (-args.size,args.size+1)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                if "down" not in C_left:
                    C_left["down"] = env.C[(new_coord,(-1,1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (-args.size,-args.size+j)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    if (j) not in T_left:
                        print("T_left is empty")
                        T_left[(j)] = env.T[(new_coord,direction)].clone()
                    
                for i in range(args.size+1):
                    vec_coord_u = (-args.size+i,-args.size)
                    new_coord_u = state.vertexToSite((coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
                    coord_shift_up= stateDL.vertexToSite((new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
                    coord_shift_down= stateDL.vertexToSite((new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
                    P2, Pt2 = ctm_get_projectors(direction, new_coord_u, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P2 = view(P2, (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord_u).size()[2],env.chi))
                    P1, Pt1 = ctm_get_projectors(direction, coord_shift_up, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P1 = view(P1, (env.chi,stateDL.site(new_coord_u).size()[0],env.chi))
                    Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    nC1 = contract(C_left["up"], env.T[(new_coord_u,(0,-1))],([1],[0]))
                    nC1 = contract(Pt1, nC1,([0,1],[0,1]))
                    C_left["up"] = nC1/nC1.abs().max()
                    env.C[(new_coord_u,(-1,-1))] = C_left["up"]
                    
                    vec_coord_d = (-args.size+i,args.size+1)
                    new_coord_d = state.vertexToSite((coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
                    coord_shift_up= stateDL.vertexToSite((new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
                    coord_shift_down= stateDL.vertexToSite((new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
                    P2, Pt2 = ctm_get_projectors(direction, new_coord_d, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P2 = view(P2, (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord_d).size()[2],env.chi))
                    P1, Pt1 = ctm_get_projectors(direction, coord_shift_up, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P1 = view(P1, (env.chi,stateDL.site(new_coord_d).size()[0],env.chi))
                    Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    nC2 = contract(C_left["down"], env.T[(new_coord_d,(0,1))],([1],[1]))
                    nC2 = contract(P2, nC2,([0,1],[0,1]))
                    C_left["down"] = nC2/nC2.abs().max()
                    env.C[(new_coord_d,(-1,1))] = C_left["down"]

                    for j in range(2*args.size+2):
                        vec_coord = (-args.size+i,-args.size+j)
                        new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_up= stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        coord_shift_down= stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        P2, Pt2 = ctm_get_projectors(direction, new_coord, stateDL, env, cfg.ctm_args, cfg.global_args)
                        P2 = view(P2, (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                        Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord).size()[2],env.chi))
                        P1, Pt1 = ctm_get_projectors(direction, coord_shift_up, stateDL, env, cfg.ctm_args, cfg.global_args)
                        P1 = view(P1, (env.chi,stateDL.site(new_coord).size()[0],env.chi))
                        Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                        if i == args.size and j == args.size and lasttime:
                            nT = contract(P1, T_left[(j)],([0],[0]))
                            dimsA = state.site(new_coord).size()
                            nT = view(nT,(dimsA[1],dimsA[1],env.chi,env.chi,dimsA[2],dimsA[2]))
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            nT = contract(nT, Aket,([0,4],[1,2]))
                            nT = contract(nT, view(Pt2,(env.chi,dimsA[3],dimsA[3],env.chi)),([2,5],[0,1]))
                            tempT = contiguous(permute(nT, (1,3,0,2,5,4,6)))

                            tempT2 = tempT.detach()
                            
                            T_left[(j)] = tempT/tempT2.abs().max()
                            # env.T[(new_coord,direction)] = T_left[(j)]
                        else:
                            nT = contract(P1, T_left[(j)],([0],[0]))
                            dimsA = state.site(new_coord).size()
                            Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                            DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord)))),\
                                      (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                            nT = contract(nT, DL,([0,3],[0,1]))
                            nT = contract(nT, Pt2,([1,2],[0,1]))
                            tempT = contiguous(permute(nT, (0,2,1)))

                            tempT2 = tempT.detach()
                            
                            T_left[(j)] = tempT/tempT2.abs().max()
                            # env.T[(new_coord,direction)] = T_left[(j)]

            elif direction==(1,0):
                vec = (0,1)
                vec_coord = (args.size+1,-args.size)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                if "up" not in C_right:
                    C_right["up"] = env.C[(new_coord,(1,-1))].clone()
                vec_coord = (args.size+1,args.size+1)
                new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                if "down" not in C_right:
                    C_right["down"] = env.C[(new_coord,(1,1))].clone()
                for j in range(2*args.size+2):
                    vec_coord = (args.size+1,-args.size+j)
                    new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                    if (j) not in T_right:
                        print("T_right is empty")
                        T_right[(j)] = env.T[(new_coord,direction)].clone()
                    
                for i in range(args.size+1):
                    vec_coord_u = (args.size-i+1,-args.size)
                    new_coord_u = state.vertexToSite((coord[0]+vec_coord_u[0], coord[1]+vec_coord_u[1]))
                    coord_shift_down = stateDL.vertexToSite((new_coord_u[0]+vec[0], new_coord_u[1]+vec[1]))
                    coord_shift_up = stateDL.vertexToSite((new_coord_u[0]-vec[0], new_coord_u[1]-vec[1]))
                    P2, Pt2 = ctm_get_projectors(direction, new_coord_u, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P2 = view(P2, (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord_u).size()[0],env.chi))
                    P1, Pt1 = ctm_get_projectors(direction, coord_shift_down, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P1 = view(P1, (env.chi,stateDL.site(new_coord_u).size()[2],env.chi))
                    Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    nC2 = contract(C_right["up"], env.T[(new_coord_u,(0,-1))],([0],[2]))
                    nC2 = contract(nC2, P2,([0,2],[0,1]))
                    C_right["up"] = nC2/nC2.abs().max()
                    env.C[(new_coord_u,(1,-1))] = C_right["up"]
                    
                    vec_coord_d = (args.size-i+1,args.size+1)
                    new_coord_d = state.vertexToSite((coord[0]+vec_coord_d[0], coord[1]+vec_coord_d[1]))
                    coord_shift_down = stateDL.vertexToSite((new_coord_d[0]+vec[0], new_coord_d[1]+vec[1]))
                    coord_shift_up = stateDL.vertexToSite((new_coord_d[0]-vec[0], new_coord_d[1]-vec[1]))
                    P2, Pt2 = ctm_get_projectors(direction, new_coord_d, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P2 = view(P2, (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                    Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord_d).size()[0],env.chi))
                    P1, Pt1 = ctm_get_projectors(direction, coord_shift_down, stateDL, env, cfg.ctm_args, cfg.global_args)
                    P1 = view(P1, (env.chi,stateDL.site(new_coord_d).size()[2],env.chi))
                    Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                    nC1 = contract(C_right["down"], env.T[(new_coord_d,(0,1))],([1],[2]))
                    nC1 = contract(Pt1, nC1,([0,1],[0,1]))
                    C_right["down"] = nC1/nC1.abs().max()
                    env.C[(new_coord_d,(1,1))] = C_right["down"]

                    for j in range(2*args.size+2):
                        vec_coord = (args.size-i+1,-args.size+j)
                        new_coord = state.vertexToSite((coord[0]+vec_coord[0], coord[1]+vec_coord[1]))
                        coord_shift_down = stateDL.vertexToSite((new_coord[0]+vec[0], new_coord[1]+vec[1]))
                        coord_shift_up = stateDL.vertexToSite((new_coord[0]-vec[0], new_coord[1]-vec[1]))
                        P2, Pt2 = ctm_get_projectors(direction, new_coord, stateDL, env, cfg.ctm_args, cfg.global_args)
                        P2 = view(P2, (env.chi,stateDL.site(coord_shift_up).size()[2],env.chi))
                        Pt2 = view(Pt2, (env.chi,stateDL.site(new_coord).size()[0],env.chi))
                        P1, Pt1 = ctm_get_projectors(direction, coord_shift_down, stateDL, env, cfg.ctm_args, cfg.global_args)
                        P1 = view(P1, (env.chi,stateDL.site(new_coord).size()[2],env.chi))
                        Pt1 = view(Pt1, (env.chi,stateDL.site(coord_shift_down).size()[0],env.chi))
                        nT = contract(Pt2, T_right[(j)],([0],[0]))
                        dimsA = state.site(new_coord).size()
                        Aket = state.site(new_coord) + lam * torch.exp(-1j*(kx*vec_coord[0]+ky*vec_coord[1])) * B_grad
                        DL = view(contiguous(einsum('mefgh,mabcd->eafbgchd',Aket,conj(state.site(new_coord)))),\
                                  (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
                        nT = contract(nT, DL,([0,2],[0,3]))
                        nT = contract(nT, P1,([1,3],[0,1]))
                        tempT = contiguous(nT)

                        tempT2 = tempT.detach()
                            
                        T_right[(j)] = tempT/tempT2.abs().max()
                        # env.T[(new_coord,direction)] = T_right[(j)]
    return C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right

def Create_Norm(state, env, C_up, T_up, C_left, T_left, C_down, T_down, C_right, T_right, args):
    Norm = dict()
    for coord in state.sites.keys():
        FL = contract(C_up["left"],C_down["left"],([0],[0]))
        FL = FL/FL.abs().max()
            
        for i in range(args.size):
            temp = contract(FL,T_up[(i)],([0],[0]))
            FL = contract(temp,T_down[(i)],([0,1],[1,0]))

            FL2 = FL.detach()
                
            FL = FL/FL2.abs().max()

        FR = contract(C_up["right"],C_down["right"],([1],[0]))
        FR = FR/FR.abs().max()
        
        for i in range(args.size+1):
            temp = contract(FR,T_up[(2*args.size+1-i)],([0],[2]))
            FR = contract(temp,T_down[(2*args.size+1-i)],([0,2],[2,0]))

            FR2 = FR.detach()
                
            FR = FR/FR2.abs().max()

        dimsA = state.site(coord).size()
        
        H1 = contract(FL,T_up[(args.size)],([0],[0]))
        H1 = contract(H1,view(T_down[(args.size)],(dimsA[3],dimsA[3],env.chi,env.chi)),([0,4],[2,0]))
        H1 = contiguous(permute(contract(H1,FR,([4,6],[0,1])),(0,1,2,4,3)))

        H12 = H1.detach()

        H1 = H1/H12.abs().max()

        Norm[coord] = H1

    return Norm
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
