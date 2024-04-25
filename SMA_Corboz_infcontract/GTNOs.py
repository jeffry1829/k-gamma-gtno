import itertools
import torch
import config as cfg
import groups.su2 as su2
import numpy as np

def G_Ising(params):
##    params: [c1...,c9]
    norm_factor = sum(i*i for i in params)
    #norm_factor+= torch.tensor(1.0,dtype=cfg.global_args.torch_dtype,device='cpu')
    #c0 = 1.0/torch.sqrt(norm_factor)
    cs = []
    for i in range(9):
        cs.append(params[i]/torch.sqrt(norm_factor))

    G = torch.zeros((2,2,2,2,2,2),dtype=cfg.global_args.torch_dtype,device='cpu')
    Id = torch.eye(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    X = s2.SP()+s2.SM()
    Z = 2 * s2.SZ()
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                for l in range(0,2):
                    tmp_array = np.array([i,j,k,l])
                    nonzero = tmp_array[np.argwhere(tmp_array != 0)]
                    nonzeroidx = np.where(tmp_array != 0)[0]

                    # print(num_zeros)
                    if len(nonzero) == 0: # (0000)
                        # print(tmp_array)
                        G[:,:,i,j,k,l] = Id + cs[0]*X
                    elif len(nonzero) == 1: # P(1000)
                        # print(tmp_array)
                        G[:,:,i,j,k,l] = cs[1]*Z
                    elif len(nonzero) == 2: 
                        if (nonzeroidx[1]-nonzeroidx[0]) == 2: ## (1010), (0101)
                            # print(tmp_array)
                            # print(nonzeroidx)
                            G[:,:,i,j,k,l] = cs[2]*Id + cs[3]*X
                        else: ## (1100), (0110), (0011), (1001)
                            # print(tmp_array)
                            G[:,:,i,j,k,l] = cs[4]*Id + cs[5]*X
                    elif len(nonzero) == 3: ## P(1110)
                        # print(tmp_array)
                        G[:,:,i,j,k,l] = cs[6]*Z
                    else:  
                        # print(tmp_array)
                        G[:,:,i,j,k,l] = cs[7]*Id + cs[8]*X

    return G, cs

def G_Heisenberg(params):
##    params: [c1...,c9]
    norm_factor = sum(i*i for i in params)
    norm_factor+= torch.tensor(1.0,dtype=cfg.global_args.torch_dtype,device='cpu')
    c0 = 1.0/torch.sqrt(norm_factor)
    cs = [c0]
    for i in range(9):
        cs.append(params[i]/torch.sqrt(norm_factor))

    G = torch.zeros((2,2,4,4,4,4),dtype=cfg.global_args.torch_dtype,device='cpu')
    Id = torch.eye(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    #nor = torch.tensor(2.0,dtype=cfg.global_args.torch_dtype,device='cpu')
    #P = s2.SP()/torch.sqrt(nor)
    #M = s2.SM()/torch.sqrt(nor)
    X = s2.SP()+s2.SM()
    Y = -(s2.SP()-s2.SM())*1j
    Z = s2.SZ()
    Op = [Id,X,Y,Z]
    for i in range(0,4):
        for j in range(0,4):
            for k in range(0,4):
                for l in range(0,4):
                    # tmp is simply the sum of all permutation of (Op[i]@Op[j]@Op[k]@Op[l])
                    tmp = Op[i]@ (Op[j]@Op[k]@Op[l]+ Op[j]@Op[l]@Op[k]+ Op[k]@Op[l]@Op[j]+ Op[k]@Op[j]@Op[l] + Op[l]@Op[k]@Op[j]+ Op[l]@Op[j]@Op[k]) /24
                    # tmp = Op[i]* (Op[j]*Op[k]*Op[l]+ Op[j]*Op[l]*Op[k]+ Op[k]*Op[l]*Op[j]+ Op[k]*Op[j]*Op[l] + Op[l]*Op[k]*Op[j]+ Op[l]*Op[j]*Op[k]) /24
                    for _ in range(0,3):
                        i,j,k,l = j,k,l,i
                        tmp += Op[i]@ (Op[j]@Op[k]@Op[l]+ Op[j]@Op[l]@Op[k]+ Op[k]@Op[l]@Op[j]+ Op[k]@Op[j]@Op[l] + Op[l]@Op[k]@Op[j]+ Op[l]@Op[j]@Op[k]) /24
                        # tmp += Op[i]* (Op[j]*Op[k]*Op[l]+ Op[j]*Op[l]*Op[k]+ Op[k]*Op[l]*Op[j]+ Op[k]*Op[j]*Op[l] + Op[l]*Op[k]*Op[j]+ Op[l]*Op[j]*Op[k]) /24
                    i,j,k,l = j,k,l,i
                    # print(i,j,k,l)
                    tmp_array = np.array([i,j,k,l])
                    nonzero = tmp_array[np.argwhere(tmp_array != 0)]
                    nonzeroidx = np.where(tmp_array != 0)[0]
                    # nonzero

                    if len(nonzero) == 0: # (0000)
                        # print(i,j,k,l)
                        # print(tmp)
                        G[:,:,i,j,k,l] = cs[0]*tmp 
                    elif len(nonzero) == 1: # P(1000), (2000), (3000)
                        # print(tmp_array)
                        # print(tmp)
                        G[:,:,i,j,k,l] = cs[1]*tmp
                    elif len(nonzero) == 2:
                        if nonzero[0] == nonzero[1]: # 11 22 33
                            # print(i,j,k,l)
                            if (nonzeroidx[1]-nonzeroidx[0]) == 2: ## P(1010), P(2020), P(3030)
                                # print(tmp_array)
                                # print(tmp)
                                G[:,:,i,j,k,l] = cs[2]*tmp
                            else: ## P(1100), P(2200), P(3300)
                                # print(tmp_array)
                                # print(tmp)
                                G[:,:,i,j,k,l] = cs[3]*tmp ## should be c3
                                # pass
                    elif len(nonzero) == 3:
                        if nonzero[0] == nonzero[1] == nonzero[2]: ## (1110), (2220), (3330)
                            # print(tmp_array)
                            # print(tmp)
                            G[:,:,i,j,k,l] = cs[4]*tmp
                        elif nonzero[0] != nonzero[1] and nonzero[2] != nonzero[1] and nonzero[2] != nonzero[0]: ## (1230)
                            # print(tmp_array)
                            # print(tmp)
                            pass
                        else: 
                            if i ==j or j == k or k == l or l == i:## (1120), (1130), (2210), (2230), (3310), (3320)
                                # print(tmp_array)
                                # print(tmp)
                                G[:,:,i,j,k,l] = cs[5]*tmp
                            else:  ## (1210), (1310), (2120), (2320), (3130), (3230)
                                # print(tmp_array)
                                # print(tmp)
                                G[:,:,i,j,k,l] = cs[6]*tmp ## should be c6
                                # pass
                    elif len(nonzero) == 4:
                        if i == j ==  k ==  l: ## (1111), (2222), (3333)
                            # print(tmp_array)
                            # print(tmp)
                            G[:,:,i,j,k,l] = cs[7]*tmp 
                        elif np.count_nonzero(tmp_array == 1) == 3 or np.count_nonzero(tmp_array == 2)== 3 or \
                        np.count_nonzero(tmp_array == 3) == 3: ## (1112) (1113) (2221) (2223) (3331) (3332)
                            # print(tmp_array)
                            # print(tmp)
                            pass
                        elif np.count_nonzero(tmp_array == 1) == 0 or np.count_nonzero(tmp_array == 2)== 0 or \
                        np.count_nonzero(tmp_array == 3) == 0 :
                            if tmp_array[nonzeroidx[2]] == tmp_array[nonzeroidx[0]]:  ## (1212), (1313), (2323)
                                # print(tmp_array)
                                # print(tmp)
                                G[:,:,i,j,k,l] = cs[8]*tmp
                            else: ## (1122), (1133), (2233)
                                # print(tmp_array)
                                # print(tmp)
                                G[:,:,i,j,k,l] = cs[9]*tmp
                                # pass
    return G, cs

def G_kitaev(params):
    
    ## This is actually the DG operator

    num_params = 2
    norm_factor = sum(i*i for i in params)
    cs = [torch.cos(params[0]), torch.sin(params[0])]
    # for i in range(num_params):
    #     # cs.append(params[i].real/torch.sqrt(norm_factor))
    #     cs.append(params[i].real)
    G = torch.zeros((2,2,2,2,2),dtype=cfg.global_args.torch_dtype,device='cpu') # (s,s,Xv,Yv,Zv)
    # G = torch.zeros((2,2,2,2,2),dtype=torch.complex64,device='cpu') # (s,s,Xv,Yv,Zv)
    Id = torch.eye(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    X = s2.SP()+s2.SM()
    Y = -(s2.SP()-s2.SM())*1j
    Z = 2*s2.SZ()
    G[:, :, 0, 0 ,0] = cs[0]*Id
    G[:, :, 1, 0, 0] = cs[1]*X
    G[:, :, 0, 1, 0] = cs[1]*Y
    G[:, :, 0, 0, 1] = cs[1]*Z
    # G[:, :, 0, 0 ,0] = np.cos(0.24*3.14159)*Id
    # G[:, :, 1, 0, 0] = np.sin(0.24*3.14159)*X
    # G[:, :, 0, 1, 0] = np.sin(0.24*3.14159)*Y
    # G[:, :, 0, 0, 1] = np.sin(0.24*3.14159)*Z
    # G[:, :, 1, 1, 1] = cs[2]*Id
    return G, cs

def G_kitaev_gamma(params):
    K = 1
    Gamma = 1

    num_params = 9
    norm_factor = sum(i*i for i in params)
    norm_factor+= torch.tensor(1.0,dtype=torch.complex128,device='cpu')
    # cs = [torch.cos(params[0]), torch.sin(params[0])]

    cs=[]
    for i in range(num_params):
        cs.append((params[i]/torch.sqrt(norm_factor)).cpu())

    # for i in range(num_params):
    #     # cs.append(params[i].real/torch.sqrt(norm_factor))
    #     cs.append(params[i].real)
    G = torch.zeros((2,2,4,4,4),dtype=torch.complex128,device='cpu') # (s,s,Xv,Yv,Zv)
    G2 = torch.zeros((2,2,4,4,4),dtype=torch.complex128,device='cpu') # (s,s,Xv,Yv,Zv)
    # G = torch.zeros((2,2,2,2,2),dtype=torch.complex64,device='cpu') # (s,s,Xv,Yv,Zv)
    Id = torch.eye(2,dtype=torch.complex128,device='cpu')
    s2 = su2.SU2(2,dtype=torch.complex128,device='cpu')
    X = s2.SP()+s2.SM()
    Y = -(s2.SP()-s2.SM())*1j
    Z = 2*s2.SZ()

    # A[0,1,2][] = A[x,y,z][]
    # B[0,1,2][] = B[x,y,z][]
    A = torch.zeros((3,2,2,4),dtype=torch.complex128,device='cpu')
    B = torch.zeros((3,2,2,4),dtype=torch.complex128,device='cpu')
    A[0, :, :, 0] = Id
    A[0, :, :, 1] = X
    A[0, :, :, 2] = Y
    A[0, :, :, 3] = Z
    A[1, :, :, 0] = Id
    A[1, :, :, 1] = Y
    A[1, :, :, 2] = Z
    A[1, :, :, 3] = X
    A[2, :, :, 0] = Id
    A[2, :, :, 1] = Z
    A[2, :, :, 2] = X
    A[2, :, :, 3] = Y

    B[0, :, :, 0] = Id
    B[0, :, :, 1] = X
    B[0, :, :, 2] = Z
    B[0, :, :, 3] = Y
    B[1, :, :, 0] = Id
    B[1, :, :, 1] = Y
    B[1, :, :, 2] = X
    B[1, :, :, 3] = Z
    B[2, :, :, 0] = Id
    B[2, :, :, 1] = Z
    B[2, :, :, 2] = Y
    B[2, :, :, 3] = X

    for i in range(0,4):
        for j in range(0,4):
            for k in range(0,4):
                Ox = A[0, :, :, i]
                Oy = A[1, :, :, j]
                Oz = A[2, :, :, k]
                permuta = itertools.permutations([Ox,Oy,Oz])
                # print("len permuta: "+str(len(list(permuta))))
                for permu in list(permuta):
                    G[:, :, i, j, k] += permu[0]@permu[1]@permu[2]
                G[:, :, i, j, k] /= 6
                lst = [i,j,k]
                # These are zero 2x2 matrix
                # (1,1,1),(1,2,3),(2,2,2),(2,3,1),(3,1,2),(3,3,3)
                # (0,1,1),(0,1,2),(0,2,2),(0,2,3),(0,3,1),(0,3,3)
                # (1,0,1),(1,0,3),(1,1,0),(1,2,0),(2,0,1),(2,0,2)
                # (2,2,0),(2,3,0),(3,0,2),(3,0,3),(3,1,0),(3,3,0)
                if set(lst) == set([0,0,0]):
                    G[:, :, i, j, k] *= cs[0]
                if set(lst) == set([1,0,0]):
                    G[:, :, i, j, k] *= cs[1]
                cntgt1 = len(list(filter(lambda x: x>1, lst)))
                zero = len(list(filter(lambda x: x==0, lst)))
                nonzero = len(list(filter(lambda x: x!=0, lst)))
                if cntgt1==1 and zero==2:
                    G[:, :, i, j, k] *= cs[2]
                # if set(lst) == set([1,1,0]):
                #     G[:, :, i, j, k] *= cs[3]
                # if set(lst) == set([1,1,1]):
                #     G[:, :, i, j, k] *= cs[3]
                if set(lst) == set([0,1,2]) or set(lst) == set([0,1,3]):
                    G[:, :, i, j, k] *= cs[3]
                if set(lst) == set([0,2,3]):
                    G[:, :, i, j, k] *= cs[4]
                if set(lst) == set([1,1,2]) or set(lst) == set([1,1,3]):
                    G[:, :, i, j, k] *= cs[5]
                if set(lst) == set([1,2,2]) or set(lst) == set([1,3,3]):
                    G[:, :, i, j, k] *= cs[6]
                if set(lst) == set([2,2,3]) or set(lst) == set([2,3,3]):
                    G[:, :, i, j, k] *= cs[7]
    
    for i in range(0,4):
        for j in range(0,4):
            for k in range(0,4):
                Ox = B[0, :, :, i]
                Oy = B[1, :, :, j]
                Oz = B[2, :, :, k]
                permuta = itertools.permutations([Ox,Oy,Oz])
                # print("len permuta: "+str(len(list(permuta))))
                for permu in list(permuta):
                    G2[:, :, i, j, k] += permu[0]@permu[1]@permu[2]
                G2[:, :, i, j, k] /= 6
    return G, G2, cs

def symmetrize(A, B):
    return (A@B + B@A)/2

def symmetrize2(A, B, C):
    return (A@symmetrize(B,C) + B@symmetrize(A,C) + C@symmetrize(A,B))/3

def symmetrize3(A, B, C, D):
    return (A@symmetrize2(B,C,D) + B@symmetrize2(A,C,D) + C@symmetrize2(A,B,D)+ D@symmetrize2(A,B,C))/4

def G_kitaev_ani(params):
    num_params = 13
    norm_factor = sum(i*i for i in params).type(cfg.global_args.torch_dtype)
    norm_factor += torch.tensor(1.0,dtype=cfg.global_args.torch_dtype,device='cpu')
    c0 = 1.0/torch.sqrt(norm_factor).cpu()
    cs = [c0]
    for i in range(num_params):
        cs.append(params[i].cpu()/torch.sqrt(norm_factor).cpu())
    count = 0
    elecount = 0
    G = torch.zeros((2,2,2,2,2),dtype=torch.complex128,device='cpu') # (s,s,Xv,Yv,Zv)
    G2 = torch.zeros((2,2,2,2,2),dtype=torch.complex128,device='cpu') # (s,s,Xv,Yv,Zv)
    Id, X, Y, Z = paulis()
    Ax = [Id, X]
    Ay = [Id, Y]
    Az = [Id, Z]
    C = [Id, X+Y+Z]
    dec = [[0],[1]]
    uniques = [[0,0,0],[0,0,1],[0,1,1]]  # Note that [1,1,1] vanishes if we have C_0 and C_1.
    for ele in uniques:
        coms = list(set(itertools.permutations(ele)))
        for com in coms:
            com=list(com)
            for i in dec[com[0]]:
                for j in dec[com[1]]:
                    for k in dec[com[2]]:
                        G[:,:,i,j,k] = cs[count]*symmetrize3(Ax[i], Ay[j], Az[k], C[0])
                        G[:,:,i,j,k] += cs[count+1]*symmetrize3(Ax[i], Ay[j], Az[k], C[1])
                        G2[:,:,i,j,k] = symmetrize3(Ax[i], Ay[j], Az[k], C[0])
                        G2[:,:,i,j,k] += cs[count+1]*symmetrize3(Ax[i], Ay[j], Az[k], C[1])
                        elecount+=2
            count+=2
    # print("# of params = ", count)
    # print("# of assigned elements = ", elecount)
    return G, G2, cs

def G_gamma_systematic(params):

    num_params = 8-1

    for i in range(7):
        params[i] = params[i].real
    # print(params)

    norm_factor = sum(i*i for i in params[:7]).type(cfg.global_args.torch_dtype)
    norm_factor += torch.tensor(1.0,dtype=cfg.global_args.torch_dtype,device='cpu')
    c0 = 1.0/torch.sqrt(norm_factor).cpu()
    cs = [c0]
    for i in range(num_params):
        cs.append((params[i]/torch.sqrt(norm_factor)).cpu())
    
    count = 0
    elecount = 0
    G = torch.zeros((2,2,4,4,4),dtype=torch.complex128,device='cpu') # (s,s,Xv,Yv,Zv)
    G2 = torch.zeros((2,2,4,4,4),dtype=torch.complex128,device='cpu') # (s,s,Xv,Yv,Zv)

    Id, X, Y, Z = paulis()
    Ax = [Id, Y, Z, X]
    Ay = [Id, Z, X, Y]
    Az = [Id, X, Y, Z]
    Bx = [Id, Z, Y, X]
    By = [Id, X, Z, Y]
    Bz = [Id, Y, X, Z]
    dec = [[0],[1,2],[3]]

    # Note that [0,2,2] & [2,2,2] vanish.
    uniques = [[0,0,0], [0,0,2], [0,0,1], [1,1,1], [0,1,1], [1,1,2], [1,2,2], [0,1,2]]
    for ele in uniques:
        coms = list(set(itertools.permutations(ele)))
        for com in coms:
            com=list(com)
            for i in dec[com[0]]:
                for j in dec[com[1]]:
                    for k in dec[com[2]]:
                        G[:,:,i,j,k] = cs[count]*symmetrize2(Ax[i], Ay[j], Az[k])
                        G2[:,:,i,j,k] = symmetrize2(Bx[i], By[j], Bz[k])
                        elecount+=1
        count+=1
    # print("# of params = ", count)
    # print("# of assigned elements = ", elecount)
    return G, G2, cs

def LG(z):
    # norm_factor = sum(i*i for i in params)
    # cs = []
    # for i in range(3):
    #     cs.append(params[i]/torch.sqrt(norm_factor))
    
    z = z.cpu()
    
    Q = torch.zeros((2,2,2,2,2),dtype=cfg.global_args.torch_dtype,device='cpu') # (s,s,Xv,Yv,Zv)
    Id = torch.eye(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    X = (s2.SP()+s2.SM()).cpu()
    Y = -(s2.SP()-s2.SM()).cpu()*1j
    Z = 2*s2.SZ().cpu()
    #print("XXXXXXXXXXXXXXXXXXXX="+str(X))
    #print("YYYYYYYYYYYYYYYYYYYY="+str(Y))
    #print("ZZZZZZZZZZZZZZZZZZZZ="+str(Z))
    Q[:, :, 0, 0 ,0] = -1j*X@Y@Z
    Q[:, :, 1, 1, 0] = z*1*Z
    Q[:, :, 1, 0, 1] = z*1*Y
    Q[:, :, 0, 1, 1] = z*1*X
    return Q

def checkC6U6(G):
    Id = torch.eye(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    X = (s2.SP()+s2.SM())
    Y = -(s2.SP()-s2.SM())*1j
    Z = 2*s2.SZ()
    U = (Id+1j*X+1j*Y+1j*Z)/2
    print(U.conj().T@X@U)
    print(U.conj().T@Y@U)
    print(U.conj().T@Z@U)
    
    for k in range(4):
        G_ = torch.einsum("abijk,ma,bn->mnijk",G,U,U.conj().T)
        G_ = torch.permute(G_,(0,1,3,4,2))
        print((G-G_).norm())
        G = G_
    return 0
    # T_=torch.permute((0)

def paulis():
    Id = torch.eye(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    X = (s2.SP()+s2.SM())
    Y = -(s2.SP()-s2.SM())*1j
    Z = 2*s2.SZ()
    return Id, X, Y, Z

def state_test():

    from scipy import linalg

    """Returns magnetized state (i.e. polarized state) |0> = (1 1 1)."""

    tensor = torch.zeros((2,1,1,1,1),dtype=cfg.global_args.torch_dtype,device='cpu') # (s,s,Xv,Yv,Zv)
    tensor.requires_grad_(False)
    # _, sx, sy, sz = paulis()
    # e, v = linalg.eigh(-(sx + sy + sz))
    # for i, x in enumerate(v[:, 0]):
    #     tensor[i,0,0,0] = x
    # print(e)
    tensor[0,0,0,0,0] = 1
    tensor[1,0,0,0,0] = 1

    # import math

    # tmpb = math.sqrt((math.sqrt(3)-1)/4)
    # tensor[0,0,0,0] = torch.tensor((1/(2*tmpb)))
    # tensor[1,0,0,0] = torch.tensor(((1+1j)*tmpb))

    return tensor
def state_111():

    from scipy import linalg

    """Returns magnetized state (i.e. polarized state) |0> = (1 1 1)."""

    tensor = torch.zeros((2,1,1,1),dtype=cfg.global_args.torch_dtype,device='cpu') # (s,s,Xv,Yv,Zv)

    # _, sx, sy, sz = paulis()
    # e, v = linalg.eigh(-(sx + sy + sz))
    # for i, x in enumerate(v[:, 0]):
    #     tensor[i,0,0,0] = x
    # print(e)
    tensor[0,0,0,0] = -0.8881+0.0000j
    tensor[1,0,0,0] = -0.3251-0.3251j

    # import math

    # tmpb = math.sqrt((math.sqrt(3)-1)/4)
    # tensor[0,0,0,0] = torch.tensor((1/(2*tmpb)))
    # tensor[1,0,0,0] = torch.tensor(((1+1j)*tmpb))

    return tensor

def state_m1m1m1():

    from scipy import linalg

    """Returns magnetized state (i.e. polarized state) |0> = (1 1 1)."""

    tensor = torch.zeros((2,1,1,1),dtype=cfg.global_args.torch_dtype,device='cpu') # (s,s,Xv,Yv,Zv)

    _, sx, sy, sz = paulis()
    _, v = linalg.eigh((sx + sy + sz))
    for i, x in enumerate(v[:, 0]):
        tensor[i,0,0,0] = x

    # tensor[0,0,0,0] = -0.8881+0.0000j
    # tensor[1,0,0,0] = -0.3251-0.3251j

    # import math

    # tmpb = math.sqrt((math.sqrt(3)-1)/4)
    # tensor[0,0,0,0] = torch.tensor((1/(2*tmpb)))
    # tensor[1,0,0,0] = torch.tensor(((1+1j)*tmpb))

    return tensor    

def state_001():

    from scipy import linalg

    """Returns magnetized state (i.e. polarized state) |0> = (1 1 1)."""

    tensor = torch.zeros((2,1,1,1),dtype=cfg.global_args.torch_dtype,device='cpu') # (s,s,Xv,Yv,Zv)

    _, sx, sy, sz = paulis()
    _, v = linalg.eigh(-(sz))
    for i, x in enumerate(v[:, 0]):
        tensor[i,0,0,0] = x

    # tensor[0,0,0,0] = -0.8881+0.0000j
    # tensor[1,0,0,0] = -0.3251-0.3251j

    # import math

    # tmpb = math.sqrt((math.sqrt(3)-1)/4)
    # tensor[0,0,0,0] = torch.tensor((1/(2*tmpb)))
    # tensor[1,0,0,0] = torch.tensor(((1+1j)*tmpb))

    return tensor    

def zigzagstate():
    from scipy import linalg

    """Returns magnetized state (i.e. polarized state) |0> = (0 1 1)."""

    tensor = torch.zeros((2,1,1,1),dtype=cfg.global_args.torch_dtype,device='cpu') # (s,s,Xv,Yv,Zv)

    _, sx, sy, sz = paulis()
    _, v = linalg.eigh(-(sy + sz))
    for i, x in enumerate(v[:, 0]):
        tensor[i,0,0,0] = x

    return tensor

def rotation(state, axis, phi):
    # state is represented by tensor(2,1,1,1)
    # axis is represented by [ , , ]
    norm = 0
    for i in range(3):
        norm+=axis[i]*axis[i]
    for i in range(3):
        axis[i] = axis[i]/norm
    _, X, Y, Z = paulis()
    oper = torch.exp(1j*(phi/2)*(axis[0]*X+axis[1]*Y+axis[2]*Z))
    return torch.einsum('ij,jabc',oper,state)

if __name__ == "__main__":
    cfg.global_args.torch_dtype = torch.complex128

    # G, G2, cs = G_kitaev_gamma([1,1,1,1,1,1,1,1,1])
    # print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
    # print(G)
    # print("G2G2G2G2G2G2G2G2G2G2G2G2G2G2G2G2")
    # print(G2)

    # for i in range(0,4):
    #     for j in range(0,4):
    #         for k in range(0,4):
    #             if len(list(filter(lambda x: x==0, [i,j,k])))>=1:
    #                 print([i,j,k])
    #                 print(G[:, :, i,j,k])

    # G,G2,cs = G_gamma([0.,0.,0.,0.,0.])
    # for i in range(0,4):
    #     for j in range(0,4):
    #         for k in range(0,4):
    #             print([i,j,k])
    #             print(G[:, :, i,j,k])

    # checkC6U6(0)

    # single_A = zigzagstate()
    # Dsingle_A = single_A.conj()
    # s2 = su2.SU2(2,dtype=cfg.global_args.torch_dtype,device='cpu')
    # X = (s2.SP()+s2.SM())
    # Y = -(s2.SP()-s2.SM())*1j
    # Z = 2*s2.SZ()
    # XA = torch.einsum('ij,jabc->iabc',X,single_A)
    # AXA = torch.einsum('iabc,idef->abcdef',Dsingle_A,XA)
    # print("AXA: "+str(AXA))
    # YA = torch.einsum('ij,jabc->iabc',Y,single_A)
    # AYA = torch.einsum('iabc,idef->abcdef',Dsingle_A,YA)
    # print("AYA: "+str(AYA))
    # ZA = torch.einsum('ij,jabc->iabc',Z,single_A)
    # AZA = torch.einsum('iabc,idef->abcdef',Dsingle_A,ZA)
    # print("AZA: "+str(AZA))

    print("state_111(): "+str(state_111()))
    print("zigzag: "+str(zigzagstate()))
    import math
    print("1_2pi: "+str(rotation(state_111(),[0,0,1],math.pi*2)))
    print("1_pi: "+str(rotation(state_111(),[0,0,1],math.pi)))
    print("1_4pi: "+str(rotation(state_111(),[0,0,1],math.pi*4)))

    print("2_2pi: "+str(rotation(zigzagstate(),[0,0,1],math.pi*2)))
    print("2_pi: "+str(rotation(zigzagstate(),[0,0,1],math.pi)))
    print("2_4pi: "+str(rotation(zigzagstate(),[0,0,1],math.pi*4)))