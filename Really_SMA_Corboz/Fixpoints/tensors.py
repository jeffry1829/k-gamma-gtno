import numpy as np
from ncon import ncon

def prod(a):
    import operator
    from functools import reduce
    return reduce(operator.mul, a, 1)

def eijk(args):
    from math import factorial
    n = len(args)
    return prod(prod(args[j] - args[i] for j in range(i + 1, n)) / factorial(i) for i in range(n))

def zbar():
    z = np.zeros([3, 3])
    z[0, 0] = 1
    z[1, 1] = 1
    z[2, 2] = -1
    return z

def rvb():
    P = np.zeros([2, 3, 3])
    P[0, 0, 2] = P[0, 2, 0] = P[1, 1, 2] = P[1, 2, 1] = 1.
    epsilon = np.zeros([3, 3, 3])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                epsilon[i, j, k] = eijk((i,j,k))
                # epsilon[i, j, k] = eijk((i,j,k))*2**(-1/2)
    epsilon[2, 2, 2] = 1
    ePPP = ncon([epsilon, P, P, P], [[1, 2, 3], [-4,1,-1], [-5,2,-2], [-6,3,-3]]).reshape(3,3,3,8)
    ePPPe = ncon([ePPP, epsilon], [[-1, 1, -2, -5], [1, -4, -3]])
    return ePPPe/np.max(ePPPe)

def rvbPx(x):
    Px = np.zeros([2, 2, 3, 3])
    Px[0, 0, 0, 2] = Px[0, 0, 2, 0] = Px[0, 1, 1, 2] = Px[0, 1, 2, 1] = 1.
    Px[1, 0, 0, 2] = Px[1, 1, 1, 2] = x
    Px[1, 0, 2, 0] = Px[1, 1, 2, 1] = -x
    epsilon = np.zeros([3, 3, 3])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                epsilon[i, j, k] = eijk((i,j,k))
                # epsilon[i, j, k] = eijk((i,j,k))*2**(-1/2)
    epsilon[2, 2, 2] = 1
    P = Px.reshape(4, 3, 3)
    ePPP = ncon([epsilon, P, P, P], [[1, 2, 3], [-4,1,-1], [-5,2,-2], [-6,3,-3]]).reshape(3,3,3,64)
    ePPPe = ncon([ePPP, epsilon], [[-1, 1, -2, -5], [1, -4, -3]])
    return ePPPe/np.max(ePPPe)

def spins():
    # sI = np.eye(3)
    # sX = np.array([[0, 1, 0], [1, 0, 1],[0, 1, 0]])
    # sY = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
    # sZ = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    sI = np.eye(2)
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0, -1]])
    return sI, sX, sY, sZ

def AKLTsquare():
    bond = np.zeros([2, 2])
    bond[0, 1] = 2**(-1/2); bond[1, 0] = -2**(-1/2)
    I = np.eye(2)
    Y = np.zeros([2, 2], dtype = np.csingle)
    Y[0, 1] = -1; Y[1, 0] = 1
    P = np.zeros([5, 2, 2, 2, 2])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if i+j+k+l == 0:
                        P[0 ,i, j, k, l] = 1
                    elif i+j+k+l == 1:
                        P[1 ,i, j, k, l] = 4**(-1/2)
                    elif i+j+k+l == 2: 
                        P[2 ,i, j, k, l] = 6**(-1/2)
                    elif i+j+k+l == 3:  
                        P[3 ,i, j, k, l] = 4**(-1/2)  
                    elif i+j+k+l == 4:   
                        P[4 ,i, j, k, l] = 1        
    # aklt = ncon([P, Y, Y, I, I], [[-1, 1, 2, 3, 4],[1, -2], [2, -3], [3, -4], [4, -5]]) ### works
    aklt = ncon([P, bond, bond, bond, bond], [[-1, 1, 2, 3, 4],[1, -2], [2, -3], [3, -4], [4, -5]])
    aklt = ncon([P, Y, Y], [[-1, 1, 2, -4, -5],[1, -2], [2, -3]]) 
    return aklt


def symmetrize(A, B):
    return (np.matmul(A,B) + np.matmul(B,A))/2

def symmetrize2(A, B, C):
    return (np.matmul(A,symmetrize(B,C)) + np.matmul(B,symmetrize(A,C)) + np.matmul(C,symmetrize(A,B)))/3

def symmetrize3(A, B, C, D):
    return (np.matmul(A,symmetrize2(B,C,D)) + np.matmul(B,symmetrize2(A,C,D)) + np.matmul(C,symmetrize2(A,B,D))+ np.matmul(D,symmetrize2(A,B,C)))/4

import itertools

def G_kitaev_ani(params):
    cs = params
    # num_params = 14
    # for i in range(num_params):
    #     cs.append(params[i])
    count = 0
    elecount = 0
    G = np.zeros([2,2,2,2,2],dtype=np.complex128) # (s,s,Xv,Yv,Zv)
    G2 = np.zeros([2,2,2,2,2],dtype=np.complex128) # (s,s,Xv,Yv,Zv)
    Id, X, Y, Z = spins()
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
            # print("com : ", com," cs0 = ", cs[count].item().real, " cs1 = ", cs[count+1].item().real)
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


def LG(z):
    # norm_factor = sum(i*i for i in params)
    # cs = []
    # for i in range(3):
    #     cs.append(params[i]/np.sqrt(norm_factor))
    Q = np.zeros((2,2,2,2,2),dtype=np.complex128) # (s,s,Xv,Yv,Zv)
    Id, X, Y, Z = spins()
    #print("XXXXXXXXXXXXXXXXXXXX="+str(X))
    #print("YYYYYYYYYYYYYYYYYYYY="+str(Y))
    #print("ZZZZZZZZZZZZZZZZZZZZ="+str(Z))
    Q[:, :, 0, 0 ,0] = -1j*X@Y@Z
    Q[:, :, 1, 1, 0] = z*1*Z
    Q[:, :, 1, 0, 1] = z*1*Y
    Q[:, :, 0, 1, 1] = z*1*X
    return Q


def state_111():

    from scipy import linalg

    """Returns magnetized state (i.e. polarized state) |0> = (1 1 1)."""

    tensor = np.zeros((2,1,1,1),dtype=np.complex128) # (s,s,Xv,Yv,Zv)

    # _, sx, sy, sz = paulis()
    # e, v = linalg.eigh(-(sx + sy + sz))
    # for i, x in enumerate(v[:, 0]):
    #     tensor[i,0,0,0] = x
    # print(e)
    tensor[0,0,0,0] = -0.8881+0.0000j
    tensor[1,0,0,0] = -0.3251-0.3251j

    # import math

    # tmpb = math.sqrt((math.sqrt(3)-1)/4)
    # tensor[0,0,0,0] = np.tensor((1/(2*tmpb)))
    # tensor[1,0,0,0] = np.tensor(((1+1j)*tmpb))

    return tensor

# def state_m1m1m1():

#     from scipy import linalg

#     """Returns magnetized state (i.e. polarized state) |0> = (1 1 1)."""

#     tensor = np.zeros((2,1,1,1),dtype=cfg.global_args.torch_dtype,device='cpu') # (s,s,Xv,Yv,Zv)

#     _, sx, sy, sz = spins()
#     _, v = linalg.eigh((sx + sy + sz))
#     for i, x in enumerate(v[:, 0]):
#         tensor[i,0,0,0] = x

#     # tensor[0,0,0,0] = -0.8881+0.0000j
#     # tensor[1,0,0,0] = -0.3251-0.3251j

#     # import math

#     # tmpb = math.sqrt((math.sqrt(3)-1)/4)
#     # tensor[0,0,0,0] = np.tensor((1/(2*tmpb)))
#     # tensor[1,0,0,0] = np.tensor(((1+1j)*tmpb))

#     return tensor    
