import itertools
import numpy as np

Id = np.zeros([2,2],dtype=np.complex128)
Id[0, 0] = 1
Id[1, 1] = 1
X = np.zeros([2,2],dtype=np.complex128)
X[1, 0] = 1
X[0, 1] = 1
Z = np.zeros([2,2],dtype=np.complex128)
Z[0, 0] = 1
Z[1, 1] = -1
Y = np.zeros([2,2],dtype=np.complex128)
Y[1, 0] = -1j
Y[0, 1] = 1j

def symmetrize(A, B):
    return (A@B + B@A)/2

def symmetrize2(A, B, C):
    return (A@symmetrize(B,C) + B@symmetrize(A,C) + C@symmetrize(A,B))/3

def symmetrize3(A, B, C, D):
    return (A@symmetrize2(B,C,D) + B@symmetrize2(A,C,D) + C@symmetrize2(A,B,D)+ D@symmetrize2(A,B,C))/4

def state_111():

    from scipy import linalg

    """Returns magnetized state (i.e. polarized state) |0> = (1 1 1)."""

    tensor = np.zeros((2,1,1,1),dtype=np.complex128) # (s,s,Xv,Yv,Zv)
    tensor[0,0,0,0] = -0.8881+0.0000j
    tensor[1,0,0,0] = -0.3251-0.3251j
    return tensor

def state_m1m1m1():

    from scipy import linalg

    """Returns magnetized state (i.e. polarized state) |0> = (1 1 1)."""

    tensor = np.zeros((2,1,1,1),dtype=np.complex128) # (s,s,Xv,Yv,Zv)
    _, v = linalg.eigh((X + Y + Z))
    for i, x in enumerate(v[:, 0]):
        tensor[i,0,0,0] = x
    return tensor   

def G_kitaev_ani_hard(params):
    num_params = 11
    cs = []
    for i in range(num_params):
        cs.append(params[i])
    count = 0
    elecount = 0
    G = np.zeros((2,2,2,2,2),dtype=np.complex128) # (s,s,Xv,Yv,Zv)
    G2 = np.zeros((2,2,2,2,2),dtype=np.complex128) # (s,s,Xv,Yv,Zv)
    
    Ax = [Id, X]
    Ay = [Id, Y]
    Az = [Id, Z]
    C = [Id, X+Y+Z]

    # Note that [1,1,0]..s vanished if we have only C_0.
    # Note that [1,1,1] vanishes if we have C_0 or C_1.
    
    # Kx=Ky=Kz non-zeros: c0 c2=c4
    # Kx=Ky!=Kz non-zeros: c0 c2 c4
    # Kx=Ky=Kz, h!=0  non-zeros: c0 c1 c2=c4 c3=c5 c6=c8 c7=c9
    # Kx=Ky!=Kz, h!=0  non-zeros: c0 c1 c2 c4 c3 c5 c6 c8 c7 c9
    
    G[:,:,0,0,0] = cs[0]*symmetrize3(Ax[0], Ay[0], Az[0], C[0])
    G[:,:,0,0,0] += cs[1]*symmetrize3(Ax[0], Ay[0], Az[0], C[1])
    
    G[:,:,1,0,0] = cs[2]*symmetrize3(Ax[1], Ay[0], Az[0], C[0])
    G[:,:,1,0,0] += cs[3]*symmetrize3(Ax[1], Ay[0], Az[0], C[1])
    
    G[:,:,0,0,1] = cs[4]*symmetrize3(Ax[0], Ay[0], Az[1], C[0])
    G[:,:,0,0,1] += cs[5]*symmetrize3(Ax[0], Ay[0], Az[1], C[1])
    
    G[:,:,0,1,0] = cs[2]*symmetrize3(Ax[0], Ay[1], Az[0], C[0])
    G[:,:,0,1,0] += cs[3]*symmetrize3(Ax[0], Ay[1], Az[0], C[1])   

    G[:,:,1,0,1] = cs[6]*symmetrize3(Ax[1], Ay[0], Az[1], C[0])
    G[:,:,1,0,1] += cs[7]*symmetrize3(Ax[1], Ay[0], Az[1], C[1]) 

    G[:,:,1,1,0] = cs[8]*symmetrize3(Ax[1], Ay[1], Az[0], C[0])
    G[:,:,1,1,0] += cs[9]*symmetrize3(Ax[1], Ay[1], Az[0], C[1]) 

    G[:,:,0,1,1] = cs[6]*symmetrize3(Ax[0], Ay[1], Az[1], C[0])
    G[:,:,0,1,1] += cs[7]*symmetrize3(Ax[0], Ay[1], Az[1], C[1]) 

    
    G2[:,:,0,0,0] = symmetrize3(Ax[0], Ay[0], Az[0], C[0])
    G2[:,:,0,0,0] += cs[1]*symmetrize3(Ax[0], Ay[0], Az[0], C[1])
    
    G2[:,:,1,0,0] = symmetrize3(Ax[1], Ay[0], Az[0], C[0])
    G2[:,:,1,0,0] += cs[3]*symmetrize3(Ax[1], Ay[0], Az[0], C[1])
    
    G2[:,:,0,0,1] = symmetrize3(Ax[0], Ay[0], Az[1], C[0])
    G2[:,:,0,0,1] += cs[5]*symmetrize3(Ax[0], Ay[0], Az[1], C[1])
    
    G2[:,:,0,1,0] = symmetrize3(Ax[0], Ay[1], Az[0], C[0])
    G2[:,:,0,1,0] += cs[3]*symmetrize3(Ax[0], Ay[1], Az[0], C[1])   

    G2[:,:,1,0,1] = symmetrize3(Ax[1], Ay[0], Az[1], C[0])
    G2[:,:,1,0,1] += cs[7]*symmetrize3(Ax[1], Ay[0], Az[1], C[1]) 

    G2[:,:,1,1,0] = symmetrize3(Ax[1], Ay[1], Az[0], C[0])
    G2[:,:,1,1,0] += cs[9]*symmetrize3(Ax[1], Ay[1], Az[0], C[1]) 

    G2[:,:,0,1,1] = symmetrize3(Ax[0], Ay[1], Az[1], C[0])
    G2[:,:,0,1,1] += cs[7]*symmetrize3(Ax[0], Ay[1], Az[1], C[1]) 
    
    # print("# of params = ", count)
    # print("# of assigned elements = ", elecount)
    return G, G2, cs

def LG_h(c0,c1,c2):
    Q = np.zeros((2,2,2,2,2),dtype = np.complex128) # (s,s,Xv,Yv,Zv)
    Q[:, :, 0, 0 ,0] = c0*Id+c1*(X+Y+Z)
    Q[:, :, 1, 1, 0] = c2*(c1*Id+c0*Z)
    Q[:, :, 1, 0, 1] = c2*(c1*Id+c0*Y)
    Q[:, :, 0, 1, 1] = c2*(c1*Id+c0*X)
    return Q