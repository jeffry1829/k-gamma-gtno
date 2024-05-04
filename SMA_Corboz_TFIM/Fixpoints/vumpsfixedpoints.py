# Follow SciPost Phys. Lect. Notes 7 (2019)
import scipy.linalg as linalg
import numpy as np
# import cytnx as cy
from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs, eigsh
from canonical import *

"""
    leftenv(A, M, FL; kwargs)
Compute the left environment tensor for MPS `A` and MPO `W`, by finding the left fixed point
of `A - M - conj(A)` contracted along the physical dimension.
"""


def leftenv(A, W, FL, tol):
    D = A.shape[0]
    dw = W.shape[0]
    if FL is None:
        FL = np.random.rand(D, dw, D)

    def lcon(v):
        return ncon([v.reshape(D, dw, D), A, W, A.conj()], [[1, 2, 3], [3, 5, -3], [2, 5, -2, 4], [1, 4, -1]]).flatten()
    lop = LinearOperator((D*dw*D, D*dw*D), matvec=lcon, dtype=np.cdouble)
    e, FL = eigs(lop, v0=FL.flatten(), k=1, which='LM', tol=tol)

    return FL.reshape(D, dw, D), e

# """
#     leftenv(A, M, FL; kwargs)
# Compute the left environment tensor for MPS `A` and MPO `W`, by finding the
# left fixed point
# of `A - M - conj(A)` contracted along the physical dimension.
# """
# function leftenv(A::Dims3Array, W::Dims4Array, FL = randn(eltype(A), size(A,1), size(W,1), size(A,1)); kwargs...)
#     λs, FLs, info = eigsolve(FL, 1, :LM; ishermitian = false, kwargs...) do FL
#         FL = ncon([FL,A,W,conj(A)], [[1,2,3], [3,5,-3], [2,5,-2,4], [1,4,-1]])
#     end
#     return FLs[1], real(λs[1]), info
# end


"""
    rightenv(A, M, FR; kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the right fixed point
of A - M - conj(A) contracted along the physical dimension.
"""


def rightenv(A, W, FR, tol):
    D = A.shape[0]
    dw = W.shape[0]
    if FR is None:
        FR = np.random.rand(D, dw, D)

    def rcon(v):
        return ncon([v.reshape(D, dw, D), A, W, A.conj()], [[1, 2, 3], [-1, 4, 1], [-2, 4, 2, 5], [-3, 5, 3]]).flatten()
    rop = LinearOperator((D*dw*D, D*dw*D), matvec=rcon, dtype=np.cdouble)
    e, FR = eigs(rop, v0=FR.flatten(), k=1, which='LM', tol=tol)

    return FR.reshape(D, dw, D), e

# """
#     rightenv(A, M, FR; kwargs...)

# Compute the right environment tensor for MPS A and MPO M, by finding the right fixed point
# of A - M - conj(A) contracted along the physical dimension.
# """
# function rightenv(A::Dims3Array, W::Dims4Array, FR = randn(eltype(A), size(A,1), size(W,1), size(A,1)); kwargs...)
#     λs, FRs, info = eigsolve(FR, 1, :LM; ishermitian = false, kwargs...) do FR
#         # @tensor FR[α,a,β] := A[α,s',α']*FR[α',a',β']*M[a,s,a',s']*conj(A[β,s,β'])
#         FR = ncon([FR,A,W,conj(A)], [[1,2,3],[-1,4,1],[-2,4,2,5],[-3,5,3]])
#     end
#     return FRs[1], real(λs[1]), info
# end


"""
    vumpsfixedpts(A,W;)
Given initial mps A and the double tensor W, return λ, AL, C, AR, FL, FR
"""


def vumpsfixedpts(A, W, verbose=True, steps=100, tol=1e-6):
    eigtol = 1e-3
    print("Eigensolver tol=", eigtol)
    print(A.shape, W.shape)
    D, d, _ = A.shape

    # Algorithm4
    AL = np.random.rand(D, d, D)
    AR = np.random.rand(D, d, D)
    C = np.random.rand(D, D)

    # AL, AR, C = mixcanonical(A)
    FL, eL = leftenv(AL, W, FL=None, tol=tol)
    FR, eR = rightenv(AR, W, FR=None, tol=tol)
    # normalize FL and FR: necessary!
    FR /= ncon([FL, C, C.conj(), FR], [[3, 2, 1], [1, 4], [3, 5], [4, 2, 5]])

    iteration = 0
    err = 1
    lambd = 0  # initial value, not important
    while (err > tol) and (iteration < steps):

        def applyH1(AC):
            return ncon([AC.reshape(D, d, D), FL, FR, W], [[1, 5, 2], [-1, 3, 1], [2, 4, -3], [3, 5, 4, -2]]).flatten()

        def applyH2(C):
            return ncon([C.reshape(D, D), FL, FR], [[1, 2], [-1, 3, 1], [2, 3, -2]]).flatten()
        applyH1op = LinearOperator(
            (D*d*D, D*d*D), matvec=applyH1, dtype=np.cdouble)
        applyH2op = LinearOperator(
            (D*D, D*D), matvec=applyH2, dtype=np.cdouble)

        AC = ncon([AL, C], [[-1, -2, 1], [1, -3]])
        eac, AC = eigs(applyH1op, v0=AC.flatten(), k=1,
                       which='LM', maxiter=1, tol=eigtol)  # update AC
        ec, C = eigs(applyH2op, v0=C.flatten(), k=1,
                     which='LM', maxiter=1, tol=eigtol)  # update C
        lambd = eac/ec
        AC = AC.reshape(D, d, D)
        C = C.reshape(D, D)

        AL, AR, errL, errR = min_AC_C(AC, C)  # transform back to AL and AR

        # AL, C, _ = leftorth(AR, C, tol = tol/10) # regauge MPS: not really necessary
        # update FL and FR
        FL, eL = leftenv(AL, W, FL=FL, tol=tol/10)
        FR, eR = rightenv(AR, W, FR=FR, tol=tol/10)
        # FR /= ncon([FL ,C, C.conj(),FR], [[3,2,1], [1,4], [3,5], [4,2,5]]) # normalize FL and FR: not necessary.

        # Convergence measure: norm of the projection of the residual onto the tangent space
        AC = ncon([AL, C], [[-1, -2, 1], [1, -3]])
        # normalize FL and FR: not necessary.
        FR /= ncon([FL, AC, W, AC.conj(), FR], [[6, 8, 7],
                   [6, 2, 1], [8, 2, 9, 4], [7, 4, 5], [1, 9, 5]])
        MAC = ncon([AC, FL, FR, W], [[1, 5, 2],
                   [-1, 3, 1], [2, 4, -3], [3, 5, 4, -2]])
        MAC -= ncon([AL, AL.conj(), MAC], [[-1, -2, 1], [2, 3, 1], [2, 3, -3]])
        err = norm(MAC)
        iteration += 1

        if verbose:
            print("Step %d: λ ≈ %.6e , err ≈ %.6e" %
                  (iteration, np.real(lambd), err))
            print("e = ", lambd)
            print("eL = ", eL)
            print("eR = ", eR)
    return lambd, AL, C, AR, FL, FR


def toLambdaGamma(AL):
    chil, d, chir = AL.shape[0], AL.shape[1], AL.shape[2]
    u, s, vt = linalg.svd(AL.reshape(chil*d, chir), full_matrices=False)
    Gamma = ncon([vt, u.reshape(chil, d, chir)], [[-1, 1], [1, -2, -3]])
    Lambda = np.diag(s)/linalg.norm(s, ord=np.inf)
    return Lambda, Gamma


def contractLambdaGamma(Lambda, Gamma):
    return ncon([Lambda, Gamma], [[-1, 1], [1, -2, -3]])


def __main__():
    D = 2
    d = 2
    AL = np.random.rand(D, d, D)
    W = np.random.rand(D, D, D, D)
    lambd, AL, C, AR, FL, FR = vumpsfixedpts(AL, W)
    ALcAL = ncon([AL, AL.conj()], [[-1, 1, -3], [-2, 1, -4]]
                 ).reshape(D**2, D**2)
    print("ALcAL spec = ", linalg.eigvals(ALcAL))
    # print("lambd = ", lambd)
    # print("AL = ", AL)
    # print("C = ", C)
    # print("AR = ", AR)
    # print("FL = ", FL)
    # print("FR = ", FR)
    Lambda, Gamma = toLambdaGamma(AL)
    # print("Lambda = ", Lambda)
    # print("Gamma = ", Gamma)
    # print("Lambda*Gamma = ", ncon([Lambda, Gamma], [[-1, 1], [1, -2, -3]]))
    AL2 = ncon([Lambda, Gamma], [[-1, 1], [1, -2, -3]])
    W2 = ncon([AL2, AL2.conj()], [[-1, 1, -3],
                                  [-2, 1, -4]]).reshape(D**2, D**2)
    print("W2 spec = ", linalg.eigvals(W2))


if __name__ == '__main__':
    __main__()
# """
#     vumpsfixedpts(A,W;)
# Given initial mps A and the double tensor W, return λ, AL, C, AR, FL, FR
# """
# function vumpsfixedpts(A::Dims3Array, W::Dims4Array; verbose = true, steps = 100, tol = 1e-6, kwargs...)
#     ## Algorithm4
#     AL, AR, C = mixcanonical(A)
#     FL, λL = leftenv(AL, W; kwargs...)
#     FR, λR = rightenv(AR, W; kwargs...)
#     FR ./= @tensor scalar(FL[c,b,a]*C[a,a']*conj(C[c,c'])*FR[a',b,c']) # normalize FL and FR: necessary!
#     iter = 0 # vumps step
#     err = 1; λ = 0 #initial value, not important
#     while err > tol && iter < steps
#         @tensor AC[a,s,b] := AL[a,s,b']*C[b',b]
#         applyH1 = (AC, FL, FR, W) -> ncon([AC,FL,FR,W], [[1,5,2], [-1,3,1], [2,4,-3], [3,5,4,-2]])
#         applyH0 = (C, FL, FR) -> ncon([C, FL, FR], [[1,2], [-1,3,1], [2,3,-2]])
#         ## update AC
#         μ1s, ACs, info1 = eigsolve(x->applyH1(x, FL, FR, W), AC, 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
#         ## update C
#         μ0s, Cs, info0 = eigsolve(x->applyH0(x, FL, FR), C, 1; ishermitian = false, maxiter = 1, kwargs...)
#         λ = real(μ1s[1]/μ0s[1])
#         AC = ACs[1]
#         C = Cs[1]
#         AL, AR, errL, errR = min_AC_C(AC,C) ## transform back to AL and AR
#         AL, C, = leftorth(AR, C; tol = tol/10, kwargs...) # regauge MPS: not really necessary
#         ## update FL and FR
#         FL, λL = leftenv(AL, W, FL; tol = tol/10, kwargs...)
#         FR, λR = rightenv(AR, W, FR; tol = tol/10, kwargs...)
#         FR ./= @tensor scalar(FL[c,b,a]*C[a,a']*conj(C[c,c'])*FR[a',b,c']) # normalize FL and FR: not really necessary
#         # Convergence measure: norm of the projection of the residual onto the tangent space
#         @tensor AC[a,s,b] := AL[a,s,b']*C[b',b];
#         MAC = applyH1(AC, FL, FR, W);
#         @tensor MAC[a,s,b] -= AL[a,s,b']*(conj(AL[a',s',b'])*MAC[a',s',b]);
#         err = norm(MAC);
#         iter += 1
#         verbose && println("Step $(iter): λ ≈ $λ ≈ $λL ≈ $λR, err ≈ $err")
#     end
#     return λ, AL, C, AR, FL, FR
# end
