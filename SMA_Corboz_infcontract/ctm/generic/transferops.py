import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
import config as cfg
import ipeps
from ctm.generic.env import ENV
from ctm.generic import corrf


def get_Top_spec(n, coord, direction, state, env, verbosity=0):
    chi = env.chi
    ad = state.get_aux_bond_dims()[0]

    # depending on the direction, get unit-cell length
    if direction == (1, 0) or direction == (-1, 0):
        N = state.lX
    elif direction == (0, 1) or direction == (0, -1):
        N = state.lY
    else:
        raise ValueError("Invalid direction: "+str(direction))

    # multiply vector by transfer-op within torch and pass the result back in numpy
    #  --0 (chi)
    # v--1 (D^2)
    #  --2 (chi)

    # if state and env are on gpu, the matrix-vector product can be performed
    # there as well. Price to pay is the communication overhead of resulting vector
    def _mv(v):
        c0 = coord
        V = torch.as_tensor(v, device=state.device)
        V = V.view(chi, ad*ad, chi)
        for i in range(N):
            V = corrf.apply_TM_1sO(c0, direction, state,
                                   env, V, verbosity=verbosity)
            c0 = (c0[0]+direction[0], c0[1]+direction[1])
        V = V.view(chi*ad*ad*chi)
        v = V.cpu().numpy()
        return v

    _test_T = torch.zeros(1, dtype=env.dtype)
    T = LinearOperator((chi*ad*ad*chi, chi*ad*ad*chi), matvec=_mv,
                       dtype="complex128" if _test_T.is_complex() else "float64")
    vals = eigs(T, k=n, v0=None, return_eigenvectors=False)

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    # sort by abs value in ascending order, then reverse order to descending
    ind_sorted = np.argsort(np.abs(vals))
    vals = vals[ind_sorted[::-1]]
    # vals= np.copy(vals[::-1]) # descending order
    vals = (1.0/np.abs(vals[0])) * vals
    L = torch.zeros((n, 2), dtype=torch.float64, device=state.device)
    L[:, 0] = torch.as_tensor(np.real(vals))
    L[:, 1] = torch.as_tensor(np.imag(vals))

    return L


def get_full_EH_spec_Ttensor(L, coord, direction, state, env,
                             verbosity=0):
    r"""
    :param L: width of the cylinder
    :type L: int
    :param coord: reference site (x,y)
    :type coord: tuple(int,int)
    :param direction: direction of the transfer operator. Either
    :type direction: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS_C4V
    :param env_c4v: corresponding environment
    :type env_c4v: ENV_C4V
    :return: leading n-eigenvalues, returned as rank-1 tensor 
    :rtype: torch.Tensor

    Compute the leading part of spectrum of :math:`exp(EH)`, where EH is boundary
    Hamiltonian. Exact :math:`exp(EH)` is given by the leading eigenvector of 
    transfer matrix::

         ...                PBC                                /
          |                  |                        |     --a*--
        --A(x,y)----       --A(x,y)------           --A-- =  /| 
        --A(x,y+1)--       --A(x,y+1)----             |       |/
        --A(x,y+2)--        ...                             --a--
          |                --A(x,y+L-1)--                    /
         ...                 |
                            PBC

        infinite exact TM; exact TM of L-leg cylinder  

    The :math:`exp(EH)` is then given by

    .. math::

        exp(-H_{ent}) = \sqrt{\sigma_R}\sigma_L\sqrt{\sigma_R}

    where :math:`\sigma_L,\sigma_R` are reshaped (D^2)^L left and right 
    leading eigenvectors of TM into :math:`D^L \times D^L` operator. Given that spectrum
    of :math:`AB` is equivalent to :math:`BA`, it is enough to diagonalize
    product :math:`\sigma_R\sigma_L` or :math:`\sigma_R\sigma_L`. 

    We approximate the :math:`\sigma_L,\sigma_R` of L-leg cylinder as MPO formed 
    by T-tensors of the CTM environment. Then, the spectrum of this approximate 
    exp(EH) is obtained through full diagonalization::

           0                    1
           |                    |
         --T[(x,y),(-1,0)]------T[(x,y),(1,0)]------
         --T[(x,y+1),(-1,0)]----T[(x,y+1),(1,0)]----
          ...                  ...
         --T[(x,y+L-1),(-1,0)]--T[(x,y+L-1),(1,0)]--
           0(PBC)               1(PBC)
    """
    chi = env.chi
    #             up        left       down      right
    dir_to_ind = {(0, -1): 1, (-1, 0): 2, (0, 1): 3, (1, 0): 4}
    ind_to_dir = dict(zip(dir_to_ind.values(), dir_to_ind.keys()))
    #
    # TM in direction (1,0) [right], grow in direction (0,1) [down]
    # TM in direction (0,1) [down], grow in direction (-1,0) [left]
    d_grow = ind_to_dir[dir_to_ind[direction] -
                        1 + ((4-dir_to_ind[direction]+1)//4)*4]
    d_opp = (-direction[0], -direction[1])
    if verbosity > 0:
        print(f"transferops.get_full_EH_spec_Ttensor direction {direction}"
              + f" growth d {d}")

    def _get_and_transform_T(c, d=direction):
        #
        # Return T-tensor as rank-4 tensor by permuting (bra,ket) aux index of T-tensor
        # to last position and then opening it
        #
        if d == (0, -1):
            #                              3
            # 0--T--2->1 => 0--T--1 ==> 0--T--1
            #    1->2          2           2
            return env.T[(c, (0, -1))].permute(0, 2, 1).contiguous().view(
                [chi]*2 + [state.site(c).size(dir_to_ind[(0, -1)])]*2)
        elif d == (-1, 0):
            # 0          0
            # T--2 => 2--T--3
            # 1          1
            return env.T[(c, (-1, 0))].view([chi]*2
                                            + [state.site(c).size(dir_to_ind[(-1, 0)])]*2)
        elif d == (0, 1):
            #    0->2         2           2
            # 1--T--2   => 0--T--1 ==> 0--T--1
            # ->0   ->1                   3
            return env.T[(c, (0, 1))].permute(1, 2, 0).contiguous().view(
                [chi]*2 + [state.site(c).size(dir_to_ind[(0, 1)])]*2)
        elif d == (1, 0):
            #       0          0         0
            # 2<-1--T =>    2--T  =>  2--T--3
            #    1<-2          1         1
            return env.T[(c, (1, 0))].permute(0, 2, 1).contiguous().view(
                [chi]*2 + [state.site(c).size(dir_to_ind[(1, 0)])]*2)

    if L == 1:
        c = state.vertexToSite(coord)
        sigma_0 = torch.einsum('iilr->lr', _get_and_transform_T(c))
        sigma_1 = torch.einsum('iilr->lr', _get_and_transform_T(c, d_opp))
        D, U = torch.linalg.eig(sigma_0@sigma_1)
        D_abs, inds = torch.sort(D.abs(), descending=True)
        D_sorted = D[inds]/D_abs[0]

        return D_sorted

    def get_sigma(d_sigma, d_grow):
        c = state.vertexToSite(coord)
        #    0
        # 2--T--3 =>    T--1,2;3
        #    1          0
        sigma = _get_and_transform_T(c, d_sigma).permute(1, 2, 3, 0)
        for i in range(1, L-1):
            #
            #    T--2i-1,2i;2i+1  =>   T--2i+1,2i+2;2i+3
            #   ...                   ...
            #    T--1,2                T--3,4
            #    0                     |
            #    0                     |
            # 2--T--3                  T--1,2
            #    1                     0
            #
            c = state.vertexToSite((c[0]+d_grow[0], c[1]+d_grow[1]))
            sigma = torch.tensordot(_get_and_transform_T(
                c, d_sigma), sigma, ([0], [0]))

        #
        #    T--2L-3,2L-2;2L-1  => T--2L-2,2L-1
        #   ...                   ...
        #    T--1,2                T--2,3
        #    0                     |
        #    0                     |
        # 2--T--3                  T--0,1
        #    1
        #
        c = state.vertexToSite((c[0]+d_grow[0], c[1]+d_grow[1]))
        sigma = torch.tensordot(_get_and_transform_T(
            c, d_sigma), sigma, ([0, 1], [0, 2*L-1]))
        sigma = sigma.permute(list(range(0, 2*L, 2))+list(range(1, 2*L+1, 2))).contiguous()\
            .view(np.prod(sigma.size()[0:L]), np.prod(sigma.size()[L:]))
        return sigma

    sigma_0 = get_sigma(direction, d_grow)
    sigma_1 = get_sigma(d_opp, d_grow)

    D, U = torch.linalg.eig(sigma_0@sigma_1)
    D_abs, inds = torch.sort(D.abs(), descending=True)
    D_sorted = D[inds]/D_abs[0]
    return D_sorted
