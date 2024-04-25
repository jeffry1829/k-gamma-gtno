
using PyCall

function kitaevgs(Jx, Jy, Jz, h)
    py"""
        import numpy as np
        npfile = np.array([1])
        def kitaevgs(Jx, Jy, Jz, h):
            return np.load("/home/petjelinux/k-gamma-gtno/SMA_Corboz_TFIM/KitaevLGJx{}Jy{}Jz{}h{}dtypecomplex128.npy".format(Jx, Jy, Jz, h))
        """
    # py"anisokgs"(Kz, icnt)
    # npfile = PyArray(py"npfile"o)
    npfile = py"kitaevgs"o(Jx, Jy, Jz, h)
    
    W = zeros(ComplexF64, size(npfile)[1], size(npfile)[2], size(npfile)[2], size(npfile)[2], size(npfile)[2])
    for i in 1:size(npfile)[1], j in 1:size(npfile)[2], k in 1:size(npfile)[2], l in 1:size(npfile)[2], m in 1:size(npfile)[2]
        W[i, j, k, l, m] = npfile[i, j, k, l, m]
    end
    W = ncon([W, conj(W)],
        [[1, -1, -3, -5, -7], [1, -2, -4, -6, -8]])
    W = reshape(W, (size(npfile)[2]^2, size(npfile)[2]^2, size(npfile)[2]^2, size(npfile)[2]^2))
    return W
end