
using PyCall

function tfimgs(hx, q=0)
    py"""
        import numpy as np
        npfile = np.load("/Users/petjelinux/Downloads/mpo_excitation_cytnx/1d/q0hx1.0.npy")
        """
    npfile = PyArray(py"npfile"o)

    W = zeros(size(npfile)[1], size(npfile)[2], size(npfile)[2], size(npfile)[2], size(npfile)[2])
    for i in 1:size(npfile)[1], j in 1:size(npfile)[2], k in 1:size(npfile)[2], l in 1:size(npfile)[2], m in 1:size(npfile)[2]
        W[i, j, k, l, m] = npfile[i, j, k, l, m]
    end
    W = ncon([W, conj(W)],
        [[1, -1, -3, -5, -7], [1, -2, -4, -6, -8]])
    W = reshape(W, (size(npfile)[2]^2, size(npfile)[2]^2, size(npfile)[2]^2, size(npfile)[2]^2))
    return W
end