## Reproducing Fig.7(c.d) in 10.1038/ncomms9284
using DelimitedFiles
using PyCall
include("../vumpsfixedpoints.jl")
include("../excitationmpo.jl")
include("tfimtensor.jl")
##
folder_out = "TFIM/"
if isdir(folder_out) == false
    mkdir(folder_out)
end
##
######### Initial states
# βx, βz = 0.4, 1.3
hx = 1.0
Dmps = 16

filename = "hx$(hx)Dmps$(Dmps)"
# sZ = [1. 0.; 0. -1.]
# sI = [1. 0; 0. 1.]
# ZZ = reshape(ncon([sZ,sZ],[[-1,-3],[-2,-4]]), 4,4)
# IZ = reshape(ncon([sI,sZ],[[-1,-3],[-2,-4]]), 4,4)
# beta_x, beta_z = 1.5, 0.5


# toric = symstringtoric(βx, βz)
# dim = size(toric)[2]
# W = ncon([toric, conj(toric)],
#          [[1, -1, -3, -5, -7], [1, -2, -4, -6, -8]])
# d = dim^2

# py"""
#     import numpy as np
#     npfile = np.load("/Users/petjelinux/Downloads/mpo_excitation_cytnx/1d/q0hx1.0.npy")
#     """
# W = PyArray(py"npfile"o)
# dim = size(W)[2]
# W = ncon([W, conj(W)],
#     [[1, -1, -3, -5, -7], [1, -2, -4, -6, -8]])
# d = dim^2
# W = reshape(W, (d, d, d, d));

##
W = tfimgs(hx)
d = size(W)[1]
AL = rand(Dmps, d, Dmps);

########## Fixed points

λ, AL, C, AR, FL, FR = vumpsfixedpts(AL, W; tol=1e-6);
W = W / λ;
##
########## Excitation
num_ω = 15
p_list = LinRange(0, 1, 11)

open(folder_out * "$(filename)_II.txt", "w") do io
    write(io, "# p ω \n")
end
# open(folder_out*"$(filename)_IZ.txt", "w") do io
#     write(io, "# p ω \n")
# end
# open(folder_out*"$(filename)_ZZ.txt", "w") do io
#     write(io, "# p ω \n")
# end

for p in p_list
    data = excitation(W, AL, AR, C, FL, FR, num_ω, p; charge=false, domain=false,
        Fstring=Nothing, verbose=true)
    ϕ = data["ϕ"]
    ω = data["ω"]
    # println(size(ω_Ceq))
    len = length(ω)
    open(folder_out * "$(filename)_II.txt", "a") do io
        writedlm(io, reshape([p; ω[1:len]], 1, len + 1))
    end
end
##
# for p in p_list
#     data = excitation(W,AL,AR,C,FL,FR, num_ω, p; charge = false,domain = true, 
#     Fstring = IZ, verbose = true)
#     ϕ = data["ϕ"]; ω = data["ω"]
#     # println(size(ω_Ceq))
#     len = length(ω)
#     open(folder_out*"$(filename)_IZ.txt", "a") do io
#         writedlm(io, reshape([p; ω[1:len]], 1,len+1))
#     end   
# end
# ##
# for p in p_list
#     data = excitation(W,AL,AR,C,FL,FR, num_ω, p; charge = false,domain = true, 
#     Fstring = ZZ, verbose = true)
#     ϕ = data["ϕ"]; ω = data["ω"]
#     # println(size(ω_Ceq))
#     len = length(ω)
#     open(folder_out*"$(filename)_ZZ.txt", "a") do io
#         writedlm(io, reshape([p; ω[1:len]], 1,len+1))
#     end   
# end