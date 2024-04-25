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
windowL = 1

filename = "hx$(hx)Dmps$(Dmps)"

W = tfimgs(hx)
d = size(W)[1]
AL = rand(Dmps, d, Dmps);

########## Fixed points

λ, AL, C, AR, FL, FR = vumpsfixedpts(AL, W; tol=1e-6);
W = W / λ;
@tensor AC[a, s, b] := AL[a, s, b'] * C[b', b]
@tensor tmpN[a, b, f, g] := FL[d, e, a] * AC[d, h, f] * W[e, h, g, b]
for i in 2:wi
    ndowL
    @tebsor tmpN[a, b, f', g'] := reshape(tmpN[a, b, f, g] * AC[f, h, f'] * W[g, h, g', b], (Dmps, d^i, Dmps, d))
end
@tensor P[a, b, c] := tmpN[a, b, f, g] * FR[f, g, c]




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