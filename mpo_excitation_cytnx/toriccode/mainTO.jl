## Reproducing Fig.7(a.b.e.f) in 10.1038/ncomms9284
using DelimitedFiles
include("../vumpsfixedpoints.jl")
include("../excitationmpo.jl")
include("torictensor.jl")
##
folder_out = "TO/"
if isdir(folder_out) == false
    mkdir(folder_out)
end
##
########## Initial states
# βx, βz = 0.4, 0.8
βx, βz = 0.6, 0.5
# βx, βz = 0.8, 0.5


Dmps = 6 ## MPS's accuracy-controlled bond dimension

filename = "bx$(βx)bz$(βz)Dmps$(Dmps)"
sZ = [1. 0.; 0. -1.]
sI = [1. 0; 0. 1.]
## building blocks of string operators acting on both ket and bra layers
ZZ = reshape(ncon([sZ,sZ],[[-1,-3],[-2,-4]]), 4,4)
IZ = reshape(ncon([sI,sZ],[[-1,-3],[-2,-4]]), 4,4)
# beta_x, beta_z = 1.5, 0.5

## get TC wavefunctions
toric = symstringtoric(βx, βz)
dim = size(toric)[2]
## Construct the double tensor
W = ncon([toric, conj(toric)],
         [[1, -1, -3, -5, -7], [1, -2, -4, -6, -8]])
d = dim^2
W = reshape(W,(d, d, d, d));
AL = rand(Dmps,d,Dmps);
##
########## Fixed points
λ, AL, C, AR, FL, FR = vumpsfixedpts(AL, W; tol = 1e-6);
# λ, AL, C, AR, FL, FR = vumpsfixedpts(AL, O; tol = 1e-10);
W = W/λ;
##
########## Excitation
num_ω = 15
p_list = LinRange(0,1,11)

open(folder_out*"$(filename)_triv_Ceq.txt", "w") do io
    write(io, "# p ω \n")
end   
open(folder_out*"$(filename)_triv_Cdiff.txt", "w") do io
    write(io, "# p ω \n")
end   

## topologically trivial excitaitons
for p in p_list
    data = excitation(W,AL,AR,C,FL,FR, num_ω, p; charge = true, Cstring = [ZZ],domain = false, 
    Fstring = Nothing, verbose = true)
    ϕ_Ceq = data["ϕ_Ceq"]; ω_Ceq = data["ω_Ceq"]
    ϕ_Cdiff = data["ϕ_Cdiff"]; ω_Cdiff = data["ω_Cdiff"]
    # println(size(ω_Ceq))
    if p == p_list[1]
        global len_Ceq = length(ω_Ceq)-5
        global len_Cdiff = length(ω_Cdiff)-5
    end
    open(folder_out*"$(filename)_triv_Ceq.txt", "a") do io
        writedlm(io, reshape([p; ω_Ceq[1:len_Ceq]], 1,len_Ceq+1))
    end   
    open(folder_out*"$(filename)_triv_Cdiff.txt", "a") do io
        writedlm(io, reshape([p; ω_Cdiff[1:len_Cdiff]], 1,len_Cdiff+1))
    end   
end
##
open(folder_out*"$(filename)_domain_Ceq.txt", "w") do io
    write(io, "# p ω \n")
end   
open(folder_out*"$(filename)_domain_Cdiff.txt", "w") do io
    write(io, "# p ω \n")
end   

## topologically non-trivial excitaitons
for p in p_list
    data = excitation(W,AL,AR,C,FL,FR, num_ω, p; charge = true, Cstring = [ZZ],domain = true, 
    Fstring = (IZ), verbose = true)
    ϕ_Ceq = data["ϕ_Ceq"]; ω_Ceq = data["ω_Ceq"]
    ϕ_Cdiff = data["ϕ_Cdiff"]; ω_Cdiff = data["ω_Cdiff"]
    # println(size(ω_Ceq))
    if p == p_list[1]
        global len_Ceq = length(ω_Ceq)-5
        global len_Cdiff = length(ω_Cdiff)-5
    end
    open(folder_out*"$(filename)_domain_Ceq.txt", "a") do io
        writedlm(io, reshape([p; ω_Ceq[1:len_Ceq]], 1,len_Ceq+1))
    end   
    open(folder_out*"$(filename)_domain_Cdiff.txt", "a") do io
        writedlm(io, reshape([p; ω_Cdiff[1:len_Cdiff]], 1,len_Cdiff+1))
    end   
end