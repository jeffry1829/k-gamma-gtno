## Reproducing Fig.7(g.f) in 10.1038/ncomms9284
using DelimitedFiles
include("../vumpsfixedpoints.jl")
include("../excitationmpo.jl")
include("torictensor.jl")
##
folder_out = "FC/"
if isdir(folder_out) == false
    mkdir(folder_out)
end
##
########## Initial states

βx, βz = 1.0, 0.5


Dmps = 6

filename = "bx$(βx)bz$(βz)Dmps$(Dmps)"
sZ = [1. 0.; 0. -1.]
sI = [1. 0; 0. 1.]
ZZ = reshape(ncon([sZ,sZ],[[-1,-3],[-2,-4]]), 4,4)
IZ = reshape(ncon([sI,sZ],[[-1,-3],[-2,-4]]), 4,4)


toric = symstringtoric(βx, βz)
dim = size(toric)[2]
W = ncon([toric, conj(toric)],
         [[1, -1, -3, -5, -7], [1, -2, -4, -6, -8]])
d = dim^2
W = reshape(W,(d, d, d, d));
##
AL = rand(Dmps,d,Dmps);

########## Fixed points

λ, AL, C, AR, FL, FR = vumpsfixedpts(AL, W; tol = 1e-5);
# λ, AL, C, AR, FL, FR = vumpsfixedpts(AL, O; tol = 1e-10);
W = W/λ;
##
########## Excitation
num_ω = 15
p_list = LinRange(0,1,11)

open(folder_out*"$(filename)_Cee.txt", "w") do io
    write(io, "# p ω \n")
end   
open(folder_out*"$(filename)_Ceo.txt", "w") do io
    write(io, "# p ω \n")
end   
open(folder_out*"$(filename)_Coo.txt", "w") do io
    write(io, "# p ω \n")
end   

for p in p_list
    data = excitation(W,AL,AR,C,FL,FR, num_ω, p; charge = true, Cstring = [ZZ,IZ],domain = false, 
    Fstring = Nothing, verbose = true)
    ω_Cee = data["ω_Cee"]; ω_Ceo = data["ω_Ceo"]; ω_Coo = data["ω_Coo"]
    if p == p_list[1]
        global len_Cee = length(ω_Cee)-2
        global len_Ceo = length(ω_Ceo)-2
        global len_Coo = length(ω_Coo)-2
    end
    open(folder_out*"$(filename)_Cee.txt", "a") do io
        writedlm(io, reshape([p; ω_Cee[1:len_Cee]], 1,len_Cee+1))
    end   
    open(folder_out*"$(filename)_Ceo.txt", "a") do io
        writedlm(io, reshape([p; ω_Ceo[1:len_Ceo]], 1,len_Ceo+1))
    end   
    open(folder_out*"$(filename)_Coo.txt", "a") do io
        writedlm(io, reshape([p; ω_Coo[1:len_Coo]], 1,len_Coo+1))
    end  
end
##