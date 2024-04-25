## Follow SciPost Phys. Lect. Notes 7 (2019)
const MPSOperator = Union{Matrix,Dims4Array} ## Dims 2 or 4
const MPSEnvironment = Union{Matrix,Dims3Array} ## Dims 2 or 3
"""
    braOket(Abra, W, Aket)
Compute the channel operator TW, where W is an Operator.
"""
function braOket(Abra::Dims3Array, W::MPSOperator, Aket::Dims3Array)
    if ndims(W) == 4
        braOket = ncon([Aket, W, conj(Abra)], [[-3, 2, -6], [-2, 2, -5, 1], [-1, 1, -4]])
    elseif ndims(W) == 2
        braOket = ncon([conj(Abra), W, (Aket)], [[-1, 1, -3], [1, 2], [-2, 2, -4]])
    end
    return braOket
end

#################### pinvL, pinvR
"""
    pinvL(x,p,Abra,W,Aket,l,r; domain)
"""
function pinvL(x::MPSEnvironment, p::Real, Abra::Dims3Array, W::MPSOperator,
    Aket::Dims3Array, l::Union{MPSEnvironment,Nothing}, r::Union{MPSEnvironment,Nothing}; domain=false)
    AbraOAket = braOket(Abra, W, Aket)
    dimO = ndims(W)
    if abs(exp(1im * p) - 1) < 1e-12 && domain == false
        if dimO == 4
            x = x - l * tensorscalar(ncon([x, r], [[1, 2, 3], [3, 2, 1]]))
        elseif dimO == 2
            x = x - l * tensorscalar(ncon([x, r], [[1, 2], [2, 1]]))
        end
    end
    y, info = linsolve(x; tol=1e-10, maxiter=400) do x
        if dimO == 4
            y = x - exp(1im * p) * ncon([x, AbraOAket], [[1, 2, 3], [1, 2, 3, -1, -2, -3]])
        elseif dimO == 2
            y = x - exp(1im * p) * ncon([x, AbraOAket], [[1, 2], [1, 2, -1, -2]])
        end
        if abs(exp(1im * p) - 1) < 1e-12 && domain == false
            if dimO == 4
                y = y + l * tensorscalar(ncon([x, r], [[1, 2, 3], [3, 2, 1]]))
            elseif dimO == 3
                y = y + l * tensorscalar(ncon([x, r], [[1, 2], [2, 1]]))
            end
        end
        return y
    end
    #@assert info.converged != 0
    info.converged == 0 ? @warn("pinvL not converge!!!") : nothing
    return y
end



"""
    pinvR(x,p,Abra,W,Aket,l,r; domain)
"""
function pinvR(x::MPSEnvironment, p, Abra::Dims3Array, O::MPSOperator,
    Aket::Dims3Array, l::Union{MPSEnvironment,Nothing}, r::Union{MPSEnvironment,Nothing}; domain=false)
    dimO = ndims(O)
    AbraOAket = braOket(Abra, O, Aket)
    if abs(exp(1im * p) - 1) < 1e-12 && domain == false
        if dimO == 4
            x = x - r * tensorscalar(ncon([x, l], [[1, 2, 3], [3, 2, 1]]))
        elseif dimO == 2
            x = x - r * tensorscalar(ncon([x, l], [[1, 2], [2, 1]]))
        end

    end
    y, info = linsolve(x; tol=1e-10, maxiter=400) do x
        if dimO == 4
            y = x - exp(1im * p) * ncon([x, AbraOAket], [[1, 2, 3], [-3, -2, -1, 3, 2, 1]])
        elseif dimO == 2
            y = x - exp(1im * p) * ncon([x, AbraOAket], [[1, 2], [-2, -1, 2, 1]])
        end
        if abs(exp(1im * p) - 1) < 1e-12 && domain == false
            if dimO == 4
                y = y + l * tensorscalar(ncon([x, r], [[1, 2, 3], [3, 2, 1]]))
            elseif dimO == 2
                y = y + l * tensorscalar(ncon([x, r], [[1, 2], [2, 1]]))
            end
        end
        return y
    end
    info.converged == 0 ? @warn("pinvR not converge!!!") : nothing
    return y
end


function pinvLJulia(x::MPSEnvironment, p, Abra::Dims3Array, O::MPSEnvironment, Aket::Dims3Array;)
    dimO = ndims(O)
    if dimO == 4
        D, dw = size(x)
        mat_braOket = reshape(braOket(Abra, O, Aket), (D^2 * dw, D^2 * dw))
        eye = Matrix(I, size(mat_braOket))
        mat_inv = pinv(eye - exp(1im * p) * transpose(mat_braOket))
        invL = reshape(mat_inv, (D, dw, D, D, dw, D))
        y = ncon([invL, x], [[-1, -2, -3, 1, 2, 3], [1, 2, 3]])
    elseif dimO == 2
        D, = size(x)
        mat_braOket = reshape(braOket(Abra, O, Aket), D^2, D^2)
        eye = Matrix(I, size(mat_braOket))
        mat_inv = pinv(eye - exp(1im * p) * transpose(mat_braOket))
        invL = reshape(mat_inv, (D, D, D, D))
        y = ncon([invL, x], [[-1, -2, 1, 2], [1, 2]])
    end
end

function pinvRJulia(x::MPSEnvironment, p, Abra::Dims3Array, O::MPSOperator, Aket::Dims3Array)
    dimO = ndims(O)
    if dimO == 4
        D, dw = size(x)
        mat_braOket = reshape(braOket(Abra, O, Aket), (D^2 * dw, D^2 * dw))
        eye = Matrix(I, size(mat_braOket))
        mat_inv = pinv(eye - exp(1im * p) * mat_braOket)
        invL = reshape(mat_inv, (D, dw, D, D, dw, D))
        y = ncon([invL, x], [[-3, -2, -1, 3, 2, 1], [1, 2, 3]])
    elseif dimO == 2
        D, = size(x)
        mat_braOket = reshape(braOket(Abra, O, Aket), (D^2, D^2))
        eye = Matrix(I, size(mat_braOket))
        mat_inv = pinv(eye - exp(1im * p) * mat_braOket)
        invL = reshape(mat_inv, (D, D, D, D))
        y = ncon([invL, x], [[-2, -1, 2, 1], [1, 2]])
    end
end


#################### apply_Heff, apply_1Deff, measure_charge
"""
    applyHeff(B,p,AL,AR,C,O,FL,FR)
MPO version of applyHeff
"""
function applyHeff(B::Dims3Array, p, AL::Dims3Array, AR::Dims3Array, C::AbstractMatrix,
    O::MPSOperator, FL::MPSEnvironment, FR::MPSEnvironment; pinv="manual")
    ALOB = braOket(AL, O, B)
    LB = ncon([FL, ALOB], [[1, 2, 3], [1, 2, 3, -1, -2, -3]])
    AROB = braOket(AR, O, B)
    RB = ncon([FR, AROB], [[1, 2, 3], [-3, -2, -1, 3, 2, 1]])

    if pinv == "manual"
        l = ncon([FL, C], [[-1, -2, 3], [3, -3]])
        r = ncon([FR, C'], [[-1, -2, 3], [3, -3]])
        LB = pinvL(LB, -p, AL, O, AR, l, r)

        l = ncon([C', FL], [[-1, 1], [1, -2, -3]])
        r = ncon([C, FR], [[-1, 1], [1, -2, -3]])
        RB = pinvR(RB, +p, AR, O, AL, l, r)
    elseif pinv == "julia"
        @warn("Uisng Julia's inverse, and the computational cost is much higher than linsolve!")
        LB = pinvLJulia(LB, -p, AL, O, AR)
        RB = pinvRJulia(RB, +p, AR, O, AL)
    end

    By = exp(-1im * p) * ncon([LB, AR, O, FR], [[-1, 1, 2], [2, 5, 4], [1, 5, 3, -2], [4, 3, -3]]) +
         exp(1im * p) * ncon([FL, AL, O, RB], [[-1, 1, 2], [2, 5, 4], [1, 5, 3, -2], [4, 3, -3]]) +
         ncon([FL, B, O, FR], [[-1, 1, 2], [2, 5, 4], [1, 5, 3, -2], [4, 3, -3]])

    return By
end

"""
    applydomainHeff(B,p,AL1,AR2,C,O,FL,FR)
"""
function applydomainHeff(B::Dims3Array, p, AL1::Dims3Array, AR2::Dims3Array, C::AbstractMatrix,
    O::Dims4Array, FL::Dims3Array, FR::Dims3Array; pinv="manual")
    AL1OB = braOket(AL1, O, B)
    LB = ncon([FL, AL1OB], [[1, 2, 3], [1, 2, 3, -1, -2, -3]])
    AR2OB = braOket(AR2, O, B)
    RB = ncon([FR, AR2OB], [[1, 2, 3], [-3, -2, -1, 3, 2, 1]])
    if pinv == "manual"
        l = r = nothing
        LB = pinvL(LB, -p, AL1, O, AR2, l, r; domain=true)
        RB = pinvR(RB, +p, AR2, O, AL1, l, r; domain=true)
    elseif pinv == "julia"
        @warn("Uisng Julia's inverse, and the computational cost is much higher than linsolve!")
        LB = pinvLJulia(LB, -p, AL1, O, AR2)
        RB = pinvRJulia(RB, +p, AR2, O, AL1)
    end

    By = exp(-1im * p) * ncon([LB, AR2, O, FR], [[-1, 1, 2], [2, 5, 4], [1, 5, 3, -2], [4, 3, -3]]) +
         exp(1im * p) * ncon([FL, AL1, O, RB], [[-1, 1, 2], [2, 5, 4], [1, 5, 3, -2], [4, 3, -3]]) +
         ncon([FL, B, O, FR], [[-1, 1, 2], [2, 5, 4], [1, 5, 3, -2], [4, 3, -3]])

    return By
end

"""
    apply1Deff(B,AL,AR,C,O,FL,FR;)
"""
function apply1Deff(B::Dims3Array, p, AL::Dims3Array, AR::Dims3Array, C::AbstractMatrix, O::AbstractMatrix,
    FL, FR; pinv="manual")
    ALOB = braOket(AL, O, B)
    LB = ncon([FL, ALOB], [[1, 2], [1, 2, -1, -2]])
    AROB = braOket(AR, O, B)
    RB = ncon([AROB, FR], [[-2, -1, 2, 1], [1, 2]])
    if pinv == "manual"
        l = ncon([FL, C], [[-1, 2], [2, -2]])
        r = ncon([FR, C'], [[-1, 2], [2, -2]])
        LB = pinvL(LB, -p, AL, O, AR, l, r)
        l = ncon([C', FL], [[-1, 1], [1, -2]])
        r = ncon([C, FR], [[-1, 1], [1, -2]])
        RB = pinvR(RB, +p, AR, O, AL, l, r)
    elseif pinv == "julia"
        @warn("Uisng Julia's inverse, and the computational cost is much higher than linsolve!")
        LB = pinvLJulia(LB, -p, AL, O, AR)
        RB = pinvRJulia(RB, +p, AR, O, AL)
    end

    By = exp(-1im * p) * ncon([LB, AR, O, FR], [[-1, 1], [1, 2, 3], [-2, 2], [3, -3]]) +
         exp(1im * p) * ncon([FL, AL, O, RB], [[-1, 1], [1, 2, 3], [-2, 2], [3, -3]]) +
         ncon([FL, B, O, FR], [[-1, 1], [1, 2, 3], [-2, 2], [3, -3]])
    return By
end


"""
    measurecharge(B,AL,AR,C,O)
"""
function measurecharge(B::Dims3Array, AL::Dims3Array, AR::Dims3Array, C::AbstractMatrix, O::AbstractMatrix)
    ALOAL = braOket(AL, O, AL)
    AROAR = braOket(AR, O, AR)
    D, = size(AL)
    FL = randn(eltype(AL), D, D)
    λs, FLs, info = eigsolve(FL, 1, :LM; ishermitian=false) do FL
        FL = ncon([FL, ALOAL], [[1, 2], [1, 2, -1, -2]])
    end
    λL = λs[1]
    FL = FLs[1]
    FR = randn(eltype(AL), D, D)
    λs, FRs, info = eigsolve(FR, 1, :LM; ishermitian=false) do FR
        FR = ncon([AROAR, FR], [[-2, -1, 2, 1], [1, 2]])
    end
    λR = λs[1]
    FR = FRs[1]
    FR ./= @tensor (FL[c, a] * C[a, a'] * conj(C[c, c']) * FR[a', c'])

    By = apply1Deff(B, 0, AL, AR, C, O, FL, FR; pinv="manual")
    return (tensorscalar(ncon([conj(B), By], [[1, 2, 3], [1, 2, 3]])))
end

"""
    getnullspace(AL)
Given `AL`, return the its nullspace `VL` and its effective dimension `nL` 
"""
function getnullspace(AL::Dims3Array)
    D_mps, d_mps, = size(AL)
    L = reshape(permutedims(conj(AL), (3, 1, 2)), (D_mps, D_mps * d_mps))
    VL = nullspace(L)
    nL = size(VL)[2]
    VL = reshape(VL, (D_mps, d_mps, nL))
    return VL, nL
end



"""
    excitation(W,AL,AR,C,FL,FR,num_ω,p;charge,Cstring,domain,Fstring,verbose)
Return a Dict `data` with data["p"] = p; data["ϕ"] = ϕ; data["ω"] = abs_ωs
"""
function excitation(W::Dims4Array, AL::Dims3Array, AR::Dims3Array, C::AbstractArray,
    FL::Dims3Array, FR::Dims3Array, num_ω::Int, p::Real;
    charge=false, Cstring=Nothing, domain=false, Fstring=Nothing, verbose=false)

    D_mps, = size(AL)
    VL, nL = getnullspace(AL)
    # global data = Dict()
    if domain == true
        applyH = applydomainHeff
        AR2 = ncon([Fstring, AR], [[-2, 2], [-1, 2, -3]])
        AR2OAR2 = braOket(AR2, W, AR2)
        FR2 = ncon([Fstring, FR], [[-2, 2], [-1, 2, -3]])
    else
        applyH = applyHeff
        AR2 = AR
        FR2 = FR
    end
    ωs, excits, info = eigsolve(rand(nL, D_mps), num_ω, :LM;) do X
        B = ncon([VL, X], [[-1, -2, 1], [1, -3]])
        By = applyH(B, p * π, AL, AR2, C, W, FL, FR2; pinv="manual")
        Heff_X = ncon([By, conj(VL)],
            [[1, 2, -2], [1, 2, -1]])
        return Heff_X
    end
    ϕ = angle.(ωs) / (π)
    abs_ωs = abs.(ωs)
    data = Dict()
    if charge == true
        if length(Cstring) == 1
            Ceq = Vector{Int}()
            Cdiff = Vector{Int}()
            ZZ = Cstring[1]
            for i = 1:(num_ω)
                excit = excits[i]
                B = ncon([VL, excit], [[-1, -2, 1], [1, -3]])
                tmp = real(measurecharge(B, AL, AR, C, ZZ))
                verbose == true ? println("charge = ", (tmp), " ") : nothing
                tmp > 0 ? push!(Ceq, i) : push!(Cdiff, i)
            end
            data["p"] = p
            data["ϕ_Ceq"] = ϕ[Ceq]
            data["ω_Ceq"] = abs_ωs[Ceq]
            data["p"] = p
            data["ϕ_Cdiff"] = ϕ[Cdiff]
            data["ω_Cdiff"] = abs_ωs[Cdiff]
        elseif length(Cstring) == 2
            Cee = Vector{Int}()
            Ceo = Vector{Int}()
            Coo = Vector{Int}()
            ZZ = Cstring[1]
            IZ = Cstring[2]
            for i = 1:(num_ω)
                excit = excits[i]
                B = ncon([VL, excit], [[-1, -2, 1], [1, -3]])
                CZZ = real(measurecharge(B, AL, AR, C, ZZ))
                # tmp = real(measure_charge(B,AL,AR,C,Cop));
                verbose == true ? println("CZZ = ", (CZZ), " ") : nothing
                CIZ = real(measurecharge(B, AL, AR, C, IZ))
                verbose == true ? println("CIZ = ", (CIZ), " ") : nothing
                if CZZ > 0
                    CIZ > 0 ? push!(Cee, i) : push!(Coo, i)
                else
                    push!(Ceo, i)
                end
            end
            data["p"] = p
            data["ϕ_Cee"] = ϕ[Cee]
            data["ω_Cee"] = abs_ωs[Cee]
            data["p"] = p
            data["ϕ_Ceo"] = ϕ[Ceo]
            data["ω_Ceo"] = abs_ωs[Ceo]
            data["p"] = p
            data["ϕ_Coo"] = ϕ[Ceo]
            data["ω_Coo"] = abs_ωs[Coo]
            # print("hi")
        end
    else
        data["p"] = p
        data["ϕ"] = ϕ[1:num_ω]
        data["ω"] = abs_ωs[1:num_ω]
        # push!(data, reshape([p; ϕ[1:5]; abs_ωs[1:num_ω]], (1,num_ω+6)))
    end
    verbose == true ? println("p = $(p),  ω =  $(abs_ωs[1]), ϕ = $(ϕ[1])",) : nothing
    return data
end