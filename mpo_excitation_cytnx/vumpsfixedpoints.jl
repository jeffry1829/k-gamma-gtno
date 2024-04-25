## Follow SciPost Phys. Lect. Notes 7 (2019)
include("canonical.jl")
#################### leftenv and rightenv
"""
    leftenv(A, M, FL; kwargs)
Compute the left environment tensor for MPS `A` and MPO `W`, by finding the left fixed point
of `A - M - conj(A)` contracted along the physical dimension.
"""
function leftenv(A::Dims3Array, W::Dims4Array, FL=randn(eltype(A), size(A, 1), size(W, 1), size(A, 1)); kwargs...)
    λs, FLs, info = eigsolve(FL, 1, :LM; ishermitian=false, kwargs...) do FL
        FL = ncon([FL, A, W, conj(A)], [[1, 2, 3], [3, 5, -3], [2, 5, -2, 4], [1, 4, -1]])
    end
    return FLs[1], real(λs[1]), info
end
"""
    rightenv(A, M, FR; kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the right fixed point
of A - M - conj(A) contracted along the physical dimension.
"""
function rightenv(A::Dims3Array, W::Dims4Array, FR=randn(eltype(A), size(A, 1), size(W, 1), size(A, 1)); kwargs...)
    λs, FRs, info = eigsolve(FR, 1, :LM; ishermitian=false, kwargs...) do FR
        # @tensor FR[α,a,β] := A[α,s',α']*FR[α',a',β']*M[a,s,a',s']*conj(A[β,s,β'])
        FR = ncon([FR, A, W, conj(A)], [[1, 2, 3], [-1, 4, 1], [-2, 4, 2, 5], [-3, 5, 3]])
    end
    return FRs[1], real(λs[1]), info
end



#################### Final algorithm
"""
    vumpsfixedpts(A,W;)
Given initial mps A and the double tensor W, return λ, AL, C, AR, FL, FR
"""
function vumpsfixedpts(A::Dims3Array, W::Dims4Array; verbose=true, steps=100, tol=1e-6, kwargs...)
    ## Algorithm4
    AL, AR, C = mixcanonical(A)
    FL, λL = leftenv(AL, W; kwargs...)
    FR, λR = rightenv(AR, W; kwargs...)
    FR ./= @tensor (FL[c, b, a] * C[a, a'] * conj(C[c, c']) * FR[a', b, c']) # normalize FL and FR: necessary!
    iter = 0 # vumps step
    err = 1
    λ = 0 #initial value, not important 
    while err > tol && iter < steps
        @tensor AC[a, s, b] := AL[a, s, b'] * C[b', b]
        applyH1 = (AC, FL, FR, W) -> ncon([AC, FL, FR, W], [[1, 5, 2], [-1, 3, 1], [2, 4, -3], [3, 5, 4, -2]])
        applyH0 = (C, FL, FR) -> ncon([C, FL, FR], [[1, 2], [-1, 3, 1], [2, 3, -2]])
        ## update AC
        μ1s, ACs, info1 = eigsolve(x -> applyH1(x, FL, FR, W), AC, 1, :LM; ishermitian=false, maxiter=1, kwargs...)
        ## update C
        μ0s, Cs, info0 = eigsolve(x -> applyH0(x, FL, FR), C, 1; ishermitian=false, maxiter=1, kwargs...)
        λ = real(μ1s[1] / μ0s[1])
        AC = ACs[1]
        C = Cs[1]
        AL, AR, errL, errR = min_AC_C(AC, C) ## transform back to AL and AR
        AL, C, = leftorth(AR, C; tol=tol / 10, kwargs...) # regauge MPS: not really necessary
        ## update FL and FR
        FL, λL = leftenv(AL, W, FL; tol=tol / 10, kwargs...)
        FR, λR = rightenv(AR, W, FR; tol=tol / 10, kwargs...)
        FR ./= @tensor (FL[c, b, a] * C[a, a'] * conj(C[c, c']) * FR[a', b, c']) # normalize FL and FR: not really necessary
        # Convergence measure: norm of the projection of the residual onto the tangent space
        @tensor AC[a, s, b] := AL[a, s, b'] * C[b', b]
        MAC = applyH1(AC, FL, FR, W)
        @tensor MAC[a, s, b] -= AL[a, s, b'] * (conj(AL[a', s', b']) * MAC[a', s', b])
        err = norm(MAC)
        iter += 1
        verbose && println("Step $(iter): λ ≈ $λ ≈ $λL ≈ $λR, err ≈ $err")
    end
    return λ, AL, C, AR, FL, FR
end


