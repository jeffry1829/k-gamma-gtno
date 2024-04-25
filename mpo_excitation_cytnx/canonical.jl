
## Follow SciPost Phys. Lect. Notes 7 (2019)
using LinearAlgebra, TensorOperations, KrylovKit
BLAS.set_num_threads(Threads.nthreads())
const Dims3Array{T} = Array{T,3}
const Dims4Array{T} = Array{T,4}
const Dims5Array{T} = Array{T,5}
#################### modifed  version of QR and LQ decomposition
safesign(x::Number) = iszero(x) ? one(x) : sign(x) # will be used to make QR decomposition unique
"""
    qrpos(A)
Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end
##
"""
    lqpos(A)
Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A::AbstractMatrix) = lqpos!(copy(A))
function lqpos!(A)
    F = qr!(Matrix(transpose(A)))
    Q = transpose(Matrix(F.Q))
    L = transpose(Matrix(F.R))
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    return L, Q
end

##################### Mixed Gauge: leftorth, rightorth, mixcanonical, and min_AC_C
"""
    leftorth(A, [C]; kwargs...)
Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL^s C = C A^s``, where an initial guess for `C` can be
provided.
"""
function leftorth(A::Dims3Array, C=Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol=1e-12, maxiter=100, kwargs...)
    # find better initial guess C
    λ2s, ρs, info = eigsolve(C' * C, 1, :LM; ishermitian=false, tol=tol, maxiter=1, kwargs...) do ρ
        @tensor ρE[a, b] := ρ[a', b'] * A[b', s, b] * conj(A[a', s, a]) ## Find the fixed pt of ∑ᵢAⁱ⊗Aⁱ      
        return ρE
    end
    ρ = ρs[1] + ρs[1]' ## ρ = C'*C is hermitian
    ρ ./= tr(ρ) ## enforce tr(ρ) = 1
    # C = cholesky!(ρ).U
    # If ρ is not exactly positive definite, cholesky will fail
    F = svd!(ρ)
    C = lmul!(Diagonal(sqrt.(F.S)), F.Vt) ## given ρ = C'*C, find C
    _, C = qrpos!(C) # I don't know why

    ## Algorithm1 
    D, d, = size(A)
    Q, R = qrpos!(reshape(C * reshape(A, D, d * D), D * d, D))
    AL = reshape(Q, D, d, D)
    λ = norm(R)
    rmul!(R, 1 / λ)
    numiter = 1
    while norm(C - R) > tol && numiter < maxiter
        # C = R
        λs, Cs, info = eigsolve(R, 1, :LM; ishermitian=false, tol=tol, maxiter=1, kwargs...) do X
            @tensor Y[a, b] := X[a', b'] * A[b', s, b] * conj(AL[a', s, a])
            return Y
        end
        _, C = qrpos!(Cs[1])
        # The previous lines can speed up the process when C is still very far from the correct
        # gauge transform, it finds an improved value of C by finding the fixed point of a
        # 'mixed' transfer matrix composed of `A` and `AL`, even though `AL` is also still not
        # entirely correct. Therefore, we restrict the number of iterations to be 1 and don't
        # check for convergence
        Q, R = qrpos!(reshape(C * reshape(A, D, d * D), D * d, D)) ## do QR decomposition iteratively
        AL = reshape(Q, D, d, D)
        λ = norm(R)
        rmul!(R, 1 / λ)
        numiter += 1
    end
    C = R
    return AL, C, λ
end
"""
    rightorth(A, [C]; kwargs...)
Given an MPS tensor `A`, return a gauge transform C, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ C AR^s = A^s C``, where an initial guess for `C` can be
provided.
"""
function rightorth(A::Dims3Array, C=Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol=1e-12, kwargs...)
    ## simply permute A and C for leftorth!
    AL, C, λ = leftorth(permutedims(A, (3, 2, 1)), permutedims(C, (2, 1)); tol=tol, kwargs...)
    return permutedims(AL, (3, 2, 1)), permutedims(C, (2, 1)), λ
end

"""
    mixcanonical(A)
Transform a mps `A` into the mixe canonical form and return `AL,AR,C` 
"""
function mixcanonical(A::Dims3Array)
    ## see Algorithm2
    AL, = leftorth(A)
    AR, C, = rightorth(AL)
    return AL, AR, C
end

"""
    min_AC_C(AC, C)
Given `AC, C` and then return `AL,AR` along with the error `errL, errR`
"""
function min_AC_C(AC::Dims3Array, C::AbstractMatrix)
    ## Algorithm5
    D, d, = size(AC)
    # F = qr(reshape(AC,(D*d, D))
    # QAC,RAC = Matrix(F.Q), F.R
    QAC, RAC = qrpos(reshape(AC, (D * d, D))) ## polar left for AC
    # F = qr(C)
    # QC,RC = Matrix(F.Q), F.R
    QC, RC = qrpos(C) ## polar left for C
    AL = reshape(QAC * QC', (D, d, D))
    errL = norm(RAC - RC) ## not sure why
    LAC, QAC = lqpos(reshape(AC, (D, d * D))) ## polar right for AC
    LC, QC = lqpos(C) ## polar right for C
    AR = reshape(QC' * QAC, (D, d, D))
    errR = norm(LAC - LC)
    return AL, AR, errL, errR
end


