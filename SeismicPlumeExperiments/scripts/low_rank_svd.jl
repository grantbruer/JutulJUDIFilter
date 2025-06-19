"""
Implement various linear algebra algorithms for low rank matrices.
"""

using LinearAlgebra
using Random

function get_approximate_basis(A::AbstractArray, q::Int, niter::Int=2, M::Union{Nothing, AbstractArray}=nothing)
    niter = niter == nothing ? 2 : niter
    m, n = size(A)
    dtype = eltype(A)

    R = randn(dtype, n, q)

    A_H = A'
    if M === nothing
        Q = Matrix(qr(A * R).Q)
        for i in 1:niter
            Q = Matrix(qr(A_H * Q).Q)
            Q = Matrix(qr(A * Q).Q)
        end
    else
        M_H = M'
        Q = Matrix(qr((A * R) - (M * R)).Q)
        for i in 1:niter
            Q = Matrix(qr((A_H * Q) - (M_H * Q)).Q)
            Q = Matrix(qr((A * Q) - (M * Q)).Q)
        end
    end

    return Q
end

function svd_lowrank(A::AbstractArray, q::Union{Nothing, Int}=6, niter::Union{Nothing, Int}=2, M::Union{Nothing, AbstractArray}=nothing)
    q = q === nothing ? 6 : q
    m, n = size(A)

    if M === nothing
        M_t = nothing
    else
        M_t = M'
    end
    A_t = A'

    if m < n || n > q
        Q = get_approximate_basis(A_t, q, niter, M_t)
        Q_c = conj(Q)
        B_t = M === nothing ? (A * Q_c) : (A * Q_c) - (M * Q_c)
        @assert size(B_t, ndims(B_t)-1) == m (size(B_t), m)
        @assert size(B_t, ndims(B_t)) == q (size(B_t), q)
        @assert size(B_t, ndims(B_t)) <= size(B_t, ndims(B_t)-1) size(B_t)
        U, S, Vh = svd(B_t, full=false)
        V = Q * Vh'
    else
        Q = get_approximate_basis(A, q, niter, M)
        Q_c = conj(Q)
        B = M === nothing ? (A_t * Q_c) : (A_t * Q_c) - (M_t * Q_c)
        B_t = B'
        U, S, Vh = svd(B_t, full=false)
        V = Vh'
        U = Q * U
    end

    return U, S, V
end

function pca_lowrank(A::AbstractArray, q::Union{Nothing, Int}=nothing, center::Bool=true, niter::Int=2)
    m, n = size(A)

    if q === nothing
        q = min(6, m, n)
    elseif !(q >= 0 && q <= min(m, n))
        throw(ArgumentError("q(=$q) must be a non-negative integer and not greater than min(m, n)=$(min(m, n))"))
    end
    if !(niter >= 0)
        throw(ArgumentError("niter(=$niter) must be a non-negative integer"))
    end

    if !center
        return svd_lowrank(A, q, niter)
    end

    C = mean(A, dims=1)
    return svd_lowrank(A .- C, q, niter)
end
