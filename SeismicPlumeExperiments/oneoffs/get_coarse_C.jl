using MultivariateStats
using Statistics
using JLD2
using JOLI: joMatrix
using LowRankMatrices
using LinearAlgebra
using Arpack

B = load("run/enkf_N256/observation_noise.jld2", "data");
B = reshape(B, (256*5, 325*341));

n_target_rows = 256
N = (325, 341)
coarse_N = (16, 16)
target_idxs = []
for coarse_i = 1:coarse_N[1]
    for coarse_j = 1:coarse_N[2]
        # The coarse grid is interior to the fine grid.
        fine_i = round(Int, coarse_i * N[1] / (coarse_N[1] + 1))
        fine_j = round(Int, coarse_j * N[2] / (coarse_N[2] + 1))
        fine_idx = (fine_i - 1) + (fine_j - 1) * N[1] + 1
        push!(target_idxs, fine_idx)
    end
end

coarse_C = B[:, target_idxs]' * B ./ (size(B, 1) - 1)
jldsave("coarse_C.jld2"; data=coarse_C)


M = fit(PCA, B'; maxoutdim=256)
jldsave("PCA.jld2"; data=M)

vals = eigvals(M)
jldsave("eigvals.jld2"; data=vals)


@time U, G, V = svd(B, full=false);
S = G .^ 2 / (size(B, 1) - 1);
gm2 = 1 / ((1/S[1] + 1/S[end])/2)
gm = sqrt(gm2)
am2 = mean(S)
am = sqrt(am2)
jldsave("svdvals.jld2"; data=G)
jldsave("svdUV.jld2"; U=U, V=V)

N_eff = sum(S)^2 / sum(S .^2)

std1 = sqrt(sum(S) / size(B, 2))
std2 = sqrt(sum(S) / size(B, 1))
std3 = sqrt(1 / mean(1 ./ S[[1,end]]))
std4 = sqrt(size(B, 1) / sum(1 ./ S))
std5 = sqrt(size(B, 2) / sum(1 ./ S))

let

    std1s = []
    std2s = []
    std3s = []
    std4s = []
    std5s = []

    e = 256
    ei = 1
    ne = 1
    for ne = 1:5
        B2 = B[1 + e*(ei-1) : 1 + e*(ei-1) + e*ne - 1, :];
        @time U2, G2, V2 = svd(B2, full=false);
        S2 = G2 .^ 2 / (size(B2, 1) - 1);
        N_eff2 = sum(S2)^2 / sum(S2 .^2)

        std1 = sqrt(sum(S2) / size(B2, 2))
        std2 = sqrt(sum(S2) / size(B2, 1))
        std3 = sqrt(1 / mean(1 ./ S2[[1,end]]))
        std4 = sqrt(size(B2, 1) / sum(1 ./ S2))
        std5 = sqrt(size(B2, 2) / sum(1 ./ S2))

        push!(std1s, std1)
        push!(std2s, std2)
        push!(std3s, std3)
        push!(std4s, std4)
        push!(std5s, std5)
    end
end

B_op = joMatrix(B)
R_op = B_op' * B_op * (1 / (size(B, 1) - 1))

function Base.show(io::IO, mime::MIME"text/plain", X::LowRankMatrix)
    Base.summary(io, X)
    println(io, ":")
    println()
    print(io, " U = ")
    lines, columns = displaysize(io)
    io = IOContext(io, :displaysize => (max(div(lines, 2), 1), columns))
    Base.show(io, mime, X.U)
    println()
    println()
    print(io, " V = ")
    Base.show(io, mime, X.V)
end

function Base.show(io::IO, X::LowRankMatrix)
    print(io, "LowRankMatrix(")
    Base.show(io, X.U)
    print(io, ", ")
    Base.show(io, X.V)
    print(io, ")")
end


@time S = LowRankMatrix(B', B');
@time begin
    s = 0.0
    d = 0.0
    for j = 1:1000
        for i = 1:1000
            d = LowRankMatrices.unsafe_getindex(S, i,j) .^ 2
            s += d
        end
    end
    println("difference: ", s)
end

function my_getindex(S, i, j)
    return dot(view(S.U, i, :), view(S.V, j, :))
end

unsafe_getindex = LowRankMatrices.unsafe_getindex

@time U, G, V = svd_lowrank(B)
@time U, G, V = svd_lowrank(B, 128)
@time U, G, V = svd_lowrank(B, 256*5)

@time U, G, V = svd(B, full=false)

# 0.563462 for 5*50000 elements
# 0.563462 / 250000 * prod(size(S))
@time sum(view(S, 1:5, 1:5))


# c = CartesianIndices((istart:[istep:]istop, jstart:[jstep:]jstop, ...))

@time dot(S.U[1, :], S.V[1, :])
@time dot(view(S.U, 1, :), view(S.V, 1, :))


@time trS = sum(B .* B)
@time trS = sum(abs2, B)
@time trS2 = sum(abs2, S)

d, nconv, niter, nmult, resid = eigs(S; nev=1, ritzvec=false, maxiter=1)
@time d, nconv, niter, nmult, resid = eigs(view(B, 1:2, 1:2); nev=1, ritzvec=false, maxiter=1)

n = 2000; @time d, nconv, niter, nmult, resid = eigs(view(S, 1:n, 1:n); nev=1, ritzvec=false, maxiter=1)

a = ones(10); b = ones(10);
@time ab = LowRankMatrix(a, b)
vals, vecs = eigs(ab; ritzvec=false)


@time ab = LowRankMatrix(B', B')
d, nconv, niter, nmult, resid = eigs(S; nev=6, ritzvec=false, maxiter=1)





using LowRankMatrices
@time U = randn(1000);
@time V = randn(1000);
@time mat = U * V';
@time mat_lazy = ApplyArray(*, U, V');
@time mat_lr = LowRankMatrix(U, V);

@time begin
    s = 0.0
    d = 0.0
    for j = 1:1000
        for i = 1:1000
            # d = mat[i,j] # - LowRankMatrices.unsafe_getindex(mat_lr, i,j)
            d = LowRankMatrices.unsafe_getindex(mat_lr, i,j) .^ 2
            s += d
        end
    end
    println("difference: ", s)
end

function LowRankMatrices.unsafe_getindex(L::LowRankMatrix, i::Int, j::Int)
    ret = zero(eltype(L))
    @inbounds for k=1:LowRankMatrices.rank(L)
        ret = muladd(L.U[i,k], L.V[j,k], ret)
    end
    return ret
end
