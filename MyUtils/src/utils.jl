using DataStructures: IntDisjointSets, union!, root_union!, find_root, in_same_set, push!

struct CartesianMesh{D, Δ, O}
    "Tuple of dimensions (nx, ny, [nz])"
    dims::D
    "Either a tuple of scalars (uniform grid) or a tuple of vectors (non-uniform grid)"
    deltas::Δ
    "Coordinate of lower left corner"
    origin::O
    "Tags on cells/faces/nodes"
    function CartesianMesh(dims::Tuple, deltas_or_size::Union{Nothing, Tuple} = nothing; origin = nothing)
        dim = length(dims)
        if isnothing(deltas_or_size)
            deltas_or_size = Tuple(ones(dim))
        end
        if isnothing(origin)
            origin = zeros(dim)
        else
            @assert length(origin) == dim
        end
        function generate_deltas(deltas_or_size)
            deltas = Vector(undef, dim)
            for (i, D) = enumerate(deltas_or_size)
                if isa(D, AbstractFloat)
                    # Deltas are actually size of domain in each direction
                    deltas[i] = D/dims[i]
                else
                    # Deltas are the actual cell widths
                    @assert length(D) == dims[i]
                    deltas[i] = D
                end
            end
            return Tuple(deltas)
        end
        @assert length(deltas_or_size) == dim
        deltas = generate_deltas(deltas_or_size)
        g = new{typeof(dims), typeof(deltas), typeof(origin)}(dims, deltas, origin)
        return g
    end
end

function get_colorrange(colorrange; make_divergent=false)
    cr = collect(colorrange)
    if cr[1] == cr[2]
        cr[1] -= 1
        cr[2] += 1
    end
    if make_divergent
        cr[1] = min(cr[1], -cr[2])
        cr[2] = max(-cr[1], cr[2])
    end
    return cr
end

function get_shot_extrema(data)
    # Combine extrema from all shots.
    es = extrema.(data)
    cr = collect(es[1])
    for e in es
        cr[1] = min(cr[1], e[1])
        cr[2] = max(cr[2], e[2])
    end
    return cr
end

function set_circle(A, value, x, y; r=1)
    A[(x-r):(x+r), y] .= value
    for i in 1:r
        ri = r - i + 1
        xl = x - ri
        xr = x + ri
        A[xl:xr, y - i] .= value
        A[xl:xr, y + i] .= value
    end
end

function get_circle(x, y; r=1)
    xpoints = []
    ypoints = []
    append!(xpoints, range(x-r, x+r))
    append!(ypoints, fill(y, 2*r + 1))
    for i in 1:r
        ri = r - i + 1
        xl = x - ri
        xr = x + ri
        append!(xpoints, xl:xr)
        append!(ypoints, fill(y - i, 2*ri + 1))

        append!(xpoints, xl:xr)
        append!(ypoints, fill(y + i, 2*ri + 1))
    end
    return xpoints, ypoints
end

struct RunningStats{T}
    mean::T
    sum_squared_deviation::T
    n
    data_keys
    other_keys
end

function RunningStats{T}(data_keys, other_keys) where T
    stats = RunningStats{T}(T(), T(), [0], data_keys, other_keys)
end

function update_running_stats!(stats::RunningStats, x)
    if stats.n[1] == 0
        for k in stats.data_keys
            stats.mean[k] = deepcopy(x[k])
            stats.sum_squared_deviation[k] = zero(x[k])
        end
        for k in stats.other_keys
            stats.mean[k] = x[k]
            stats.sum_squared_deviation[k] = x[k]
        end
    end
    orig_mean = stats.mean
    stats.n[1] += 1
    for k in stats.data_keys
        stats.mean[k] = orig_mean[k] .+ (x[k] .- orig_mean[k]) ./ stats.n[1]
        stats.sum_squared_deviation[k] = stats.sum_squared_deviation[k] .+ (x[k] .- orig_mean[k]) .* (x[k] .- stats.mean[k])
    end
end

function get_sample_std(stats::RunningStats)
    stds = deepcopy(stats.sum_squared_deviation)
    for k in stats.data_keys
        stds[k] = sqrt.(stds[k] ./ (stats.n[1] - 1))
    end
    return stds
end

function get_connected_mask(arr, plume_ci)
    mask = (arr .> 0)
    return get_connected_mask(mask, plume_ci)
end

function get_connected_mask(mask::AbstractArray{Bool, N}, plume_ci::AbstractArray) where N
    mask[plume_ci] .= true
    return get_connected_mask(mask, plume_ci[1])
end

function get_connected_mask(mask::AbstractArray{Bool, N}, plume_ci::CartesianIndex) where N
    mask[plume_ci] = true

    dj_sets = IntDisjointSets(length(mask))
    LI = LinearIndices(mask)
    CI = CartesianIndices(mask)

    neighbors = CartesianIndex.((
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, -1),
        (1, -1),
    ))
    for (li, ci) in enumerate(CI)
        # Union each neighbor with this i.
        for n_offset in neighbors
            n_ci = ci + n_offset
            if checkbounds(Bool, mask, n_ci) && mask[ci] == mask[n_ci]
                union!(dj_sets, li, LI[n_ci])
            end
        end
    end

    set_labels = zeros(Int, size(mask))
    for (li, ci) in enumerate(CI)
        set_labels[ci] = find_root(dj_sets, li)
    end

    plume_li = LI[plume_ci]
    plume_set = find_root(dj_sets, plume_li)
    mask = (set_labels .== plume_set)
    return mask
end

function get_next_jump_idx(times, idx=1)
    """Advance idx until two consecutive times are not strictly increasing.

    Specifically, times[idx:get_next_jump_idx(times, idx)] is strictly increasing.

    julia> get_next_jump_idx([1, 2, 3])
    3
    julia> get_next_jump_idx([1, 2, 3, 1])
    3
    julia> get_next_jump_idx([1, 2, 3, 3])
    3
    julia> get_next_jump_idx([1, 2, 3, 1, 2, 3, 4, 5])
    3
    julia> get_next_jump_idx([1, 2, 3, 1, 2, 3, 4, 5, 1, 2], idx=4)
    9
    """
    jump_idx = idx + 1
    while jump_idx <= length(times) && times[jump_idx] > times[jump_idx - 1]
        jump_idx += 1
    end
    return jump_idx - 1
end