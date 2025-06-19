using CairoMakie
using Format

# Normal units for source term are Pa/m^2.
# Normal units for temporal source term are Pa in 2D, Pa⋅m in 3D.
# JUDI units for temporal source term are Pa/m^2, and need to be scaled by cell volume to get normal units.

# The source term I give to JUDI is unitless, so I need to scale it by this to get the right units.
ORIGINAL_SOURCE_WAVELET_JUDI_TO_SI = 1e9 # units are Pa / m^2

# I want to rescale the source term to give it more normal magnitude.
TARGET_WAVELET_TEMPORAL_MAGNITUDE = 1e6 / 78.125 # units are Pa/m^2
ORIGINAL_PA_TO_NEW_PA = TARGET_WAVELET_TEMPORAL_MAGNITUDE / ORIGINAL_SOURCE_WAVELET_JUDI_TO_SI # Unitless

# Compute unitless rescaler for source term and all pressure outputs.
global WAVELET_JUDI_TEMPORAL_TO_TEMPORAL = 12.5 * 6.25 # m^2
global WAVELET_JUDI_TEMPORAL_TO_SI = TARGET_WAVELET_TEMPORAL_MAGNITUDE * WAVELET_JUDI_TEMPORAL_TO_TEMPORAL # Pa

# try
#     import MyUtils: plot_heatmap_from_grid
# catch e
#     @warn "Couldn't load MyUtils: $e"
    function plot_heatmap_from_grid!(ax, a, grid; kwargs...)
        plot_heatmap_from_grid!(ax, a; dims=grid.dims, deltas=grid.deltas, origin=grid.origin, kwargs...)
    end
# end

function plot_heatmap_from_grid(args...; make_colorbar=false, kwargs...)
    fig = Figure()
    ax = Axis(fig[1,1], yreversed=true)
    hm = plot_heatmap_from_grid!(ax, args...; kwargs...)
    if make_colorbar
        Colorbar(fig[:, end+1], hm)
    end
    return fig, ax, hm
end

function rescale_heatmap_to_grid!(fig; dims, deltas=(1, 1), origin=(0, 0))
    aspect = (dims[1] * deltas[1]) / (dims[end] * deltas[end])
    colsize!(fig.layout, 1, Aspect(1, aspect))
    resize_to_layout!(fig)
end

function get_coordinate_corners(; dims, deltas, origin)
    xs = range(0; length = dims[1]+1, step = deltas[1]) .- origin[1]
    ys = range(0; length = dims[end]+1, step = deltas[end]) .- origin[end]
    return xs, ys
end

function get_coordinates_cells(; dims, deltas, origin)
    xs = deltas[1]/2 .+ range(0; length = dims[1], step = deltas[1]) .- origin[1]
    ys = deltas[end]/2 .+ range(0; length = dims[end], step = deltas[end]) .- origin[end]
    return xs, ys
end

function plot_heatmap_from_grid!(ax, a; dims, deltas=(1, 1), origin=(0, 0), colorrange=nothing, fix_colorrange=true, make_divergent=false, make_heatmap = false, kwargs...)
    if make_heatmap
        xs, ys = get_coordinate_corners(; dims, deltas, origin)
    else
        xs, ys = get_coordinates_cells(; dims, deltas, origin)
    end

    if isnothing(colorrange)
        m1 = minimum(x -> isfinite(x) ? x : Inf, a)
        m2 = maximum(x -> isfinite(x) ? x : -Inf, a)
        colorrange = (m1, m2)
    end
    if fix_colorrange
        colorrange = get_colorrange(colorrange; make_divergent)
    end

    if make_heatmap
        hm = heatmap!(ax, xs, ys, a; colorrange, kwargs...)
    else
        levels = pop!(Dict(kwargs), :levels, 10)
        mode = pop!(Dict(kwargs), :normal, 10)
        if isa(levels, Int)
            # TODO: this doesn't work because colorrange may be an Observable.
            levels = range(colorrange[1], colorrange[2], levels)
        elseif mode == :relative
            levels = levels .* (colorrange[2] - colorrange[1]) .+ colorrange[1]
        end
        hm = contourf!(ax, xs, ys, a; levels, kwargs...)
    end
    xlims!(ax, - origin[1], dims[1] * deltas[1] - origin[1])
    ylims!(ax, dims[end] * deltas[end] - origin[end], - origin[end])
    rescale_heatmap_to_grid!(ax.parent; dims, deltas, origin)
    return hm
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

function plot_disjoint_lines!(ax, times, ys; kwargs...)
    end_idx = 0
    color = get(kwargs, :color, nothing)
    while end_idx + 1 <= length(times)
        start_idx = end_idx + 1
        end_idx = get_next_jump_idx(times, start_idx)
        if isnothing(color)
            sc = scatterlines!(ax, times[start_idx:end_idx], ys[start_idx:end_idx]; kwargs...)
            color = sc.color
        else
            sc = scatterlines!(ax, times[start_idx:end_idx], ys[start_idx:end_idx]; color, kwargs...)
        end
        color = sc.color
    end
end

function plot_disjoint_lines(times, ys; kwargs...)
    start_idx = 1
    end_idx = get_next_jump_idx(times, start_idx)
    fig, ax, sc = scatterlines(times[start_idx:end_idx], ys[start_idx:end_idx]; kwargs...)
    plot_disjoint_lines!(ax, times[end_idx+1:end], ys[end_idx+1:end]; color=sc.color, kwargs...)
    return fig, ax, sc
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

function add_colorbar_from_labels(ax; kwargs...)
    lplots, labels = Makie.get_labeled_plots(ax; merge=false, unique=true)
    limits = (0, length(labels))
    ticks_pos = (1:length(labels)) .- 0.5
    ticks = (collect(ticks_pos), labels)
    lcolors = [to_value(p.color)[1] for p in lplots]
    lcolormap = cgrad(Makie.ColorScheme(lcolors), categorical=true)
    cb = Colorbar(fig[1, end+1]; limits, colormap=lcolormap, ticks, kwargs...)
    return cb
end

get_tickformat(a) = a
get_tickformat(a::AbstractString) = values -> [latexstring(cfmt(a, v)) for v in values]

set_theme!(theme_latexfonts())
function axis_setup(ax; cb_label=nothing, cb_tickformat=nothing, delete_colorbar=true, text_size_normal=28, text_size_smaller=24,
    xtickformat = nothing, ytickformat = nothing,cb_rotation=0.0)
    hidespines!(ax)
    ax.xticklabelsize = text_size_normal
    ax.xlabelsize = text_size_normal
    ax.yticklabelsize = text_size_normal
    ax.ylabelsize = text_size_normal
    if ! isa(ax.xlabel[], LaTeXString) && ax.xlabel[] != ""
        ax.xlabel = LaTeXString(ax.xlabel[])
    end
    if ! isa(ax.ylabel[], LaTeXString) && ax.ylabel[] != ""
        ax.ylabel = LaTeXString(ax.ylabel[])
    end
    println("Title: $(ax.title[])")
    if !isnothing(xtickformat)
        ax.xtickformat = get_tickformat(xtickformat)
    end
    if !isnothing(ytickformat)
        ax.ytickformat = get_tickformat(ytickformat)
    end

    # Process colorbar.
    idx = findfirst(x -> x isa Colorbar, ax.parent.content)
    if !isnothing(idx)
        cb = ax.parent.content[idx]
        if ! isnothing(cb_label)
            cb.label = isa(cb_label, LaTeXString) ? cb_label : LaTeXString(cb_label)
            if cb.labelrotation[] == Makie.Automatic()
                cb.labelrotation = cb_rotation
            end
        end
        cb.labelsize = text_size_smaller
        cb.ticklabelsize = text_size_smaller
        # if cb.tickformat[] == Makie.Automatic() && isnothing(cb_tickformat)
        #     cb_tickformat = "%g"
        # end
        # if isa(cb_tickformat, AbstractString)
        #     cb_tickformat = values -> [latexstring(cfmt(cb_tickformat, v)) for v in values]
        # end
        if !isnothing(cb_tickformat)
            cb.tickformat = get_tickformat(cb_tickformat)
        end
        if delete_colorbar
            delete!(cb)
        end
    end
    resize_to_layout!(ax.parent)

    # Process legend.
    # idx = findfirst(x -> x isa Legend, ax.parent.content)
    # if !isnothing(idx)
    #     cb = ax.parent.content[idx]
    #     if ! isnothing(cb_label)
    #         cb.label = LaTeXString(cb_label)
    #         cb.labelrotation = 0.0
    #     end
    #     cb.labelsize = text_size_smaller
    #     cb.ticklabelsize = text_size_smaller
    #     if isnothing(cb_tickformat)
    #         if cb.tickformat[] == Makie.Automatic()
    #             cb_tickformat = values -> [latexstring(cfmt("%.1f", v)) for v in values]
    #         end
    #     elseif isa(cb_tickformat, AbstractString)
    #         formatter = generate_formatter(cb_tickformat)
    #         cb_tickformat = values -> [latexstring(formatter(v)) for v in values]
    #     end
    #     if !isnothing(cb_tickformat)
    #         cb.tickformat = cb_tickformat
    #     end
    #     if delete_colorbar
    #         delete!(cb)
    #     end
    #     resize_to_layout!(ax.parent)
    # end
end
