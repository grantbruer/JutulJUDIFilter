using .CairoMakie
using Printf
using LaTeXStrings
using FileIO

function plot_heatmap_from_grid(a, grid; fix_colorrange=true, make_divergent=false, make_colorbar=true, make_heatmap = false, kwargs...)
    # xs = range(grid.deltas[1]/2; length = grid.dims[1], step = grid.deltas[1]) .- grid.origin[1]
    # ys = range(grid.deltas[end]/2; length = grid.dims[end], step = grid.deltas[end]) .- grid.origin[end]
    if make_heatmap
        # For heatmap
        xs = range(0; length = grid.dims[1]+1, step = grid.deltas[1]) .- grid.origin[1]
        ys = range(0; length = grid.dims[end]+1, step = grid.deltas[end]) .- grid.origin[end]
    else
        # For contourf
        xs = grid.deltas[1]/2 .+ range(0; length = grid.dims[1], step = grid.deltas[1]) .- grid.origin[1]
        ys = grid.deltas[end]/2 .+ range(0; length = grid.dims[end], step = grid.deltas[end]) .- grid.origin[end]
    end

    if haskey(Dict(kwargs), :colorrange)
        colorrange = kwargs[:colorrange]
    else
        m1 = minimum(x -> isfinite(x) ? x : Inf, a)
        m2 = maximum(x -> isfinite(x) ? x : -Inf, a)
        colorrange = (m1, m2)
    end
    if fix_colorrange
        colorrange = get_colorrange(colorrange; make_divergent)
    end

    if haskey(Dict(kwargs), :aspect)
        aspect = kwargs[:aspect]
    else
        aspect = nothing
    end
    if make_heatmap
        fig, ax, hm = heatmap(xs, ys, a, axis=(yreversed=true,); colorrange, aspect, kwargs...)
    else
        levels = pop!(Dict(kwargs), :levels, 10)
        mode = pop!(Dict(kwargs), :normal, 10)
        if isa(levels, Int)
            # TODO: this doesn't work because colorrange may be an Observable.
            levels = range(colorrange[1], colorrange[2], levels)
        elseif mode == :relative
            levels = levels .* (colorrange[2] - colorrange[1]) .+ colorrange[1]
        end
        fig, ax, hm = contourf(xs, ys, a, axis=(yreversed=true,); levels, aspect, kwargs...)
    end
    if make_colorbar
        Colorbar(fig[:, end+1], hm)
    end
    aspect = (grid.dims[1] * grid.deltas[1]) / (grid.dims[end] * grid.deltas[end])
    xlims!(ax, - grid.origin[1], grid.dims[1] * grid.deltas[1] - grid.origin[1])
    ylims!(ax, grid.dims[end] * grid.deltas[end] - grid.origin[end], - grid.origin[end])
    colsize!(fig.layout, 1, Aspect(1, aspect))
    resize_to_layout!(fig)
    return fig, ax
end


function plot_shot_record(data; 
    dtR,
    timeR,
    nsrc,
    nrec,
    fix_colorrange = false,
    make_divergent = false,
    kwargs...
)
    s = 2
    fig = Figure(resolution = (1600*s, 900*s));
    grid = (2, 4)
    CI = CartesianIndices(grid)
    kwargs = Dict(kwargs)
    text_size = pop!(kwargs, :text_size, 28)

    if haskey(kwargs, :colorrange)
        colorrange = pop!(kwargs, :colorrange)
    else
        colorrange = get_shot_extrema(data)
    end

    if fix_colorrange
        colorrange = get_colorrange(colorrange; make_divergent)
    end

    local hm
    times = range(start=0, stop=timeR/1e3, step=dtR/1e3)
    src_range = StepRange(1, nsrc ÷ prod(grid), nsrc)
    for (i, ci) in zip(src_range, CI)
        ax = CairoMakie.Axis(fig[Tuple(ci)...], title = "Source $i", yreversed=true)
        ax.titlesize = text_size
        xs = @lift(1:size($data[i], 2))
        ys = times
        a = @lift($data[i]')
        hm = heatmap!(ax, xs, ys, a; colorrange, kwargs...)
        if ci[1] == grid[1]
            # Show x label on bottom row.
            ax.xlabel = "receiver index"
            ax.xticklabelsize = text_size
            ax.xlabelsize = text_size
        else
            # Hide x ticks everywhere else.
            ax.xticklabelsvisible = false
        end
        if ci[2] == 1
            # Show y label on left column.
            ax.ylabel = "time (seconds)"
            ax.ylabelsize = text_size
            ax.yticklabelsize = text_size
        else
            # Hide y ticks everywhere else.
            ax.yticklabelsvisible = false
        end
        hidespines!(ax)
    end
    Colorbar(fig[:, end+1], hm; label = "amplitude (Pa)", ticklabelsize = text_size, labelsize = text_size)
    return fig
end


function anim_reservoir_plotter(idx, getter, states;
    post_plot=(fig, ax)->nothing,
    colormap=parula,
    colorrange=nothing,
    grid,
    # inj_zidx,
    params,
    divergent=false,
    kwargs...
)
    dt = params["transition"]["dt"]
    inj_loc = params["transition"]["injection"]["loc"]
    prod_loc = params["transition"]["production"]["loc"]
    d_3d = params["transition"]["d"]
    injection_length = params["transition"]["injection"]["length"]

    data = @lift(getter(states[$idx]))
    function get_time_string(state)
        if haskey(state, :time_str)
            return state[:time_str]
        end
        return @sprintf "%5d days" state[:step] * dt
    end
    time_str = @lift(get_time_string(states[$idx]))

    if isnothing(colorrange)
        colorrange = @lift(extrema($data))
    elseif ! isa(colorrange, Observable)
        colorrange = Observable(colorrange)
    end
    colorrange = @lift(get_colorrange($colorrange; make_divergent=divergent))
    fig, ax = plot_heatmap_from_grid(data, grid; colormap, colorrange, fix_colorrange=false, kwargs...)

    # xi = [inj_loc[1], inj_loc[1]]
    # xp = [prod_loc[1], prod_loc[1]]

    # startz = inj_zidx * d_3d[3]
    # endz = startz + injection_length

    # y = [startz, endz]

    # lines!(ax, xi, y, markersize=20, label="Injector")
    # lines!(ax, xp, y, markersize=20, label="Producer")
    # axislegend()

    Label(fig[1, 1, Top()], time_str, halign = :center, valign = :bottom, font = :bold)
    post_plot(fig, ax)

    return fig
end


function anim_shot_record_plotter(idx, getter, states;
    colormap=parula,
    grid,
    params,
    divergent=false,
    title="",
    kwargs...
)
    dt = params["transition"]["dt"]
    dtR = params["observation"]["dtR"]
    timeR = params["observation"]["timeR"]
    nsrc = params["observation"]["nsrc"]
    nrec = params["observation"]["nrec"]

    data = @lift(getter(states[$idx]))
    function get_time_string(state)
        if haskey(state, :time_str)
            return state[:time_str]
        end
        return @sprintf "%5d days" state[:step] * dt
    end
    time_str = @lift(get_time_string(states[$idx]))

    colorrange = @lift begin
        return get_colorrange(get_shot_extrema($data); make_divergent=divergent)
    end
    fig = plot_shot_record(data;
        dtR,
        timeR,
        nsrc,
        nrec,
        colormap,
        colorrange,
        fix_colorrange = false,
    )
    if title != ""
        Label(fig[1, 1:end-1, Top()], title, valign = :top, font = :bold, fontsize=30, padding = (0, 0, 5, 0))
    end
    Label(fig[1, 1, Top()], time_str, halign = :left, valign = :bottom, font = :bold, fontsize=30)
    return fig
end


function my_step!(s::Makie.FolderStepper)
    Makie.update_state_before_display!(s.figlike)
    # s = @sprintf("%04d"), s.step
    FileIO.save(joinpath(s.folder, basename(s.folder) * "-$(@sprintf("%04d", s.step)).$(s.format)"), Makie.colorbuffer(s.screen))
    s.step += 1
    return s
end

function record_png(update_idx, fig, foldername, idx_iterator; framerate)
    """Same arguments as Make.record, but makes image files instead of a video."""
    st = Stepper(fig, foldername)
    for i in idx_iterator
        update_idx(i)
        my_step!(st)
    end
end

function plot_anim(states, getter, plotter, filename; extras=(), framerate=2, plot_png=true, plot_mp4=true, n=0)
    """Makes a video and a folder of images.

    plotter is called with an Observable idx which is incremented from 1 to the length of states.
    """
    function update_idx(i)
        idx[] = i
    end
    if n == 0
        n = size(states, 1)
    end

    if n == 1
        # Just plot one png.
        stem = splitext(filename)[1]
        if stem == filename
            stem *= "_png"
        end
        filename = stem * ".png"

        idx = Observable(1)
        fig = plotter(idx, getter, states; extras...)
        save(filename, fig)
        return
    end

    if plot_mp4
        idx = Observable(1)
        idx_iterator = collect(1:n)
        idx_iterator = append!(idx_iterator, fill(n, framerate))

        fig = plotter(idx, getter, states; extras...)
        Makie.record(update_idx, fig, filename, idx_iterator; framerate = framerate)
    end

    if plot_png
        foldername = splitext(filename)[1]
        if foldername == filename
            foldername *= "_png"
        end

        idx = Observable(1)
        idx_iterator = collect(1:n)

        fig = plotter(idx, getter, states; extras...)
        record_png(update_idx, fig, foldername, idx_iterator; framerate = framerate)
    end
end
