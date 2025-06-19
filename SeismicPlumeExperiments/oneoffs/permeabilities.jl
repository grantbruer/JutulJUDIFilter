import Pkg
# Pkg.activate("plotting"; shared=true)

import TOML

using Statistics
using LinearAlgebra
using JLD2
using Printf
using MyUtils
using CairoMakie
using LaTeXStrings
using Format
include("../lib/seismic_plume_params.jl")
include("utils.jl")


function my_anim_reservoir_plotter(idx, getter, states;
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

function plot_ensemble_params(states, save_dir_root; params, plot_png=false, plot_mp4=true)
    if length(states) == 0
        error("This is ridiculous. I can't plot plumes without any data.")
    end
    n_3d = Tuple(params["transition"]["n"])
    d_3d = Tuple(params["transition"]["d"])
    dt = params["transition"]["dt"]

    # Use mesh in kilometers instead of meters.
    origin = (params["transition"]["injection"]["loc"][1] / 1000, 0, 0)
    mesh_3d = CartesianMesh(n_3d, d_3d .* n_3d ./ 1000.0; origin)

    cutoff_idx = Int(1200 ÷ d_3d[end])
    cutoff_depth = cutoff_idx * d_3d[end]
    n_3d_cutoff = n_3d .- (0, 0, cutoff_idx - 1)
    origin_cutoff = origin .- (0, 0, cutoff_depth ./ 1000)
    mesh_3d_cutoff = CartesianMesh(n_3d_cutoff, d_3d .* n_3d_cutoff ./ 1000.0; origin=origin_cutoff)

    function axis_setup(ax; cb_label=nothing, cb_tickformat=nothing, delete_colorbar=true)
        hidespines!(ax)
        ax.xticklabelsize = text_size_normal
        ax.xlabelsize = text_size_normal
        ax.yticklabelsize = text_size_normal
        ax.ylabelsize = text_size_normal
        if ! isa(ax.xlabel[], LaTeXString)
            ax.xlabel = LaTeXString(ax.xlabel[])
        end
        if ! isa(ax.ylabel[], LaTeXString)
            ax.ylabel = LaTeXString(ax.ylabel[])
        end
        println("Title: $(ax.title)")
        if ax.xtickformat[] == Makie.Automatic()
            ax.xtickformat = values -> [latexstring(@sprintf("%.0f", v)) for v in values]
        end
        if ax.ytickformat[] == Makie.Automatic()
            ax.ytickformat = values -> [latexstring(@sprintf("%.1f", v)) for v in values]
        end

        idx = findfirst(x -> x isa Colorbar, ax.parent.content)
        if !isnothing(idx)
            cb = ax.parent.content[idx]
            if ! isnothing(cb_label)
                # cb.label = cb_label
                cb.label = isa(cb_label, LaTeXString) ? cb_label : LaTeXString(cb_label)
                cb.labelrotation = 0.0
            end
            cb.labelsize = text_size_smaller
            cb.ticklabelsize = text_size_smaller
            if isnothing(cb_tickformat)
                if cb.tickformat[] == Makie.Automatic()
                    cb_tickformat = values -> [latexstring(cfmt("%.0f", v)) for v in values]
                end
            elseif isa(cb_tickformat, AbstractString)
                formatter = generate_formatter(cb_tickformat)
                cb_tickformat = values -> [latexstring(formatter(v)) for v in values]
            end
            if !isnothing(cb_tickformat)
                cb.tickformat = cb_tickformat
            end
            if delete_colorbar
                delete!(cb)
                resize_to_layout!(ax.parent)
            end
        end
        # ax.title = ""
    end


    for state in states
        # state[:time_str] = @sprintf "%.2f years" state[:step] * dt / 365.2425
        state[:time_str] = ""
    end

    base_extras = (; params,
        make_colorbar = false,
        make_heatmap = true,
        grid = mesh_3d_cutoff,
    )
    framerate = 2

    # Plot permeability
    get_simple = state -> state[:Permeability][:, cutoff_idx:end] ./ mD_to_meters2
    file_path = joinpath(save_dir_root, "permeability.mp4")
    post_plot = function (fig, ax)
        ax.xlabel = "Horizontal (km)"
        ax.ylabel = "Depth (km)"
        axis_setup(ax)
    end
    extras = (; post_plot, colormap = Reverse(:Purples), colorrange = (0, 1700), make_colorbar = false, base_extras...)
    plot_anim(
        states, get_simple, my_anim_reservoir_plotter, file_path;
        plot_png, plot_mp4, extras, framerate,
    )
end

text_size_normal = 28
text_size_smaller = 24

params_file = "params/enkf/base.toml"
job_dir = joinpath(pwd(), "run")

if ! (@isdefined params) || isnothing(params)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    params["transition"]["dt"] *= params["filter"]["update_interval"]

    params["transition"] = merge(params["transition"], params["filter"]["transition"])
end


if ! (@isdefined Ks) || isnothing(Ks)
    Ks = get_permeability(params)
    states = [Dict{Symbol, Any}(:Permeability=> a) for a in eachslice(Ks; dims=1)]
end

# Make directory for saving figures.
save_dir = joinpath("figs", "filters")
mkpath(save_dir)

set_theme!(theme_latexfonts())

plot_ensemble_params(states, save_dir; params, plot_png=true, plot_mp4=true)


# file_prefix_path = joinpath(save_dir, "mean")
# plot_ensemble_params(means, file_prefix_path; params, plot_png=true, plot_mp4=!single_step)

# file_prefix_path = joinpath(save_dir, "std")
# plot_ensemble_params(stds, file_prefix_path; params, plot_png=true, plot_mp4=!single_step)
