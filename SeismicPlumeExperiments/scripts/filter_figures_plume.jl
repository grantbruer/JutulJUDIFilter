import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.instantiate()

import TOML

using Statistics
using LinearAlgebra
using JLD2
using Printf
using MyUtils
using CairoMakie
using LaTeXStrings
include("../lib/seismic_plume_params.jl")

text_size_normal = 28
text_size_smaller = 24

function filter_figures_plume(params_file, job_dir)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    params["transition"]["dt"] *= params["filter"]["update_interval"]

    # Load filter.
    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)
    filepath = joinpath(work_path, "filter_$(0)_posterior")
    filter = load_filter(params, filepath)

    # Make directory for saving figures.
    save_dir = joinpath(job_dir, "figs", work_dir)
    mkpath(save_dir)

    running_stats = Vector{RunningStats}()
    folder_path = joinpath(save_dir, "ensemble")
    mkpath(folder_path)
    data_keys = [:Saturation]
    other_keys = [:step]
    for i = 1:filter.params.ensemble_size
        states = load_ensemble_member_states(job_dir, filter, i)
        for i in 1:length(states)
            if i > length(running_stats)
                push!(running_stats, RunningStats{Dict{Symbol, Any}}(data_keys, other_keys))
            end
            update_running_stats!(running_stats[i], states[i])
        end
        println("Ensemble member $(i) has $(length(states)) states")
        if length(states) <= 1
            continue
        end
        file_prefix_path = joinpath(folder_path, "$(i)")
        plot_ensemble_plume_data(states, file_prefix_path; params)
        if i > 5
            break
        end
    end
    stats = running_stats
    means = [stats.mean for stats in running_stats]
    stds = [get_sample_std(stats) for stats in running_stats]

    file_prefix_path = joinpath(save_dir, "mean")
    plot_ensemble_plume_data(means, file_prefix_path; params, plot_png=true)

    file_prefix_path = joinpath(save_dir, "std")
    plot_ensemble_plume_data(stds, file_prefix_path; params, plot_png=true)
end

function plot_ensemble_plume_data(states, file_prefix_path; params, plot_png=false)
    if length(states) == 0
        error("This is ridiculous. I can't plot plumes without any data.")
    end
    n_3d = Tuple(params["transition"]["n"])
    d_3d = Tuple(params["transition"]["d"])
    dt = params["transition"]["dt"]

    # Use mesh in kilometers instead of meters.
    # origin = (params["transition"]["injection"]["loc"][1] / 1000, 0, 0)
    origin = (params["transition"]["injection"]["loc"][1] / 1000 - d_3d[1]/2000, 0, 0 - d_3d[end]/2000)
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
        if ! isa(ax.xlabel[], LaTeXString) && ax.xlabel[] != ""
            ax.xlabel = LaTeXString(ax.xlabel[])
        end
        if ! isa(ax.ylabel[], LaTeXString) && ax.ylabel[] != ""
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
                cb.label = cb_label
                cb.labelrotation = 0.0
            end
            cb.labelsize = text_size_smaller
            cb.ticklabelsize = text_size_smaller
            if isnothing(cb_tickformat)
                if cb.tickformat[] == Makie.Automatic()
                    cb_tickformat = values -> [latexstring(cfmt("%.1f", v)) for v in values]
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
    end


    for state in states
        # state[:time_str] = @sprintf "%.2f years" state[:step] * dt / 365.2425
        state[:time_str] = ""
    end

    extras = (; params, grid=mesh_3d_cutoff)
    framerate = 2

    get_saturation_simple(state) = ifelse.(state[:Saturation] .== 0, NaN, state[:Saturation])[:, cutoff_idx:end]
    get_pressure_simple(state) = state[:Pressure]

    sat_file = "$(file_prefix_path)_saturation.mp4"
    delete_colorbar = true
    post_plot = function (fig, ax)
        ax.xlabel = "Horizontal (km)"
        ax.ylabel = "Depth (km)"
        # ax.title = "CO₂ saturation"
        axis_setup(ax; delete_colorbar)
    end
    levels = range(0, 1, length = 11)
    extras2 = (; post_plot, levels, make_colorbar=false, extras...)
    plot_anim(states, get_saturation_simple, anim_reservoir_plotter, sat_file; plot_png, extras=extras2, framerate)

    if haskey(states[1], :Pressure)
        pre_file = "$(file_prefix_path)_pressure.mp4"
        post_plot = function (fig, ax)
            ax.xlabel = "Horizontal (km)"
            ax.ylabel = "Depth (km)"
            # ax.title = "Pressure (Pa)"
        end
        extras2 = (; post_plot, extras...)
        plot_anim(states, get_pressure_simple, anim_reservoir_plotter, pre_file; plot_png, extras=extras2, framerate)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    filter_figures_plume(params_file, job_dir)
end
