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
using Format
include("../lib/seismic_plume_params.jl")


text_size_normal = 28
text_size_smaller = 24

function filter_figures_params(params_file, job_dir)
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
    data_keys = [:Permeability, :Porosity]
    other_keys = [:step]
    single_step = true
    for i = 1:filter.params.ensemble_size
        states = load_ensemble_member_params(job_dir, filter, i)
        for i in 1:length(states)
            if i > length(running_stats)
                push!(running_stats, RunningStats{Dict{Symbol, Any}}(data_keys, other_keys))
            end
            update_running_stats!(running_stats[i], states[i])
        end
        println("Ensemble member $(i) has $(length(states)) states")
        all_same = all(key -> all(state -> all(state[key] .== states[1][key]), states), data_keys)
        if all_same
            println("  but they're all identical to the first state")
            states = states[1:1]
        else
            single_step = false
        end
        file_prefix_path = joinpath(folder_path, "$(i)")
        # if i < 5
            plot_ensemble_params(states, file_prefix_path; params, plot_png=all_same, plot_mp4=!all_same)
        #end
        if i > 5 && !single_step
            break
        end
        # break
    end
    stats = running_stats
    if single_step
        means = [running_stats[1].mean]
        stds = [get_sample_std(running_stats[1])]
    else
        means = [stats.mean for stats in running_stats]
        stds = [get_sample_std(stats) for stats in running_stats]
    end

    file_prefix_path = joinpath(save_dir, "mean")
    plot_ensemble_params(means, file_prefix_path; params, plot_png=true, plot_mp4=!single_step)

    file_prefix_path = joinpath(save_dir, "std")
    plot_ensemble_params(stds, file_prefix_path; params, plot_png=true, plot_mp4=!single_step)
end

function plot_ensemble_params(states, file_prefix_path; params, plot_png=false, plot_mp4=true)
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
    file_path = "$(file_prefix_path)_permeability.mp4"
    post_plot = function (fig, ax)
        ax.xlabel = "Horizontal (km)"
        ax.ylabel = "Depth (km)"
        axis_setup(ax)
    end
    extras = (; post_plot, colormap = Reverse(:Purples), colorrange = (0, 1600), make_colorbar = false, extendhigh = :yellow, base_extras...)
    plot_anim(
        states, get_simple, anim_reservoir_plotter, file_path;
        plot_png, plot_mp4, extras, framerate,
    )

    # Plot porosity
    get_simple = state -> state[:Porosity][:, cutoff_idx:end]
    file_path = "$(file_prefix_path)_porosity.mp4"
    post_plot = function (fig, ax)
        ax.xlabel = "Horizontal (km)"
        ax.ylabel = "Depth (km)"
        axis_setup(ax)
    end
    extras = (; post_plot, colorrange=(0,1), base_extras...)
    plot_anim(
        states, get_simple, anim_reservoir_plotter, file_path;
        plot_png, plot_mp4, extras, framerate,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    filter_figures_params(params_file, job_dir)
end
