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
include("../lib/seismic_plume_params.jl")

function filter_figures_seismic(params_file, job_dir)
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

    # Load baseline RTM.
    stem = get_ground_truth_seismic_stem(params)
    stem = joinpath(job_dir, stem)
    baseline, extra_data = read_ground_truth_seismic_baseline(stem; state_keys = [:rtm_born, :rtm_born_noisy])

    # Make directory for saving figures.
    save_dir = joinpath(job_dir, "figs", work_dir)
    mkpath(save_dir)

    running_stats = Vector{RunningStats}()
    folder_path = joinpath(save_dir, "ensemble")
    mkpath(folder_path)
    data_keys = [:rtm, :rtm_noisy, :rtm_offset, :rtm_offset_noisy]
    other_keys = [:step]
    for i = 1:filter.params.ensemble_size
        states = load_ensemble_member_rtms(job_dir, filter, i)
        for i in 1:length(states)
            if i > length(running_stats)
                push!(running_stats, RunningStats{Dict{Symbol, Any}}(data_keys, other_keys))
            end
            states[i][:rtm_offset] = states[i][:rtm] .- baseline[:rtm_born]
            states[i][:rtm_offset_noisy] = states[i][:rtm_noisy] .- baseline[:rtm_born_noisy]
            update_running_stats!(running_stats[i], states[i])
        end
        println("Ensemble member $(i) has $(length(states)) states")
        if length(states) == 0
            continue
        end
        file_prefix_path = joinpath(folder_path, "$(i)")
        plot_rtm_seismic_data(baseline, states, file_prefix_path; params)
        if i > 5
            break
        end
    end

    global stats = running_stats
    global means = [stats.mean for stats in running_stats]
    global stds = [get_sample_std(stats) for stats in running_stats]

    file_prefix_path = joinpath(save_dir, "mean")
    plot_rtm_seismic_data(baseline, means, file_prefix_path; params, plot_png=true)

    file_prefix_path = joinpath(save_dir, "std")
    plot_rtm_seismic_data(baseline, stds, file_prefix_path; params, plot_png=true, colormap=Reverse(:inferno), divergent=false)

end

function plot_rtm_seismic_data(baseline, states, file_prefix_path; params, plot_png=false, colormap=:balance, divergent=true)
    if length(states) == 0
        error("This is ridiculous. I can't plot rtms without any data.")
    end
    n_3d = Tuple(params["transition"]["n"])
    d_3d = Tuple(params["transition"]["d"])
    dt = params["transition"]["dt"]

    # Use mesh in kilometers instead of meters.
    mesh_3d = CartesianMesh(n_3d, d_3d .* n_3d ./ 1000.0)

    for state in states
        state[:time_str] = @sprintf "%.2f years" state[:step] * dt / 365.2425
    end

    extras = (; params, grid=mesh_3d)
    framerate = 2

    get_rtm_plain(state) = state[:rtm]
    get_rtm_offset(state) = state[:rtm_offset]

    file_path = "$(file_prefix_path)_rtm_plain.mp4"
    post_plot = function (fig, ax)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        # ax.title = "RTM: diag(z) W J₀ᵀ (J₀(ρ₀v₀ - ρv) + η)"
    end
    extras2 = (; post_plot, extras..., colormap, divergent)
    plot_anim(states, get_rtm_plain, anim_reservoir_plotter, file_path; plot_png, extras=extras2, framerate)

    file_path = "$(file_prefix_path)_rtm_offset.mp4"
    post_plot = function (fig, ax)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        # ax.title = "RTM: diag(z) W J₀ᵀ (J₀(ρ₁v₁ - ρv) + η - ϵ)"
    end
    extras2 = (; post_plot, extras..., colormap, divergent)
    plot_anim(states, get_rtm_offset, anim_reservoir_plotter, file_path; plot_png, extras=extras2, framerate)

    get_rtm_plain_noisy(state) = state[:rtm_noisy]
    get_rtm_offset_noisy(state) = state[:rtm_offset_noisy]

    file_path = "$(file_prefix_path)_rtm_plain_noisy.mp4"
    post_plot = function (fig, ax)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        # ax.title = "RTM: diag(z) W J₀ᵀ (J₀(ρ₀v₀ - ρv) + η)"
    end
    extras2 = (; post_plot, extras..., colormap, divergent)
    plot_anim(states, get_rtm_plain_noisy, anim_reservoir_plotter, file_path; plot_png, extras=extras2, framerate)

    file_path = "$(file_prefix_path)_rtm_offset_noisy.mp4"
    post_plot = function (fig, ax)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        # ax.title = "RTM: diag(z) W J₀ᵀ (J₀(ρ₁v₁ - ρv) + η - ϵ)"
    end
    extras2 = (; post_plot, extras..., colormap, divergent)
    plot_anim(states, get_rtm_offset_noisy, anim_reservoir_plotter, file_path; plot_png, extras=extras2, framerate)
end

if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    filter_figures_seismic(params_file, job_dir)
end

