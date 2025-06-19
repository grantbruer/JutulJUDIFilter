
import Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

import TOML
import Random

using CairoMakie
using EnsembleFilters: assimilate_data, get_filter_work_dir
using JLD2
using LinearAlgebra
using MyUtils

import KernelMatrices
using EnsembleKalmanFilters
using EnsembleKalmanFilters: assimilate_data

include("../lib/ensemble_actions.jl")

function main(params_file, job_dir)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    mkpath(job_dir)
    ground_truth_dir = joinpath(job_dir, "ground_truth")
    mkpath(ground_truth_dir)

    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)
    mkpath(work_path)

    figs_dir = joinpath(job_dir, "figs")
    mkpath(figs_dir)

    gt_figs_dir = joinpath(figs_dir, "ground_truth")
    mkpath(gt_figs_dir)

    filter_figs_dir = joinpath(figs_dir, work_dir)
    mkpath(filter_figs_dir)

    # Read ground_truth.
    file_path = joinpath(ground_truth_dir, "ground_truth.jld2")
    ground_truth_states = load(file_path, "states")
    ground_truth_observations = load(file_path, "observations")

    # Plot ground truth.
    xs = [state[1] for state in ground_truth_states]
    ys = [state[2] for state in ground_truth_states]
    zs = [state[3] for state in ground_truth_states]
    plot_lorenz_views(xs, ys, zs, gt_figs_dir)

    # Read filter.
    data_keys = [
        :state,
        :state_error,
    ]
    other_keys = [:step]
    running_stats = Vector{RunningStats}()
    for i = 0:(length(ground_truth_states) - 1)
        file_path = joinpath(work_path, "$(i)-pred.jld2")
        filter = load(file_path, "filter")
        ensemble_states = [Dict(
            :state => em.state,
            :state_error => em.state .- ground_truth_states[i+1],
            :step => i,
        ) for em in filter.ensemble]

        stats = RunningStats{Dict{Symbol, Any}}(data_keys, other_keys)
        for j in length(ensemble_states)
            update_running_stats!(stats, ensemble_states[j])
        end
        push!(running_stats, stats)
        file_path = joinpath(work_path, "$(i)-pred.jld2")
        if isfile(file_path)
            filter = load(file_path, "filter")
            ensemble_states = [Dict(
                :state => em.state,
                :state_error => em.state .- ground_truth_states[i+1],
                :step => i,
            ) for em in filter.ensemble]

            stats = RunningStats{Dict{Symbol, Any}}(data_keys, other_keys)
            for j in length(ensemble_states)
                update_running_stats!(stats, ensemble_states[j])
            end
            push!(running_stats, stats)
        end
    end
    means = [stats.mean for stats in running_stats]
    stds = [get_sample_std(stats) for stats in running_stats]

    # Plot filter error.
    times = [state[:step] for state in means]
    kwargs = (;
        linewidth = 1,
        markersize = 0,
        colormap = :balance,
    )
    text_size_normal = 28
    text_size_smaller = 24
    err = [norm(state[:state_error]) for state in means]
    fig, ax, sc = plot_disjoint_lines(times, times, err; kwargs...)
    # fig, ax, sc = lines(err; kwargs...)
    ax.xlabel = "time step"
    ax.ylabel = "error"
    hidespines!(ax)
    ax.xticklabelsize = text_size_normal
    ax.xlabelsize = text_size_normal
    ax.yticklabelsize = text_size_normal
    ax.ylabelsize = text_size_normal
    save(joinpath(filter_figs_dir, "error.png"), fig)

    times = [state[:step] for state in means]
    kwargs = (;
        linewidth = 1,
        markersize = 0,
        colormap = :balance,
    )
    text_size_normal = 28
    text_size_smaller = 24
    err = [norm(state[:state_error]) for state in means]
    fig, ax, sc = plot_disjoint_lines(times, times, err; kwargs...)
    # fig, ax, sc = lines(err; kwargs...)
    ax.xlabel = "time step"
    ax.ylabel = "error"
    ax.yscale = log10
    hidespines!(ax)
    ax.xticklabelsize = text_size_normal
    ax.xlabelsize = text_size_normal
    ax.yticklabelsize = text_size_normal
    ax.ylabelsize = text_size_normal
    save(joinpath(filter_figs_dir, "error-log.png"), fig)

    # Plot 3D filter estimate.
    xs = [state[:state][1] for state in means]
    ys = [state[:state][2] for state in means]
    zs = [state[:state][3] for state in means]
    plot_lorenz_views(xs, ys, zs, joinpath(filter_figs_dir, "mean"))

    # Plot 3D filter state error.
    xs = [state[:state_error][1] for state in means]
    ys = [state[:state_error][2] for state in means]
    zs = [state[:state_error][3] for state in means]
    plot_lorenz_views(xs, ys, zs, joinpath(filter_figs_dir, "mean_error"))

end

function plot_disjoint_lines!(ax, times, xs, ys; kwargs...)
    end_idx = 0
    kwargs = Dict(kwargs)
    color = pop!(kwargs, "color", 1:length(xs))
    @assert length(color) == length(times)
    @assert length(color) == length(xs)
    @assert length(color) == length(ys)
    while end_idx + 1 <= length(times)
        start_idx = end_idx + 1
        end_idx = MyUtils.get_next_jump_idx(times, start_idx)
        sc = scatterlines!(ax, xs[start_idx:end_idx], ys[start_idx:end_idx]; kwargs..., color=color[start_idx:end_idx])
    end
end

function plot_disjoint_lines(times, xs, ys; kwargs...)
    start_idx = 1
    end_idx = MyUtils.get_next_jump_idx(times, start_idx)
    kwargs = Dict(kwargs)
    color = pop!(kwargs, "color", 1:length(xs))
    @assert length(color) == length(times)
    @assert length(color) == length(xs)
    @assert length(color) == length(ys)
    fig, ax, sc = scatterlines(xs[start_idx:end_idx], ys[start_idx:end_idx]; kwargs..., color=color[start_idx:end_idx])
    plot_disjoint_lines!(ax, times[end_idx+1:end], times[end_idx+1:end], ys[end_idx+1:end]; kwargs...)
    return fig, ax, sc
end


function plot_lorenz_views(xs, ys, zs, save_dir, times=nothing)
    mkpath(save_dir)

    common_kwargs = (;
        linewidth = 1,
        markersize = 0,
        colormap = :balance,
        color = 1:length(xs),
    )
    text_size_normal = 28
    text_size_smaller = 24

    if isnothing(times)
        fig, ax, sc = scatterlines(xs, ys; common_kwargs...)
    else
        fig, ax, sc = MyUtils.plot_disjoint_lines(times, xs, ys; common_kwargs...)
    end
    ax.xlabel = "x"
    ax.ylabel = "y"
    hidespines!(ax)
    ax.xticklabelsize = text_size_normal
    ax.xlabelsize = text_size_normal
    ax.yticklabelsize = text_size_normal
    ax.ylabelsize = text_size_normal
    save(joinpath(save_dir, "x_vs_y.png"), fig)


    if isnothing(times)
        fig, ax, sc = scatterlines(xs, zs; common_kwargs...)
    else
        fig, ax, sc = MyUtils.plot_disjoint_lines(times, xs, zs; common_kwargs...)
    end
    ax.xlabel = "x"
    ax.ylabel = "z"
    hidespines!(ax)
    ax.xticklabelsize = text_size_normal
    ax.xlabelsize = text_size_normal
    ax.yticklabelsize = text_size_normal
    ax.ylabelsize = text_size_normal
    save(joinpath(save_dir, "x_vs_z.png"), fig)


    if isnothing(times)
        fig, ax, sc = scatterlines(ys, zs; common_kwargs...)
    else
        fig, ax, sc = MyUtils.plot_disjoint_lines(times, ys, zs; common_kwargs...)
    end
    ax.xlabel = "y"
    ax.ylabel = "z"
    hidespines!(ax)
    ax.xticklabelsize = text_size_normal
    ax.xlabelsize = text_size_normal
    ax.yticklabelsize = text_size_normal
    ax.ylabelsize = text_size_normal
    save(joinpath(save_dir, "y_vs_z.png"), fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    main(params_file, job_dir)
end
