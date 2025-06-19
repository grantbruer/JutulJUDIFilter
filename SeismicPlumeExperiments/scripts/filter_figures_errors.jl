import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.instantiate()

import TOML

using Statistics
using LinearAlgebra
using JLD2
using Printf
using MyUtils
using Images
using CairoMakie
using LaTeXStrings
include("../lib/seismic_plume_params.jl")


text_size_normal = 28
text_size_smaller = 24

function ssim_error(a, b)
    return 1 - assess_ssim(a, b)
end

function l2_error(a, b)
    return sum((a .- b).^2)
end

function filter_figures_errors(params_file, job_dir)
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
    folder_path = joinpath(save_dir, "ensemble")
    mkpath(folder_path)

    # Load ground-truth states.
    gt_states, extra_data = read_ground_truth_plume_all(params, job_dir)
    gt_obs, extra_data = read_ground_truth_seismic_all(params, job_dir; state_keys=[:rtm_born, :rtm_born_noisy])
    for (state, obs) in zip(gt_states, gt_obs)
        merge!(state, obs)
    end

    function compute_errors!(gt_states, states)
        for i = 1:length(states)
            state = states[i]
            gt_step = state[:step] * params["filter"]["update_interval"]
            gt_state = gt_states[gt_step + 1]

            a = state[:Saturation]
            b = reshape(gt_state[:Saturation], size(state[:Saturation]))
            state[:Saturation_field_error] = a .- b
            state[:Saturation_ssim_error] = ssim_error(a, b)
            state[:Saturation_l2_error] = l2_error(a, b)

            if haskey(state, :rtm)
                a = state[:rtm]
                b = reshape(gt_state[:rtm_born], size(state[:rtm]))
                state[:rtm_born_field_error] = a .- b
                state[:rtm_born_ssim_error] = ssim_error(a, b)
                state[:rtm_born_l2_error] = l2_error(a, b)

                a = state[:rtm_noisy]
                b = reshape(gt_state[:rtm_born_noisy], size(state[:rtm_noisy]))
                state[:rtm_born_noisy_field_error] = a .- b
                state[:rtm_born_noisy_ssim_error] = ssim_error(a, b)
                state[:rtm_born_noisy_l2_error] = l2_error(a, b)
           end
        end
    end

    data_keys0 = [
        :Saturation,
        :Saturation_field_error,
        :Saturation_ssim_error,
        :Saturation_l2_error,
    ]
    data_keys = [
        :Saturation,
        :Saturation_field_error,
        :Saturation_ssim_error,
        :Saturation_l2_error,
        :rtm,
        :rtm_born_field_error,
        :rtm_born_ssim_error,
        :rtm_born_l2_error,
        :rtm_noisy,
        :rtm_born_noisy_field_error,
        :rtm_born_noisy_ssim_error,
        :rtm_born_noisy_l2_error,
    ]
    other_keys = [:step]
    single_step = true
    running_stats = Vector{RunningStats}()
    global jd = job_dir
    global fi = filter
    global gt = gt_states
    for i = 1:filter.params.ensemble_size
        states = load_ensemble_member_states(job_dir, filter, i)
        obs = load_ensemble_member_rtms(job_dir, filter, i)

        # There is one observation for every pre-assimilation state except for the first one.
        for (i, obs) in enumerate(obs)
            state = states[2*i]
            @assert state[:step] == obs[:step]
            merge!(states[2*i], obs)
        end
        global st = states
        global ob = obs
        compute_errors!(gt_states, states)
        for i in 1:length(states)
            if i > length(running_stats)
                dks = haskey(states[i], :rtm) ? data_keys : data_keys0
                push!(running_stats, RunningStats{Dict{Symbol, Any}}(dks, other_keys))
            end
            update_running_stats!(running_stats[i], states[i])
        end
        println("Ensemble member $(i) has $(length(states)) states")
        all_same = false # all(key -> all(state -> all(state[key] .== states[1][key]), states), data_keys)
        if all_same
            println("  but they're all identical to the first state")
            states = states[1:1]
        else
            single_step = false
        end
        file_prefix_path = joinpath(folder_path, "$(i)")
        plot_ensemble_errors(states, file_prefix_path; params, plot_png=all_same, plot_mp4=!all_same)
        if i > 5 && !single_step
            break
        end
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
    plot_ensemble_errors(means, file_prefix_path; params, plot_png=true, plot_mp4=!single_step)

    file_prefix_path = joinpath(save_dir, "std")
    plot_ensemble_errors(stds, file_prefix_path; params, plot_png=true, plot_mp4=!single_step, colormap=parula, divergent=false)

    file_prefix_path = joinpath(save_dir, "mean_ensemble")
    plot_ensemble_scalar_errors(means, gt_states, file_prefix_path; params)

    compute_errors!(gt_states, means)
    file_prefix_path = joinpath(save_dir, "mean")
    plot_ensemble_scalar_errors(means, gt_states, file_prefix_path; params)
end

function plot_ensemble_scalar_errors(states, gt_states, file_prefix_path; params)
    if length(states) == 0
        error("This is ridiculous. I can't plot plumes without any data.")
    end
    dt = params["transition"]["dt"]

    times = [state[:step] * dt / 365.2425 for state in states]

    ys = [state[:Saturation_l2_error] for state in states]
    file_path = "$(file_prefix_path)_saturation_l2_error"
    fig, ax, sc = CairoMakie.lines(times, ys)
    ax.xlabel = "Time (years)"
    ax.ylabel = "error"
    println("Saving to $(file_path)")
    jldsave("$(file_path).jld2"; times, ys)
    save("$(file_path).png", fig)

    ys = [state[:Saturation_ssim_error] for state in states]
    file_path = "$(file_prefix_path)_saturation_ssim_error"
    fig, ax, sc = CairoMakie.lines(times, ys)
    ax.xlabel = "Time (years)"
    ax.ylabel = "error"
    println("Saving to $(file_path)")
    jldsave("$(file_path).jld2"; times, ys)
    save("$(file_path).png", fig)

    rtm_states = filter(s -> haskey(s, :rtm), states)
    if length(rtm_states) == 0
        return
    end
    states = rtm_states
    times = [state[:step] * dt / 365.2425 for state in states]

    ys = [state[:rtm_born_l2_error] for state in states]
    file_path = "$(file_prefix_path)_rtm_born_l2_error"
    fig, ax, sc = CairoMakie.lines(times, ys)
    ax.xlabel = "Time (years)"
    ax.ylabel = "error"
    println("Saving to $(file_path)")
    jldsave("$(file_path).jld2"; times, ys)
    save("$(file_path).png", fig)

    ys = [state[:rtm_born_ssim_error] for state in states]
    file_path = "$(file_prefix_path)_rtm_born_ssim_error"
    fig, ax, sc = CairoMakie.lines(times, ys)
    ax.xlabel = "Time (years)"
    ax.ylabel = "error"
    println("Saving to $(file_path)")
    jldsave("$(file_path).jld2"; times, ys)
    save("$(file_path).png", fig)

    ys = [state[:rtm_born_noisy_l2_error] for state in states]
    file_path = "$(file_prefix_path)_rtm_born_noisy_l2_error"
    fig, ax, sc = CairoMakie.lines(times, ys)
    ax.xlabel = "Time (years)"
    ax.ylabel = "error"
    println("Saving to $(file_path)")
    jldsave("$(file_path).jld2"; times, ys)
    save("$(file_path).png", fig)

    ys = [state[:rtm_born_noisy_ssim_error] for state in states]
    file_path = "$(file_prefix_path)_rtm_born_noisy_ssim_error"
    fig, ax, sc = CairoMakie.lines(times, ys)
    ax.xlabel = "Time (years)"
    ax.ylabel = "error"
    println("Saving to $(file_path)")
    jldsave("$(file_path).jld2"; times, ys)
    save("$(file_path).png", fig)
end

function plot_ensemble_errors(states, file_prefix_path; params, plot_png=false, plot_mp4=true, colormap=:balance, divergent=true)
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
        return

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

    framerate = 2
    post_plot = function (fig, ax)
        ax.xlabel = "Horizontal (km)"
        ax.ylabel = "Depth (km)"
        axis_setup(ax)
    end
    extras = (; grid = mesh_3d_cutoff, params, post_plot, colormap, divergent)

    let extras = (; extras..., colorrange = (-1, 1), make_heatmap = true, make_colorbar = false)
        get_simple = state -> ifelse.(state[:Saturation_field_error] .== 0, NaN, state[:Saturation_field_error])[:, cutoff_idx:end]
        file_path = "$(file_prefix_path)_saturation_field_error.mp4"
        plot_anim(
            states, get_simple, anim_reservoir_plotter, file_path;
            plot_png, plot_mp4, extras, framerate,
        )
    end
    return
    let extras = (; extras..., make_heatmap = true, make_colorbar = true)
        rtm_states = filter(s -> haskey(s, :rtm), states)
        get_simple = state -> ifelse.(state[:rtm_born_field_error] .== 0, NaN, state[:rtm_born_field_error])[:, cutoff_idx:end]
        file_path = "$(file_prefix_path)_rtm_born_field_error.mp4"
        plot_anim(
            rtm_states, get_simple, anim_reservoir_plotter, file_path;
            plot_png, plot_mp4, extras, framerate,
        )

        get_simple = state -> ifelse.(state[:rtm_born_noisy_field_error] .== 0, NaN, state[:rtm_born_noisy_field_error])[:, cutoff_idx:end]
        file_path = "$(file_prefix_path)_rtm_born_noisy_field_error.mp4"
        plot_anim(
            rtm_states, get_simple, anim_reservoir_plotter, file_path;
            plot_png, plot_mp4, extras, framerate,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    filter_figures_errors(params_file, job_dir)
end

