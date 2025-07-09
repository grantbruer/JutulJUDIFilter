
import TOML
import Pkg
if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    params = TOML.parsefile(params_file);
    algorithm = params["filter"]["algorithm"]
    if algorithm == "enkf"
        project = "envs/EnKF"
    elseif algorithm == "no_observations"
        project = "envs/NoObservations"
    elseif algorithm == "just_observations"
        project = "envs/JustObservations"
    elseif algorithm == "nf"
        project = "envs/NormalizingFlow"
    else
        error("No environment set up for this algorithm yet: '$(algorithm)'")
    end
    Pkg.activate(project)
    Pkg.resolve()
end

if Base.active_project() == Base.load_path_expand("@v#.#")
    error("Please activate the project for this filter.")
end

Pkg.instantiate()

using JLD2: load
using CairoMakie
using MyUtils

using EnsembleFilters: EnsembleFilters, get_filter_work_dir, load_filter, save_filter

name = splitpath(Base.active_project())[end-1]
if name == "EnKF"
    import KernelMatrices
    using EnsembleKalmanFilters
    using EnsembleKalmanFilters: assimilate_data
elseif name == "NormalizingFlow"
    import InvertibleNetworks
    import UNet
    import MLUtils
    import Flux
    using EnsembleNormalizingFlowFilters
    using EnsembleNormalizingFlowFilters: assimilate_data
elseif name == "JustObservations"
    import SeismicPlumeEnsembleFilter
    import SlimOptim
    using EnsembleJustObsFilters
    using EnsembleJustObsFilters: assimilate_data
elseif name == "NoObservations"
    using EnsembleFilters
    using EnsembleFilters: assimilate_data
end

include("../lib/params.jl")

function filter_assimilate(params_file, job_dir, step_index)
    # Load parameters.
    params = TOML.parsefile(params_file);

    println("======================== params ========================")
    TOML.print(params);
    println("========================================================")

    # Read filter files.
    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)

    filepath = joinpath(work_path, "filter_$(step_index)_prior")
    filter = load_filter(params, filepath);

    filepath = joinpath(work_path, "filter_obs_$(step_index)_prior")
    obs_filter = load_filter(params, filepath);


    # Read observation data.
    y_obs, obs_filter = get_observations(job_dir, obs_filter, step_index; params);

    # Run filter.
    assimilate_data(filter, obs_filter, y_obs, job_dir, step_index; params);
end

function get_observations(job_dir, obs_filter, step_index; params)
    observation_type = params["filter"]["observation_type"]
    stem = get_ground_truth_seismic_stem(params)
    stem = joinpath(job_dir, stem)
    gt_step_idx = step_index * params["filter"]["update_interval"]
    work_dir = get_filter_work_dir(params)

    if observation_type == "born_rtm_depth_noise"
        # Load baseline RTM.
        baseline, extra_data = read_ground_truth_seismic_baseline(stem; state_keys = [:rtm_born, :rtm_born_noisy])

        ground_truth_observation_noisy = params["filter"]["ground_truth_observation_noisy"]
        baseline_noisy = baseline[:rtm_born_noisy]
        baseline_clean = baseline[:rtm_born]
        if ground_truth_observation_noisy == 1
            baseline = baseline_noisy
        elseif ground_truth_observation_noisy == 0
            baseline = baseline_clean
        else
            baseline = baseline_clean + (baseline_noisy - baseline_clean) * ground_truth_observation_noisy
        end

        # Get the ground truth data and ensemble observations into the same space.
        for (i, em) in enumerate(obs_filter.ensemble)
            em.state[1] .-= baseline
            em.state[2] .-= baseline
        end
    elseif observation_type == "none"
    else
        error("Invalid observation_type: '$(observation_type)'")
    end

    ground_truth_observation_type = params["filter"]["ground_truth_observation_type"]
    if ground_truth_observation_type == "same"
        ground_truth_observation_type = observation_type
    end
    if ground_truth_observation_type == "born_rtm_depth_noise"
        obs_data = read_ground_truth_seismic(stem, params, gt_step_idx; state_keys = [:rtm_born, :rtm_born_noisy])
        obs_data[:delta_rtm_born_noisy] = obs_data[:rtm_born_noisy] - baseline
        obs_data[:delta_rtm_born] = obs_data[:rtm_born] - baseline

        ground_truth_observation_noisy = params["filter"]["ground_truth_observation_noisy"]
        y_obs_noisy = obs_data[:delta_rtm_born_noisy]
        y_obs_clean = obs_data[:delta_rtm_born]
        if ground_truth_observation_noisy == 1
            y_obs = y_obs_noisy
        elseif ground_truth_observation_noisy == 0
            y_obs = y_obs_clean
        else
            y_obs = y_obs_clean + (y_obs_noisy - y_obs_clean) * ground_truth_observation_noisy
        end

        if params["filter"]["make_assimilation_figures"]
            # Plot the situation before assimilating.
            save_dir = joinpath(job_dir, "figs", work_dir, "assimilation")
            mkpath(save_dir)

            label_axes = function(ax)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
            end

            n = Tuple(params["observation"]["n"])
            d = Tuple(params["observation"]["d"])

            # Use mesh in kilometers instead of meters.
            grid = CartesianMesh(n, d .* n ./ 1000.0)

            # Plot RTM.
            colormap = :balance
            fig, ax = plot_heatmap_from_grid(obs_data[:rtm_born], grid; colormap, make_divergent=true)
            label_axes(ax)
            file_path = joinpath(save_dir, "ground_truth_rtm_$(step_index).png")
            save(file_path, fig)

            fig, ax = plot_heatmap_from_grid(obs_data[:rtm_born_noisy], grid; colormap, make_divergent=true)
            label_axes(ax)
            file_path = joinpath(save_dir, "ground_truth_rtm_noisy_$(step_index).png")
            save(file_path, fig)

            fig, ax = plot_heatmap_from_grid(obs_data[:delta_rtm_born], grid; colormap, make_divergent=true)
            label_axes(ax)
            file_path = joinpath(save_dir, "ground_truth_delta_rtm_$(step_index).png")
            save(file_path, fig)

            fig, ax = plot_heatmap_from_grid(obs_data[:delta_rtm_born_noisy], grid; colormap, make_divergent=true)
            label_axes(ax)
            file_path = joinpath(save_dir, "ground_truth_delta_rtm_noisy_$(step_index).png")
            save(file_path, fig)

            fig, ax = plot_heatmap_from_grid(y_obs, grid; colormap, make_divergent=true)
            label_axes(ax)
            file_path = joinpath(save_dir, "ground_truth_y_obs_$(step_index).png")
            save(file_path, fig)
        end
    elseif ground_truth_observation_type == "shot"
        baseline, extra_data = read_ground_truth_seismic_baseline(stem; state_keys = [:dshot_born_noisy, :dshot_born, :rtm_born, :rtm_born_noisy])
        obs_data = read_ground_truth_seismic(stem, params, gt_step_idx; state_keys = [:dshot_born_noisy, :dshot_born, :rtm_born, :rtm_born_noisy])
        obs_data[:delta_shot_born_noisy] = obs_data[:dshot_born_noisy] - baseline[:dshot_born_noisy]
        obs_data[:delta_shot_born] = obs_data[:dshot_born] - baseline[:dshot_born]
        obs_data[:delta_rtm] = obs_data[:rtm_born] - baseline[:rtm_born]
        obs_data[:delta_rtm_noisy] = obs_data[:rtm_born_noisy] - baseline[:rtm_born_noisy]

        ground_truth_observation_noisy = params["filter"]["ground_truth_observation_noisy"]
        y_obs_noisy = obs_data[:delta_shot_born_noisy]
        y_obs_clean = obs_data[:delta_shot_born]
        if ground_truth_observation_noisy == 1
            y_obs = y_obs_noisy
        elseif ground_truth_observation_noisy == 0
            y_obs = y_obs_clean
        else
            y_obs = y_obs_clean + (y_obs_noisy - y_obs_clean) * ground_truth_observation_noisy
        end
        obs_data[:y_obs] = y_obs

        if params["filter"]["make_assimilation_figures"]
            save_dir = joinpath(job_dir, "figs", work_dir, "assimilation")
            mkpath(save_dir)

            label_axes = function(ax)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
            end

            n = Tuple(params["observation"]["n"])
            d = Tuple(params["observation"]["d"])

            # Use mesh in kilometers instead of meters.
            grid = CartesianMesh(n, d .* n ./ 1000.0)

            # Plot RTM.
            colormap = :balance
            fig, ax = plot_heatmap_from_grid(obs_data[:rtm_born], grid; colormap, make_divergent=true)
            label_axes(ax)
            file_path = joinpath(save_dir, "ground_truth_rtm_$(step_index).png")
            save(file_path, fig)

            fig, ax = plot_heatmap_from_grid(obs_data[:rtm_born_noisy], grid; colormap, make_divergent=true)
            label_axes(ax)
            file_path = joinpath(save_dir, "ground_truth_rtm_noisy_$(step_index).png")
            save(file_path, fig)

            fig, ax = plot_heatmap_from_grid(obs_data[:delta_rtm], grid; colormap, make_divergent=true)
            label_axes(ax)
            file_path = joinpath(save_dir, "ground_truth_delta_rtm_$(step_index).png")
            save(file_path, fig)

            fig, ax = plot_heatmap_from_grid(obs_data[:delta_rtm_noisy], grid; colormap, make_divergent=true)
            label_axes(ax)
            file_path = joinpath(save_dir, "ground_truth_delta_rtm_noisy_$(step_index).png")
            save(file_path, fig)

            # Plot shot record.
            extras = (; params, extra_data..., grid, colormap, divergent=true)
            file_path = joinpath(save_dir, "ground_truth_delta_dshot_$(step_index).mp4")
            plot_anim([obs_data], s->s[:delta_shot_born].data, anim_shot_record_plotter, file_path; extras, framerate=2)

            file_path = joinpath(save_dir, "ground_truth_delta_dshot_noisy_$(step_index).mp4")
            plot_anim([obs_data], s->s[:delta_shot_born_noisy].data, anim_shot_record_plotter, file_path; extras, framerate=2)

            file_path = joinpath(save_dir, "ground_truth_dshot_$(step_index).mp4")
            plot_anim([obs_data], s->s[:dshot_born].data, anim_shot_record_plotter, file_path; extras, framerate=2)

            file_path = joinpath(save_dir, "ground_truth_dshot_noisy_$(step_index).mp4")
            plot_anim([obs_data], s->s[:dshot_born_noisy].data, anim_shot_record_plotter, file_path; extras, framerate=2)

            file_path = joinpath(save_dir, "ground_truth_y_obs_$(step_index).mp4")
            plot_anim([obs_data], s->s[:y_obs].data, anim_shot_record_plotter, file_path; extras, framerate=2)
        end
    elseif ground_truth_observation_type == "none"
        y_obs = nothing
    else
        error("Invalid ground_truth_observation_type: '$(ground_truth_observation_type)'")
    end
    return y_obs, obs_filter
end

function filter_assimilate(args)
    params_file = args[1]
    job_dir = args[2]
    step_index = parse(Int64, args[3])
    filter_assimilate(params_file, job_dir, step_index)
end

if abspath(PROGRAM_FILE) == @__FILE__
    filter_assimilate(ARGS)
end
