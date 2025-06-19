
import Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

import TOML
import Random

using JLD2
using EnsembleFilters: assimilate_data, get_filter_work_dir

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

    params["transition"] = merge(params["transition"], get(params["filter"], "transition", Dict()))
    params["observation"] = merge(params["observation"], get(params["filter"], "observation", Dict()))

    mkpath(job_dir)
    ground_truth_dir = joinpath(job_dir, "ground_truth")
    mkpath(ground_truth_dir)

    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)
    mkpath(work_path)

    # Generate ground truth
    Random.seed!(params["ground_truth"]["seed"])
    transitioner = Lorenz63Model(; params)
    initial_state = [1., 1., 1.]

    num_burnin_steps = params["transition"]["burnin_steps"]
    state = initial_state
    for i = 1:num_burnin_steps
        state = transitioner(state)
    end

    ground_truth_states = [state]
    num_simulation_steps = params["transition"]["nt"]
    for i = 1:num_simulation_steps
        push!(ground_truth_states, transitioner(ground_truth_states[end]))
    end

    observer = Lorenz63Observer(; params)
    ground_truth_observations = [observer(state) for state in ground_truth_states]

    file_path = joinpath(ground_truth_dir, "ground_truth.jld2")
    jldsave(file_path; states = ground_truth_states, observations = ground_truth_observations)

    # Run filter
    Random.seed!(params["filter"]["initialization_seed"])
    filter = initialize_filter(params)

    transitioner = Lorenz63Model(; params)
    observer = Lorenz63Observer(; params)

    num_burnin_steps = params["transition"]["burnin_steps"]
    for k = 1:num_burnin_steps
        for em in filter.ensemble
            em.state .= transitioner(em.state)
        end
    end

    file_path = joinpath(work_path, "$(0)-pred.jld2")
    jldsave(file_path; filter)

    update_interval = params["filter"]["update_interval"]
    for k = 1:num_simulation_steps
        for em in filter.ensemble
            em.state .= transitioner(em.state)
        end
        file_path = joinpath(work_path, "$(k)-pred.jld2")
        jldsave(file_path; filter)
        if k % update_interval == 0
            obs_ensemble = [eltype(filter.ensemble)(observer(em.state), nothing, nothing) for em in filter.ensemble]
            obs_filter = typeof(filter)(obs_ensemble, filter.params, nothing)
            y_obs = ground_truth_observations[k][2] # noisy data

            filter = assimilate_data(filter, obs_filter, y_obs, job_dir, k; params, save_update=false);
            file_path = joinpath(work_path, "$(k)-post.jld2")
            jldsave(file_path; filter)
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    main(params_file, job_dir)
end
