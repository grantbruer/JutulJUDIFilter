import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.instantiate()

import TOML
import Random

using EnsembleFilters: get_filter_work_dir, load_filter
using EnsembleKalmanFilters, EnsembleNormalizingFlowFilters
using SeismicPlumeEnsembleFilter: transition_filter

include("../lib/seismic_plume_params.jl")

function filter_transition(params_file, job_dir, step_index; closer, job_id, num_jobs)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    previous_step_index = step_index - 1

    dt = params["filter"]["update_interval"] * params["transition"]["dt"]
    t0 = previous_step_index * dt
    t = step_index * dt

    # Read filter file.
    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)
    filepath = joinpath(work_path, "filter_$(previous_step_index)_posterior")
    if isfile(filepath)
        filter = load_filter(params, filepath)
    else
        filepath = joinpath(work_path, "filter_$(previous_step_index)_state_update")
        filter_update = load_filter(params, filepath)

        filepath = joinpath(work_path, "filter_$(previous_step_index)_prior")
        filter = load_filter(params, filepath)

        for (em, em_update) in zip(filter.ensemble, filter_update.ensemble)
	    state = em_update.state
	    if isa(state, Dict) && haskey(state, :Saturation)
	        if haskey(state, :Permeability)
                    set_permeability!(em.params.M, em_update.state[:Permeability])
                end
	        state = state[:Saturation]
            end
            em.state .= state
        end

        filepath = joinpath(work_path, "filter_$(previous_step_index)_posterior")
        save_filter(filepath, filter)
    end

    filepath = joinpath(job_dir, work_dir, "filter_$(step_index)_prior")
    transition_filter(filter, t0, t, "intermediate_trans_$(previous_step_index)_to_$(step_index)", filepath; params, closer, job_id, num_jobs)
end

function filter_transition(args)
    params_file = args[1]
    job_dir = args[2]
    step_index = parse(Int64, args[3])
    name = args[4]

    helper_pattern = r"^helper-(\d+)-(\d+)$"
    match_result = match(helper_pattern, name)
    if name == "closer"
        closer = true
	job_id = 1
	num_jobs = 1
    elseif ! isnothing(match_result)
        closer = false
	job_id = parse(Int, match_result.captures[1])
	num_jobs = parse(Int, match_result.captures[2])
    else
        error("Name should be 'closer' or 'helper-X-Y'. Got '$(name)'")
    end
    filter_transition(params_file, job_dir, step_index; closer, job_id, num_jobs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    filter_transition(ARGS)
end
