import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.resolve()
Pkg.instantiate()

import TOML
import Random

using EnsembleFilters: get_filter_work_dir, load_filter
using EnsembleKalmanFilters, EnsembleNormalizingFlowFilters
using SeismicPlumeEnsembleFilter: transition_filter

include("../lib/seismic_plume_params.jl")

function filter_process_assimilation(params_file, job_dir, step_index)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    dt = params["filter"]["update_interval"] * params["transition"]["dt"]
    t0 = step_index * dt

    # Read filter file.
    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)
    filepath = joinpath(work_path, "filter_$(step_index)_posterior")
    if ! isfile(filepath)
        filepath = joinpath(work_path, "filter_$(step_index)_state_update")
        filter_update = load_filter(params, filepath)

        filepath = joinpath(work_path, "filter_$(step_index)_prior")
        filter = load_filter(params, filepath)

        for (em, em_update) in zip(filter.ensemble, filter_update.ensemble)
            state = em_update.state
            mask = (
		(!hasfield(typeof(em.params), :update_mask) || isnothing(em.params.update_mask))
                ? trues(size(state))
                : em.params.update_mask
            )
            if isa(state, Dict) && haskey(state, :Saturation)
                if haskey(state, :Permeability)
                    em_update.state[:Permeability][mask] = get_permeability(em.params.M)[mask]
                    set_permeability!(em.params.M, em_update.state[:Permeability])
                end
                state = state[:Saturation]
            end
            em.state[mask] .= state[mask]
        end

        filepath = joinpath(work_path, "filter_$(step_index)_posterior")
        save_filter(filepath, filter)
    end
end

function filter_process_assimilation(args)
    params_file = args[1]
    job_dir = args[2]
    step_index = parse(Int64, args[3])
    filter_process_assimilation(params_file, job_dir, step_index)
end

if abspath(PROGRAM_FILE) == @__FILE__
    filter_process_assimilation(ARGS)
end
