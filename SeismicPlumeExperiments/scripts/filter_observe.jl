import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.instantiate()

import TOML

# JUDI must be imported first to avoid cuInit error.
using SeismicPlumeEnsembleFilter: observe_filter
# import JUDI

using EnsembleFilters: get_filter_work_dir, load_filter
using EnsembleKalmanFilters, EnsembleNormalizingFlowFilters

include("../lib/seismic_plume_params.jl")


function filter_observe(params_file, job_dir, step_index; closer, job_id, num_jobs)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    if haskey(params["filter"], "observation")
        params["observation"] = merge(params["observation"], params["filter"]["observation"])
    end

    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)
    filepath = joinpath(work_path, "filter_$(step_index)_prior")
    filter = load_filter(params, filepath)


    observer = get_observer(params)
    function observer_t(args...)
        return observer(args...)
    end

    filepath = joinpath(job_dir, work_dir, "filter_obs_$(step_index)_prior")
    obs_filter = observe_filter(observer_t, filter, "intermediate_obs_$(step_index)", filepath; params, closer, job_id, num_jobs)
end

function filter_observe(args)
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
    filter_observe(params_file, job_dir, step_index; closer, job_id, num_jobs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    filter_observe(ARGS)
end
