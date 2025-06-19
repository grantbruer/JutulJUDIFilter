
import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.resolve()
Pkg.instantiate()

import TOML
import Random

using EnsembleFilters: get_filter_work_dir, save_filter

include("../lib/seismic_plume_params.jl")


function filter_initialize(params_file, job_dir)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    params["transition"] = merge(params["transition"], get(params["filter"], "transition", Dict()))
    params["observation"] = merge(params["observation"], get(params["filter"], "observation", Dict()))

    Random.seed!(params["filter"]["initialization_seed"])

    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)
    mkpath(work_path)

    filter = initialize_filter(params)

    filepath = joinpath(work_path, "filter_0_posterior")
    save_filter(filepath, filter)
end

if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    filter_initialize(params_file, job_dir)
end
