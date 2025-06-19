import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.instantiate()

import TOML
import Jutul

using JLD2

include("../lib/seismic_plume_params.jl")

function generate_ground_truth_plume(params_file, job_dir)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    params["transition"] = merge(params["transition"], params["ground_truth"]["transition"])
    params["observation"] = merge(params["observation"], params["ground_truth"]["observation"])

    K, phi = get_permeability_porosity(params)

    mkpath(job_dir)
    file_stem = get_ground_truth_plume_stem(params)
    path_stem = joinpath(job_dir, file_stem)

    generate_plume_data(K, phi; path_stem, params)
end

function generate_plume_data(K, phi; path_stem, params)
    nbatches = params["transition"]["nbatches"]

    M, Msetup = initialize_plume_model(K, phi, params)

    extra_data = (
        idx_wb = M.idx_wb,
        inj_zidx = M.inj_idx[3],
        format = "v1.1",
    )

    j = 0
    state0 = Jutul.get_output_state(Msetup.sim)
    filepath = "$(path_stem)_time$(j).jld2"
    jldsave(filepath; state0=state0, extra_data=extra_data)

    for j = 1:nbatches
        @time result = M(Msetup)
        filepath = "$(path_stem)_time$(j).jld2"
        jldsave(filepath; result=result)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    generate_ground_truth_plume(params_file, job_dir)
end
