import Pkg
# Pkg.activate("envs/SeismicPlume")
# Pkg.instantiate()

import TOML

using Statistics
using LinearAlgebra
using JLD2
using Printf
using MyUtils
using CairoMakie
include("../lib/seismic_plume_params.jl")

function filter_extract_observation_noise(params_file, job_dir)
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

    running_stats = Vector{RunningStats}()
    rtm_noise_all = fill(NaN, filter.params.ensemble_size, 5, size(baseline[:rtm_born])...)
    for i = 1:filter.params.ensemble_size
        states = load_ensemble_member_rtms(job_dir, filter, i)
        for j in 1:length(states)
            rtm_noise = states[j][:rtm_noisy] - states[j][:rtm]
            selectdim(selectdim(rtm_noise_all, 1, i), 1, j) .= rtm_noise
        end
        println("Ensemble member $(i) has $(length(states)) states")
        if length(states) == 0
            continue
        end
    end

    filepath = joinpath(work_path, "observation_noise.jld2")
    jldsave(filepath; data=rtm_noise_all)
end

if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    filter_extract_observation_noise(params_file, job_dir)
end

