import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.resolve()
Pkg.instantiate()

# JUDI must be imported first to avoid cuInit error.
using SeismicPlumeEnsembleFilter
# import JUDI

import TOML
import Random
import CUDA

using JLD2

include("../lib/seismic_plume_params.jl")

function generate_ground_truth_seismic(params_file, job_dir)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    params["transition"] = merge(params["transition"], params["ground_truth"]["transition"])
    params["observation"] = merge(params["observation"], params["ground_truth"]["observation"])

    K, phi = get_permeability_porosity(params)
    vel, rho = get_velocity_density(params)

    mkpath(job_dir)
    file_stem = get_ground_truth_plume_stem(params)
    input_path_stem = joinpath(job_dir, file_stem)

    file_stem = get_ground_truth_seismic_stem(params)
    output_path_stem = joinpath(job_dir, file_stem)

    Random.seed!(params["ground_truth"]["observation"]["noise_seed"])
    generate_seismic_data(K, phi, vel, rho; input_path_stem, output_path_stem, params)
end

function generate_seismic_data(K, phi, vel, rho; input_path_stem, output_path_stem, params)
    n = Tuple(params["observation"]["n"])
    FT = Float32
    phi = FT.(phi)
    vel = FT.(vel)
    rho = FT.(rho)

    nbatches = Int(params["transition"]["nbatches"])

    idx_wb = maximum(find_water_bottom_immutable(log.(K) .- log(K[1,1])))

    v0, rho0 = get_background_velocity_density(vel, rho, idx_wb; params)

    observation_type = :born_shot_rtm_depth_noise
    M = SeismicModel_params(phi, vel, rho; params, observation_type)
    M0 = SeismicModel(M, phi, v0, rho0)
    P = PatchyModel(vel, rho, phi; params)

    S = SeismicCO2Observer(M0, P)

    extra_data = (
        idx_wb = idx_wb,
    )

    function compute_seismic!(state, vel, rho, k)
        # This is a fully linearized version.
        dshot, rtm, dshot_noisy, rtm_noisy = M0(vel, rho)
        state[Symbol("dshot_born_$(k)")] = dshot
        state[Symbol("rtm_born_$(k)")] = rtm
        state[Symbol("dshot_born_noisy_$(k)")] = dshot_noisy
        state[Symbol("rtm_born_noisy_$(k)")] = rtm_noisy
        return state
    end

    state = Dict{Symbol, Any}()

    @time obs_baseline = M0(v0, rho0, Val(:shot))
    state[:shot_baseline] = obs_baseline

    batch_idx = 0
    println("Batch $(batch_idx)")
    if CUDA.functional()
        @show CUDA.memory_status()
    end
    println()

    file_path = "$(output_path_stem)_time$(batch_idx).jld2"
    state[Symbol("vel_$(0)")] = vel
    state[Symbol("rho_$(0)")] = rho
    compute_seismic!(state, vel, rho, 0)
    jldsave(file_path; state..., extra_data=extra_data)


    for batch_idx = 1:nbatches
        println("Batch $(batch_idx)")
        if CUDA.functional()
            @show CUDA.memory_status()
        end
        println()

        # Read saturation data.
        file_path = "$(input_path_stem)_time$(batch_idx).jld2"

        local sim_result
        try
            sim_result = load(file_path, "result")
        catch e
            println("Error loading $(file_path), so I'll give up on this sample.")
            break
        end

        # Compute seismic observations for each time step.
        state = Dict{Symbol, Any}()
        for k = 1:length(sim_result.time)
            println("Running simulation batch $(batch_idx), step $(k)")
            plume = sim_result.states[k]
            sat = reshape(plume[:Saturations][1, :], n)

            v_t, rho_t = P(sat)
            state[Symbol("vel_$(k)")] = v_t
            state[Symbol("rho_$(k)")] = rho_t

            compute_seismic!(state, v_t, rho_t, k)
        end

        file_path = "$(output_path_stem)_time$(batch_idx).jld2"
        jldsave(file_path; state...)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    generate_ground_truth_seismic(params_file, job_dir)
end
