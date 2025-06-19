import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.resolve()
Pkg.instantiate()

# JUDI must be imported first to avoid cuInit error.
using SeismicPlumeEnsembleFilter
using SeismicPlumeEnsembleFilter.JUDI
using LinearAlgebra
using Statistics

import TOML
import Random
import CUDA

using JLD2

include("../lib/seismic_plume_params.jl")

compute_covariance_row(M::SeismicModel, idx::Int) = compute_covariance_row(M, idx, M.observation_type)
function compute_covariance_row(M::SeismicModel, idx::Int, ::Val{:born_rtm_depth_noise})
    # For this observer, the noise comes from
    #    n1 = generate_noise(M, dshot, M.params.snr) .* norm(dshot)
    #    n2 = M(n1, Val(:rtm))
    # So if I have noise n from a multivariate unit normal,
    #   and I know n1 = A*n and n2 = B*n1, then the covariance is
    #   cov(n1) = A*A' and cov(n2) = B*A*A'*B'.
    # I, in fact, know A is approximately Q'DQ, where Q is the Fourier
    #   transform and D is a diagonal matrix of the spectrum of the
    #   source wavelet, so A*A' = Q'DDQ
    # I also know B = Mr1*Mr1 * J', so the full covariance is
    #   C = Mr1*Mr1 * J' * Q'DDQ * J * Mr1*Mr1.

    source = M.q.data[1]
    source_fft = fft(source)
    source_fft2 = source_fft .^ 2
    # noise_norm_samples = [norm(randn(size(source)) .* source_fft) for i = 1:30]
    # noise_norm_mean = mean(noise_norm_samples)
    # s = std(noise_norm_samples)
    # println("Mean noise norm: $noise_norm_mean")
    # println("Mean noise std : $s")
    noise_norm_mean = 38f0

    n = size(M.model)
    v = zeros(Float32, prod(n))
    v[idx] = 1
    v_shot = M.J * (M.Mr1 * M.Mr1 * v)

    # Let's assume that the division by norm(noise) in the generate_noise
    # function is approximately a constant.
    for v_shot_i in v_shot.data
        v_shot_i .= real.(ifft(fft(v_shot_i) .* source_fft2))
    end
    v_shot = v_shot / (noise_norm_mean.^2) * 10f0^(-M.params.snr/10f0)

    v_rtm = M.J' * v_shot
    v = M.Mr1 * M.Mr1 * vec(v_rtm.data)

    # Let's fix units now.
    
    # J has JUDI units of pressure per impedance.
    conversion = (JUDI_to_SI.pressure / JUDI_to_SI.specific_acoustic_impedance) .^ 2

    # Mr1^2 has JUDI units of length.
    conversion *= JUDI_to_SI.distance .^ 2

    return v .* conversion
end


function generate_ground_truth_seismic_covariance(params_file, job_dir)
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

    file_stem = get_ground_truth_seismic_stem(params)
    output_path_stem = joinpath(job_dir, file_stem)

    generate_seismic_covariance_data(K, phi, vel, rho; output_path_stem, params)
end

function generate_seismic_covariance_data(K, phi, vel, rho; output_path_stem, params)
    N = Tuple(params["observation"]["n"])
    FT = Float32
    phi = FT.(phi)
    vel = FT.(vel)
    rho = FT.(rho)

    nbatches = Int(params["transition"]["nbatches"])

    idx_wb = maximum(find_water_bottom_immutable(log.(K) .- log(K[1,1])))

    v0, rho0 = get_background_velocity_density(vel, rho, idx_wb; params)

    observation_type = :born_rtm_depth_noise
    M = SeismicModel_params(phi, vel, rho; params, observation_type)
    M0 = SeismicModel(M, phi, v0, rho0)

    file_path = "$(output_path_stem)_covariance.jld2"

    coarse_C = nothing
    target_idxs = nothing
    try
        coarse_C, target_idxs = load(file_path, "coarse_C", "target_idxs")
    catch e
        if isfile(file_path)
            rethrow()
        end
        println("Checkpoint file not found. Starting from scratch.")

        coarse_N = (16, 16)
        target_idxs = []
        for coarse_i = 1:coarse_N[1]
            for coarse_j = 1:coarse_N[2]
                # The coarse grid is interior to the fine grid.
                fine_i = round(Int, coarse_i * N[1] / (coarse_N[1] + 1))
                fine_j = round(Int, coarse_j * N[2] / (coarse_N[2] + 1))
                fine_idx = (fine_i - 1) + (fine_j - 1) * N[1] + 1
                push!(target_idxs, fine_idx)
            end
        end
        coarse_C = fill(NaN, (prod(coarse_N), prod(size(K))))
    end
    for (i, t_idx) in enumerate(target_idxs)
        if !isnan(coarse_C[i, 1])
            continue
        end
        println("Computing coarse row $i/$(size(coarse_C,1)), fine row $t_idx/$(size(coarse_C,2))")
        row = compute_covariance_row(M0, t_idx)
        coarse_C[i, :] .= row
        jldsave(file_path; coarse_C, target_idxs)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    generate_ground_truth_seismic_covariance(params_file, job_dir)
end
