using EnsembleFilters

using Lorenz63

struct Lorenz63Model
    kwargs
end

function Lorenz63Model(; params)
    kwargs = (;
        σ = Float64(params["transition"]["sigma"]),
        ρ = Float64(params["transition"]["rho"]),
        β = Float64(params["transition"]["beta"]),
        s = Float64(params["transition"]["scaling"]),
        Δt = Float64(params["transition"]["ministep_dt"]),
        N = params["transition"]["ministep_nt"],
    )
    return Lorenz63Model(kwargs)
end

function (M::Lorenz63Model)(state; kwargs...)
    states = L63(; M.kwargs..., kwargs..., xyz=state)
    return Dict(:state=>states[:, end])
end

struct Lorenz63Observer
    noise_level
end

function Lorenz63Observer(; params)
    noise_level = params["observation"]["noise_level"]
    return Lorenz63Observer(noise_level)
end

function (M::Lorenz63Observer)(state)
    noise = M.noise_level .* randn(size(state))
    return state, state .+ noise
end

function generate_ensemble(params::Dict, ::Val{T}) where T <: AbstractEnsembleMember
    transitioner = Lorenz63Model(; params)

    ensemble_size = params["filter"]["ensemble_size"]
    initial_noise_level = params["filter"]["initial_noise_level"]
    Δt = params["filter"]["initial_ministep_dt"]
    N = params["filter"]["initial_ministep_nt"]

    ensemble = Vector{T}(undef, ensemble_size)

    for i = 1:ensemble_size
        # Simulate for some steps.
        state = initial_noise_level * randn(3)
        state = transitioner(state; Δt, N)
        ensemble[i] = T(state, nothing)
    end
    return ensemble
end

function initialize_filter(params)
    algorithm = params["filter"]["algorithm"]
    if algorithm == "enkf"
        return EnsembleKalmanFilter(params, generate_ensemble)
    elseif algorithm == "no_observations"
        return EnsembleNoObsFilter(params, generate_ensemble)
    elseif algorithm == "just_observations"
        return EnsembleJustObsFilter(params, generate_ensemble)
    elseif algorithm == "ekf"
        return initialize_kf(params, generate_ensemble)
    elseif algorithm == "nf"
        return EnsembleNormFlowFilter(params, generate_ensemble)
    end
    error("Unsupported algorithm: $(algorithm)")
end


function EnsembleFilters.get_filter_work_dir(params::Dict)
    algorithm = params["filter"]["algorithm"]
    if algorithm == "enkf"
        return get_filter_work_dir(params, Val(EnsembleKalmanFilter))
    elseif algorithm == "no_observations"
        return get_filter_work_dir(params, Val(EnsembleNoObsFilter))
    elseif algorithm == "just_observations"
        return get_filter_work_dir(params, Val(EnsembleJustObsFilter))
    elseif algorithm == "ekf"
        return get_filter_work_dir(params, Val(ExtendedKalmanFilter))
    elseif algorithm == "nf"
        return get_filter_work_dir(params, Val(EnsembleNormFlowFilter))
    end
    error("Unsupported algorithm: $(algorithm)")
end
