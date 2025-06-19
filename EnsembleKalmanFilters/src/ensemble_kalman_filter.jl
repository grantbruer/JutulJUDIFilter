import EnsembleFilters
using EnsembleFilters: AbstractEnsembleMember, AbstractEnsembleFilterParams, AbstractEnsembleFilter
using EnsembleFilters: get_ensemble, get_ensemble_params, get_ensemble_work_dir, get_ensemble_size, make_ensemble_member
using JLD2: ReconstructedMutable

struct EnsembleKalmanMember <: AbstractEnsembleMember
    state
    params
    extra
end

struct EnsembleKalmanParams <: AbstractEnsembleFilterParams
    ensemble_size
    observation_noise_stddev # TODO: figure out a better place to store this.
end

struct EnsembleKalmanFilter <: AbstractEnsembleFilter
    ensemble::Vector{EnsembleKalmanMember}
    params::EnsembleKalmanParams
    work_dir
end

let oldType = ReconstructedMutable{:EnsembleKalmanMember, (:state, :params), Tuple{Any, Any}}
    EnsembleFilters.make_ensemble_member(data::oldType, ::Val{EnsembleKalmanFilter}) = EnsembleKalmanMember(getfield(data, 1)[1], getfield(data, 1)[2])
end

EnsembleKalmanMember(state, params) = EnsembleKalmanMember(state, params, nothing)
EnsembleFilters.make_ensemble_member(data, ::Val{EnsembleKalmanFilter}) = EnsembleKalmanMember(data, nothing)
EnsembleFilters.get_state(x::EnsembleKalmanMember) = x.state
EnsembleFilters.get_params(x::EnsembleKalmanMember) = x.params
EnsembleFilters.get_extra(x::EnsembleKalmanMember) = x.extra

EnsembleFilters.get_ensemble_size(params::EnsembleKalmanParams) = params.ensemble_size

EnsembleFilters.get_ensemble(filter::EnsembleKalmanFilter) = filter.ensemble
EnsembleFilters.get_ensemble_params(filter::EnsembleKalmanFilter) = filter.params
EnsembleFilters.get_ensemble_work_dir(filter::EnsembleKalmanFilter) = filter.work_dir

EnsembleFilters.get_filter_work_dir(params::Dict, ::Val{EnsembleKalmanFilter}) = "enkf_$(params["filter"]["unique_name"])"
EnsembleFilters.get_filter_work_dir(filter::EnsembleKalmanFilter) = filter.work_dir

function EnsembleKalmanFilter(params::Dict, generate_ensemble)
    ensemble_size = params["filter"]["ensemble_size"]
    observation_noise_stddev = params["filter"]["observation_noise_stddev"]

    ensemble = generate_ensemble(params, Val(EnsembleKalmanMember))
    ensemble_params = EnsembleKalmanParams(ensemble_size, observation_noise_stddev)
    work_dir = EnsembleFilters.get_filter_work_dir(params, Val(EnsembleKalmanFilter))

    filter = EnsembleKalmanFilter(ensemble, ensemble_params, work_dir)
    return filter
end

