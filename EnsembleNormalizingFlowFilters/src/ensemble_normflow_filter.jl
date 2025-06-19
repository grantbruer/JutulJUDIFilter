using EnsembleFilters

struct EnsembleNormFlowMember <: AbstractEnsembleMember
    state
    params
    extra
end

struct EnsembleNormFlowParams <: AbstractEnsembleFilterParams
    ensemble_size
    observation_noise_stddev # TODO: figure out a better place to store this.
end

struct EnsembleNormFlowFilter <: AbstractEnsembleFilter
    ensemble::Vector{EnsembleNormFlowMember}
    params::EnsembleNormFlowParams
    work_dir
end

EnsembleNormFlowMember(state, params) = EnsembleNormFlowMember(state, params, nothing)
EnsembleFilters.make_ensemble_member(data, ::Val{EnsembleNormFlowFilter}) = EnsembleNormFlowMember(data, nothing)
EnsembleFilters.get_state(x::EnsembleNormFlowMember) = x.state
EnsembleFilters.get_params(x::EnsembleNormFlowMember) = x.params
EnsembleFilters.get_extra(x::EnsembleNormFlowMember) = x.extra

EnsembleFilters.get_ensemble_size(params::EnsembleNormFlowParams) = params.ensemble_size

EnsembleFilters.get_ensemble(filter::EnsembleNormFlowFilter) = filter.ensemble
EnsembleFilters.get_ensemble_params(filter::EnsembleNormFlowFilter) = filter.params
EnsembleFilters.get_ensemble_work_dir(filter::EnsembleNormFlowFilter) = filter.work_dir

EnsembleFilters.get_filter_work_dir(params::Dict, ::Val{EnsembleNormFlowFilter}) = "nf_$(params["filter"]["unique_name"])"
EnsembleFilters.get_filter_work_dir(filter::EnsembleNormFlowFilter) = filter.work_dir

function EnsembleNormFlowFilter(params::Dict, generate_ensemble)
    ensemble_size = params["filter"]["ensemble_size"]
    observation_noise_stddev = params["filter"]["observation_noise_stddev"]

    ensemble = generate_ensemble(params, Val(EnsembleNormFlowMember))
    ensemble_params = EnsembleNormFlowParams(ensemble_size, observation_noise_stddev)
    work_dir = EnsembleFilters.get_filter_work_dir(params, Val(EnsembleNormFlowFilter))

    filter = EnsembleNormFlowFilter(ensemble, ensemble_params, work_dir)
    return filter
end

