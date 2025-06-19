using EnsembleFilters
using JLD2: ReconstructedMutable

struct EnsembleJustObsMember <: AbstractEnsembleMember
    state
    params
    extra
end

struct EnsembleJustObsParams <: AbstractEnsembleFilterParams
    ensemble_size
end

struct EnsembleJustObsFilter <: AbstractEnsembleFilter
    ensemble::Vector{EnsembleJustObsMember}
    params::EnsembleJustObsParams
    work_dir
end

let oldType = ReconstructedMutable{:EnsembleJustObsMember, (:state, :params), Tuple{Any, Any}}
    global EnsembleFilters.make_ensemble_member(data::oldType, ::Val{EnsembleJustObsFilter}) = EnsembleJustObsMember(getfield(data, 1)[1], getfield(data, 1)[2])
end

EnsembleJustObsMember(state, params) = EnsembleJustObsMember(state, params, nothing)
EnsembleFilters.make_ensemble_member(data, ::Val{EnsembleJustObsFilter}) = EnsembleJustObsMember(data, nothing)
EnsembleFilters.get_state(x::EnsembleJustObsMember) = x.state
EnsembleFilters.get_params(x::EnsembleJustObsMember) = x.params
EnsembleFilters.get_extra(x::EnsembleJustObsMember) = x.extra

EnsembleFilters.get_ensemble_size(params::EnsembleJustObsParams) = params.ensemble_size

EnsembleFilters.get_ensemble(filter::EnsembleJustObsFilter) = filter.ensemble
EnsembleFilters.get_ensemble_params(filter::EnsembleJustObsFilter) = filter.params
EnsembleFilters.get_ensemble_work_dir(filter::EnsembleJustObsFilter) = filter.work_dir

EnsembleFilters.get_filter_work_dir(params::Dict, ::Val{EnsembleJustObsFilter}) = "justobs_$(params["filter"]["unique_name"])"
EnsembleFilters.get_filter_work_dir(filter::EnsembleJustObsFilter) = filter.work_dir

function EnsembleJustObsFilter(params::Dict, generate_ensemble)
    ensemble_size = params["filter"]["ensemble_size"]

    ensemble = generate_ensemble(params, Val(EnsembleJustObsMember))
    ensemble_params = EnsembleJustObsParams(ensemble_size)
    work_dir = get_filter_work_dir(params, Val(EnsembleJustObsFilter))

    filter = EnsembleJustObsFilter(ensemble, ensemble_params, work_dir)
    return filter
end
