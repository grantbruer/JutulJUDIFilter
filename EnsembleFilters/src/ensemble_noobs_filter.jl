using JLD2: ReconstructedMutable

struct EnsembleNoObsMember <: AbstractEnsembleMember
    state
    params
    extra
end

struct EnsembleNoObsParams <: AbstractEnsembleFilterParams
    ensemble_size
end

struct EnsembleNoObsFilter <: AbstractEnsembleFilter
    ensemble::Vector{EnsembleNoObsMember}
    params::EnsembleNoObsParams
    work_dir
end

let oldType = ReconstructedMutable{:EnsembleNoObsMember, (:state, :params), Tuple{Any, Any}}
    global make_ensemble_member(data::oldType, ::Val{EnsembleNoObsFilter}) = EnsembleNoObsMember(getfield(data, 1)[1], getfield(data, 1)[2])
end

EnsembleNoObsMember(state, params) = EnsembleNoObsMember(state, params, nothing)
EnsembleFilters.make_ensemble_member(data, ::Val{EnsembleNoObsFilter}) = EnsembleNoObsMember(data, nothing)
get_state(x::EnsembleNoObsMember) = x.state
get_params(x::EnsembleNoObsMember) = x.params
get_extra(x::EnsembleNoObsMember) = x.extra

EnsembleFilters.get_ensemble_size(params::EnsembleNoObsParams) = params.ensemble_size

EnsembleFilters.get_ensemble(filter::EnsembleNoObsFilter) = filter.ensemble
EnsembleFilters.get_ensemble_params(filter::EnsembleNoObsFilter) = filter.params
EnsembleFilters.get_ensemble_work_dir(filter::EnsembleNoObsFilter) = filter.work_dir

EnsembleFilters.get_filter_work_dir(params::Dict, ::Val{EnsembleNoObsFilter}) = "noobs_$(params["filter"]["unique_name"])"
EnsembleFilters.get_filter_work_dir(filter::EnsembleNoObsFilter) = filter.work_dir

function EnsembleNoObsFilter(params::Dict, generate_ensemble)
    ensemble_size = params["filter"]["ensemble_size"]

    ensemble = generate_ensemble(params, Val(EnsembleNoObsMember))
    ensemble_params = EnsembleNoObsParams(ensemble_size)
    work_dir = get_filter_work_dir(params, Val(EnsembleNoObsFilter))

    filter = EnsembleNoObsFilter(ensemble, ensemble_params, work_dir)
    return filter
end

function EnsembleFilters.assimilate_data(prior::EnsembleNoObsFilter, obs_filter::EnsembleNoObsFilter, y_obs, job_dir, step_index; params, save_update=true)
    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)

    if save_update
        posterior_states = [typeof(em)(em.state, nothing, nothing) for em in prior.ensemble]
        posterior = EnsembleNoObsFilter(posterior_states, prior.params, prior.work_dir)
        filepath = joinpath(work_path, "filter_$(step_index)_state_update")
        save_filter(filepath, posterior)
    else
        posterior = prior
    end
    return posterior
end

