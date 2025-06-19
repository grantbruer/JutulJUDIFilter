using JLD2: jldsave, load

abstract type AbstractEnsembleMember end
abstract type AbstractEnsembleFilterParams end
abstract type AbstractEnsembleFilter end

function get_filter_work_dir(::AbstractEnsembleFilter)
    return "ensemble_filter"
end

function get_filter_work_dir(params::Dict, ::Val{<:AbstractEnsembleFilter})
    return "ensemble_filter"
end

function get_ensemble(filter::AbstractEnsembleFilter)
    error("abstract method")
end

function get_ensemble_path(file_path, ::Val{<:AbstractEnsembleFilter})
    prefix = splitext(file_path)[1]
    ensemble_dir = "$(prefix)_ensemble"
    return ensemble_dir
end

function get_ensemble_work_dir(filter::AbstractEnsembleFilter)
    error("abstract method")
end

function get_ensemble_params(filter::AbstractEnsembleFilter)
    error("abstract method")
end

function get_ensemble_size(params::AbstractEnsembleFilterParams)
    error("abstract method")
end

make_ensemble_member(data::AbstractEnsembleMember, T::Type) = data
make_ensemble_member(data, T::Type) = make_ensemble_member(data, Val(T))

function load_ensemble(folder_path, N, ::Val{T}) where {T<:AbstractEnsembleFilter}
    ensemble_data = [load(joinpath(folder_path, "$(i).jld2"), "ensemble_member") for i = 1:N]
    ensemble = [make_ensemble_member(data, T) for data in ensemble_data]
    return ensemble
end

function save_ensemble(filter::AbstractEnsembleFilter, folder_path, N)
    ensemble = get_ensemble(filter)
    mkpath(folder_path)
    for i = 1:N
        file_path = joinpath(folder_path, "$(i).jld2")
        ensemble_member = ensemble[i]
        jldsave(file_path; ensemble_member)
    end
end

function load_filter(file_path, ::Val{T}) where {T<:AbstractEnsembleFilter}
    # Read parameters
    params = load(file_path, "filter_parameters")
    N = get_ensemble_size(params)

    # Read ensemble members.
    ensemble_dir = get_ensemble_path(file_path, Val(T))
    ensemble = load_ensemble(ensemble_dir, N, Val(T))

    work_dir = splitdir(file_path)[1]
    return T(ensemble, params, work_dir)
end

function save_filter(file_path, filter::AbstractEnsembleFilter; ensemble=true)
    # Save parameters
    params = get_ensemble_params(filter)
    jldsave(file_path, filter_parameters=params)

    if ensemble
        # Save ensemble members.
        N = get_ensemble_size(params)
        ensemble_dir = get_ensemble_path(file_path, Val(typeof(filter)))
        save_ensemble(filter, ensemble_dir, N)
    end
end

function load_ensemble_member_params(job_dir, filter::AbstractEnsembleFilter, id)
    # Ensemble member is saved at "filter_$(k)_posterior_ensemble/i.jld2"
    #   and "filter_$(k+1)_prior_ensemble/i.jld2" from k = 0 to something.
    #
    # If "filter_$(k+1)_prior" doesn't exist, the ensemble member may be
    #   at "intermediate_trans_$(k)_to_$(k+1)_ensemble/i.jld2".

    function get_state(e)
        params = e.params
        model = params.M

        state = Dict{Symbol, Any}(
            :Permeability => model.K,
            :Porosity => model.phi,
        )
        return state
    end

    w = get_filter_work_dir(filter)
    states = Vector{Dict{Symbol, Any}}()
    step_index = 0
    while (step_index <= 100000)
        filename = "filter_$(step_index)_posterior_ensemble/$(id).jld2"
        filepath = joinpath(w, filename)
        if !isfile(filepath)
            println("nothing at $(filepath)")
            break
        end
        e = load(filepath, "ensemble_member")
        state = merge(get_state(e), Dict{Symbol, Any}(:step => step_index))
        push!(states, state)

        step_index += 1

        filename = "filter_$(step_index)_prior_ensemble/$(id).jld2"
        filepath = joinpath(w, filename)
        if !isfile(filepath)
            println("nothing at $(filepath)")
            break

            filename = "intermediate_trans_$(step_index-1)_to_$(step_index)_ensemble/$(id).jld2"
            filepath = joinpath(w, filename)
            if !isfile(filepath)
                println("nothing at $(filepath)")
                break
            end
            e = load(filepath, "ensemble_member")
            state = merge(get_state(e), Dict{Symbol, Any}(:step => step_index))
            push!(states, state)
            break
        end
        e = load(filepath, "ensemble_member")
        state = merge(get_state(e), Dict{Symbol, Any}(:step => step_index))
        push!(states, state)
    end
    return states
end

function load_ensemble_member_states(job_dir, filter::AbstractEnsembleFilter, id)
    # Ensemble member is saved at "filter_$(k)_posterior_ensemble/i.jld2"
    #   and "filter_$(k+1)_prior_ensemble/i.jld2" from k = 0 to something.
    #
    # If "filter_$(k+1)_prior" doesn't exist, the ensemble member may be
    #   at "intermediate_trans_$(k)_to_$(k+1)_ensemble/i.jld2".

    w = get_filter_work_dir(filter)
    states = Vector{Dict{Symbol, Any}}()
    step_index = 0
    while (step_index <= 100000)
        filename = "filter_$(step_index)_posterior_ensemble/$(id).jld2"
        filepath = joinpath(w, filename)
        if !isfile(filepath)
            println("nothing at $(filepath)")
            break
        end
        e = load(filepath, "ensemble_member")
        state = Dict{Symbol, Any}(
            :step => step_index,
            :Saturation => e.state,
        )
        push!(states, state)

        step_index += 1

        filename = "filter_$(step_index)_prior_ensemble/$(id).jld2"
        filepath = joinpath(w, filename)
        if !isfile(filepath)
            println("nothing at $(filepath)")
            break

            filename = "intermediate_trans_$(step_index-1)_to_$(step_index)_ensemble/$(id).jld2"
            filepath = joinpath(w, filename)
            if !isfile(filepath)
                println("nothing at $(filepath)")
                break
            end
            e = load(filepath, "ensemble_member")
            state = Dict{Symbol, Any}(
                :step => step_index,
                :Saturation => e.state,
            )
            push!(states, state)
            break
        end
        e = load(filepath, "ensemble_member")
        state = Dict{Symbol, Any}(
            :step => step_index,
            :Saturation => e.state,
        )
        push!(states, state)
    end
    return states
end

function load_ensemble_member_rtms(job_dir, filter::AbstractEnsembleFilter, id)
    # Ensemble member is saved at "filter_obs_$(k)_prior_ensemble/i.jld2" k = 1 to something.
    #
    # If "filter_$(k+1)_prior" doesn't exist, the ensemble member may be
    #   at "filter_intermediate_obs_$(k)_prior_ensemble/i.jld2".

    w = get_filter_work_dir(filter)
    states = Vector{Dict{Symbol, Any}}()
    step_index = 1
    while (step_index <= 100000)
        filename = "filter_obs_$(step_index)_prior_ensemble/$(id).jld2"
        filepath = joinpath(w, filename)
        if !isfile(filepath)
            println("nothing at $(filepath)")
            filename = "intermediate_obs_$(step_index)_ensemble/$(id).jld2"
            filepath = joinpath(w, filename)
            if isfile(filepath)
                e = load(filepath, "ensemble_member")
                state = Dict{Symbol, Any}(
                    :step => step_index,
                    :rtm => e[1],
                    :rtm_noisy => e[2],
                )
                push!(states, state)
            end
            break
        end
        e = load(filepath, "ensemble_member")
        state = Dict{Symbol, Any}(
            :step => step_index,
            :rtm => e[1],
            :rtm_noisy => e[2],
        )
        push!(states, state)
        step_index += 1
    end
    return states
end

function generate_ensemble(params::Dict)
    error("abstract method")
end

function assimilate_data(prior::AbstractEnsembleFilter, obs_filter::AbstractEnsembleFilter, y_obs, job_dir, step_index; params, save_update=true)
    error("abstract method")
end
