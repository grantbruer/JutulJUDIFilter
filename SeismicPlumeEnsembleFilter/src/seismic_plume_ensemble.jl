using EnsembleFilters
using Distributed: pmap
using JLD2

function no_transition_member(x::T, t0, t) where T <: AbstractEnsembleMember
    return x
end

function transition_member(x::T, t0, t) where T <: AbstractEnsembleMember
    dt = t - t0
    state = get_state(x)
    extra = get_extra(x)
    params = get_params(x)
    M = params.M
    if isnothing(extra)
        result = M(state, [dt])
    else
        result = M(state, extra.pressure, [dt])
    end
    state .= reshape(result.states[end][:Saturations][1, :], size(state))
    extra = (; pressure = result.states[end][:Pressure])
    x = T(state, params, extra)
    return x
end

function (S::SeismicCO2Observer)(x::AbstractEnsembleMember)
    state = get_state(x)
    return S(state)
end

function make_transitioner_wrapper(transitioner, t0, t, ensemble_dir; params)
    function transitioner_wrapper(args)
        (i, ensemble_member) = args

        # Skip if file already exists.
        filepath = joinpath(ensemble_dir, "$(i).jld2")
        if isfile(filepath)
            return
        end

        println("Advancing ensemble member $(i) from time $(t0) to time $(t)")
        new_ensemble_member = transitioner(ensemble_member, t0, t)
        jldsave(filepath; ensemble_member=new_ensemble_member)
    end
    return transitioner_wrapper
end

function divvy_up_ensemble(N, job_id, num_jobs)
    base_work, extra_work = divrem(N, num_jobs)

    # Distribute work almost evenly
    work_per_job = fill(base_work, num_jobs)
    work_per_job[1:extra_work] .+= 1

    # Calculate the starting index for this process
    start_index = sum(work_per_job[1:job_id-1]) + 1

    # Calculate the ending index for this process
    end_index = start_index + work_per_job[job_id] - 1

    return start_index, end_index
end

function transition_filter(filter::AbstractEnsembleFilter, t0, t, identifier, target_path; params, closer, job_id, num_jobs)
    if ispath(target_path)
        error("Can't write to $(target_path). It already exists")
    end

    work_dir = get_ensemble_work_dir(filter)
    ensemble = get_ensemble(filter)

    ensemble_dir = joinpath(work_dir, "$(identifier)_ensemble")
    mkpath(ensemble_dir)

    println("Advancing filter from time $(t0) to time $(t)")
    if closer
        my_slice = enumerate(ensemble)
    else
        N = length(ensemble)
        s, e = divvy_up_ensemble(N, job_id, num_jobs)
        println("  - Doing $(e-s+1) ensemble members: $(s) through $(e).")
        my_slice = zip(s:e, ensemble[s:e])
    end
    if params["filter"]["transition_type"] == "plume"
        transitioner = transition_member
    elseif params["filter"]["transition_type"] == "none"
        transitioner = no_transition_member
    else
        error("Invalid transition_type: '$(params["filter"]["transition_type"])'")
    end
    transitioner_wrapper = make_transitioner_wrapper(transitioner, t0, t, ensemble_dir; params)
    ensemble = pmap(transitioner_wrapper, my_slice)

    if closer
        # Save filter to target_path.
        mv(ensemble_dir, "$(target_path)_ensemble")
        save_filter(target_path, filter; ensemble=false)
    end
end

function make_observer_wrapper(observer, ensemble_dir)
    function observer_wrapper(args)
        (i, em) = args

        # Skip if file already exists.
        filepath = joinpath(ensemble_dir, "$(i).jld2")
        if isfile(filepath)
            return
        end

        println("Simulating observations for ensemble member $(i)")
        y_em = observer(em)
        jldsave(filepath; ensemble_member=y_em)
    end
end

function observe_filter(observer, filter::AbstractEnsembleFilter, identifier, target_path; params, closer, job_id, num_jobs)
    work_dir = get_ensemble_work_dir(filter)
    ensemble = get_ensemble(filter)

    ensemble_dir = joinpath(work_dir, "$(identifier)_ensemble")
    mkpath(ensemble_dir)

    println("Observing filter")
    if closer
        my_slice = enumerate(filter.ensemble)
    else
        N = length(ensemble)
        s, e = divvy_up_ensemble(N, job_id, num_jobs)
        println("  - Doing $(e-s+1) ensemble members: $(s) through $(e).")
        my_slice = zip(s:e, ensemble[s:e])
    end
    observer_wrapper = make_observer_wrapper(observer, ensemble_dir)
    obs_ensemble = pmap(observer_wrapper, my_slice)

    if closer
        # Save filter to target_path.
        mv(ensemble_dir, "$(target_path)_ensemble")
        save_filter(target_path, filter; ensemble=false)
    end
end
