using JLD2

function read_ground_truth_plume_all(job_dir::String, sample_idx::Int; nbatches, nt)
    return read_ground_truth_plume(job_dir, -1, sample_idx; nbatches, nt)
end

function read_ground_truth_plume(job_dir::String, target_step_index::Int, sample_idx::Int; nbatches, nt)
    stem = "states_sample$(sample_idx)"
    return read_ground_truth_plume(stem, target_step_index; nbatches, nt)
end

function read_ground_truth_plume(stem::String, target_step_index::Int; nbatches, nt)
    vec_states = Vector{Dict{Symbol, Any}}()
    plume_state_keys = [:Saturations]

    if target_step_index == 0
        target_batch_idx = 0
        target_t_idx = 0
    else
        target_batch_idx, target_t_idx = divrem(target_step_index - 1, nt) .+ 1
    end

    batch_idx = 0
    filepath = "$(stem)_time$(batch_idx).jld2"

    step = 0
    local state0
    local extra_data
    # try
        state0, extra_data = load(filepath, "state0", "extra_data")
    # catch e
        # println("Error loading $(filepath)")
    # else
        if target_batch_idx == 0 || target_step_index == -1
            compressed = Dict(
                :Saturation => state0[:Reservoir][:Saturations][1, :],
                :Pressure => state0[:Reservoir][:Pressure],
                :step => step,
            )
            push!(vec_states, compressed)
        end
    # end
    for batch_idx = 1:nbatches
        if ! (target_step_index == -1 || batch_idx == target_batch_idx)
            continue
        end
        filepath = "$(stem)_time$(batch_idx).jld2"

        local result
        # try
            result = load(filepath, "result")
        # catch e
            # println("Error loading $(filepath)")
            # break
        # else
            if target_step_index == -1
                compressed = [Dict(
                    # :Saturation => state[:Reservoir][:Saturations][1, :],
                    # :Pressure => state[:Reservoir][:Pressure],
                    :Saturation => state[:Saturations][1, :],
                    :Pressure => state[:Pressure],
                    :step => step + i,
                ) for (i, state) in enumerate(result.states)]
                append!(vec_states, compressed)
                if length(compressed) != nt
                    error("$(filepath) expected to have $(nt) states: $(length(compressed)) != $(nt)")
                end
            else
                i = target_t_idx
                state = result.states[i]
                compressed = Dict(
                    # :Saturation => state[:Reservoir][:Saturations][1, :],
                    # :Pressure => state[:Reservoir][:Pressure],
                    :Saturation => state[:Saturations][1, :],
                    :Pressure => state[:Pressure],
                    :step => step + i,
                )
                push!(vec_states, compressed)
                break
            end
        # end
        step += nt
    end
    states_dict = vcat(vec_states)
    return states_dict, extra_data
end

function read_ground_truth_seismic_baseline(stem::String; baseline=false, state_keys=[:obs])
    batch_idx = 0
    k = 0
    file_path = "$(stem)_time$(batch_idx).jld2"
    step_index = 0

    filedata = load(file_path)
    state = Dict{Symbol, Any}()
    for key in state_keys
        state[key] = filedata["$(key)_$(k)"]
    end
    state[:step] = step_index
    if baseline
        state[:shot_baseline] = filedata["shot_baseline"]
    end
    extra_data = filedata["extra_data"]

    return state, extra_data
end

function read_ground_truth_seismic(stem::String, time_step; nbatches, nt, state_keys=[:obs])
    batch_idx, t_idx = divrem(time_step - 1, nt) .+ 1

    file_path = "$(stem)_time$(batch_idx).jld2"

    local filedata
    try
        filedata = load(file_path)
    catch e
        println("Error loading $(file_path)")
        return nothing
    end

    state = Dict{Symbol, Any}()
    state[:step] = time_step
    for key in state_keys
        state[key] = filedata["$(key)_$(t_idx)"]
    end
    return state
end

function read_ground_truth_seismic_all(stem::String; nbatches, nt, baseline=true, state_keys=[:obs])
    baseline, extra_data = read_ground_truth_seismic_baseline(stem; baseline, state_keys)

    states = Vector{Dict{Symbol, Any}}()
    push!(states, baseline)

    # Load all the time step files.
    step = 0
    for batch_idx = 1:nbatches
        file_path = "$(stem)_time$(batch_idx).jld2"

        local filedata
        try
            filedata = load(file_path)
        catch e
            println("Error loading $(file_path)")
            break
        else
            for k in 1:nt
                step += 1
                state = Dict{Symbol, Any}()
                state[:step] = step
                for key in state_keys
                    state[key] = filedata["$(key)_$(k)"]
                end
                push!(states, state)
            end
        end
    end
    return states, extra_data
end
