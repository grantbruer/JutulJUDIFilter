import TOML
using JUDI
using JLD2
using Random
using JOLI
using Images
using Distributed
using Statistics
using LinearAlgebra
include("../lib/patchy.jl")
include("../lib/params.jl")
include("../lib/ensemble_filters.jl")

function filter_compute_stats(params_file, job_dir, step_index; closer, job_id, num_jobs)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    # Read filter file.
    previous_step_index = step_index - 1

    dt = params["filter"]["update_interval"] * params["transition"]["dt"]
    t0 = previous_step_index * dt
    t = step_index * dt

    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)
    filepath = joinpath(work_path, "filter_$(previous_step_index)_posterior")
    filter = load_filter(params, filepath)

    # Get fine-grained truth.
    gt_params = deepcopy(params)
    gt_params["transition"] = merge(gt_params["transition"], gt_params["ground_truth"]["transition"])
    gt_params["observation"] = merge(gt_params["observation"], gt_params["ground_truth"]["observation"])

    global gt_previous_step_index = previous_step_index * params["filter"]["update_interval"]
    gt_states, extra_data = read_ground_truth_plume(params, job_dir, gt_previous_step_index)
    global gt_state0 = gt_states[1]

    global gt_step_index = step_index * params["filter"]["update_interval"]
    gt_states, extra_data = read_ground_truth_plume(params, job_dir, gt_step_index)
    global gt_state = gt_states[1]

    fine_nt = params["filter"]["update_interval"] * params["stats"]["finer_scale"]
    global gt_states = get_fine_grained_truth(gt_state0, gt_state, step_index, t0, t, fine_nt; params=gt_params)

    error("I am here.")

    filepath = joinpath(work_path, stats, "filter_$(step_index)_prior")
    transition_filter_errors(filter, gt_states, t0, t, "intermediate_filter_$(previous_step_index)_to_$(step_index)_stats", filepath; params, closer, job_id, num_jobs)

    # For each step_index
    #     1. Advance the ground truth from t0 = previous_step_index * dt to t = step_index * dt.
    #         - Save many intermediate states.
    #     For each ensemble member em
    #         1. Advance em from t0 to t.
    #             - Save many intermediate states. 
    #         2. Compute error statistics compared to ground truth.
    #         3. Save these in a new file called "enkf_N256/stats/intermediate_filter_0_1_stats/$(em).jld2"

    #     2. The closer should save these in a new file called enkf_N256/stats/filter_0_1_stats.jld2
end

function get_fine_grained_truth(state0, state, step_index, t0, t, nt; params)
    K, phi = get_permeability_porosity(params)
    M, _, _ = initialize_plume_model(K, phi, params; setup_sim=false)
    dt = (t - t0) / nt
    tstep = fill(dt, nt)

    Msetup = PlumeModelSetup(M, state0[:Saturation], state0[:Pressure], tstep)
    global result = M(Msetup)

    states = [state0]
    compressed = [Dict(
        :Saturation => state[:Saturations][1, :],
        :Pressure => state[:Pressure],
        :t => t0 + dt * i
    ) for (i, state) in enumerate(result.states)]
    append!(states, compressed)

    for k in [:Saturation, :Pressure]
	err = sum(abs.(compressed[end][k] .- state[k])) / length(vec(state[k]))
        println("The average error in $(k) at time $(t) is $(err)")
    end

    return states
end


function filter_compute_stats(args)
    params_file = args[1]
    job_dir = args[2]
    step_index = parse(Int64, args[3])
    name = args[4]

    helper_pattern = r"^helper-(\d+)-(\d+)$"
    match_result = match(helper_pattern, name)
    if name == "closer"
        closer = true
	job_id = 1
	num_jobs = 1
    elseif ! isnothing(match_result)
        closer = false
	job_id = parse(Int, match_result.captures[1])
	num_jobs = parse(Int, match_result.captures[2])
    else
        error("Name should be 'closer' or 'helper-X-Y'. Got '$(name)'")
    end
    filter_compute_stats(params_file, job_dir, step_index; closer, job_id, num_jobs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    filter_compute_stats(ARGS)
end
