import Pkg
using JLD2
using MyUtils
using EnsembleFilters
using EnsembleFilters: get_filter_work_dir, load_filter


const Darcy_to_meters2 = 9.869232667160130e-13
const mD_to_meters2 = Darcy_to_meters2 * 1e-3
    
const day_to_seconds = 24*3600.0

function load_uncertain_params(params)
    Ks = get_permeability(params)
    phis = get_porosity(params)
    if ndims(phis) == 2
        phis = reshape(phis, 1, size(phis)...)
        phis = repeat(phis, outer=(size(Ks, 1),))
    end
    @assert size(Ks)[2:end] == size(phis)[2:end] "$(size(Ks)) != $(size(phis))"
    return Ks, phis
end

function maybe_pad_porosity(phi, params)
    if params["transition"]["porosity"]["pad_boundary"]
        val = params["transition"]["porosity"]["pad_value"]
        phi[1, :] .= val
        phi[end, :] .= val
        phi[:, end] .= val
    end
    return phi
end

function get_ground_truth_plume_stem(params::Dict)
    stem = "ground_truth_plume"
    return stem
end

function get_ground_truth_seismic_stem(params::Dict)
    stem = "ground_truth_seismic"
    return stem
end

function get_initial_saturation(inj_idx, params)
    sat0_radius_cells = params["transition"]["sat0_radius_cells"]
    sat0_range = params["transition"]["sat0_range"]
    n = params["transition"]["n"]
    sat0 = zeros(Float64, n[1], n[end]);
    value = sat0_range[1] + rand(Float64) * (sat0_range[2] - sat0_range[1]);
    set_circle(sat0, value, inj_idx[1], inj_idx[end], r = sat0_radius_cells)
    return sat0
end

function read_ground_truth_plume_all(params::Dict, job_dir::String)
    return read_ground_truth_plume(params, job_dir, -1)
end

function read_ground_truth_plume(params::Dict, job_dir::String, step_index::Int)
    stem = get_ground_truth_plume_stem(params)
    nt = params["transition"]["nt"]
    nbatches = params["transition"]["nbatches"]
    stem = joinpath(job_dir, stem)
    return MyUtils.read_ground_truth_plume(stem, step_index::Int; nbatches, nt)
end

function read_ground_truth_seismic_all(params::Dict, job_dir::String; state_keys=[:rtm], kwargs...)
    stem = get_ground_truth_seismic_stem(params)
    nt = params["transition"]["nt"]
    nbatches = params["transition"]["nbatches"]
    stem = joinpath(job_dir, stem)
    return MyUtils.read_ground_truth_seismic_all(stem; nbatches, nt, state_keys)
end

function read_ground_truth_seismic(params::Dict, job_dir::String, step_index::Int; state_keys=[:rtm], kwargs...)
    stem = get_ground_truth_seismic_stem(params)
    nt = params["transition"]["nt"]
    nbatches = params["transition"]["nbatches"]
    stem = joinpath(job_dir, stem)
    return MyUtils.read_ground_truth_seismic(stem, step_index; nbatches, nt, state_keys)
end

function read_ground_truth_seismic(stem::String, params::Dict, step_index::Int; state_keys=[:rtm], kwargs...)
    nt = params["transition"]["nt"]
    nbatches = params["transition"]["nbatches"]
    return MyUtils.read_ground_truth_seismic(stem, step_index; nbatches, nt, state_keys)
end

