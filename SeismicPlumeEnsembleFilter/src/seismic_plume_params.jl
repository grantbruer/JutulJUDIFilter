include("params.jl")

import EnsembleFilters
using Images: imfilter, imresize, Kernel
using Random: rand, randperm

function EnsembleFilters.generate_ensemble(params::Dict, ::Val{T}) where T <: AbstractEnsembleMember
    Ks, phis = load_uncertain_params(params)

    ensemble_size = params["filter"]["ensemble_size"]

    ensemble = Vector{T}(undef, ensemble_size)
    if ensemble_size != size(Ks, 1)
        params_idxs = randperm(size(Ks, 1))
        @assert ensemble_size <= length(params_idxs)
    else
        params_idxs = collect(1:size(Ks, 1))
    end

    n = size(Ks)[2:end]
    for i = 1:ensemble_size
        K = Ks[params_idxs[i], :, :]
        phi = phis[params_idxs[i], :, :]

        println("Generating ensemble member $(i)")
        M, saturation0, tstep = initialize_plume_model(K, phi, params; setup_sim=false)

        K_min = 500.0
        K_minned = max.(K_min, K./mD_to_meters2)
        K_blur = imfilter(K_minned, Kernel.gaussian(10));
        K_lower_mD = max.(K_blur .- K_min, 0)
        idx = find_water_bottom_immutable(K_lower_mD)
        rows = 1:n[end]
        masks = [rows .>= idx[i] for i = 1:n[1]];
        update_mask = hcat(masks...)';

        state = saturation0
        em_params = (;
            M,
            update_mask,
        )
        ensemble[i] = T(state, em_params)
    end
    return ensemble
end


function get_observer(params)
    if params["filter"]["observation_type"] == "none"
        observer = x -> nothing
    else
        observation_type = Symbol(params["filter"]["observation_type"])
        M, M0 = SeismicModel_params(params; observation_type)

        # Patchy model still uses unblurred velocity and density.
        P = PatchyModel(M.vel, M.rho, M.phi; params)

        observer = SeismicCO2Observer(M0, P)
    end
end

function get_permeability(params)
    compass_dir = params["compass_dir"]
    n = Tuple(params["transition"]["n"])
    type = params["transition"]["permeability"]["type"]
    key = get(params["transition"]["permeability"], "key", "K")
    if type == "file_Kphi"
        idx = get(params["transition"]["permeability"], "idx", 0)
        if idx == 0
            idx_file = params["transition"]["permeability"]["idx_file"]
            idx_file = joinpath(compass_dir, idx_file)
            idx = load(idx_file, "idx")
        end
        file_name = params["transition"]["permeability"]["Kphi_file"]
        file_path = joinpath(compass_dir, file_name)

        K = jldopen(file_path, "r") do file
            K = file[key][idx, :, :]
            K = K * mD_to_meters2 # Convert from millidarcy to square meters.
            return K
        end
        if ndims(K) == 2
            K = imresize(K, n[1], n[end])
        else
            K = cat(collect(reshape(imresize(K[i, :, :], n[1], n[end]), 1, n[1], n[end]) for i in 1:size(K, 1))..., dims=1)
        end
        return K
    end
    if type == "file_K"
        file_name = params["transition"]["permeability"]["K_file"]
        file_path = joinpath(compass_dir, file_name)
        K = load(file_path, key)
        K = K * mD_to_meters2 # Convert from millidarcy to square meters.
        if ndims(K) == 3 && size(K, 2) == 1
            K = dropdims(K; dims=2)
        end
        K = imresize(K, n[1], n[end])
        return K
    end
    error("Unknown permeability type: '$(type)'")
end

function get_porosity(params)
    type = params["transition"]["porosity"]["type"]
    n = Tuple(params["transition"]["n"])
    if type == "constant"
        value = params["transition"]["porosity"]["value"]
        phi = fill(value, n)
        if n[2] == 1
            phi = dropdims(phi; dims=2)
        end
        phi = maybe_pad_porosity(phi, params)
        return phi
    end
    key = get(params["transition"]["porosity"], "key", "K")
    if type == "file_Kphi"
        compass_dir = params["compass_dir"]

        idx = params["transition"]["porosity"]["idx"]
        file_name = params["transition"]["porosity"]["Kphi_file"]
        file_path = joinpath(compass_dir, file_name)

        phi = jldopen(file_path, "r") do file
            phi = file["phi"][idx, :, :]
            return phi
        end
        phi = maybe_pad_porosity(phi, params)
        return phi
    end
    if type == "file_phi"
        file_name = params["transition"]["porosity"]["phi_file"]
        file_path = joinpath(compass_dir, file_name)
        phi = load(file_path, key)
        if ndims(phi) == 3 && size(phi, 2) == 1
            phi = dropdims(phi; dims=2)
        end
        phi = imresize(phi, n[1], n[end])
        phi = maybe_pad_porosity(phi, params)
        return phi
    end
    error("Unknown porosity type: '$(type)'")
end

function get_permeability_porosity(params)
    K = get_permeability(params)
    phi = get_porosity(params)
    return K, phi
end

function get_velocity(params)
    compass_dir = params["compass_dir"]
    type = params["observation"]["velocity"]["type"]

    if type == "file_velrho"
        idx = params["observation"]["velocity"]["idx"]
        file_name = params["observation"]["velocity"]["velrho_file"]
        file_path = joinpath(compass_dir, file_name)

        vel = jldopen(file_path, "r") do file
            # Speed is already m/s.
            vel = file["v"][idx, :, :]
            return vel
        end
        return vel
    end
    if type == "file_mrho"
        file_name = params["observation"]["velocity"]["mrho_file"]
        file_path = joinpath(compass_dir, file_name)

        vel = jldopen(file_path, "r") do file
            # Squared slowness is s²km². Need to get velocity in m/s.
            m = file["m"]
            vel = (1 ./ m) .^ 0.5 .* 1e3
            return vel
        end
        n = Tuple(params["observation"]["n"])
        vel = imresize(vel, n)
        return vel
    end
    error("Unknown velocity type: '$(type)'")
end

function get_density(params)
    compass_dir = params["compass_dir"]
    type = params["observation"]["density"]["type"]

    if type == "file_velrho"
        idx = params["observation"]["density"]["idx"]
        file_name = params["observation"]["density"]["velrho_file"]
        file_path = joinpath(compass_dir, file_name)

        rho = jldopen(file_path, "r") do file
            rho = file["rho"][idx, :, :]
            # Convert density from g/cm³ to kg/m³.
            rho = rho .* 1f3
            return rho
        end
        n = Tuple(params["observation"]["n"])
        rho = imresize(rho, n)
        return rho
    end
    if type == "file_mrho"
        file_name = params["observation"]["density"]["mrho_file"]
        file_path = joinpath(compass_dir, file_name)
        rho = load(file_path, "rho")
        # Convert density from g/cm³ to kg/m³.
        rho = rho .* 1f3
        n = Tuple(params["observation"]["n"])
        rho = imresize(rho, n[1], n[end])
        return rho
    end
    error("Unknown density type: '$(type)'")
end

function get_velocity_density(params)
    vel = get_velocity(params)
    rho = get_density(params)
    return vel, rho
end

function get_background_velocity_density(vel, rho, idx_wb; params)
    type = params["observation"]["background"]["type"]
    v0 = deepcopy(vel);
    rho0 = deepcopy(rho);
    if type == "blur"
        # This part blurs the data to get a background model.
        blur_cells = Int(params["observation"]["background"]["blur_cells"])
        v0[:, idx_wb+1:end] = 1f0 ./ imfilter(1f0./v0[:, idx_wb+1:end], Kernel.gaussian(blur_cells));
        rho0[:, idx_wb+1:end] = 1f0 ./ imfilter(1f0./rho0[:, idx_wb+1:end], Kernel.gaussian(blur_cells));
        return v0, rho0
    end
    error("Unknown background model type: '$(type)'")
end

function initialize_plume_model(K, phi, params; setup_sim=true)
    plume_params = Dict{Symbol, Any}(
        :injection_loc => Tuple(params["transition"]["injection"]["loc"]),
        :production_loc => Tuple(params["transition"]["production"]["loc"]),
        :injection_search_zrange => params["transition"]["injection"]["search_zrange"],
        :production_search_zrange => params["transition"]["production"]["search_zrange"],
        :n_3d => Tuple(Int.(params["transition"]["n"])),
        :d_3d => Tuple(params["transition"]["d"]),
        :injection_length => params["transition"]["injection"]["length"],
        :production_length => params["transition"]["production"]["length"],
        :production_active => params["transition"]["production"]["active"],
        :production_bottom_hole_pressure_target => params["transition"]["production"]["bottom_hole_pressure_target"],
        :kv_over_kh => params["transition"]["kv_over_kh"],
        :dt => params["transition"]["dt"],
        :visCO2 => params["transition"]["viscosity_CO2"],
        :visH2O => params["transition"]["viscosity_H2O"],
        :ρCO2 => params["transition"]["density_CO2"],
        :ρH2O => params["transition"]["density_H2O"],
        :g => params["transition"]["g"],
        :p_ref => params["transition"]["reference_pressure"],
        :compCO2 => params["transition"]["compressibility_CO2"],
        :compH2O => params["transition"]["compressibility_H2O"],
    )

    if haskey(params["transition"]["injection"], "irate")
        irate = params["transition"]["injection"]["irate"]
    else
        irate_mtons_year = params["transition"]["injection"]["rate_mtons_year"]
        irate_kg_s = irate_mtons_year * 1e9 / (365.2425 * 24 * 60 * 60) # kilograms per second
        irate = irate_kg_s / params["transition"]["injection"]["density_CO2"] # m^3/s
    end
    plume_params[:injection_rate] = irate

    M = PlumeModel(K, phi; plume_params...)

    saturation0 = get_initial_saturation(M.inj_idx, params)
    tstep = params["transition"]["dt"] * ones(params["transition"]["nt"])
    if ! setup_sim
        return M, saturation0, tstep
    end
    Msetup = PlumeModelSetup(M, saturation0, tstep)
    return M, Msetup
end

function SeismicModel_params(params; return_blurred=true, kwargs...)
    gt_params = deepcopy(params)
    gt_params["transition"] = merge(gt_params["transition"], gt_params["ground_truth"]["transition"])
    gt_params["observation"] = merge(gt_params["observation"], gt_params["ground_truth"]["observation"])

    K, phi = get_permeability_porosity(gt_params)
    vel, rho = get_velocity_density(gt_params)

    FT = Float32
    phi = FT.(phi)
    vel = FT.(vel)
    rho = FT.(rho)

    # Create initial model with unblurred velocity, then copy stuff to blurred model.
    M = SeismicModel_params(phi, vel, rho; params, kwargs...)
    if ! return_blurred
        return M
    end

    idx_wb = maximum(find_water_bottom_immutable(log.(K) .- log(K[1,1])))
    v0, rho0 = get_background_velocity_density(vel, rho, idx_wb; params)
    M0 = SeismicModel(M, M.phi, v0, rho0)
    return M, M0
end

function SeismicModel_params(phi, vel, rho; params, kwargs...)
    FT = Float32
    phi = FT.(phi)
    vel = FT.(vel)
    rho = FT.(rho)

    d = Tuple(Float32.(params["observation"]["d"]))
    n = Tuple(Int.(params["observation"]["n"]))
    nsrc = Int(params["observation"]["nsrc"])
    nrec = Int(params["observation"]["nrec"])
    timeR = Float32(params["observation"]["timeR"])
    dtR = Float32(params["observation"]["dtR"])
    setup_type = Symbol(params["observation"]["setup_type"])
    f0 = Float32(params["observation"]["f0"])
    snr = Float32(params["observation"]["snr"])
    nbl = Int(params["observation"]["nbl"])

    params_seismic = (;
        kwargs...,
        n,
        d,
        nsrc,
        nrec,
        dtR,
        timeR,
        setup_type,
        f0,
        nbl,
        snr,
    )

    M = SeismicModel(phi, vel, rho; params_seismic...)
    return M
end
