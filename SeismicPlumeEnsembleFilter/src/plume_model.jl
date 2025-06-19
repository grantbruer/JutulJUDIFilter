using Jutul
using JutulDarcy
using JutulDarcyRules

struct PlumeModel
    K
    phi
    dt
    idx_wb
    inj_idx
    force
    model_wrap
    params
end

function find_max_permeability_index(loc, search_z_range, d_3d, K)
    idx_2d = round.(Int, loc ./ d_3d[1:2])
    search_z_idx = range(round.(Int, search_z_range ./ d_3d[3] .- (0, 1))...)
    z_idx = search_z_idx[1] + argmax(K[idx_2d[1], search_z_idx]) - 1
    idx = (idx_2d..., z_idx)
    return idx
end

function PlumeModel(K, phi; params...)
    params = (; params...)

    (;
        injection_loc,
        production_loc,
        injection_rate,
        injection_search_zrange,
        production_search_zrange,
        n_3d,
        d_3d,
        injection_length,
        production_length,
        production_active,
        production_bottom_hole_pressure_target,
        kv_over_kh,
        dt,
    ) = params

    idx_wb = maximum(find_water_bottom(log.(K) .- log(K[1,1])))
    phi[:,1:idx_wb] .= 1.0

    inj_idx = find_max_permeability_index(injection_loc, injection_search_zrange, d_3d, K)
    startz = inj_idx[3] * d_3d[3]
    endz = startz + injection_length

    well_locs = [injection_loc]
    well_startzs = [startz]
    well_endzs = [endz]

    if production_active
        prod_idx = find_max_permeability_index(production_loc, production_search_zrange, d_3d, K)
        startz = prod_idx[3] * d_3d[3]
        endz = startz + injection_length

        push!(well_locs, production_loc)
        push!(well_startzs, startz)
        push!(well_endzs, endz)
    end

    force = jutulVWell(injection_rate, well_locs, startz = well_startzs, endz = well_endzs)

    phi_vec = convert(Vector{Float64}, vec(phi))
    model_wrap = jutulModel(n_3d, d_3d, phi_vec, K1to3(K, kvoverkh=kv_over_kh))

    return PlumeModel(K, phi, dt, idx_wb, inj_idx, force, model_wrap, params)
end

struct PlumeModelSetup
    model
    sim
    forces
    config
    tstep
end

function PlumeModelSetup(M::PlumeModel, saturation0::Array{T, N}, tstep) where {T, N}
    state0 = jutulSimpleState(M.model_wrap)
    state0[1:length(M.K)] .= vec(saturation0)
    return PlumeModelSetup(M, state0, tstep)
end

function PlumeModelSetup(M::PlumeModel, saturation0::Array{T, N}, pressure0::Array{T, N}, tstep) where {T, N}
    state0 = jutulSimpleState(M.model_wrap)
    state0[1:length(M.K)] .= vec(saturation0)
    state0[length(M.K)+1:end] .= vec(pressure0)
    return PlumeModelSetup(M, state0, tstep)
end

function PlumeModelSetup(M::PlumeModel, state0::jutulSimpleState, tstep)
    tstep = JutulDarcyRules.day * tstep
    params = (; M.params...)

    model, parameters, state0_, forces = JutulDarcyRules.setup_well_model(
        M.model_wrap,
        M.force,
        tstep;
        params.ρCO2,
        params.ρH2O,
        params.visCO2,
        params.visH2O,
        params.g,
    )

    p_ref = params.p_ref
    density_ref = [params.ρCO2, M.params.ρH2O]
    compressibility = [params.compCO2, M.params.compH2O]
    ρ = ConstantCompressibilityDensities(; p_ref, density_ref, compressibility)
    JutulDarcyRules.replace_variables!(model, PhaseMassDensities = ρ)
    syst = model.models.Reservoir.system
    JutulDarcyRules.replace_variables!(model, RelativePermeabilities = BrooksCoreyRelPerm(syst, [2.0, 2.0], [0.1, 0.1], 1.0))

    state0_[:Reservoir] = JutulDarcyRules.get_Reservoir_state(state0)

    sim, config = JutulDarcyRules.setup_reservoir_simulator(model, state0_, parameters);
    return PlumeModelSetup(model, sim, forces, config, tstep)
end

(M::PlumeModel)(saturation0) = M(saturation0, [M.dt])
# (M::PlumeModel)(saturation0, pressure0) = M(saturation0, pressure0, [M.dt])
(M::PlumeModel)(saturation0, tstep) = M(PlumeModelSetup(M, saturation0, tstep))
(M::PlumeModel)(saturation0, pressure0, tstep) = M(PlumeModelSetup(M, saturation0, reshape(pressure0, size(M.K)), tstep))

function (M::PlumeModel)(Msetup::PlumeModelSetup)
    @time result = simulate!(Msetup.sim, Msetup.tstep, forces = Msetup.forces, config = Msetup.config, max_timestep_cuts = 1000, info_level=1);
    result = ReservoirSimResult(Msetup.model, result, Msetup.forces)
    return result
end

get_permeability(M::PlumeModel) = M.K
function set_permeability!(M::PlumeModel, K)
    K = reshape(K, size(M.K))
    M.K .= K
    M.model_wrap.K .= K1to3(K, kvoverkh=M.params.kv_over_kh)
end

