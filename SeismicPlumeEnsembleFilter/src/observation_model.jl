struct PatchyModel
    vel
    rho
    phi
    boundary_mask
    d
    patchy_kwargs
end

function PatchyModel(vel, rho, phi; params, idx_wb=-1)
    d = Tuple(Float32.(params["observation"]["d"]))
    bulk_H2O = Float32(params["observation"]["bulk_H2O"])
    bulk_CO2 = Float32(params["observation"]["bulk_CO2"])
    ρCO2 = Float32(params["observation"]["density_CO2"])
    ρH2O = Float32(params["observation"]["density_H2O"])

    patchy_constant_kwargs = (;
        B_fl1 = bulk_H2O,
        ρ_fl1 = ρH2O,
        B_fl2 = bulk_CO2,
        ρ_fl2 = ρCO2,
    )
    M_sat1, M_sat2, idx_wb1 = compute_patchy_constants(vel, rho, phi, d; patchy_constant_kwargs...)
    if idx_wb == -1
        idx_wb = idx_wb1
    end
    boundary_mask = (phi .> 1e1)
    patchy_kwargs = (; patchy_constant_kwargs..., M_sat1, M_sat2, idx_wb)
    return PatchyModel(vel, rho, phi, boundary_mask, d, patchy_kwargs)
end

function (P::PatchyModel)(saturation)
    v_t, rho_t = Patchy(Float32.(saturation), P.vel, P.rho, P.phi, P.d; P.patchy_kwargs...)
    return v_t, rho_t
end

struct SeismicCO2Observer
    M::SeismicModel
    P::PatchyModel
end

function (S::SeismicCO2Observer)(saturation)
    saturation = reshape(saturation, S.M.params.n)
    saturation = ifelse.(S.P.boundary_mask, 0, saturation)
    v_t, rho_t = S.P(saturation)
    return S.M(v_t, rho_t)
end
