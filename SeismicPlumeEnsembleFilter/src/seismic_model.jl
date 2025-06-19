using JOLI
using JUDI
using LinearAlgebra: norm
using Random
import ChainRulesCore

FT = Float32

Base.@kwdef struct UnitConverter
    time::FT
    area::FT
    volume::FT
    acceleration::FT
    force::FT
    distance::FT
    mass::FT
    velocity::FT
    specific_acoustic_impedance::FT
    density::FT
    pressure::FT
end

function InitializeUnitConverter(; time, distance, mass)
    time = FT(time)
    distance = FT(distance)
    mass = FT(mass)

    area = distance * distance
    volume = distance * area
    density = mass / volume

    velocity = distance / time
    acceleration = velocity / time
    force = mass * acceleration
    pressure = force / area

    specific_acoustic_impedance = velocity * density

    kwargs = (;
        time,
        distance,
        mass,
        area,
        volume,
        density,
        velocity,
        acceleration,
        force,
        pressure,
        specific_acoustic_impedance,
    )
    return UnitConverter(; kwargs...)
end

# JUDI uses the following base units: meters, milliseconds, and megagrams.
const SI_to_JUDI = InitializeUnitConverter(distance = 1, time = 1e3, mass = 1e-3)
const JUDI_to_SI = InitializeUnitConverter(distance = 1, time = 1e-3, mass = 1e3)

function JUDI._worker_pool()
    return nothing
end

struct SeismicModel{A, B, C, D, E, F}
    model::A
    q::B
    Mr1::C
    phi::Matrix{FT}
    vel::Matrix{FT}
    rho::Matrix{FT}
    imp::Matrix{FT}
    F::D
    J::E
    observation_type::Val{F}
    params::NamedTuple
end

function SeismicModel(phi, vel, rho;
    n,
    d,
    nsrc,
    nrec,
    dtR,
    timeR,
    f0,
    nbl,
    setup_type,
    snr,
    observation_type = :shot,
    kwargs...
)
    d = FT.(d)
    dtR = FT(dtR)
    timeR = FT(timeR)
    f0 = FT(f0)
    params = (;
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

    idx_wb = maximum(find_water_bottom(log.(vel) .- log(vel[1,1])))
    srcGeometry, recGeometry = build_source_receiver_geometry(; idx_wb, params...)

    # Set up source term.
    wavelet = ricker_wavelet(timeR, dtR, f0)
    q = judiVector(srcGeometry, wavelet)

    # Create model.
    origin = (0, 0)
    rho_judi = rho * SI_to_JUDI.density
    vel_judi = vel * SI_to_JUDI.velocity
    m = (1 ./ vel_judi).^2f0
    model = Model(n, d, origin, m; rho=rho_judi, nb=nbl)

    # Mute the water column and do depth scaling.
    Tm = judiTopmute(n, idx_wb, 1)
    S = judiDepthScaling(model)
    Mr1 = joCoreBlock(S*Tm)

    # Set up modeling operators.
    options = Options(IC="isic")
    F = judiModeling(model, srcGeometry, recGeometry; options)
    J = judiJacobian(F, q)

    imp = vel .* rho
    observation_type = Val(observation_type)
    return SeismicModel(model, q, Mr1, phi, vel, rho, imp, F, J, observation_type, params)
end

function SeismicModel(M::SeismicModel, phi, vel, rho)
    srcGeometry = M.F.qInjection.op.geometry
    recGeometry = M.F.rInterpolation.geometry
    n = size(M.model)
    d = spacing(M.model)
    origin = M.model.G.o
    nbl = M.model.G.nb

    vel_judi = vel * SI_to_JUDI.velocity
    rho_judi = rho * SI_to_JUDI.density

    m = (1 ./ vel_judi).^2f0

    model = Model(n, d, origin, m; rho=rho_judi, nb=nbl)
    options = Options(IC="isic")
    F = judiModeling(model, srcGeometry, recGeometry; options)
    J = judiJacobian(F, M.q)
    imp = vel .* rho
    return SeismicModel(model, M.q, M.Mr1, phi, vel, rho, imp, F, J, M.observation_type, M.params)
end

function (M::SeismicModel)(vel, rho; kwargs...)
    return M(vel, rho, M.observation_type; kwargs...)
end

function (M::SeismicModel)(vel, rho, ::Val{:shot})
    vel_judi = vel * SI_to_JUDI.velocity
    m = (1 ./ vel_judi) .^ 2
    obs = M.F(m, M.q)
    return obs * JUDI_to_SI.pressure
end

function (M::SeismicModel)(vel, rho, ::Val{:born})
    imp = vel .* rho
    dimp = vec(imp .- M.imp)
    # J has JUDI units of pressure per impedance.
    conversion = JUDI_to_SI.pressure / JUDI_to_SI.specific_acoustic_impedance
    return M.J * dimp .* conversion
end

function ChainRulesCore.rrule(::typeof(*), F::JUDI.judiPropagator, x::AbstractArray{T}) where T
    """The lazy evaluation in JUDI's rrule doesn't work right, so I got rid of it."""
    ra = F.options.return_array
    y = F*x
    postx = ra ? (dx -> reshape(dx, size(x))) : identity
    function Fback(Δy)
        dx = postx(F' * Δy)
        # F is m parametric
        dF = JUDI.∇prop(F, x, Δy)
        return ChainRulesCore.NoTangent(), dF, dx
    end
    y = F.options.return_array ? reshape_array(y, F) : y
    return y, Fback
end

function (M::SeismicModel)(dshot, ::Val{:rtm})
    n = size(M.model)
    rtm = M.J' * dshot
    rtm = reshape(M.Mr1 * M.Mr1 * vec(rtm.data), n)

    # J has JUDI units of pressure per impedance.
    conversion = JUDI_to_SI.pressure / JUDI_to_SI.specific_acoustic_impedance

    # Mr1^2 has JUDI units of length.
    conversion *= JUDI_to_SI.distance

    return rtm * conversion
end

function (M::SeismicModel)(vel, rho, ::Val{:born_rtm_depth})
    dshot, rtm, dshot_noisy, rtm_noisy = M(vel, rho, Val(:born_shot_rtm_depth_noise))
    return rtm
end

function (M::SeismicModel)(vel, rho, ::Val{:born_rtm_depth_noise})
    dshot, rtm, dshot_noisy, rtm_noisy = M(vel, rho, Val(:born_shot_rtm_depth_noise))
    return rtm, rtm_noisy
end

function (M::SeismicModel)(vel, rho, ::Val{:born_shot_rtm_depth_noise})
    dshot = M(vel, rho, Val(:born))
    rtm = M(dshot, Val(:rtm))

    dshot_noisy = dshot + generate_noise(M, dshot, M.params.snr) .* norm(dshot)
    println("Noise norm: $(norm(dshot_noisy - dshot))")
    println("SNR: $(M.params.snr)")
    rtm_noisy = M(dshot_noisy, Val(:rtm))
    return dshot, rtm, dshot_noisy, rtm_noisy
end

function build_source_receiver_geometry(; params...)
    (;
        setup_type,
        d,
        n,
        nsrc,
        dtR,
        timeR,
        nrec,
        idx_wb
    ) = (;params...)

    # Set up source and receiver geometries.
    if setup_type == :surface
        xrange = (d[1], (n[1] - 1) * d[1])
        y = 0f0
        z = 10f0
        xsrc = range(xrange[1], stop=xrange[2], length=nsrc)
        ysrc = range(y, stop=y, length=nsrc)
        zsrc = range(z, stop=z, length=nsrc)
        srcGeometry = Geometry(convertToCell.([xsrc, ysrc, zsrc])...; dt=dtR, t=timeR)

        y = 0f0
        z = (idx_wb - 1) * d[2]
        xrec = range(xrange[1], stop=xrange[2], length=nrec)
        yrec = range(y, stop=y, length=nrec)
        zrec = range(z, stop=z, length=nrec)
        recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)
    elseif setup_type == :left_right
        x = 0f0
        y = 0f0
        zrange = (d[2], (n[2] - 1) * d[2])
        xsrc = range(x, stop=x, length=nsrc)
        ysrc = range(y, stop=y, length=nsrc)
        zsrc = range(zrange[1], stop=zrange[2], length=nsrc)
        srcGeometry = Geometry(convertToCell.([xsrc, ysrc, zsrc])...; dt=dtR, t=timeR)

        x = (n[1] - 1) * d[1]
        y = 0f0
        xrec = range(x, stop=x, length=nrec)
        yrec = range(y, stop=y, length=nrec)
        zrec = range(zrange[1], stop=zrange[2], length=nrec)
        recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)
    else
        error("Unknown setup_type $(setup_type)")
    end
    return srcGeometry, recGeometry
end

function generate_noise(M::SeismicModel, ref, snr)
    source = M.q.data[1]
    noise = deepcopy(ref)
    for noise_i in noise.data
        v = randn(FT, size(noise_i))
        noise_i .= real.(ifft(fft(v) .* fft(source)))
    end
    noise = noise/norm(noise) * 10f0^(-snr/20f0)
    return noise
end

function compute_noise_info(params_seismic)
    timeR = FT(params_seismic[:timeR])
    dtR = FT(params_seismic[:dtR])
    f0 = FT(params_seismic[:f0])
    wavelet = ricker_wavelet(timeR, dtR, f0)
    return wavelet
end

function generate_noise(source, ref, snr)
    noise = deepcopy(ref)
    for noise_i in noise.data
        v = randn(FT, size(noise_i))
        noise_i .= real.(ifft(fft(v) .* fft(source)))
    end
    noise = noise/norm(noise) * 10f0^(-snr/20f0)
    return noise
end
