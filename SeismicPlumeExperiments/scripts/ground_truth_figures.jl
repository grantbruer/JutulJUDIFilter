import Pkg
Pkg.activate("envs/SeismicPlume")
Pkg.instantiate()

import TOML

using Statistics
using LinearAlgebra
using JLD2
using Printf
using MyUtils
using CairoMakie
using SeismicPlumeEnsembleFilter: fft, ifft, ricker_wavelet
include("../lib/seismic_plume_params.jl")

function generate_ground_truth_figures(params_file, job_dir)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    # Make directory for saving figures.
    save_dir = joinpath(job_dir, "figs", "ground_truth")
    mkpath(save_dir)

    # Plot static data.
    params["transition"] = merge(params["transition"], params["ground_truth"]["transition"])
    params["observation"] = merge(params["observation"], params["ground_truth"]["observation"])

    K, phi = get_permeability_porosity(params)
    vel, rho = get_velocity_density(params)

    # Plot stuff that doesn't change.
    plot_static_data(K, phi, vel, rho, save_dir; params)

    # Plot plume.
    states, extra_data = read_ground_truth_plume_all(params, job_dir)
    plot_plume_data(states, save_dir; params, extra_data)

    # Print out some nice stats, just in case they're useful.
    print_plume_stats(states; params)

    # Plot rtms and shots.
    state_keys = [:rtm_born, :rtm_born_noisy, :dshot_born, :dshot_born_noisy, :vel, :rho]
    states, extra_data = read_ground_truth_seismic_all(params, job_dir; state_keys)
    plot_dynamic_seismic_parameters(states, save_dir; params, extra_data)
    plot_rtm_data(states, save_dir; params, extra_data)
    plot_shot_data(states, save_dir; params, extra_data)

    # Print out some nice stats, just in case they're useful.
    print_seismic_stats(states; params)
end

function get_horizontal_gradient(a, d, n)
    a = reshape(a, n)
    return (a[2:end, :] - a[1:end-1, :]) / d[1]
end

function get_vertical_gradient(a, d, n)
    a = reshape(a, n)
    return (a[:, 2:end] - a[:, 1:end-1]) / d[end]
end

function print_plume_stats(states; params)
    d = params["transition"]["d"]
    n = Tuple(params["transition"]["n"])[1:2:3]
    todo = [
        ("Saturation", s -> s[:Saturation]),
        ("Saturation horizontal gradient", s -> get_horizontal_gradient(s[:Saturation], d, n)),
        ("Saturation vertical gradient", s -> get_vertical_gradient(s[:Saturation], d, n)),
        ("Pressure", s -> s[:Pressure]),
    ]
    println("*"^80)
    for (name, getter) in todo
        for state in states
            i = state[:step]
            println("Step $(i) -- $(name)")
            print_scalar_stats(getter(state))
            println()
        end
        println()
    end
    println("*"^80)
end

function print_seismic_stats(states; params)
    d = params["observation"]["d"]
    n = Tuple(params["observation"]["n"])
    todo = [
        ("Velocity", s -> s[:vel]),
        ("Velocity horizontal gradient", s -> get_horizontal_gradient(s[:vel], d, n)),
        ("Velocity vertical gradient", s -> get_vertical_gradient(s[:vel], d, n)),

        ("Delta velocity", s -> s[:dvel]),
        ("Delta velocity horizontal gradient", s -> get_horizontal_gradient(s[:dvel], d, n)),
        ("Delta velocity vertical gradient", s -> get_vertical_gradient(s[:dvel], d, n)),

        ("Density", s -> s[:rho]),
        ("Density horizontal gradient", s -> get_horizontal_gradient(s[:rho], d, n)),
        ("Density vertical gradient", s -> get_vertical_gradient(s[:rho], d, n)),

        ("Delta density", s -> s[:drho]),
        ("Delta density horizontal gradient", s -> get_horizontal_gradient(s[:drho], d, n)),
        ("Delta density vertical gradient", s -> get_vertical_gradient(s[:drho], d, n)),

        ("Impedance", s -> s[:imp]),
        ("Impedance horizontal gradient", s -> get_horizontal_gradient(s[:imp], d, n)),
        ("Impedance vertical gradient", s -> get_vertical_gradient(s[:imp], d, n)),

        ("Delta impedance", s -> s[:dimp]),
        ("Delta impedance horizontal gradient", s -> get_horizontal_gradient(s[:dimp], d, n)),
        ("Delta impedance vertical gradient", s -> get_vertical_gradient(s[:dimp], d, n)),

        ("Delta RTM", s -> s[:rtm_offset]),
        ("Delta RTM horizontal gradient", s -> get_horizontal_gradient(s[:rtm_offset], d, n)),
        ("Delta RTM vertical gradient", s -> get_vertical_gradient(s[:rtm_offset], d, n)),

        ("Delta RTM noisy", s -> s[:rtm_offset_noisy]),
        ("Delta RTM noisy horizontal gradient", s -> get_horizontal_gradient(s[:rtm_offset_noisy], d, n)),
        ("Delta RTM noisy vertical gradient", s -> get_vertical_gradient(s[:rtm_offset_noisy], d, n)),

        ("RTM noise", s -> s[:rtm_offset_noise]),
        ("RTM noise horizontal gradient", s -> get_horizontal_gradient(s[:rtm_offset_noise], d, n)),
        ("RTM noise vertical gradient", s -> get_vertical_gradient(s[:rtm_offset_noise], d, n)),

        ("Shot", s -> s[:shot_born]),
        ("Noisy shot", s -> s[:shot_born_noisy]),
        ("Delta shot", s -> s[:delta_shot_born]),
        ("Delta noisy shot", s -> s[:delta_shot_born_noisy]),
        ("Shot noise", s -> s[:noise_shot]),
    ]
    println("*"^80)
    for (name, getter) in todo
        for state in states
            i = state[:step]
            println("Step $(i) -- $(name)")
            print_scalar_stats(vec(getter(state)))
            println()
        end
        println()
    end
    println("*"^80)
end

function print_scalar_stats(a)
    m = mean(a)
    @printf("  %-10s: % 10.6e\n", "mean", m)

    v = stdm(a, m)
    @printf("  %-10s: % 10.6e\n", "std", v)

    m = median(a)
    @printf("  %-10s: % 10.6e\n", "median", m)

    v = mean(abs.(a .- m))
    @printf("  %-10s: % 10.6e\n", "MAD", v)
end

function plot_points_of_interest!(ax;
    n,
    inj_idx,
    inj_zidx,
    inj_loc,
    prod_loc,
    d_3d,
    mesh_3d,
    injection_length,
    srcGeometry,
    recGeometry,
    sat0_idx_radius,
    sat0_radius,
    extra_kwargs...
)
    custom_legend_entries = Vector()

    # # Plot wells.
    # startz = inj_zidx * d_3d[3]
    # endz_idx = Int((startz + injection_length) / d_3d[3])
    # endz = endz_idx * d_3d[3]

    # l = endz_idx - inj_zidx + 1
    # # xi = collect(range(inj_loc[1], inj_loc[1]; length=l))
    # # xp = collect(range(prod_loc[1], prod_loc[1]; length=l))
    # # y = collect(range(startz, endz; length=l))

    # # scatter!(ax, xi, y, color=:purple, marker=:rect, markersize=10, label="Injector", alpha=0.0)
    # # scatter!(ax, xp, y, color=:orange, marker=:rect, markersize=10, label="Producer", alpha=0.0)

    # prod_idx = round.(Int, prod_loc ./ d_3d[1:2])

    # wells = fill(NaN, n)
    # yidx = inj_zidx:endz_idx

    # xinj = fill(inj_idx[1], l)
    # for (xi, yi) in zip(xinj, yidx)
    #     wells[xi, yi] = 0
    # end

    # xpro = fill(prod_idx[1], l)
    # for (xi, yi) in zip(xpro, yidx)
    #     wells[xi, yi] = 1
    # end
    # plot_heatmap!(ax, wells, mesh_3d, colormap=[:purple, :orange])

    # le_inj = ("Injector", MarkerElement(color = :purple, marker = :rect, markersize = 10))
    # le_pro = ("Producer", MarkerElement(color = :orange, marker = :rect, markersize = 10))
    # push!(custom_legend_entries, le_inj)
    # push!(custom_legend_entries, le_pro)

    # Plot seismic sources.
    nsrc = length(srcGeometry.xloc)
    xsrc = Vector{Float64}(undef, nsrc)
    ysrc = Vector{Float64}(undef, nsrc)
    for i in 1:nsrc
        xsrc[i] = srcGeometry.xloc[i][1]
        ysrc[i] = srcGeometry.zloc[i][1]
    end
    scatter!(ax, xsrc, ysrc, marker=:xcross, strokewidth=1, markersize=20, color=:yellow, label="Sources")

    # Plot seismic receivers.
    nrec = length(recGeometry.xloc[1])
    xrec = recGeometry.xloc[1]
    yrec = recGeometry.zloc[1]
    scatter!(ax, xrec, yrec, marker=:circle, strokewidth=1, markersize=10, color=:red, label="Receivers")

    # # Plot initial saturation circle.
    # xidx, yidx = get_circle(inj_idx[1], inj_zidx, r = sat0_idx_radius)
    # sat0 = ones(n) * NaN
    # for (xi, yi) in zip(xidx, yidx)
    #     sat0[xi, yi] = 1
    # end
    # plot_heatmap!(ax, sat0, mesh_3d, colormap=[(:white, 0.5)])

    # le = ("Initial CO2", MarkerElement(color = :gray90, marker = :rect, markersize = 10))
    # push!(custom_legend_entries, le)

    # Add some entries to the legend group.
    leg = axislegend()
    entrygroups = leg.entrygroups[]
    entries = entrygroups[1][2]
    append!(entries, [LegendEntry(label, els, leg) for (label, els) in custom_legend_entries])
    leg.entrygroups[] = entrygroups
end

function plot_static_data(K, phi, vel, rho, save_dir_root; params)
    idx_wb = maximum(find_water_bottom_immutable(log.(K) .- log(K[1,1])))
    v0, rho0 = get_background_velocity_density(vel, rho, idx_wb; params)

    save_dir = joinpath(save_dir_root, "static")
    mkpath(save_dir)

    n_3d = Tuple(params["transition"]["n"])
    d_3d = Tuple(params["transition"]["d"])

    # Display mesh in kilometers.
    mesh_3d = CartesianMesh(n_3d, d_3d .* n_3d ./ 1000.0)

    # Plot fully saturated velocity and density.
    phi = ifelse.(phi .> 1, 1, phi)
    S = PatchyModel(Float32.(vel), Float32.(rho), Float32.(phi); params)
    ones_sat = ones(Float32, size(vel))
    vel2, rho2 = S(ones_sat)

    # Plot density.
    colormap = parula
    vmin = min(minimum(rho), minimum(rho2))
    vmax = max(maximum(rho), maximum(rho2))
    fig, ax = plot_heatmap_from_grid(rho, mesh_3d; colormap, colorrange=(vmin, vmax))
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Density (kg/m³) with no CO2"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "density.png")
    save(filepath, fig)

    fig, ax = plot_heatmap_from_grid(rho2, mesh_3d; colormap, colorrange=(vmin, vmax))
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Density (kg/m³) with all CO2"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "density_saturated.png")
    save(filepath, fig)

    # Plot velocity.
    vmin = min(minimum(vel), minimum(x -> isfinite(x) ? x : Inf, vel2))
    vmax = max(maximum(vel), maximum(x -> isfinite(x) ? x : -Inf, vel2))
    fig, ax = plot_heatmap_from_grid(vel, mesh_3d; colormap, colorrange=(vmin, vmax))
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "P-wave velocity (m/s) with no CO2"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity.png")
    save(filepath, fig)

    fig, ax = plot_heatmap_from_grid(vel2, mesh_3d; colormap, colorrange=(vmin, vmax))
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "P-wave velocity (m/s) with all CO2"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_saturated.png")
    save(filepath, fig)

    # Plot change in velocity and density caused by CO2.
    fig, ax = plot_heatmap_from_grid(vel2 .- vel, mesh_3d; colormap=:balance, make_divergent=true)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "P-wave velocity (m/s) difference due to max CO2"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_co2_difference.png")
    save(filepath, fig)

    fig, ax = plot_heatmap_from_grid(rho2 .- rho, mesh_3d; colormap=:balance, make_divergent=true)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Density (kg/m³) difference due to max CO2"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "density_co2_difference.png")
    save(filepath, fig)

    # Plot permeability.
    fig, ax = plot_heatmap_from_grid(K ./ mD_to_meters2, mesh_3d; colormap=parula)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Permeability (millidarcies)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "permeability.png")
    save(filepath, fig)

    # Plot log permeability.
    fig, ax = plot_heatmap_from_grid(log10.(K ./ mD_to_meters2), mesh_3d; colormap=parula)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Log (base 10) Permeability (relative to 1 millidarcy)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "log10_permeability.png")
    save(filepath, fig)

    # Plot porosity.
    fig, ax = plot_heatmap_from_grid(phi, mesh_3d; colormap=parula, colorrange=(0, 1))
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Porosity"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "porosity.png")
    save(filepath, fig)

    # Plot density.
    fig, ax = plot_heatmap_from_grid(rho, mesh_3d; colormap=parula)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Density (kg/m³) with no CO2"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "density.png")
    save(filepath, fig)

    # Plot velocity.
    fig, ax = plot_heatmap_from_grid(vel, mesh_3d; colormap=parula)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "P-wave velocity (m/s) with no CO2"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity.png")
    save(filepath, fig)

    # Plot impedance.
    fig, ax = plot_heatmap_from_grid(rho .* vel, mesh_3d; colormap=parula)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Impedance (N s / m³)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "impedance.png")
    save(filepath, fig)

    # Plot background density.
    fig, ax = plot_heatmap_from_grid(rho0, mesh_3d; colormap=parula)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Density (kg/m³) with no CO2"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "background_density.png")
    save(filepath, fig)

    # Plot background velocity.
    fig, ax = plot_heatmap_from_grid(v0, mesh_3d; colormap=parula)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "P-wave velocity (m/s)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "background_pwave_velocity.png")
    save(filepath, fig)

    # Plot background impedance.
    fig, ax = plot_heatmap_from_grid(rho0 .* v0, mesh_3d; colormap=parula)
    ax.xlabel = "Length (km)"
    ax.ylabel = "Depth (km)"
    ax.title = "Impedance (N s / m³)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "background_impedance.png")
    save(filepath, fig)

    # Plot source wavelet.
    dtR = params["observation"]["dtR"]
    timeR = params["observation"]["timeR"]
    f0 = params["observation"]["f0"]
    wavelet = ricker_wavelet(timeR, dtR, f0)

    dtR /= 1f3
    timeR /= 1f3
    f0 *= 1f3

    y_time = vec(wavelet)
    x_time = range(start=0, stop=timeR, step=dtR)

    y_extrema = extrema(y_time)
    y_mag = max(-y_extrema[1],  y_extrema[2])
    colorrange = (-y_mag, y_mag)

    fig, ax = lines(x_time, y_time; color=y_time, colormap=:balance, colorrange)
    ax.backgroundcolor = :gray90
    ax.xlabel = "Time (seconds)"
    ax.ylabel = "GPa / m^2"
    ax.title = "Seismic source ($(f0) Hz)"
    filepath = joinpath(save_dir, "source_time.png")
    save(filepath, fig)

    y_freq_comp = fft(y_time)
    y_freq = abs.(y_freq_comp)
    y_extrema = extrema(y_freq)
    y_mag = max(-y_extrema[1],  y_extrema[2])
    colorrange = (-y_mag, y_mag)

    x_freq = (0:length(y_freq)-1) / length(y_freq) / dtR

    fig, ax = lines(x_freq, y_freq; color=y_freq, colormap=:balance, colorrange);
    lines!(ax, [f0, f0], collect(y_extrema); color=:black, linestyle=:dash)
    ax.backgroundcolor = :gray90
    ax.xlabel = "Frequency (Hz)"
    ax.ylabel = "GPa / m^2"
    ax.title = "Seismic source ($(f0) Hz)"
    filepath = joinpath(save_dir, "source_frequency.png")
    save(filepath, fig)

    fig, ax = lines(abs.(y_freq_comp) .^ 2);
    ax.backgroundcolor = :gray90
    ax.xlabel = "Frequency (Hz)"
    ax.ylabel = "1 / (GPa / m^2)"
    ax.title = "Noise covariance in Fourier domain"
    filepath = joinpath(save_dir, "noise_fourier_covariance.png")
    save(filepath, fig)

    fig, ax = lines(abs.(y_freq_comp) .^ (-2));
    ax.backgroundcolor = :gray90
    ax.xlabel = "Frequency (Hz)"
    ax.ylabel = "1 / (GPa / m^2)"
    ax.title = "Noise precision in Fourier domain"
    filepath = joinpath(save_dir, "noise_fourier_precision.png")
    save(filepath, fig)
end

function plot_plume_data(states, save_dir_root; params, extra_data)
    n_3d = Tuple(params["transition"]["n"])
    d_3d = Tuple(params["transition"]["d"])
    dt = params["transition"]["dt"]

    save_dir = joinpath(save_dir_root, "plume")
    mkpath(save_dir)

    # Display mesh in kilometers.
    mesh_3d = CartesianMesh(n_3d, d_3d .* n_3d ./ 1000.0)

    colorbar_label = ""
    post_plot = function (fig, ax)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        idx = findfirst(x -> x isa Colorbar, fig.content)
        if ! isnothing(idx)
            cb = fig.content[idx]
            cb.label = colorbar_label
        end
    end

    for state in states
        state[:Pressure_diff] = state[:Pressure] .- states[1][:Pressure]
        state[:time_str] = @sprintf "%.2f years" state[:step] * dt / 365.2425
    end

    extras_base = (; params, extra_data..., grid=mesh_3d)
    framerate = 2

    # Plot saturation.
    get_simple = state -> reshape(ifelse.(state[:Saturation] .== 0, NaN, state[:Saturation]), n_3d[1], n_3d[end])
    file_path = joinpath(save_dir, "saturation.mp4")
    extras = (; post_plot, colorrange = (0, 1), extras_base...)
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

    # Plot pressure.
    get_simple = state -> reshape(state[:Pressure], n_3d[1], n_3d[end])
    file_path = joinpath(save_dir, "pressure.mp4")
    colorbar_label = "Pa"
    extras = (; post_plot, extras_base...)
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

    # Plot difference from initial pressure.
    get_simple = state -> reshape(state[:Pressure_diff], n_3d[1], n_3d[end])
    file_path = joinpath(save_dir, "pressure_diff.mp4")
    colorbar_label = "Pa"
    extras = (; post_plot, divergent=true, colormap=:balance, extras_base...)
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)
end

function plot_rtm_data(states, save_dir_root; params, extra_data)
    n = Tuple(Int.(params["observation"]["n"]))
    d = Tuple(Float32.(params["observation"]["d"]))
    dt = params["transition"]["dt"]

    save_dir = joinpath(save_dir_root, "rtm")
    mkpath(save_dir)

    println(collect(keys(states[1])))
    for state in states
        state[:rtm_offset] = state[:rtm_born] - states[1][:rtm_born]
        state[:rtm_offset_noisy] = state[:rtm_born_noisy] - states[1][:rtm_born_noisy]
        state[:rtm_offset_noise] = state[:rtm_offset_noisy] - state[:rtm_offset]
        state[:time_str] = @sprintf "%.2f years" state[:step] * dt / 365.2425
    end

    # Display mesh in kilometers.
    grid = CartesianMesh(n, d .* n ./ 1000.0)
    colorbar_label = ""
    post_plot = function (fig, ax)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        idx = findfirst(x -> x isa Colorbar, fig.content)
        if ! isnothing(idx)
            cb = fig.content[idx]
            cb.label = colorbar_label
        end
    end

    colormap = :balance
    extras = (; post_plot, params, extra_data..., grid, colormap, divergent=true)
    framerate = 2

    # Plot baseline RTM.
    file_path = joinpath(save_dir, "rtm_baseline.png")
    colorbar_label = "N/s"
    fig, ax = plot_heatmap_from_grid(states[1][:rtm_born], grid; colormap, make_divergent=true)
    post_plot(fig, ax)
    save(file_path, fig)

    # Plot RTMs.
    get_simple = state -> state[:rtm_born]
    file_path = joinpath(save_dir, "rtms_plain.mp4")
    colorbar_label = "N/s"
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

    # Plot offset from baseline RTM.
    get_simple = state -> state[:rtm_offset]
    file_path = joinpath(save_dir, "rtms_offset.mp4")
    colorbar_label = "N/s"
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

    # Plot noisy baseline RTM.
    file_path = joinpath(save_dir, "rtm_baseline_noisy.png")
    colorbar_label = "N/s"
    fig, ax = plot_heatmap_from_grid(states[1][:rtm_born_noisy], grid; colormap, make_divergent=true)
    post_plot(fig, ax)
    save(file_path, fig)

    # Plot noisy offsets from noisy baseline RTM.
    get_simple = state -> state[:rtm_offset_noisy]
    file_path = joinpath(save_dir, "rtms_offset_noisy.mp4")
    colorbar_label = "N/s"
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

    # Plot noisy RTMs.
    get_simple = state -> state[:rtm_born_noisy]
    file_path = joinpath(save_dir, "rtms_plain_noisy.mp4")
    colorbar_label = "N/s"
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

    # Plot baseline's noise RTM.
    file_path = joinpath(save_dir, "rtm_baseline_noise.png")
    colorbar_label = "N/s"
    fig, ax = plot_heatmap_from_grid(states[1][:rtm_born_noisy] .- states[1][:rtm_born], grid; colormap, make_divergent=true)
    post_plot(fig, ax)
    save(file_path, fig)

    # Plot RTMs' noise.
    get_simple = state -> state[:rtm_offset_noise]
    file_path = joinpath(save_dir, "rtm_offset_noise.mp4")
    colorbar_label = "N/s"
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)
end

function plot_dynamic_seismic_parameters(states, save_dir_root; params, extra_data)
    n = Tuple(Int.(params["observation"]["n"]))
    d = Tuple(Float32.(params["observation"]["d"]))
    dt = params["transition"]["dt"]

    # Display mesh in kilometers.
    grid = CartesianMesh(n, d .* n ./ 1000.0)

    save_dir = joinpath(save_dir_root, "seismic_params_dynamic")
    mkpath(save_dir)

    # Compute plot data.
    println(collect(keys(states[1])))
    for state in states
        state[:imp] = state[:vel] .* state[:rho];
        state[:dvel] = state[:vel] .- states[1][:vel];
        state[:drho] = state[:rho] .- states[1][:rho];
        state[:dimp] = state[:imp] .- states[1][:imp];
        state[:time_str] = @sprintf "%.2f years" state[:step] * dt / 365.2425
    end

    # Set up plot parameters.
    framerate = 2
    extras = (; params, extra_data..., grid, colormap=parula)

    # Plot things.
    let extras = extras
        colorbar_label = ""
        post_plot = function (fig, ax)
            ax.xlabel = "Length (km)"
            ax.ylabel = "Depth (km)"
            idx = findfirst(x -> x isa Colorbar, fig.content)
            if ! isnothing(idx)
                cb = fig.content[idx]
                cb.label = colorbar_label
            end
        end
        extras=(; post_plot, extras...)

        # Plot the parameters.
        file_path = joinpath(save_dir, "velocity.mp4")
        colorbar_label = "m/s"
        get_simple = x -> x[:vel]
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "density.mp4")
        colorbar_label = "kg/m³"
        get_simple = x -> x[:rho]
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "impedance.mp4")
        colorbar_label = "Pa s/m"
        get_simple = x -> x[:imp]
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)


        # Plot the deltas relative to the initial time.
        extras=(; extras..., colormap=:balance, divergent=true)

        file_path = joinpath(save_dir, "delta_velocity.mp4")
        colorbar_label = "m/s"
        get_simple = x -> x[:dvel]
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "delta_density.mp4")
        colorbar_label = "kg/m³"
        get_simple = x -> x[:drho]
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "delta_impedance.mp4")
        colorbar_label = "Pa s/m"
        get_simple = x -> x[:dimp]
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)
    end
end

function plot_shot_data(states, save_dir_root; params, extra_data)
    n = Tuple(Int.(params["observation"]["n"]))
    d = Tuple(Float32.(params["observation"]["d"]))
    dt = params["transition"]["dt"]

    # Display mesh in kilometers.
    grid = CartesianMesh(n, d .* n ./ 1000.0)

    save_dir = joinpath(save_dir_root, "shots")
    mkpath(save_dir)

    # Compute plot data.
    println(collect(keys(states[1])))
    for state in states
        state[:shot_born] = states[1][:shot_baseline] + state[:dshot_born]
        state[:shot_born_noisy] = states[1][:shot_baseline] + state[:dshot_born_noisy]
        state[:delta_shot_born] = state[:dshot_born] - states[1][:dshot_born]
        state[:delta_shot_born_noisy] = state[:dshot_born_noisy] - states[1][:dshot_born_noisy]
        state[:noise_shot] = state[:dshot_born_noisy] - state[:dshot_born]
        state[:time_str] = @sprintf "%.2f years" state[:step] * dt / 365.2425
    end

    # Set up plot parameters.
    framerate = 2
    extras = (; params, extra_data..., grid, colormap=:balance, divergent=true)

    # Plot standard shot records.
    let extras = extras
        file_path = joinpath(save_dir, "shot_baseline.png")
        get_simple = x -> x[:shot_baseline].data
        idx = Observable(1)
        fig = anim_shot_record_plotter(idx, get_simple, states; extras...)
        save(file_path, fig)

        file_path = joinpath(save_dir, "dshot.mp4")
        get_simple = x -> x[:delta_shot_born].data
        plot_anim(states, get_simple, anim_shot_record_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "dshot_noisy.mp4")
        get_simple = x -> x[:delta_shot_born_noisy].data
        plot_anim(states, get_simple, anim_shot_record_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "shot.mp4")
        get_simple = x -> x[:shot_born].data
        plot_anim(states, get_simple, anim_shot_record_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "shot_noisy.mp4")
        get_simple = x -> x[:shot_born_noisy].data
        plot_anim(states, get_simple, anim_shot_record_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "dshot_baseline.mp4")
        get_simple = x -> x[:dshot_born].data
        plot_anim(states, get_simple, anim_shot_record_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "dshot_baseline_noisy.mp4")
        get_simple = x -> x[:dshot_born_noisy].data
        plot_anim(states, get_simple, anim_shot_record_plotter, file_path; extras, framerate)

        file_path = joinpath(save_dir, "shot_noise.mp4")
        get_simple = x -> x[:noise_shot].data
        plot_anim(states, get_simple, anim_shot_record_plotter, file_path; extras, framerate)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    params_file = ARGS[1]
    job_dir = ARGS[2]
    generate_ground_truth_figures(params_file, job_dir)
end
