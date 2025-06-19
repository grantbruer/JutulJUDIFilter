import Pkg
# Pkg.activate("envs/SeismicPlume")
# Pkg.instantiate()

import TOML

using Statistics
using LinearAlgebra
using JLD2
using Printf
using MyUtils
using CairoMakie
using SeismicPlumeEnsembleFilter: fft, ifft, ricker_wavelet
using LaTeXStrings
using Format
using FFTW: rfft, rfftfreq
include("../lib/seismic_plume_params.jl")
include("utils.jl")

text_size_normal = 28
text_size_smaller = 24

params_file = "params/base.toml"
job_dir = "run"

if ! (@isdefined params) || isnothing(params)
    # Load parameters.
    params = TOML.parsefile(params_file)

    println("======================== params ========================")
    TOML.print(params)
    println("========================================================")

    params["transition"] = merge(params["transition"], params["ground_truth"]["transition"])
    params["observation"] = merge(params["observation"], params["ground_truth"]["observation"])

    # Compute unitless rescaler for source term and all pressure outputs.
    global WAVELET_JUDI_TEMPORAL_TO_TEMPORAL = prod(params["observation"]["d"]) # m^2 or m^3
    global WAVELET_JUDI_TEMPORAL_TO_SI = TARGET_WAVELET_TEMPORAL_MAGNITUDE * WAVELET_JUDI_TEMPORAL_TO_TEMPORAL # Pa or Pa⋅m
end

if ! (@isdefined K) || isnothing(K)
    K, phi = get_permeability_porosity(params)
end

if ! (@isdefined vel) || isnothing(vel)
    vel, rho = get_velocity_density(params)
end

# Make directory for saving figures.
save_dir = joinpath("figs", "ground_truth")
mkpath(save_dir)

with_theme(theme_latexfonts()) do
    println("Finished early"); return

    # Plot rtms and shots.
    state_keys = [:rtm_born, :rtm_born_noisy, :dshot_born, :dshot_born_noisy, :vel, :rho]
    states, extra_data = read_ground_truth_seismic_all(params, job_dir; state_keys)
    plot_rtm_data(states, save_dir; params, extra_data)
    println("Finished early"); return

    # Plot static data.
    plot_static_data(K, phi, vel, rho, save_dir; params)
    println("Finished early"); return

    # Plot plume.
    states, extra_data = read_ground_truth_plume_all(params, job_dir)
    plot_plume_data(states, save_dir; params, extra_data)


    println("Finished early"); return

    plot_dynamic_seismic_parameters(states, save_dir; params, extra_data)
    plot_shot_data(states, save_dir; params, extra_data)

    # Print out some nice stats, just in case they're useful.
    print_plume_stats(states; params)

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

function plot_points_of_interest!(ax; params, idx_wb, idx_unconformity)
    ORANGE = "#fc8d62"
    BLUE = "#8da0cb"
    GREEN = "#66c2a5"
    PINK = "#e78ac3"
    LIGHTGREEN = "#a6d854"
    BLACK = "#222"

    n_3d = Tuple(params["transition"]["n"])
    d_3d = Tuple(params["transition"]["d"])
    sat0_radius_cells = params["transition"]["sat0_radius_cells"] 

    # inj_idx = params["transition"]["injection"]["idx"] 
    # inj_zidx = params["transition"]["injection"]["zidx"] 
    inj_search_zrange = params["transition"]["injection"]["search_zrange"] 
    inj_loc = params["transition"]["injection"]["loc"] 
    inj_length = params["transition"]["injection"]["length"] 

    origin = (params["transition"]["injection"]["loc"][1] / 1000, 0, 0)
    mesh_3d = CartesianMesh(n_3d, d_3d .* n_3d ./ 1e3; origin)
    xs = range(mesh_3d.deltas[1]/2; length = mesh_3d.dims[1], step = mesh_3d.deltas[1]) .- mesh_3d.origin[1]
    ys = range(mesh_3d.deltas[end]/2; length = mesh_3d.dims[end], step = mesh_3d.deltas[end]) .- mesh_3d.origin[end]

    geom_params = (;
        setup_type = Symbol(params["observation"]["setup_type"]),
        d = d_3d[1:2:3],
        n = n_3d[1:2:3],
        nsrc = params["observation"]["nsrc"],
        dtR = params["observation"]["dtR"],
        timeR = params["observation"]["timeR"],
        nrec = params["observation"]["nrec"],
        idx_wb,
    )

    srcGeometry, recGeometry = build_source_receiver_geometry(; idx_wb, geom_params...)

    water_layer = zeros(n_3d[1:2:3]) .+ NaN
    water_layer[:, 1:idx_wb] .= 0.0
    heatmap!(ax, xs, ys, water_layer; colormap=[BLUE], colorrange=(0, 1))
    le_water_layer = ("Water layer", MarkerElement(color = BLUE, marker = :rect, markersize = 30))


    unconformity = zeros(n_3d[1:2:3]) .+ NaN
    for (row, col) in enumerate(idx_unconformity)
        unconformity[row, (col-8):col] .= 1.0
    end
    heatmap!(ax, xs, ys, unconformity; colormap=[BLACK], colorrange=(0, 1))
    le_unconformity = ("Reservoir seal", MarkerElement(color = BLACK, marker = :rect, markersize = 24))


    x = [inj_loc[1], inj_loc[1]] ./ 1e3 .- origin[1]
    y = inj_search_zrange ./ 1e3 .- origin[end]
    lines!(ax, x, y; linewidth=3, color=LIGHTGREEN)
    le_injection = ("Injection range", MarkerElement(color = LIGHTGREEN, marker = :rect, markersize = 10))

    # # Plot wells.
    # startz = inj_zidx * d_3d[3]
    # endz_idx = Int((startz + inj_length) / d_3d[3])
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
        xsrc[i] = srcGeometry.xloc[i][1] ./ 1e3 .- mesh_3d.origin[1]
        ysrc[i] = srcGeometry.zloc[i][1] ./ 1e3 .- mesh_3d.origin[end]
    end
    sc_sources = scatter!(ax, xsrc, ysrc, marker=:xcross, strokewidth=1, markersize=25, color=ORANGE)
    le_sources = ("Sources", sc_sources)

    # Plot seismic receivers.
    nrec = length(recGeometry.xloc[1])
    xrec = recGeometry.xloc[1] ./ 1e3 .- mesh_3d.origin[1]
    yrec = recGeometry.zloc[1] ./ 1e3 .- mesh_3d.origin[end]
    sc_receivers = scatter!(ax, xrec, yrec, marker=:circle, strokewidth=1, markersize=15, color=PINK)
    le_receivers = ("Receivers", sc_receivers)

    # # Plot initial saturation circle.
    # xidx, yidx = get_circle(inj_idx[1], inj_zidx, r = sat0_radius_cells)
    # sat0 = ones(n) * NaN
    # for (xi, yi) in zip(xidx, yidx)
    #     sat0[xi, yi] = 1
    # end
    # plot_heatmap!(ax, sat0, mesh_3d, colormap=[(:white, 0.5)])

    # le = ("Initial CO2", MarkerElement(color = :gray90, marker = :rect, markersize = 10))
    # push!(custom_legend_entries, le)

    # Add some entries to the legend group.
    custom_legend_entries = [
        le_sources,
        le_water_layer,
        le_receivers,
        le_unconformity,
        le_injection,
    ]

    markers = last.(custom_legend_entries)
    labels = first.(custom_legend_entries)
    labels = [isa(label, LaTeXString) ? label : LaTeXString(label) for label in labels]
    leg = axislegend(ax,
        markers,
        labels,
        position = :rc,
        margin = (10, 10, 10, -80),
        labelsize = text_size_smaller,
    )
    # Legend(
    #     f[1, 1],
    #     markers,
    #     labels,
    #     tellheight = false,
    #     tellwidth = false,
    #     margin = (10, 10, 10, 10),
    #     halign = :right, valign = :center, orientation = :horizontal,
    # )
    # entrygroups = leg.entrygroups[]
    # entries = entrygroups[1][2]
    # append!(entries, [LegendEntry(label, els, leg) for (label, els) in custom_legend_entries])
    # leg.entrygroups[] = entrygroups
end


function plot_points_of_interest_arrows!(ax; params, idx_wb, idx_unconformity)
    ORANGE = "#fc8d62"
    BLUE = "#8da0cb"
    GREEN = "#66c2a5"
    PINK = "#e78ac3"
    LIGHTGREEN = "#a6d854"
    BLACK = "#222"

    n_3d = Tuple(params["transition"]["n"])
    d_3d = Tuple(params["transition"]["d"])
    sat0_radius_cells = params["transition"]["sat0_radius_cells"] 

    # inj_idx = params["transition"]["injection"]["idx"] 
    # inj_zidx = params["transition"]["injection"]["zidx"] 
    inj_search_zrange = params["transition"]["injection"]["search_zrange"] 
    inj_loc = params["transition"]["injection"]["loc"] 
    inj_length = params["transition"]["injection"]["length"] 

    origin = (params["transition"]["injection"]["loc"][1] / 1000, 0, 0)
    mesh_3d = CartesianMesh(n_3d, d_3d .* n_3d ./ 1e3; origin)
    xs = range(mesh_3d.deltas[1]/2; length = mesh_3d.dims[1], step = mesh_3d.deltas[1]) .- mesh_3d.origin[1]
    ys = range(mesh_3d.deltas[end]/2; length = mesh_3d.dims[end], step = mesh_3d.deltas[end]) .- mesh_3d.origin[end]
    custom_legend_entries = Vector()

    geom_params = (;
        setup_type = Symbol(params["observation"]["setup_type"]),
        d = d_3d[1:2:3],
        n = n_3d[1:2:3],
        nsrc = params["observation"]["nsrc"],
        dtR = params["observation"]["dtR"],
        timeR = params["observation"]["timeR"],
        nrec = params["observation"]["nrec"],
        idx_wb,
    )

    srcGeometry, recGeometry = build_source_receiver_geometry(; idx_wb, geom_params...)



    water_layer = zeros(n_3d[1:2:3]) .+ NaN
    water_layer[:, 1:idx_wb] .= 0.0
    heatmap!(ax, xs, ys, water_layer; colormap=[BLUE], colorrange=(0, 1))
    le = ("Water layer", MarkerElement(color = BLUE, marker = :rect, markersize = 15))
    push!(custom_legend_entries, le)


    unconformity = zeros(n_3d[1:2:3]) .+ NaN
    for (row, col) in enumerate(idx_unconformity)
        unconformity[row, (col-4):col] .= 1.0
    end
    heatmap!(ax, xs, ys, unconformity; colormap=[BLACK], colorrange=(0, 1))
    le = ("Unconformity", MarkerElement(color = BLACK, marker = :rect, markersize = 15))
    push!(custom_legend_entries, le)


    ix = [inj_loc[1], inj_loc[1]] ./ 1e3 .- origin[1]
    iy = inj_search_zrange ./ 1e3 .- origin[end]
    lines!(ax, ix, iy; linewidth=3, color=LIGHTGREEN)
    le = ("Injection search range", MarkerElement(color = LIGHTGREEN, marker = :rect, markersize = 10))
    push!(custom_legend_entries, le)



    # Plot seismic sources.
    nsrc = length(srcGeometry.xloc)
    xsrc = Vector{Float64}(undef, nsrc)
    ysrc = Vector{Float64}(undef, nsrc)
    for i in 1:nsrc
        xsrc[i] = srcGeometry.xloc[i][1] ./ 1e3 .- mesh_3d.origin[1]
        ysrc[i] = srcGeometry.zloc[i][1] ./ 1e3 .- mesh_3d.origin[end]
    end
    sc_sources = scatter!(ax, xsrc, ysrc, marker=:xcross, strokewidth=1, markersize=20, color=ORANGE, label="Sources")


    # Plot seismic receivers.
    nrec = length(recGeometry.xloc[1])
    xrec = recGeometry.xloc[1] ./ 1e3 .- mesh_3d.origin[1]
    yrec = recGeometry.zloc[1] ./ 1e3 .- mesh_3d.origin[end]
    sc_receivers = scatter!(ax, xrec, yrec, marker=:circle, strokewidth=1, markersize=10, color=PINK, label="Receivers")

    # TEXT_COLOR = "#333"
    # text!(
    #     Point2f(-1, 1),
    #     text = "Hello",
    #     color = TEXT_COLOR,
    #     align = (:left, :baseline),
    #     fontsize = 26,
    #     # markerspace = :data
    # )
end

function plot_static_data(K, phi, vel, rho, save_dir_root; params)
    idx_wb = maximum(find_water_bottom_immutable(log.(K) .- log(K[1,1])))
    idx_unconformity = find_water_bottom_immutable((vel .- 3500f0) .* (vel .≥ 3500f0))

    v0, rho0 = get_background_velocity_density(vel, rho, idx_wb; params)

    save_dir = joinpath(save_dir_root, "static")
    mkpath(save_dir)

    n_3d = Tuple(params["transition"]["n"])
    d_3d = Tuple(params["transition"]["d"])

    # Display mesh in kilometers.
    origin = (params["transition"]["injection"]["loc"][1] / 1000, 0, 0)
    mesh_3d = CartesianMesh(n_3d, d_3d .* n_3d ./ 1e3; origin)

    cutoff_idx = Int(1200 ÷ d_3d[end])
    cutoff_depth = cutoff_idx * d_3d[end]
    n_3d_cutoff = n_3d .- (0, 0, cutoff_idx - 1)
    origin_cutoff = origin .- (0, 0, cutoff_depth ./ 1e3)
    mesh_3d_cutoff = CartesianMesh(n_3d_cutoff, d_3d .* n_3d_cutoff ./ 1e3; origin=origin_cutoff)

    heatmap_kwargs = (;
        # levels = 10,
        make_heatmap=true,
    )

    # Plot velocity.
    vel_parts = ifelse.(vel .>= 3850, 3, ifelse.(vel .>= 3650, 2, 1))
    fig, ax = plot_heatmap_from_grid(vel_parts, mesh_3d; colormap=:jet, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_partitions.png")
    axis_setup(ax; delete_colorbar=false)
    save(filepath, fig)

    # Plot velocity.
    vel_parts = ifelse.(vel .>= 3850, vel .- 3700, ifelse.(vel .>= 3650, vel .- 3350, vel .- 3500))
    fig, ax = plot_heatmap_from_grid(vel_parts ./ 1e3, mesh_3d; colormap=:jet, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_params.png")
    axis_setup(ax; delete_colorbar=false)
    save(filepath, fig)

    # Plot velocity.
    vel_parts = ifelse.(vel .>= 3850, vel .- 3700, NaN)
    fig, ax = plot_heatmap_from_grid(vel_parts ./ 1e3, mesh_3d; colormap=:jet, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_params1.png")
    axis_setup(ax; delete_colorbar=false)
    save(filepath, fig)

    # Plot velocity.
    vel_parts = ifelse.(vel .>= 3850, NaN, ifelse.(vel .>= 3650, vel .- 3350, NaN))
    fig, ax = plot_heatmap_from_grid(vel_parts ./ 1e3, mesh_3d; colormap=:jet, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_params2.png")
    axis_setup(ax; delete_colorbar=false)
    save(filepath, fig)


    # Plot velocity.
    vel_parts = ifelse.(vel .>= 3850, NaN, ifelse.(vel .>= 3650, NaN, vel .- 3500))
    fig, ax = plot_heatmap_from_grid(vel_parts ./ 1e3, mesh_3d; colormap=:jet, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_params3.png")
    axis_setup(ax; delete_colorbar=false)
    save(filepath, fig)

    error("I'm done")

    # Compute fully saturated velocity and density.
    phi = ifelse.(phi .> 1, 1, phi)
    S = PatchyModel(Float32.(vel), Float32.(rho), Float32.(phi); params)
    ones_sat = ones(Float32, size(vel))
    vel2, rho2 = S(ones_sat)

    # Plot experiment set up.
    colormap = parula
    tmp = zeros(size(rho)) .+ NaN
    fig, ax = plot_heatmap_from_grid(tmp, mesh_3d; colormap, colorrange=(0, 1), make_heatmap=true)
    plot_points_of_interest!(ax; params, idx_wb, idx_unconformity)
    delete!(fig.content[2])

    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    filepath = joinpath(save_dir, "experiment_setup.png")
    axis_setup(ax)
    save(filepath, fig)


    # tmp = zeros(size(rho)) .+ NaN
    # fig, ax = plot_heatmap_from_grid(tmp, mesh_3d; colormap, colorrange=(0, 1))
    # plot_points_of_interest_arrows!(ax; params, idx_wb, idx_unconformity)
    # delete!(fig.content[2])

    # ax.xlabel = "Horizontal (km)"
    # ax.ylabel = "Depth (km)"
    # filepath = joinpath(save_dir, "experiment_setup_arrows.png")
    # axis_setup(ax)
    # save(filepath, fig)


    # Plot density.
    colormap = :YlOrBr
    vmin = min(minimum(rho), minimum(rho2)) ./ 1e3
    vmax = max(maximum(rho), maximum(rho2)) ./ 1e3
    fig, ax = plot_heatmap_from_grid(rho ./ 1e3, mesh_3d; colormap, colorrange=(vmin, vmax), heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "density.png")
    axis_setup(ax; cb_label = "g/cm³", cb_tickformat = "%.1f", delete_colorbar=false)
    # axis_setup(ax; cb_label = L"\frac{\text{g}}{\text{m³}}")
    save(filepath, fig)

    fig, ax = plot_heatmap_from_grid(rho2 ./ 1e3, mesh_3d; colormap, colorrange=(vmin, vmax), heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "density_saturated.png")
    axis_setup(ax; cb_label = "g/cm³", cb_tickformat = "%.1f", delete_colorbar=false)
    save(filepath, fig)

    # Plot velocity.
    colormap = :YlOrRd
    vmin = min(minimum(vel), minimum(x -> isfinite(x) ? x : Inf, vel2)) ./ 1e3
    vmax = max(maximum(vel), maximum(x -> isfinite(x) ? x : -Inf, vel2)) ./ 1e3
    fig, ax = plot_heatmap_from_grid(vel ./ 1e3, mesh_3d; colormap, colorrange=(vmin, vmax), heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity.png")
    axis_setup(ax; cb_label = "km/s", delete_colorbar=false)
    save(filepath, fig)

    # Plot velocity.
    colormap = :YlOrRd
    vmin = min(minimum(vel), minimum(x -> isfinite(x) ? x : Inf, vel2)) ./ 1e3
    vmax = max(maximum(vel), maximum(x -> isfinite(x) ? x : -Inf, vel2)) ./ 1e3
    fig, ax = plot_heatmap_from_grid(vel ./ 1e3, mesh_3d; colormap, colorrange=(vmin, vmax), heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = ""
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_no_ylabel.png")
    axis_setup(ax; cb_label = "km/s", delete_colorbar=false)
    save(filepath, fig)

    fig, ax = plot_heatmap_from_grid(vel2 ./ 1e3, mesh_3d; colormap, colorrange=(vmin, vmax), heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_saturated.png")
    axis_setup(ax; cb_label = "km/s", delete_colorbar=false)
    save(filepath, fig)

    # Plot change in velocity and density caused by CO2.
    fig, ax = plot_heatmap_from_grid((vel2 .- vel) ./ 1e3, mesh_3d; colormap=:balance, make_divergent=true, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "pwave_velocity_co2_difference.png")
    axis_setup(ax; cb_label = "km/s", cb_tickformat = "%.1f", delete_colorbar=false)
    save(filepath, fig)

    fig, ax = plot_heatmap_from_grid((rho2 .- rho) ./ 1e3, mesh_3d; colormap=:balance, make_divergent=true, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "density_co2_difference.png")
    axis_setup(ax; cb_label = "g/cm³", cb_tickformat = "%.1f", delete_colorbar=false)
    save(filepath, fig)

    # Plot permeability.
    colormap = Reverse(:Purples)
    fig, ax = plot_heatmap_from_grid(K[:, cutoff_idx:end] ./ mD_to_meters2 ./ 1e3, mesh_3d_cutoff;
        colormap,
        make_colorbar = false,
        # extendlow = "#111",
        # extendhigh = "#CCC",
        # levels = range(1, 1600, 10),
        colorrange = (0, 1.6),
        heatmap_kwargs...
    )
    @show extrema(K[:, cutoff_idx:end] ./ mD_to_meters2)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "permeability.png")
    axis_setup(ax; cb_label = "darcy", cb_tickformat = "%.1f", delete_colorbar=false)
    save(filepath, fig)

    # Plot permeability colorbar.
    fig, ax = plot_heatmap_from_grid(K[:, cutoff_idx:end] ./ mD_to_meters2 ./ 1e3, mesh_3d_cutoff;
        colormap,
        make_colorbar = true,
        colorrange = (0, 1.6),
        # extendlow = "#111",
        # extendhigh = "#CCC",
        # levels = range(1, 1600, 10),
        heatmap_kwargs...
    )
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    filepath = joinpath(save_dir, "permeability_colorbar.png")
    axis_setup(ax; cb_label = "darcy", cb_tickformat = "%.1f", delete_colorbar=false)
    delete!(fig.content[1])
    cb = fig.content[1]
    cb.vertical = false
    # cb.ticks = levels
    colsize!(fig.layout, 1, 0.0)
    resize_to_layout!(fig)
    save(filepath, fig)

    # Plot log permeability.
    fig, ax = plot_heatmap_from_grid(log10.(K[:, cutoff_idx:end] ./ mD_to_meters2), mesh_3d_cutoff;
        colormap,
        # extendlow = "#111",
        # extendhigh = "#CCC",
        # levels = range(1, 4, 10),
        heatmap_kwargs...
    )
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "log10_permeability.png")
    formatter = generate_formatter("%.1f")
    cb_tickformat = values -> [latexstring("10^{$(formatter(v))}") for v in values]
    axis_setup(ax; cb_label = "mD", cb_tickformat, delete_colorbar=false)
    save(filepath, fig)

    # Plot porosity.
    fig, ax = plot_heatmap_from_grid(phi, mesh_3d; colormap=parula, colorrange=(0, 1), heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "porosity.png")
    axis_setup(ax, delete_colorbar=false)
    save(filepath, fig)

    # Plot impedance.
    fig, ax = plot_heatmap_from_grid(rho .* vel ./ 1e6, mesh_3d; colormap=parula, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "impedance.png")
    axis_setup(ax; cb_label = L"\text{GPa}\cdot\text{s/km}", delete_colorbar=false)
    save(filepath, fig)

    # Plot background density.
    colormap = :YlOrBr
    fig, ax = plot_heatmap_from_grid(rho0 ./ 1e3, mesh_3d; colormap, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "background_density.png")
    axis_setup(ax; cb_label = "g/m³", cb_tickformat = "%.1f", delete_colorbar=false)
    save(filepath, fig)

    # Plot background velocity.
    colormap = :YlOrRd
    fig, ax = plot_heatmap_from_grid(v0 ./ 1e3, mesh_3d; colormap, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "background_pwave_velocity.png")
    axis_setup(ax; cb_label = "km/s", delete_colorbar=false)
    save(filepath, fig)

    # Plot background impedance.
    fig, ax = plot_heatmap_from_grid(rho0 .* v0 ./ 1e6, mesh_3d; colormap=parula, heatmap_kwargs...)
    ax.xlabel = "Horizontal (km)"
    ax.ylabel = "Depth (km)"
    # plot_points_of_interest!(ax; params...)
    filepath = joinpath(save_dir, "background_impedance.png")
    axis_setup(ax; cb_label = L"\text{GPa}\cdot\text{s/km}", delete_colorbar=false)
    save(filepath, fig)

    # Plot source wavelet.
    dtR = params["observation"]["dtR"]
    timeR = params["observation"]["timeR"]
    f0 = params["observation"]["f0"]
    wavelet = WAVELET_JUDI_TEMPORAL_TO_SI .* ricker_wavelet(timeR, dtR, f0) ./ 1e6 # MPa or MPa⋅m
    @show WAVELET_JUDI_TEMPORAL_TO_SI

    dtR /= 1f3
    timeR /= 1f3
    f0 *= 1f3

    y_time = vec(wavelet) # MPa
    x_time = range(start=0, stop=timeR, step=dtR)

    y_extrema = extrema(y_time)
    y_mag = max(-y_extrema[1],  y_extrema[2])
    colorrange = (-y_mag, y_mag)

    fig, ax = lines(x_time, y_time; color=y_time, colormap=:balance, colorrange)
    ax.backgroundcolor = :gray90
    ax.xlabel = "Time (seconds)"
    ax.ylabel = "MPa"
    filepath = joinpath(save_dir, "source_time.png")
    axis_setup(ax; ytickformat="%.1f")
    ax.xtickformat = values -> [latexstring(@sprintf("%.1f", v)) for v in values]
    save(filepath, fig)

    y_freq = abs.(rfft(y_time))
    y_extrema = extrema(y_freq)
    y_mag = max(-y_extrema[1],  y_extrema[2])
    colorrange = (-y_mag, y_mag)

    x_freq = rfftfreq(length(y_time), 1 / dtR)

    fig, ax = lines(x_freq, y_freq; color=y_freq, colormap=:balance, colorrange);
    lines!(ax, [f0, f0], collect(y_extrema); color=:black, linestyle=:dash)
    ax.backgroundcolor = :gray90
    ax.xlabel = "Frequency (Hz)"
    ax.ylabel = "MPa"
    filepath = joinpath(save_dir, "source_frequency.png")
    axis_setup(ax; ytickformat="%.1f")
    save(filepath, fig)

    fig, ax = lines(x_freq, y_freq .^ 2);
    ax.backgroundcolor = :gray90
    ax.xlabel = "Frequency (Hz)"
    ax.ylabel = L"\text{MPa}^2"
    filepath = joinpath(save_dir, "noise_fourier_variance.png")
    axis_setup(ax)
    save(filepath, fig)

    fig, ax = lines(x_freq, y_freq .^ (-2));
    ax.backgroundcolor = :gray90
    ax.xlabel = "Frequency (Hz)"
    ax.ylabel = L"\text{MPa}^{-2}"
    filepath = joinpath(save_dir, "noise_fourier_precision.png")
    axis_setup(ax)
    save(filepath, fig)
end

function plot_plume_data(states, save_dir_root; params, extra_data)
    n_3d = Tuple(params["transition"]["n"])
    d_3d = Tuple(params["transition"]["d"])
    dt = params["transition"]["dt"]

    save_dir = joinpath(save_dir_root, "plume")
    mkpath(save_dir)

    # Display mesh in kilometers.
    # origin = (params["transition"]["injection"]["loc"][1] / 1000, 0, 0)
    origin = (params["transition"]["injection"]["loc"][1] / 1000 - d_3d[1]/2000, 0, 0 - d_3d[end]/2000)
    mesh_3d = CartesianMesh(n_3d, d_3d .* n_3d ./ 1e3; origin)

    cutoff_idx = Int(1200 ÷ d_3d[end])
    cutoff_depth = cutoff_idx * d_3d[end]
    n_3d_cutoff = n_3d .- (0, 0, cutoff_idx - 1)
    origin_cutoff = origin .- (0, 0, cutoff_depth ./ 1e3)
    mesh_3d_cutoff = CartesianMesh(n_3d_cutoff, d_3d .* n_3d_cutoff ./ 1e3; origin=origin_cutoff)

    function axis_setup(ax; cb_label=nothing, cb_tickformat=nothing, delete_colorbar=true)
        hidespines!(ax)
        ax.xticklabelsize = text_size_normal
        ax.xlabelsize = text_size_normal
        ax.yticklabelsize = text_size_normal
        ax.ylabelsize = text_size_normal
        if ! isa(ax.xlabel[], LaTeXString) && ax.xlabel[] != ""
            ax.xlabel = LaTeXString(ax.xlabel[])
        end
        if ! isa(ax.ylabel[], LaTeXString) && ax.ylabel[] != ""
            ax.ylabel = LaTeXString(ax.ylabel[])
        end
        println("Title: $(ax.title)")
        if ax.xtickformat[] == Makie.Automatic()
            ax.xtickformat = values -> [latexstring(@sprintf("%.0f", v)) for v in values]
        end
        if ax.ytickformat[] == Makie.Automatic()
            ax.ytickformat = values -> [latexstring(@sprintf("%.1f", v)) for v in values]
        end

        idx = findfirst(x -> x isa Colorbar, ax.parent.content)
        if !isnothing(idx)
            cb = ax.parent.content[idx]
            if ! isnothing(cb_label)
                cb.label = cb_label
                cb.labelrotation = 0.0
            end
            cb.labelsize = text_size_smaller
            cb.ticklabelsize = text_size_smaller
            if isnothing(cb_tickformat)
                if cb.tickformat[] == Makie.Automatic()
                    cb_tickformat = values -> [latexstring(cfmt("%.1f", v)) for v in values]
                end
            elseif isa(cb_tickformat, AbstractString)
                formatter = generate_formatter(cb_tickformat)
                cb_tickformat = values -> [latexstring(formatter(v)) for v in values]
            end
            if !isnothing(cb_tickformat)
                cb.tickformat = cb_tickformat
            end
            if delete_colorbar
                delete!(cb)
                resize_to_layout!(ax.parent)
            end
        end
    end

    colorbar_label = ""
    delete_colorbar = true
    post_plot = function (fig, ax)
        ax.xlabel = "Horizontal (km)"
        ax.ylabel = "Depth (km)"
        axis_setup(ax; cb_label = colorbar_label, delete_colorbar)
    end

    for state in states
        state[:Pressure_diff] = state[:Pressure] .- states[1][:Pressure]
        state[:time_str] = ""
        # state[:time_str] = LaTeXString(@sprintf "%.2f years" state[:step] * dt / 365.2425)
    end

    extras_base = (; params, extra_data..., grid=mesh_3d_cutoff)
    framerate = 2

    # Plot saturation.
    delete_colorbar = true
    get_simple = state -> reshape(ifelse.(state[:Saturation] .== 0, -1, state[:Saturation]), n_3d[1], n_3d[end])[:, cutoff_idx:end]
    levels = range(0, 1, length = 11)
    extras = (; post_plot, levels, make_colorbar=false, extendlow = :transparent, extras_base...)
    # extras = (; post_plot, colorrange = (0, 1), make_colorbar=false, extras_base...)
    file_path = joinpath(save_dir, "saturation.mp4")
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

    # Plot saturation colorbar.
    delete_colorbar = false
    extras = (; extras..., make_colorbar=true)
    fig = anim_reservoir_plotter(1, get_simple, states; extras...)
    # axis_setup(fig.content[1]; cb_label = nothing, delete_colorbar = false)
    delete!(fig.content[1])
    delete!(fig.content[2])
    cb = fig.content[1]
    cb.vertical = false
    cb.ticks = levels
    colsize!(fig.layout, 1, 0.0)
    file_path = joinpath(save_dir, "saturation_colorbar.png")
    resize_to_layout!(fig)
    save(file_path, fig)

    # Plot saturation error colorbar.
    delete_colorbar = false
    extras = (; extras..., colorrange = (-1, 1), colormap = :balance, make_heatmap = true)
    fig = anim_reservoir_plotter(1, get_simple, states; extras...)
    # axis_setup(fig.content[1]; cb_label = nothing, delete_colorbar = false)
    delete!(fig.content[1])
    delete!(fig.content[2])
    cb = fig.content[1]
    cb.vertical = false
    # cb.ticks = levels
    colsize!(fig.layout, 1, 0.0)
    file_path = joinpath(save_dir, "saturation_error_colorbar.png")
    resize_to_layout!(fig)
    save(file_path, fig)
    return

    # Plot pressure.
    delete_colorbar = false
    get_simple = state -> reshape(state[:Pressure], n_3d[1], n_3d[end])[:, cutoff_idx:end] ./ 1e6
    colorbar_label = L"\text{MPa}"
    extras = (; post_plot, colorrange = (12, 21.5), levels = range(12, 21.5, 10), extras_base...)
    file_path = joinpath(save_dir, "pressure.mp4")
    plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

    # Plot difference from initial pressure.
    delete_colorbar = false
    get_simple = state -> reshape(state[:Pressure_diff], n_3d[1], n_3d[end])[:, cutoff_idx:end] ./ 1e6
    colorbar_label = L"\text{MPa}"
    extras = (; post_plot, colorrange = (0, 2), levels = range(0, 2, 10), divergent=true, colormap=:amp, extras_base...)
    file_path = joinpath(save_dir, "pressure_diff.mp4")
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
        state[:rtm_born] .*= ORIGINAL_PA_TO_NEW_PA
        state[:rtm_born_noisy] .*= ORIGINAL_PA_TO_NEW_PA

        state[:rtm_offset] = state[:rtm_born] - states[1][:rtm_born]
        state[:rtm_offset_noisy] = state[:rtm_born_noisy] - states[1][:rtm_born_noisy]
        state[:rtm_offset_noise] = state[:rtm_offset_noisy] - state[:rtm_offset]
        state[:time_str] = ""
        # state[:time_str] = @sprintf "%.2f years" state[:step] * dt / 365.2425
    end

    # Display mesh in kilometers.
    grid = CartesianMesh(n, d .* n ./ 1e3)
    colorbar_label = ""
    cb_rotation = pi / 2

    delete_colorbar = false
    post_plot = function (fig, ax)
        ax.xlabel = "Horizontal (km)"
        ax.ylabel = "Depth (km)"
        axis_setup(ax; cb_label = colorbar_label, delete_colorbar, cb_rotation)
    end

    for colormap in [:balance, :grays]
        extras = (; post_plot, params, extra_data..., grid, colormap, divergent=true, make_heatmap=true)
        framerate = 2

        # Plot baseline RTM.
        file_path = joinpath(save_dir, "rtm_baseline_$colormap.png")
        colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
        fig, ax = plot_heatmap_from_grid(states[1][:rtm_born] ./ 1e9, grid; colormap, make_divergent=true, make_heatmap=true)
        post_plot(fig, ax)
        save(file_path, fig)

        # Plot RTMs.
        get_simple = state -> state[:rtm_born] ./ 1e9
        file_path = joinpath(save_dir, "rtms_plain_$colormap.mp4")
        colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

        # Plot offset from baseline RTM.
        get_simple = state -> state[:rtm_offset] ./ 1e9
        file_path = joinpath(save_dir, "rtms_offset_$colormap.mp4")
        colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

        # Plot noisy baseline RTM.
        file_path = joinpath(save_dir, "rtm_baseline_noisy_$colormap.png")
        colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
        fig, ax = plot_heatmap_from_grid(states[1][:rtm_born_noisy] ./ 1e9, grid; colormap, make_divergent=true, make_heatmap=true)
        post_plot(fig, ax)
        save(file_path, fig)

        # Plot noisy offsets from noisy baseline RTM.
        get_simple = state -> state[:rtm_offset_noisy] ./ 1e9
        file_path = joinpath(save_dir, "rtms_offset_noisy_$colormap.mp4")
        colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

        # Plot noisy RTMs.
        get_simple = state -> state[:rtm_born_noisy] ./ 1e9
        file_path = joinpath(save_dir, "rtms_plain_noisy_$colormap.mp4")
        colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

        # Plot baseline's noise RTM.
        file_path = joinpath(save_dir, "rtm_baseline_noise_$colormap.png")
        colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
        fig, ax = plot_heatmap_from_grid((states[1][:rtm_born_noisy] .- states[1][:rtm_born]) ./ 1e9, grid; colormap, make_divergent=true, make_heatmap=true)
        post_plot(fig, ax)
        save(file_path, fig)

        # Plot RTMs' noise.
        get_simple = state -> state[:rtm_offset_noise] ./ 1e9
        file_path = joinpath(save_dir, "rtm_offset_noise_$colormap.mp4")
        colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
        plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

        let extras = (; extras..., colorrange=(-160, 160))
            # Plot offset from baseline RTM.
            get_simple = state -> state[:rtm_offset] ./ 1e9
            file_path = joinpath(save_dir, "rtms_offset_cb_$colormap.mp4")
            colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
            plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

            # Plot noisy offsets from noisy baseline RTM.
            get_simple = state -> state[:rtm_offset_noisy] ./ 1e9
            file_path = joinpath(save_dir, "rtms_offset_noisy_cb_$colormap.mp4")
            colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
            plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)
        end

        let extras = (; extras..., colorrange=(-160, 160))
            # Plot RTMs' noise.
            get_simple = state -> state[:rtm_offset_noise] ./ 1e9
            file_path = joinpath(save_dir, "rtm_offset_noise_cb_$colormap.mp4")
            colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
            plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)
        end

        let extras = (; extras..., colorrange=(-700, 700))
            # Plot RTMs.
            get_simple = state -> state[:rtm_born] ./ 1e9
            file_path = joinpath(save_dir, "rtms_plain_cb_$colormap.mp4")
            colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
            plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)

            # Plot noisy RTMs.
            get_simple = state -> state[:rtm_born_noisy] ./ 1e9
            file_path = joinpath(save_dir, "rtms_plain_noisy_cb_$colormap.mp4")
            colorbar_label = L"\text{MPa}^2 \cdot \text{km} / \text{MRayl}"
            plot_anim(states, get_simple, anim_reservoir_plotter, file_path; extras, framerate)
        end
    end
end

function plot_dynamic_seismic_parameters(states, save_dir_root; params, extra_data)
    n = Tuple(Int.(params["observation"]["n"]))
    d = Tuple(Float32.(params["observation"]["d"]))
    dt = params["transition"]["dt"]

    # Display mesh in kilometers.
    grid = CartesianMesh(n, d .* n ./ 1e3)

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
            ax.xlabel = "Horizontal (km)"
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
    grid = CartesianMesh(n, d .* n ./ 1e3)

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
