using Format
using LaTeXStrings
using Printf
using MultivariateStats
using JLD2
using CairoMakie
using LinearAlgebra
using Statistics
using Images: imresize
include("utils.jl")

if ! @isdefined load_data
    let cache = Dict{Any, Any}()
        global function load_data(data_file)
            if ! (data_file in keys(cache))
                cache[data_file] = load(data_file, "data")
            end
            return cache[data_file]
        end
    end
end

if ! @isdefined load_compass_model
    let cache = Dict{Any, Any}()
        global function load_compass_model(filename, name)
            key = (filename, name)
            if ! (key in keys(cache))
                cache[key] = load(filename, name)
            end
            return cache[key]
        end
    end
end

# save_dir = "figs/noise_covariance_exact"
save_dir = "figs/noise_covariance_sampled"
mkpath(save_dir)

coarse_C = load_data("coarse_C.jld2");
# coarse_C = load_compass_model("ground_truth_seismic_covariance.jld2", "coarse_C");

# eigvals = load_data("eigvals.jld2");
# PCA = load_data("PCA.jld2");
coarse_var = maximum(coarse_C; dims = 2)[:,];

N = (325, 341)
grid_params = (dims = N, deltas = (12.5, 6.25) ./ 1e3, origin = (0, 0))
domain_size = N .* grid_params.deltas
xs, ys = get_coordinates_cells(; grid_params...)
coords_grid = reshape([[x,y] for y in ys for x in xs], N);
# coords_grid = reshape(coords_grid', N);
fine_LI = LinearIndices(N)
fine_CI = CartesianIndices(N)


coarse_N = (16, 16)
coarse_LI = LinearIndices(coarse_N)
coarse_CI = CartesianIndices(coarse_N)
coarse_grid_params = (dims = coarse_N, deltas = domain_size ./ coarse_N, origin = (0, 0))
coarse_xs, coarse_ys = get_coordinates_cells(; coarse_grid_params...)
coarse_coords_grid = reshape([[x,y] for y in coarse_ys for x in coarse_xs], coarse_N)


coarse_to_fine_LI = []
coarse_to_fine_CI = []
for coarse_i = 1:coarse_N[1]
    for coarse_j = 1:coarse_N[2]
        # The coarse grid is interior to the fine grid.
        fine_i = round(Int, coarse_i * N[1] / (coarse_N[1] + 1))
        fine_j = round(Int, coarse_j * N[2] / (coarse_N[2] + 1))
        fine_idx = (fine_i - 1) + (fine_j - 1) * N[1] + 1
        c = CartesianIndex((fine_i, fine_j))
        push!(coarse_to_fine_LI, fine_idx)
        push!(coarse_to_fine_CI, c)
    end
end

# I need to scale each row of coarse_C by the depth at each entry of the row.
# I need to scale each column of coarse_C by the depth at each entry of the column.
# So I need to know the fine depth and the coarse depth for all grid points.
coarse_P = Diagonal(vec(1e-3 ./ last.(coarse_coords_grid)))
fine_P = Diagonal(vec(1e-3 ./ last.(coords_grid)))
# coarse_P = Diagonal(vec(1e3 * last.(coarse_coords_grid)))
# fine_P = Diagonal(vec(1e3 * last.(coords_grid)))
depth_coarse_C = coarse_P * coarse_C * fine_P

common_kwargs = (;
    linewidth = 3,
    markersize = 15,
)

text_size_normal = 28
text_size_smaller = 24
ORANGE = "#fc8d62"
BLUE = "#8da0cb"
GREEN = "#66c2a5"


plot_bools = Dict(pairs((;
    variance_1D = true,
    variance = true,
    eigvals = false,
    coarse_C_1D = true,
    coarse_C_1D_correlation = true,
    coarse_C_rows_2D = false,
    coarse_C_rows_2D_cb = false,
    coarse_C_rows_2D_yscaled = false,
    coarse_C_horizontal = false,
    coarse_C_horizontal_yscaled = false,
    coarse_C_horizontal_yscaled_centered = false,
    coarse_C_horizontal_yscaled_centered_zoomed = false,
    coarse_C_horizontal_xyscaled_centered_zoomed = false,
    coarse_C_horizontal_yscaled_centered_zoomed_bad_idxs = false,
    # coarse_C_horizontal_xyscaled_centered_zoomed_bad_idxs = false,
    coarse_C_vertical = false,
    coarse_C_vertical_yscaled = false,
    coarse_C_vertical_yscaled_centered = false,
    coarse_C_vertical_yscaled_centered_zoomed = false,
    coarse_C_vertical_xyscaled_centered_zoomed = false,
    coarse_C_vertical_yscaled_centered_zoomed_bad_idxs = false,
    coarse_C_vertical_xyscaled_centered_zoomed_depth = false,
    coarse_C_vertical_yscaled_centered_zoomed_depth_bad_idxs = false,
    coarse_offset_scales_horizontal = false,
    coarse_offset_scales_vertical = false,
    coarse_offset_scales_vertical_depth = false,

    coarse_C_vertical_vels = true,
    coarse_offset_scales_vertical_velocity = false,
    coarse_velocity = false,
    coarse_impedance = false,
    coarse_density = false,
    coarse_offset_scales_vertical_impedance = false,
    coarse_offset_scales_vertical_density = false,
)))

# Plot variance in 1D.
let
    if plot_bools[:variance_1D]
        println("Plotting variance_1D")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        plot!(coarse_var)

        ax.xlabel = L"\text{Noise index}"
        ax.ylabel = L"\text{Noise variance (N/s)}^2"

        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "variance_1D.png"), fig)
    end
end

# Plot variance in 2D.
let
    if plot_bools[:variance]
        println("Plotting variance")
        fig = Figure()
        ax = Axis(fig, yreversed=true)
        fig[1, 1] = ax

        coarse_var_field = reshape(coarse_var, coarse_N)'
        hm = plot_heatmap_from_grid!(ax, coarse_var_field; make_divergent=false, make_heatmap=true, colormap=:viridis, coarse_grid_params...)

        ax.xlabel = L"\text{horizontal (km)}"
        ax.ylabel = L"\text{vertical (km)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

        Colorbar(fig[:, end+1], hm; label=L"\text{(N/s)}^2")

        # display(fig)

        save(joinpath(save_dir, "variance.png"), fig)
    end
end

# Plot eigenvalues.
let
    if plot_bools[:eigvals]
        println("Plotting eigvals")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        plot!(eigvals)

        ax.xlabel = L"\text{Eigenvalue index}"
        ax.ylabel = L"\text{Eigenvalue (N/s)}^2"

        ax.yscale = log10

        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        save(joinpath(save_dir, "eigvals.png"), fig)
    end
end


# Plot rows of coarse_C in 1D.
let
    if plot_bools[:coarse_C_1D]
        println("Plotting coarse_C_1D")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            lines!(row; color=(:black, 0.01))
        end
        ax.xlabel = L"\text{Noise vector index}"
        ax.ylabel = L"\text{Cross-variance (N/s)}^2"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        save(joinpath(save_dir, "coarse_C_1D.png"), fig)
    end
end


# Plot yscaled rows of coarse_C in 1D.
let
    if plot_bools[:coarse_C_1D_correlation]
        println("Plotting coarse_C_1D_correlation")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            m = maximum(row)
            if m != 0
                lines!(row ./ m; color=(:black, 0.01))
            end
        end
        ax.xlabel = L"\text{Noise vector index}"
        ax.ylabel = L"\text{Similar to correlation}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_1D_correlation.png"), fig)
    end
end


# Plot rows of coarse_C in 2D.
let
    if plot_bools[:coarse_C_rows_2D]
        println("Plotting coarse_C_rows_2D")
        save_dir_C = "$save_dir/2D"
        mkpath(save_dir_C)
        for (i, row) in enumerate(eachrow(coarse_C))
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            fig = Figure()
            ax = Axis(fig, yreversed=true)
            fig[1, 1] = ax

            row = reshape(row, N)
            hm = plot_heatmap_from_grid!(ax, row; make_divergent=true, make_heatmap=true, colormap=:balance, grid_params...)

            ax.xlabel = L"\text{horizontal (km)}"
            ax.ylabel = L"\text{vertical (km)}"
            axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

            Colorbar(fig[:, end+1], hm; label=L"\text{(N/s)}^2")

            file_name = @sprintf("coarse_%04d.png", i)
            save(joinpath(save_dir_C, file_name), fig)
        end
    end
end

# Plot rows of coarse_C in 2D with all the same colorbar.
let
    if plot_bools[:coarse_C_rows_2D_cb]
        println("Plotting coarse_C_rows_2D_cb")
        save_dir_C = "$save_dir/2D_cb"
        mkpath(save_dir_C)

        a = findfirst(!isnan, coarse_C)
        colorrange = get_colorrange(extrema(ifelse.(isnan.(coarse_C), coarse_C[a], coarse_C)); make_divergent=true)
        for (i, row) in enumerate(eachrow(coarse_C))
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            fig = Figure()
            ax = Axis(fig, yreversed=true)
            fig[1, 1] = ax

            row = reshape(row, N)
            hm = plot_heatmap_from_grid!(ax, row; make_heatmap=true, colormap=:balance, colorrange, grid_params...)

            ax.xlabel = L"\text{horizontal (km)}"
            ax.ylabel = L"\text{vertical (km)}"
            axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

            rescale_heatmap_to_grid!(fig; grid_params...)

            Colorbar(fig[:, end+1], hm; label=L"\text{(N/s)}^2")

            file_name = @sprintf("coarse_%04d.png", i)
            save(joinpath(save_dir_C, file_name), fig)
        end
        fig = Figure()
        Colorbar(fig[1, 1]; colorrange, colormap=:balance, vertical=false, label=L"\text{(N/s)}^2")
        file_name = @sprintf("coarse_%04d_colorbar.png", 0)
        save(joinpath(save_dir_C, file_name), fig)
    end
end


# Plot yscaled rows of coarse_C in 2D with all the same colorbar.
let
    if plot_bools[:coarse_C_rows_2D_yscaled]
        println("Plotting coarse_C_rows_2D_yscaled")
        save_dir_C = "$save_dir/2D_correlation"
        mkpath(save_dir_C)

        colorrange = (-1, 1)
        for (i, row) in enumerate(eachrow(coarse_C))
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            fig = Figure()
            ax = Axis(fig, yreversed=true)
            fig[1, 1] = ax

            row = reshape(row, N) ./ maximum(row)
            hm = plot_heatmap_from_grid!(ax, row; make_heatmap=true, colormap=:balance, colorrange, grid_params...)

            ax.xlabel = L"\text{horizontal (km)}"
            ax.ylabel = L"\text{vertical (km)}"
            axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

            rescale_heatmap_to_grid!(fig; grid_params...)

            Colorbar(fig[:, end+1], hm)

            file_name = @sprintf("coarse_%04d.png", i)
            save(joinpath(save_dir_C, file_name), fig)
        end
        fig = Figure()
        Colorbar(fig[1, 1]; colorrange, colormap=:balance, vertical=false)
        file_name = @sprintf("coarse_%04d_colorbar.png", 0)
        save(joinpath(save_dir_C, file_name), fig)
    end
end

# Plot vertical parts of coarse_C.
let
    if plot_bools[:coarse_C_vertical]
        println("Plotting coarse_C_vertical")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)
            lines!(ax, ys, row[idx.I[1], :]; color=(:black, 0.01))
        end
        ax.xlabel = L"\text{vertical (km)}"
        ax.ylabel = L"\text{Covariance}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_vertical.png"), fig)
    end
end


# Plot horizontal parts of coarse_C.
let
    if plot_bools[:coarse_C_horizontal]
        println("Plotting coarse_C_horizontal")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)
            lines!(ax, xs, row[:, idx.I[2]]; color=(:black, 0.01))
        end
        ax.xlabel = L"\text{horizontal (km)}"
        ax.ylabel = L"\text{Covariance}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_horizontal.png"), fig)
    end
end

# Plot yscaled vertical parts of coarse_C.
let
    if plot_bools[:coarse_C_vertical_yscaled]
        println("Plotting coarse_C_vertical_yscaled")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)
            lines!(ax, ys, row[idx.I[1], :] ./ val; color=(:black, 0.01))
        end
        ax.xlabel = L"\text{vertical (km)}"
        ax.ylabel = L"\text{Correlation}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_vertical_yscaled.png"), fig)
    end
end


# Plot yscaled horizontal parts of coarse_C.
let
    if plot_bools[:coarse_C_horizontal_yscaled]
        println("Plotting coarse_C_horizontal_yscaled")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)
            lines!(ax, xs, row[:, idx.I[2]] ./ val; color=(:black, 0.01))
        end
        ax.xlabel = L"\text{horizontal (km)}"
        ax.ylabel = L"\text{Correlation}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_horizontal_yscaled.png"), fig)
    end
end


# Plot yscaled centered vertical parts of coarse_C.
let
    if plot_bools[:coarse_C_vertical_yscaled_centered]
        println("Plotting coarse_C_vertical_yscaled_centered")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)

            # Choose x to be at idx and plot the value as a function of y.
            lines!(ax, ys .- (idx.I[2] - 0.5) * grid_params.deltas[2], row[idx.I[1], :] ./ val; color=(:black, 0.01))
        end
        ax.xlabel = L"\text{vertical offset (km)}"
        ax.ylabel = L"\text{Correlation}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_vertical_yscaled_centered.png"), fig)
    end
end

# Plot zoomed yscaled centered vertical parts of coarse_C.
let
    if plot_bools[:coarse_C_vertical_yscaled_centered_zoomed]
        println("Plotting coarse_C_vertical_yscaled_centered_zoomed")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)

            # Choose x to be at idx and plot the value as a function of y.
            lines!(ax, ys .- (idx.I[2] - 0.5) * grid_params.deltas[2], row[idx.I[1], :] ./ val; color=(:black, 0.01))
        end
        ax.xlabel = L"\text{vertical offset (km)}"
        ax.ylabel = L"\text{Correlation}"
        xlims!(ax, (-0.2, 0.2))
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_vertical_yscaled_centered_zoomed.png"), fig)
    end
end

# Plot yscaled centered horizontal parts of coarse_C.
let
    if plot_bools[:coarse_C_horizontal_yscaled_centered]
        println("Plotting coarse_C_horizontal_yscaled_centered")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)

            # Choose y to be at idx and plot the value as a function of x.
            lines!(ax, xs .- (idx.I[1] - 0.5) * grid_params.deltas[1], row[:, idx.I[2]] ./ val; color=(:black, 0.01))
        end
        ax.xlabel = L"\text{horizontal offset (km)}"
        ax.ylabel = L"\text{Correlation}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_horizontal_yscaled_centered.png"), fig)
    end
end

# Plot zoomed yscaled centered horizontal parts of coarse_C.
let
    if plot_bools[:coarse_C_horizontal_yscaled_centered_zoomed]
        println("Plotting coarse_C_horizontal_yscaled_centered_zoomed")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        for row in eachrow(coarse_C)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)

            # Choose y to be at idx and plot the value as a function of x.
            lines!(ax, xs .- (idx.I[1] - 0.5) * grid_params.deltas[1], row[:, idx.I[2]] ./ val; color=(:black, 0.01))
        end
        ax.xlabel = L"\text{horizontal offset (km)}"
        ax.ylabel = L"\text{Correlation}"
        xlims!(ax, (-0.2, 0.2))
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_horizontal_yscaled_centered_zoomed.png"), fig)
    end
end

function plot_vertical_xyscaled_centered!(ax, coarse_C, coarse_offset_scales_vertical, coarse_offset_scales_vertical_bad_idxs)
    for (coarse_idx, row) in zip(CartesianIndices(coarse_N), eachrow(coarse_C))
        if maximum(row) == 0 || any(isnan, row)
            continue
        end
        row = reshape(row, N)
        val, idx = findmax(row)

        # Choose x to be at idx and plot the value as a function of y.
        slice = row[idx.I[1], :] ./ val
        shifted_offsets = ys .- (idx.I[2] - 0.5) * grid_params.deltas[2]


        # Scale the offsets to line them up. Put the first cutoff correlation value at x = ±(1 - cutoff).
        # Do a linear interpolation to find the zero crossing.
        # Average the results for the left crossing and the right crossing.
        half_slice_l = slice[1:idx.I[2]]
        cutoff = 0.5
        idx_dist_l = findlast(x -> x < cutoff, half_slice_l)
        mid_dist = 0.0
        if idx_dist_l != nothing
            # Interpolate between (0, half_slice_l[idx_dist_l]) and (1, half_slice_l[idx_dist_l+1]) to find (mid_dist01, cutoff).
            mid_dist01 = (half_slice_l[idx_dist_l+1] - cutoff) / (half_slice_l[idx_dist_l+1] - half_slice_l[idx_dist_l])
            mid_dist = idx.I[2] - 1 - (idx_dist_l - mid_dist01)
        end

        half_slice_r = slice[idx.I[2]:end]
        idx_dist_r = findfirst(x -> x < cutoff, half_slice_r)
        if idx_dist_r != nothing
            # Interpolate between (0, half_slice_r[idx_dist_r-1]) and (1, half_slice_r[idx_dist_r]) to find (mid_dist01, cutoff).
            mid_dist01 = (half_slice_r[idx_dist_r-1] - cutoff) / (half_slice_r[idx_dist_r-1] - half_slice_r[idx_dist_r])
            mid_dist_r = idx_dist_r - 2 + mid_dist01
            if idx_dist_l != nothing
                if abs(mid_dist - mid_dist_r) > 1
                    println("ver: Cell $coarse_idx: $(abs(mid_dist - mid_dist_r))")
                    push!(coarse_offset_scales_vertical_bad_idxs, (coarse_idx, row, mid_dist, mid_dist_r))
                end
            end
            mid_dist += mid_dist_r
        end
        mid_dist /= (idx_dist_l != nothing) + (idx_dist_r != nothing)

        # Scale so that the correlation is cutoff at x = ±(1 - cutoff)
        # mid_dist is the distance from zero offset in number of grid cells.
        coarse_offset_scales_vertical[coarse_idx] = (mid_dist * grid_params.deltas[2]) / (1 - cutoff)
        scaled_shifted_offsets = shifted_offsets ./ coarse_offset_scales_vertical[coarse_idx]
        scatterlines!(ax, scaled_shifted_offsets, slice; color=(:black, 0.05))
    end
end

# Plot zoomed xyscaled centered vertical parts of coarse_C.
let
    if plot_bools[:coarse_C_vertical_xyscaled_centered_zoomed]
        println("Plotting coarse_C_vertical_xyscaled_centered_zoomed")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        global coarse_offset_scales_vertical = fill(NaN, coarse_N)
        global coarse_offset_scales_vertical_bad_idxs = []
        plot_vertical_xyscaled_centered!(ax, coarse_C, coarse_offset_scales_vertical, coarse_offset_scales_vertical_bad_idxs)
        ax.xlabel = L"\text{scaled vertical offset (unitless)}"
        ax.ylabel = L"\text{Correlation}"
        xlims!(ax, (-8, 8))
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_vertical_xyscaled_centered_zoomed.png"), fig)
    end
end

let
    if plot_bools[:coarse_offset_scales_vertical]
        println("Plotting coarse_offset_scales_vertical")
        # Plot the scales.
        fig = Figure()
        ax = Axis(fig, yreversed=true)
        fig[1, 1] = ax

        m = mean(filter(!isnan, coarse_offset_scales_vertical))
        s = std(filter(!isnan, coarse_offset_scales_vertical))
        cutoff = m + 3*s
        ma = maximum(coarse_offset_scales_vertical[coarse_offset_scales_vertical .<= cutoff])
        mi = minimum(filter(!isnan, coarse_offset_scales_vertical))
        nanned_data = ifelse.(coarse_offset_scales_vertical .> cutoff, NaN, coarse_offset_scales_vertical)
        data = coarse_offset_scales_vertical'
        hm = plot_heatmap_from_grid!(ax, data; colorrange=(mi, ma), highclip = :cyan, make_divergent=false, make_heatmap=true, colormap=:viridis, coarse_grid_params...)

        ax.xlabel = L"\text{horizontal (km)}"
        ax.ylabel = L"\text{vertical (km)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

        Colorbar(fig[:, end+1], hm; label=L"\text{km}")

        # display(fig)

        save(joinpath(save_dir, "coarse_offset_scales_vertical.png"), fig)
    end
end


function plot_horizontal_xyscaled_centered!(ax, coarse_C, coarse_offset_scales_horizontal, coarse_offset_scales_horizontal_bad_idxs)
    for (coarse_idx, row) in zip(CartesianIndices(coarse_N), eachrow(coarse_C))
        if maximum(row) == 0 || any(isnan, row)
            continue
        end
        row = reshape(row, N)
        val, idx = findmax(row)

        # Choose x to be at idx and plot the value as a function of y.
        slice = row[:, idx.I[2]] ./ val
        shifted_offsets = xs .- (idx.I[1] - 0.5) * grid_params.deltas[1]


        # Scale the offsets to line them up. Put the first cutoff correlation value at x = ±(1 - cutoff).
        # Do a linear interpolation to find the zero crossing.
        # Average the results for the left crossing and the right crossing.
        half_slice_l = slice[1:idx.I[1]]
        cutoff = 0.5
        idx_dist_l = findlast(x -> x < cutoff, half_slice_l)
        mid_dist = 0.0
        if idx_dist_l != nothing
            # Interpolate between (0, half_slice_l[idx_dist_l]) and (1, half_slice_l[idx_dist_l+1]) to find (mid_dist01, cutoff).
            mid_dist01 = (half_slice_l[idx_dist_l+1] - cutoff) / (half_slice_l[idx_dist_l+1] - half_slice_l[idx_dist_l])
            mid_dist = idx.I[1] - 1 - (idx_dist_l - mid_dist01)
        end

        half_slice_r = slice[idx.I[1]:end]
        idx_dist_r = findfirst(x -> x < cutoff, half_slice_r)
        if idx_dist_r != nothing
            # Interpolate between (0, half_slice_r[idx_dist_r-1]) and (1, half_slice_r[idx_dist_r]) to find (mid_dist01, cutoff).
            mid_dist01 = (half_slice_r[idx_dist_r-1] - cutoff) / (half_slice_r[idx_dist_r-1] - half_slice_r[idx_dist_r])
            mid_dist_r = idx_dist_r - 2 + mid_dist01
            if idx_dist_l != nothing
                if abs(mid_dist - mid_dist_r) > 1
                    println("hor: Cell $coarse_idx: $(abs(mid_dist - mid_dist_r))")
                    push!(coarse_offset_scales_horizontal_bad_idxs, (coarse_idx, row, mid_dist, mid_dist_r))
                end
            end
            mid_dist += mid_dist_r
        end
        mid_dist /= (idx_dist_l != nothing) + (idx_dist_r != nothing)

        # Scale so that the correlation is cutoff at x = ±(1 - cutoff)
        # mid_dist is the distance from zero offset in number of grid cells.
        coarse_offset_scales_horizontal[coarse_idx] = (mid_dist * grid_params.deltas[1]) / (1 - cutoff)
        scaled_shifted_offsets = shifted_offsets ./ coarse_offset_scales_horizontal[coarse_idx]
        scatterlines!(scaled_shifted_offsets, slice; color=(:black, 0.05))
    end
end

# Plot zoomed xyscaled centered horizontal parts of coarse_C.
let
    if plot_bools[:coarse_C_horizontal_xyscaled_centered_zoomed]
        println("Plotting coarse_C_horizontal_xyscaled_centered_zoomed")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        global coarse_offset_scales_horizontal = fill(NaN, coarse_N)
        global coarse_offset_scales_horizontal_bad_idxs = []
        plot_horizontal_xyscaled_centered!(ax, coarse_C, coarse_offset_scales_horizontal, coarse_offset_scales_horizontal_bad_idxs)
        ax.xlabel = L"\text{scaled horizontal offset (unitless)}"
        ax.ylabel = L"\text{Correlation}"
        xlims!(ax, (-6, 6))
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_horizontal_xyscaled_centered_zoomed.png"), fig)
    end
end

let
    if plot_bools[:coarse_C_horizontal_yscaled_centered_zoomed_bad_idxs]
        println("Plotting coarse_C_horizontal_yscaled_centered_zoomed_bad_idxs")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        colors = resample_cmap(Makie.wong_colors(), length(coarse_offset_scales_horizontal_bad_idxs); alpha=0.3)
        for (i, (coarse_idx, row, mid_dist_l, mid_dist_r)) in enumerate(coarse_offset_scales_horizontal_bad_idxs)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)

            # Choose y to be at idx and plot the value as a function of x.
            shifted_offsets = xs .- (idx.I[1] - 0.5) * grid_params.deltas[1]
            scatterlines!(ax, shifted_offsets, row[:, idx.I[2]] ./ val; color = colors[i], label="$(coarse_idx.I)")
        end
        ax.xlabel = L"\text{horizontal offset (km)}"
        ax.ylabel = L"\text{Correlation}"
        xlims!(ax, (-0.2, 0.2))
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        if length(colors) != 0
            Legend(fig[1, 2], ax)
            resize_to_layout!(fig)
        end

        # display(fig)

        save(joinpath(save_dir, "coarse_C_horizontal_yscaled_centered_zoomed_bad_idxs.png"), fig)
    end
end


let
    if plot_bools[:coarse_C_vertical_yscaled_centered_zoomed_bad_idxs]
        println("Plotting coarse_C_vertical_yscaled_centered_zoomed_bad_idxs")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        colors = resample_cmap(Makie.wong_colors(), length(coarse_offset_scales_vertical_bad_idxs); alpha=0.3)
        for (i, (coarse_idx, row, mid_dist_l, mid_dist_r)) in enumerate(coarse_offset_scales_vertical_bad_idxs)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)

            # Choose x to be at idx and plot the value as a function of y.
            shifted_offsets = ys .- (idx.I[2] - 0.5) * grid_params.deltas[2]
            scatterlines!(ax, shifted_offsets, row[idx.I[1], :] ./ val; color = colors[i], label="$(coarse_idx.I)")
        end
        ax.xlabel = L"\text{vertical offset (km)}"
        ax.ylabel = L"\text{Correlation}"
        xlims!(ax, (-0.2, 0.2))
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        if length(colors) != 0
            Legend(fig[1, 2], ax)
            resize_to_layout!(fig)
        end

        # display(fig)

        save(joinpath(save_dir, "coarse_C_vertical_yscaled_centered_zoomed_bad_idxs.png"), fig)
    end
end

let
    if plot_bools[:coarse_offset_scales_horizontal]
        println("Plotting coarse_offset_scales_horizontal")
        # Plot the scales.
        fig = Figure()
        ax = Axis(fig, yreversed=true)
        fig[1, 1] = ax

        m = mean(filter(!isnan, coarse_offset_scales_horizontal))
        s = std(filter(!isnan, coarse_offset_scales_horizontal))
        cutoff = m + 3*s
        ma = maximum(coarse_offset_scales_horizontal[coarse_offset_scales_horizontal .<= cutoff])
        mi = minimum(filter(!isnan, coarse_offset_scales_horizontal))
        nanned_data = ifelse.(coarse_offset_scales_horizontal .> cutoff, NaN, coarse_offset_scales_horizontal)
        data = coarse_offset_scales_horizontal'
        hm = plot_heatmap_from_grid!(ax, data; colorrange=(mi, ma), highclip = :cyan, make_divergent=false, make_heatmap=true, colormap=:viridis, coarse_grid_params...)

        ax.xlabel = L"\text{horizontal (km)}"
        ax.ylabel = L"\text{vertical (km)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

        Colorbar(fig[:, end+1], hm; label=L"\text{km}")

        display(fig)

        save(joinpath(save_dir, "coarse_offset_scales_horizontal.png"), fig)
    end
end


# Plot depth-corrected zoomed xyscaled centered vertical parts of coarse_C.
let
    if plot_bools[:coarse_C_vertical_xyscaled_centered_zoomed_depth]
        println("Plotting coarse_C_vertical_xyscaled_centered_zoomed_depth")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        global coarse_offset_scales_vertical_depth = fill(NaN, coarse_N)
        global coarse_offset_scales_vertical_depth_bad_idxs = []
        plot_vertical_xyscaled_centered!(ax, depth_coarse_C, coarse_offset_scales_vertical_depth, coarse_offset_scales_vertical_depth_bad_idxs)
        ax.xlabel = L"\text{scaled vertical offset (unitless)}"
        ax.ylabel = L"\text{Correlation}"
        xlims!(ax, (-8, 8))
        # xlims!(ax, (-0.1, 0.1))
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        # display(fig)

        save(joinpath(save_dir, "coarse_C_vertical_xyscaled_centered_zoomed_depth.png"), fig)
    end
end

let
    if plot_bools[:coarse_offset_scales_vertical_depth]
        println("Plotting coarse_offset_scales_vertical_depth")
        # Plot the scales.
        fig = Figure()
        ax = Axis(fig, yreversed=true)
        fig[1, 1] = ax

        m = mean(filter(!isnan, coarse_offset_scales_vertical_depth))
        s = std(filter(!isnan, coarse_offset_scales_vertical_depth))
        cutoff = m + 3*s
        ma = maximum(coarse_offset_scales_vertical_depth[coarse_offset_scales_vertical_depth .<= cutoff])
        mi = minimum(filter(!isnan, coarse_offset_scales_vertical_depth))
        nanned_data = ifelse.(coarse_offset_scales_vertical_depth .> cutoff, NaN, coarse_offset_scales_vertical_depth)
        data = coarse_offset_scales_vertical_depth'
        hm = plot_heatmap_from_grid!(ax, data; colorrange=(mi, ma), highclip = :cyan, make_divergent=false, make_heatmap=true, colormap=:viridis, coarse_grid_params...)

        ax.xlabel = L"\text{horizontal (km)}"
        ax.ylabel = L"\text{vertical (km)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

        Colorbar(fig[:, end+1], hm; label=L"\text{km}")

        # display(fig)

        save(joinpath(save_dir, "coarse_offset_scales_vertical_depth.png"), fig)
    end
end

let
    if plot_bools[:coarse_C_vertical_yscaled_centered_zoomed_depth_bad_idxs]
        println("Plotting coarse_C_vertical_yscaled_centered_zoomed_depth_bad_idxs")
        fig = Figure()
        ax = Axis(fig)
        fig[1, 1] = ax

        colors = resample_cmap(Makie.wong_colors(), length(coarse_offset_scales_vertical_depth_bad_idxs); alpha=0.3)
        for (i, (coarse_idx, row, mid_dist_l, mid_dist_r)) in enumerate(coarse_offset_scales_vertical_depth_bad_idxs)
            if maximum(row) == 0 || any(isnan, row)
                continue
            end
            row = reshape(row, N)
            val, idx = findmax(row)

            # Choose x to be at idx and plot the value as a function of y.
            shifted_offsets = ys .- (idx.I[2] - 0.5) * grid_params.deltas[2]
            scatterlines!(ax, shifted_offsets, row[idx.I[1], :] ./ val; color = colors[i], label="$(coarse_idx.I)")
        end
        ax.xlabel = L"\text{vertical offset (km)}"
        ax.ylabel = L"\text{Correlation}"
        xlims!(ax, (-0.2, 0.2))
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing)

        if length(colors) != 0
            Legend(fig[1, 2], ax)
            resize_to_layout!(fig)
        end

        # display(fig)

        save(joinpath(save_dir, "coarse_C_vertical_yscaled_centered_zoomed_depth_bad_idxs.png"), fig)
    end
end


# Plot one vertical part of coarse_C with the velocity.
if ! @isdefined vel
    squared_slowness = load_compass_model("compass/BGCompass_tti_625m.jld2", "m");
    println("VEL WASN'T DEFINED")
    vel = ((1 ./ squared_slowness) .^ 0.5) * 1e3;
    # vel = (vel[1:2:end, :] .+ vel[2:2:end, :]) ./ 2;
    vel = imresize(vel, N);
    coarse_vel = imresize(vel, coarse_N);
end

if ! @isdefined rho
    rho = load_compass_model("compass/BGCompass_tti_625m.jld2", "rho");
    rho *= 1e3
    rho = imresize(rho, N);
    coarse_rho = imresize(rho, coarse_N);
end

if ! @isdefined impedance
    impedance = rho .* vel
    coarse_impedance = imresize(impedance, coarse_N);
end

coarse_to_fine_CI2 = argmax.(reshape(row, N) for row in eachrow(coarse_C))

let
    if plot_bools[:coarse_C_vertical_vels]
        println("Plotting coarse_C_vertical_vels")

        for coarse_li in [2, 4, 6, 22, 131, 135, 139]
            fig = Figure()
            ax = Axis(fig[1, 1])
            ax_vel = Axis(fig[1, 1], ylabelcolor = (:brown, 0.5), yticklabelcolor = (:brown, 0.5), yaxisposition = :right)
            hidespines!(ax_vel)
            hidexdecorations!(ax_vel)

            coarse_ci = coarse_CI[coarse_li]
            row = coarse_C[coarse_li, :]
            depth_row = depth_coarse_C[coarse_li, :]

            idx = coarse_to_fine_CI[coarse_li]

            row = reshape(row, N)
            val_max, idx_max = findmax(row)
            val = row[idx]
            @show (idx, val, idx_max, val_max)
            scatterlines!(ax, ys, row[idx_max.I[1], :] ./ val_max; color=(:green, 0.3), markersize=3)

            depth_row = reshape(depth_row, N)
            val_max, idx_max = findmax(depth_row)
            val = depth_row[idx]
            @show (idx, val, idx_max, val_max)
            scatterlines!(ax, ys, depth_row[idx_max.I[1], :] ./ val_max; color=(:purple, 0.3), markersize=3)

            scatterlines!(ax_vel, ys, vel[idx_max.I[1], :] .* 1e-3; color=(:brown, 0.1), markersize=3)

            ax.xlabel = L"\text{vertical (km)}"
            ax.ylabel = L"\text{Correlation}"
            axis_setup(ax; xtickformat=nothing, ytickformat=nothing)


            ax_vel.ylabel = L"\text{Velocity (km/s)}"
            axis_setup(ax_vel; xtickformat=nothing, ytickformat=nothing)

            # display(fig)

            save(joinpath(save_dir, "coarse_C_vertical_vel$(coarse_li).png"), fig)
        end
    end
end

let
    if plot_bools[:coarse_velocity]
        println("Plotting coarse_velocity")
        fig = Figure()
        ax = Axis(fig[1, 1])
        
        hm = plot_heatmap_from_grid!(ax, coarse_vel; make_divergent=false, make_heatmap=true, colormap=:viridis, coarse_grid_params...)

        ax.xlabel = L"\text{horizontal (km)}"
        ax.ylabel = L"\text{vertical (km)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

        Colorbar(fig[:, end+1], hm; label=L"\text{m/s}")

        display(fig)

        save(joinpath(save_dir, "coarse_velocity.png"), fig)
    end
end


let
    if plot_bools[:coarse_density]
        println("Plotting coarse_density")
        fig = Figure()
        ax = Axis(fig[1, 1])
        
        hm = plot_heatmap_from_grid!(ax, coarse_rho; make_divergent=false, make_heatmap=true, colormap=:viridis, coarse_grid_params...)

        ax.xlabel = L"\text{horizontal (km)}"
        ax.ylabel = L"\text{vertical (km)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

        Colorbar(fig[:, end+1], hm; label=L"\text{(kg/m}^3\text{)}")

        display(fig)

        save(joinpath(save_dir, "coarse_density.png"), fig)
    end
end

let
    if plot_bools[:coarse_impedance]
        println("Plotting coarse_impedance")
        fig = Figure()
        ax = Axis(fig[1, 1])
        
        hm = plot_heatmap_from_grid!(ax, coarse_impedance; make_divergent=false, make_heatmap=true, colormap=:viridis, coarse_grid_params...)

        ax.xlabel = L"\text{horizontal (km)}"
        ax.ylabel = L"\text{vertical (km)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)
        Colorbar(fig[:, end+1], hm; label=L"\text{(kg/m}^3\text{*m/s)}")

        display(fig)

        save(joinpath(save_dir, "coarse_impedance.png"), fig)
    end
end


let
    if plot_bools[:coarse_offset_scales_vertical_velocity]
        println("Plotting coarse_offset_scales_vertical_velocity")
        fig = Figure()
        ax = Axis(fig[1, 1])

        scatter!(ax, vec(coarse_offset_scales_vertical[:, 3:end]'), vec(coarse_vel[:, 3:end]) * 1e-3)

        ax.xlabel = L"\text{vertical offset (km)}"
        ax.ylabel = L"\text{velocity (km/s)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

        display(fig)

        save(joinpath(save_dir, "coarse_offset_scales_vertical_velocity.png"), fig)
    end
end


let
    if plot_bools[:coarse_offset_scales_vertical_density]
        println("Plotting coarse_offset_scales_vertical_density")
        fig = Figure()
        ax = Axis(fig[1, 1])

        scatter!(ax, vec(coarse_offset_scales_vertical[:, 3:end]'), vec(coarse_rho[:, 3:end]) * 1e-3)

        ax.xlabel = L"\text{vertical offset (km)}"
        ax.ylabel = L"\text{density (g/cm}^3\text{)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

        display(fig)

        save(joinpath(save_dir, "coarse_offset_scales_vertical_density.png"), fig)
    end
end


let
    if plot_bools[:coarse_offset_scales_vertical_impedance]
        println("Plotting coarse_offset_scales_vertical_impedance")
        fig = Figure()
        ax = Axis(fig[1, 1])

        scatter!(ax, vec(coarse_offset_scales_vertical[:, 3:end]'), vec(coarse_impedance[:, 3:end]) * 1e-6)

        ax.xlabel = L"\text{vertical offset (km)}"
        ax.ylabel = L"\text{impedance (g/cm}^3\text{*km/s)}"
        axis_setup(ax; xtickformat=nothing, ytickformat=nothing, delete_colorbar=false)

        display(fig)

        save(joinpath(save_dir, "coarse_offset_scales_vertical_impedance.png"), fig)
    end
end

