import CairoMakie

using EnsembleFilters
using LinearAlgebra
using IterativeSolvers: cg
using JLD2
using JOLI: joLinearFunction, @joNF, joMatrix, jo_iterative_solver4square_set
using .KernelMatrices: KernelMatrix
using MyUtils: CartesianMesh, plot_heatmap_from_grid
using Statistics

function EnsembleFilters.assimilate_data(prior::EnsembleKalmanFilter, obs_filter::EnsembleKalmanFilter, y_obs, job_dir, step_index; params, save_update=true)
    ensemble = prior.ensemble
    obs_ensemble = obs_filter.ensemble

    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)

    println("Beginning to assimilate data...")

    N = length(ensemble)

    println("  - Creating operators...")

    if params["filter"]["make_observation_square"]
        ny = size(obs_ensemble[1].state[1])
        start = ny[2] - ny[1] + 1

        y_obs = y_obs[:, start:end]
        obs_ensemble = [(; state = (yi.state[1][:, start:end], yi.state[2][:, start:end])) for yi in obs_ensemble]

        ny = size(obs_ensemble[1].state[1])
        @assert ny[1] == ny[2]
    end

    # Create observation covariance matrix.
    ny = length(obs_ensemble[1].state[1])
    obs_std = params["filter"]["observation_noise_stddev"]
    if params["filter"]["observation_noise_type"] == "diagonal"
        R = Diagonal(fill(Float64(obs_std)^2, ny))
        R_op = joMatrix(R)
    elseif params["filter"]["observation_noise_type"] == "kernel_matrix"
        function kernfun(x1, x2, parms)
            scale = parms[1]
            dx, dy = parms[2:3]
            out = scale * exp((x1[1] - x2[1])^2 / (2 * dx^2) + (x1[2] - x2[2])^2 / (2 * dy^2))
            if x1 == x2
                out += 1.0e-12
            end
            return out
        end
        function get_coords_matrix(xs, ys)
            # https://stackoverflow.com/questions/43866322/generating-rectilinear-grid-coordinates-in-julia
            lx, ly = length(xs), length(ys)
            res = Array{Base.promote_eltype(xs, ys), 2}(undef, lx*ly, 2)
            ind = 1
            for y in ys, x in xs
                res[ind, 1] = x
                res[ind, 2] = y
                ind += 1
            end
            res
        end
        obs_length = (60, 12.5)
        obs_length = params["filter"]["observation_noise_length"]
        # kernprms = SVector{2, Float64}(obs_std, obs_length)
        d = params["observation"]["d"]
        coords = [[x, y] .* d for x in 0:ny[1]-1 for y in 0:ny[2]-1]
        kernprms = (; scale = params.observation_noise_stddev, length = noise_length)
        kernprms = (obs_std, obs_length...)
        K = KernelMatrix(coords, coords, kernprms, kernfun)
        R_op = joMatrix(full(K))
    elseif params["filter"]["observation_noise_type"] == "joli_stencil"
        x_stencil_radius = 5
        y_stencil_radius = 2
        function make_stencil_1d(f, r)
            # f(0) > 0 and f(1) = 0
            stencil = zeros(2*r+1)
            stencil[r+1:end] = f.(collect(0:r) ./ (r + 1))
            stencil[1:r] = stencil[r+2:end]
            return stencil
        end
        linear_stencil(t) = 1 - t
        quadratic_stencil(t) = 1 - t^2
        quartic_stencil(t) = (1 - t^2)^2
        x_stencil = make_stencil_1d(linear_stencil, x_stencil_radius)
        y_stencil = make_stencil_1d(linear_stencil, y_stencil_radius)
        full_stencil = x_stencil .* y_stencil'
        @show size(full_stencil)
        function fop(v)
            v = reshape(v, ny)
            out = similar(v)

            srx, sry = x_stencil_radius, y_stencil_radius

            # Do interior
            for x_idx in 1:ny[1]
                for y_idx in 1:ny[2]
                    total = zero(eltype(v))
                    for sx in 1:srx
                        for sy in 1:sry
                            mx, my = x_idx - srx + sx - 1, y_idx - sry + sy - 1
                            if 1 <= mx <= ny[1] && 1 <= my <= ny[2]
                                total += v[mx, my] * full_stencil[sx, sy]
                            end
                        end
                    end
                    out[x_idx,  y_idx] = total
                end
            end
            return vec(out)
        end
        make_op(name, m, n, fop, T) = joLinearFunction{T,T}(name, m, n, fop,
            @joNF, @joNF, @joNF, false, @joNF, @joNF, @joNF, @joNF, false,
        )
        R_op = obs_std * make_op("R", ny, ny, fop, Float64)
    end


    yvec = hcat([vec(yi.state[1]) for yi in obs_ensemble]...)
    yvec_noisy = hcat([vec(yi.state[2]) for yi in obs_ensemble]...)
    ymean = mean(yvec, dims=2)
    if params["filter"]["include_noise_in_y_covariance"]
        dY = yvec_noisy .- ymean
    else
        dY = yvec .- ymean
    end
    dY_op = joMatrix(Float64.(dY))

    deltas = compute_deltas(ensemble; params)
    dX_op = joMatrix(Float64.(deltas.dX))

    # dX and dY are typically divided by sqrt(N - 1), but I prefer moving that to R.
    # (dX dY' / a) ((dY' dY / a) + R)^{-1} == dX dY' (dY' dY + a * R)^{-1}
    obs_covariance = dY_op * dY_op' + (N - 1) * R_op
    cross_covariance = dX_op * dY_op'
    println("  - Applying operators...")

    jo_iterative_solver4square_set((A, v) -> cg(A, v))

    pred_err = Float64.(vec(y_obs) .- yvec)
    if params["filter"]["make_assimilation_figures"]
        # Plot the situation before assimilating.
        work_dir = get_filter_work_dir(params)
        save_dir = joinpath(job_dir, "figs", work_dir, "assimilation")
        mkpath(save_dir)

        ny = size(obs_ensemble[1].state[1])
        d = Tuple(params["observation"]["d"])
	colormap = :balance

        # Use mesh in kilometers instead of meters.
        grid = CartesianMesh(ny, d .* ny ./ 1000.0)

        mean_y = reshape(mean(yvec, dims=2), ny)
        fig, ax = plot_heatmap_from_grid(mean_y, grid; colormap, make_divergent=true)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        ax.title = "mean obs"
        file_path = joinpath(save_dir, "mean_obs_$(step_index).png")
        save(file_path, fig)

        mean_y_noisy = reshape(mean(yvec_noisy, dims=2), ny)
        fig, ax = plot_heatmap_from_grid(mean_y_noisy, grid; colormap, make_divergent=true)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        ax.title = "mean obs noisy"
        file_path = joinpath(save_dir, "mean_obs_noisy_$(step_index).png")
        save(file_path, fig)

        mean_err = reshape(mean(pred_err, dims=2), ny)
        fig, ax = plot_heatmap_from_grid(mean_err, grid; colormap, make_divergent=true)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        ax.title = "mean error"
        file_path = joinpath(save_dir, "mean_obs_error_$(step_index).png")
        save(file_path, fig)

        n = Tuple(Int.(params["transition"]["n"][1:2:3]))
        mean_update = reshape(cross_covariance * (obs_covariance \ vec(mean_err)), n)
        @show grid
        fig, ax = plot_heatmap_from_grid(mean_update, grid; colormap, make_divergent=true, make_heatmap=true)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        ax.title = "mean update"
        file_path = joinpath(save_dir, "mean_update_$(step_index).png")
        save(file_path, fig)

        fig, ax = plot_heatmap_from_grid(mean_update, grid; colormap, colorrange=(-1, 1), make_heatmap=true)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        ax.title = "mean update"
        file_path = joinpath(save_dir, "mean_update_clamped_$(step_index).png")
        save(file_path, fig)
    end

    posterior_states = Vector{eltype(ensemble)}(undef, length(ensemble))
    if params["filter"]["print_progress"]
        for (i, (em, pred_err_i)) in enumerate(zip(ensemble, eachcol(pred_err)))
            println("      - Updating member $(i)")
            @time ensemble_contributions_i = dY' * (obs_covariance \ pred_err_i)
            @time posterior_states[i] = update_ensemble_member!(em, ensemble_contributions_i, deltas; params)
        end
    else
        @time ensemble_contributions = dY' * (obs_covariance \ pred_err)
        println("  - Updating ensemble members...")

        # Update ensemble members.
        for (i, (em, ensemble_contributions_i)) in enumerate(zip(ensemble, eachcol(ensemble_contributions)))
            posterior_states[i] = update_ensemble_member!(em, ensemble_contributions_i, deltas; params)
        end
    end

    if save_update
        println("  - Creating posterior filter object")
        posterior = EnsembleKalmanFilter(posterior_states, prior.params, prior.work_dir)
        println("Finished assimilating data.")
        filepath = joinpath(work_path, "filter_$(step_index)_state_update")
        save_filter(filepath, posterior)
    else
        posterior = EnsembleKalmanFilter(ensemble, prior.params, prior.work_dir)
    end
    return posterior
end

function compute_delta(ensemble, getter)
    dX = hcat([vec(getter(em)) for em in ensemble]...)
    xmean = mean(dX, dims=2)
    dX .-= xmean
    return dX
end

function compute_deltas(ensemble; params)
    dX = compute_delta(ensemble, em -> em.state)
    deltas = (;
        dX = dX,
    )
    if params["filter"]["update_permeability"]
        dK = compute_delta(ensemble, em -> get_permeability(em.params.M))
        deltas = (;
            deltas...,
            dK = dK,
        )
    end
    return deltas
end

function update_ensemble_member!(em, ensemble_contributions_i, deltas; params)
    max_change = params["filter"]["max_saturation_update_size"]
    update_i = deltas.dX * ensemble_contributions_i
    if max_change >= 0
        clamp!(update_i, -max_change, max_change)
    end
    em.state .+= reshape(update_i, size(em.state))
    saturation_range = Tuple(params["filter"]["saturation_range"])
    clamp!(em.state, saturation_range...)

    state = Dict{Symbol, Any}()
    state[:Saturation] = em.state

    if params["filter"]["update_permeability"]
        max_change = params["filter"]["max_permeability_update_size"]
        permeability_range = Tuple(params["filter"]["permeability_range"])
        scale = params["filter"]["permeability_update_scale"]

        update_i = deltas.dK * (scale .* ensemble_contributions_i)
        if max_change >= 0
            clamp!(update_i, -max_change, max_change)
        end
        K = get_permeability(em.params.M) .+ update_i
        clamp!(K, permeability_range...)
        set_permeability!(em.params.M, K)
        state[:Permeability] = K
    end
    em = typeof(em)(state, nothing, nothing)
    return em
end
