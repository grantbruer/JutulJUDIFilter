import Flux
import CairoMakie
import ChainRulesCore
import EnsembleFilters
import Optim: Flat, Manifold, project_tangent!, retract!, BFGS, LBFGS, GradientDescent, Options, optimize
# import SetIntersectionProjection

using LinearAlgebra
using CairoMakie
using MyUtils
using Printf
using Statistics
using JLD2
using .SeismicPlumeEnsembleFilter
using .SlimOptim: pqn_options, pqn

function EnsembleFilters.assimilate_data(prior::EnsembleJustObsFilter, obs_filter::EnsembleJustObsFilter, y_obs, job_dir, step_index; params, save_update=true)
    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)

    # Build observer.
    observation_type = :born
    M, M0 = SeismicModel_params(params; observation_type)

    # Patchy model still uses unblurred velocity and density.
    phi = clamp.(M.phi, 0, 1)
    # phi = M.phi
    P = PatchyModel(M.vel, M.rho, phi; params)

    # Until we do JRM, we need to use the unblurred velocity and density here, too.
    observer = SeismicCO2Observer(M, P)
    observer0 = SeismicCO2Observer(M0, P)

    # Invert observations.
    x0 = mean(em.state for em in prior.ensemble)
    get_mask = em -> (
        (!hasfield(typeof(em.params), :update_mask) || isnothing(em.params.update_mask))
        ? trues(size(x0))
        : em.params.update_mask
    )
    mask = reduce(.|, get_mask(em) for em in prior.ensemble)
    x = invert_observation(observer, y_obs, x0; params, job_dir, step_index, work_dir, mask)
    println("Finished inverting data.")


    println("  - Creating posterior filter object")
    if save_update
        posterior_states = [eltype(prior.ensemble)(x, nothing, nothing) for em in prior.ensemble]
        posterior = EnsembleJustObsFilter(posterior_states, prior.params, prior.work_dir)
        println("Finished assimilating data.")

        filepath = joinpath(work_path, "filter_$(step_index)_state_update")
        save_filter(filepath, posterior)
    else
        ensemble = prior.ensemble
        for (i, em) in enumerate(ensemble)
            em.state .= x
        end
        posterior = EnsembleJustObsFilter(ensemble, prior.params, prior.work_dir)
    end
    return posterior
end

struct BoxManifold <: Manifold
    lower
    upper
end

function project_tangent!(M::BoxManifold, g, x)
    too_big = (x .>= M.upper) .&& (g .> 0)
    too_low = (x .<= M.lower) .&& (g .< 0)
    g .= ifelse.(too_big .|| too_low, 0, g)
    return g
end

function retract!(M::BoxManifold, x)
    x .= clamp.(x, M.lower, M.upper)
    return x
end

struct MaskManifold <: Manifold
    mask
    x0
end

function project_tangent!(M::MaskManifold, g, x)
    g[M.mask] .= 0
    return g
end

function retract!(M::MaskManifold, x)
    x = reshape(x, size(M.mask))
    x[M.mask] .= M.x0[M.mask]
    return x
end

struct ConnectedNonzerosManifold <: Manifold
    mask
    root_ci
    threshold
end

function ConnectedNonzerosManifold(params)
    d = Tuple(params["transition"]["d"])
    n = Tuple(Int.(params["transition"]["n"]))
    inj_loc = Tuple(params["transition"]["injection"]["loc"])
    inj_idx = round.(Int, inj_loc ./ d[1:2])
    inj_search_zrange = Tuple(params["transition"]["injection"]["search_zrange"])
    inj_search_z_idx = range(round.(Int, inj_search_zrange ./ d[3] .- (0, 1))...)
    inj_ci = collect(CartesianIndex(inj_idx[1], z_idx) for z_idx in inj_search_z_idx)
    mask = zeros(Bool, n[1:2:3])
    threshold = params["filter"]["optimization"]["nonzero_threshold"]
    return ConnectedNonzerosManifold(mask, inj_ci, threshold)
end

function project_tangent!(M::ConnectedNonzerosManifold, g, x)
    g[(.!M.mask) .&& (g .> 0)] .= 0
    return g
end

function retract!(M::ConnectedNonzerosManifold, x)
    x = reshape(x, size(M.mask))
    x[x .< M.threshold] .= 0
    M.mask .= get_connected_mask(x, M.root_ci)
    x[.!M.mask] .= 0
    return x
end

function l1_norm(x)
    return sum(abs.(x))
end

function l2_norm(x)
    return sum(x.^2)
end

function hybrid_l1_l2_norm(x)
    return sum(sqrt.(x.^2 .+ 1) .- 1)
end

function get_norm(norm_type)
    if norm_type == "l1"
        return l1_norm
    elseif norm_type == "l2"
        return l2_norm
    elseif norm_type == "hybrid_l1_l2"
        return hybrid_l1_l2_norm
    elseif norm_type == "none"
        return x -> 0.0
    end
    error("Invalid norm type: '$(norm_type)'")
end

function get_manifold(constraints_type; params)
    if constraints_type == "box"
        r = params["filter"]["optimization"]["box_range"]
        manifold = BoxManifold(r...)
    elseif constraints_type == "connected"
        manifold = ConnectedNonzerosManifold(params)
    elseif constraints_type == "none"
        manifold = Flat()
    else
        error("Invalid contraints type: '$(constraints_type)'")
    end
end

function invert_observation(observer, y_obs, x0; params, kwargs...)
    package = params["filter"]["optimization"]["package"]
    if package == "Optim"
        return invert_observation_optim(observer, y_obs, x0; params, kwargs...)
    elseif package == "SlimOptim"
        return invert_observation_slimoptim(observer, y_obs, x0; params, kwargs...)
    end
    error("Invalid filter optimization package: '$(package)'")
end

get_horizontal_spacing(observer) = observer.P.d[1]
get_vertical_spacing(observer) = observer.P.d[2]

function invert_observation_slimoptim(observer, y_obs, x0; params, job_dir, step_index, work_dir, mask=nothing, restart=true)
    # The scale factor in the l2 norm is the root mean squared deviation.
    # The scale factor in the l1 norm is the mean absolute deviation.
    # They differ by about a factor of 2, so don't worry too much about it.
    n = size(x0)

    observation_noise_stddev = params["filter"]["observation_noise_stddev"]
    observer_full = x -> observer(x)
    function misfit_func(x)
        x = reshape(x, size(x0))
        obs = observer_full(x)
        J = 0.5 * norm(obs - y_obs)^2 / (observation_noise_stddev^2)
        return J
    end

    horizontal_gradient_norm_type = params["filter"]["optimization"]["horizontal_gradient_norm_type"]
    if horizontal_gradient_norm_type == "none"
        hreg_func = x -> 0.0
    else
        horizontal_gradient_norm = get_norm(horizontal_gradient_norm_type)
        horizontal_gradient_scale = params["filter"]["optimization"]["horizontal_gradient_scale"]
        hstep = get_horizontal_spacing(observer)
        hreg_func = function (x)
            x = reshape(x, size(x0))
            grad_horizontal = (x[2:end, :] .- x[1:end-1, :]) ./ hstep
            J = horizontal_gradient_norm(grad_horizontal ./ horizontal_gradient_scale)
            return J
        end
    end

    vertical_gradient_norm_type = params["filter"]["optimization"]["vertical_gradient_norm_type"]
    if vertical_gradient_norm_type == "none"
        vreg_func = x -> 0.0
    else
        vertical_gradient_norm = get_norm(vertical_gradient_norm_type)
        vertical_gradient_scale = params["filter"]["optimization"]["vertical_gradient_scale"]
        vstep = get_vertical_spacing(observer)
        vreg_func = function (x)
            x = reshape(x, size(x0))
            grad_vertical = (x[:, 2:end] .- x[:, 1:end-1]) ./ vstep
            J = vertical_gradient_norm(grad_vertical ./ vertical_gradient_scale)
            return J
        end
    end

    inject_connect_norm_type = params["filter"]["optimization"]["inject_connect_norm_type"]
    if inject_connect_norm_type == "none"
        conn_func = x -> 0.0
    else
        inject_connect_norm = get_norm(inject_connect_norm_type)
        conn_M = ConnectedNonzerosManifold(params)
        conn_func = function(x)
            x = reshape(x, size(x0))
            mask = ChainRulesCore.ignore_derivatives() do
                # Get a mask for all elements that are connected to the injection location.
                mask = get_connected_mask(x, conn_M.root_ci)
            end
            # Penalize all saturations not connected to the injection location.
            J = inject_connect_norm(x[.!mask] ./ inject_connect_scale)
            return J
        end
    end

    function objective(x)
        misfit = misfit_func(x)
        hreg = hreg_func(x)
        vreg = vreg_func(x)
        conn = conn_func(x)
        ChainRulesCore.ignore_derivatives() do
            @printf("misfit: %10.6g, hreg: %10.6g, vreg: %10.6g, conn: %10.6g\n", misfit, hreg, vreg, conn)
            @show extrema(x)
        end
        J = misfit + hreg + vreg + conn
        return J
    end

    save_dir_training_data = joinpath(job_dir, work_dir, "training_data")
    save_dir_training_states = joinpath(job_dir, work_dir, "training_states_$(step_index)")
    file_path_opt_info = joinpath(save_dir_training_data, "opt_info_$(step_index).jld2")
    mkpath(save_dir_training_data)
    mkpath(save_dir_training_states)

    restarted = false
    opt_info = nothing
    if restart && isfile(file_path_opt_info)
        opt_info = load(file_path_opt_info, "info")
        iter = length(opt_info)
        iter_str = @sprintf("%04d", iter)
        file_path_training_states = joinpath(save_dir_training_states, "state_$(iter_str).jld2")
        x0 .= reshape(load(file_path_training_states, "state"), size(x0))
	restarted = true
    end

    if params["filter"]["make_assimilation_figures"]
        save_dir_training_fig = joinpath(job_dir, "figs", work_dir, "assimilation", "training")
        mkpath(save_dir_training_fig)

        save_dir_fig = joinpath(job_dir, "figs", work_dir, "assimilation")
        mkpath(save_dir_fig)

        n = Tuple(params["observation"]["n"])
        d = Tuple(params["observation"]["d"])

        # Use mesh in kilometers instead of meters.
        grid = CartesianMesh(n, d .* n ./ 1000.0)
    end

    if params["filter"]["make_assimilation_figures"] && !restarted
        let
            # Plot shot record.
            extras = (; params, grid, colormap=:balance, divergent=true)
            x0_obs = observer_full(x0)
            file_path = joinpath(save_dir_fig, "x0_shot_$(step_index).mp4")
            plot_anim([Dict(:obs=>x0_obs, :step=>step_index)], s->s[:obs].data, anim_shot_record_plotter, file_path; extras, framerate=2)
        end
    end

    constraints_type = params["filter"]["optimization"]["constraints"]
    if ! isa(constraints_type, Array)
        constraints_type = [constraints_type]
    end
    manifolds = Vector{Manifold}()
    append!(manifolds, [get_manifold(c; params) for c in constraints_type])
    if ! isnothing(mask)
        push!(manifolds, MaskManifold(.!mask, x0))
    end

    # Build method options and optimization options.
    method_type = params["filter"]["optimization"]["method"]
    method_kwargs = (; Dict(Symbol(k) => v for (k,v) in params["filter"]["optimization"]["method_kwargs"])...)

    if method_type == "PQN"
        options = pqn_options(;method_kwargs...)
    else
        error("Invalid method type: '$(method_type)'")
    end

    function objective_grad(X)
        X = reshape(X, n)
        J, dJdx = Flux.withgradient(objective, X)
        dJdx = vec(dJdx[1])
        return J, dJdx
    end

    function project_x(X::T) where T
        for manifold in manifolds
            retract!(manifold, X)
        end
        return X
    end

    function get_callback(x0, opt_info=nothing)
        opt_info = isnothing(opt_info) ? [] : opt_info
        old_x = deepcopy(vec(x0))
        function callback(res)
            x = res.x
            misfit, misfit_grad = Flux.withgradient(misfit_func, x)
            hreg, hreg_grad = Flux.withgradient(hreg_func, x)
            vreg, vreg_grad = Flux.withgradient(vreg_func, x)
            conn, conn_grad = Flux.withgradient(conn_func, x)
            misfit_grad = misfit_grad[1]
            hreg_grad = hreg_grad[1]
            vreg_grad = vreg_grad[1]
            conn_grad = conn_grad[1]
            hreg_grad = isnothing(hreg_grad) ? zeros(size(x)) : hreg_grad
            vreg_grad = isnothing(vreg_grad) ? zeros(size(x)) : vreg_grad
            conn_grad = isnothing(conn_grad) ? zeros(size(x)) : conn_grad
            project_grad = function (x, g) return project_x(vec(x) - vec(g)) - vec(x) end
            norm_proj = function (x, g) return norm(project_grad(x, g)) end
            info = Dict(
                :J => res.ϕ,
                :dJdx_norm => norm(res.g),
                :dJdx_norm_proj => norm_proj(res.x, res.g),
                :optCond_step => norm(vec(res.x) - vec(old_x)),
                :misfit => misfit,
                :dmisfit_dx_norm => norm(misfit_grad),
                :dmisfit_dx_norm_proj => norm_proj(x, misfit_grad),
                :hreg => hreg,
                :dhreg_dx_norm => norm(hreg_grad),
                :dhreg_dx_norm_proj => norm_proj(x, hreg_grad),
                :vreg => vreg,
                :dvreg_norm => norm(vreg_grad),
                :dvreg_norm_proj => norm_proj(x, vreg_grad),
                :conn => conn,
                :dconn_norm => norm(conn_grad),
                :dconn_norm_proj => norm_proj(x, conn_grad),
            )
            push!(opt_info, info)

            iter = length(opt_info)
            iter_str = @sprintf("%04d", iter)
            file_path_training_states = joinpath(save_dir_training_states, "state_$(iter_str).jld2")

            # Save to file.
            jldsave(file_path_opt_info; info=opt_info)
            jldsave(file_path_training_states; state=x, grad=res.g, misfit_grad, hreg_grad, vreg_grad, conn_grad)

            # TODO: stochasticity. The callback should also change the sources used in the observer.

            if params["filter"]["make_assimilation_figures"]
                n = Tuple(params["observation"]["n"])

                # 1D plots.
                xs = collect(0:(iter-1))
                fix_limits = function (ax, ys)
                    """If the values are too close together, Makie throws an error trying to add axis ticks."""
                    ymin, ymax = extrema(ys)
                    target_diff = 1e-6
                    yref = max(abs(ymin), abs(ymax), 1)
                    ydiff = ymax - ymin
                    if ydiff <= target_diff * yref
                        offset = (target_diff * yref - ydiff) / 2
                        ylims!(ax, ymin - offset, ymax + offset)
                    end
                end

                for s in keys(info)
                    ys = [info[s] for info in opt_info]
                    fig, ax, sc = scatterlines(xs, ys, markersize=10, linewidth=2)
                    fix_limits(ax, ys)
                    file_path = joinpath(save_dir_fig, "opt_$(s)_$(step_index).png")
                    save(file_path, fig)
                end

                # 2D plots.
                data = reshape(deepcopy(x), n)
                data[data .== 0] .= NaN
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=parula, colorrange=(0,1))
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_state_$(iter_str).png")
                save(file_path, fig)

                data = reshape(vec(x) - vec(old_x), n)
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=:balance, make_divergent=true)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_step_$(iter_str).png")
                save(file_path, fig)

                data = reshape(-res.g, n)
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=:balance, make_divergent=true)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_grad_$(iter_str).png")
                save(file_path, fig)

                data = reshape(project_grad(x, res.g), n)
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=:balance, make_divergent=true)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_grad_proj_$(iter_str).png")
                save(file_path, fig)

                data = reshape(-hreg_grad, n)
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=:balance, make_divergent=true)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_hreg_grad_$(iter_str).png")
                save(file_path, fig)

                data = reshape(project_grad(x, hreg_grad), n)
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=:balance, make_divergent=true)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_hreg_grad_proj_$(iter_str).png")
                save(file_path, fig)

                data = reshape(-vreg_grad, n)
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=:balance, make_divergent=true)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_vreg_grad_$(iter_str).png")
                save(file_path, fig)

                data = reshape(project_grad(x, vreg_grad), n)
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=:balance, make_divergent=true)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_vreg_grad_proj_$(iter_str).png")
                save(file_path, fig)

                data = reshape(-conn_grad, n)
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=:balance, make_divergent=true)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_conn_grad_$(iter_str).png")
                save(file_path, fig)

                data = reshape(project_grad(x, conn_grad), n)
                fig, ax = plot_heatmap_from_grid(data, grid; colormap=:balance, make_divergent=true)
                ax.xlabel = "Length (km)"
                ax.ylabel = "Depth (km)"
                ax.title = "Time $(step_index), iteration $(iter)"
                file_path = joinpath(save_dir_training_fig, "time_$(step_index)_conn_grad_proj_$(iter_str).png")
                save(file_path, fig)
            end
            old_x .= vec(res.x)
        end
        return callback, opt_info
    end

    callback, opt_info = get_callback(x0, opt_info)

    # set_verbosity(false)
    res = pqn(objective_grad, vec(deepcopy(x0)), project_x, options; callback=callback)

    objective(res.x)
    callback(res)

    x = reshape(res.x, size(x0))
    return x
end

function invert_observation_optim(observer, y_obs, x0; params)
    error("deprecated")

    # The scale factor in the l2 norm is the root mean squared deviation.
    # The scale factor in the l1 norm is the mean absolute deviation.
    # They differ by about a factor of 1.4, so don't worry too much about it.
    horizontal_gradient_norm = get_norm(params["filter"]["optimization"]["horizontal_gradient_norm_type"])
    horizontal_gradient_scale = params["filter"]["optimization"]["horizontal_gradient_scale"]
    vertical_gradient_norm = get_norm(params["filter"]["optimization"]["vertical_gradient_norm_type"])
    vertical_gradient_scale = params["filter"]["optimization"]["vertical_gradient_scale"]

    observation_noise_stddev = params["filter"]["observation_noise_stddev"]
    function objective(x)
        obs = observer(x)
        misfit = 0.5 * norm(obs - y_obs)^2 / (observation_noise_stddev^2)
        grad_horizontal = (x[2:end, :] - x[1:end-1, :]) / observer.P.d[1]
        grad_vertical = (x[:, 2:end] - x[:, 1:end-1]) / observer.P.d[2]
        hreg = horizontal_gradient_norm(grad_horizontal ./ horizontal_gradient_scale)
        vreg = vertical_gradient_norm(grad_vertical ./ vertical_gradient_scale)
        @printf("misfit: %10.6g, hreg: %10.6g, vreg: %10.6g\n", misfit, hreg, vreg)
        J = misfit + hreg + vreg
        return J
    end

    function objective_gradient!(G, x)
        grad = Flux.gradient(objective, x)
        G .= grad[1]
        return G
    end

    # Build constraints
    constraints_type = params["filter"]["optimization"]["constraints"]
    if constraints_type == "box"
        manifold = BoxManifold(0, 1)
    elseif constraints_type == "none"
        manifold = Flat()
    else
        error("Invalid contraints type: '$(constraints_type)'")
    end

    # Build method options and optimization options.
    method_type = params["filter"]["optimization"]["method"]
    method_kwargs = (; Dict(Symbol(k) => v for (k,v) in params["filter"]["optimization"]["method_kwargs"])...)
    opt_kwargs = (; Dict(Symbol(k) => v for (k,v) in params["filter"]["optimization"]["opt_kwargs"])...)
    if method_type == "BFGS"
        method = BFGS(;
            manifold,
            method_kwargs...
        )
    elseif method_type == "L-BFGS"
        method = LBFGS(;
            manifold,
            method_kwargs...
        )
    elseif method_type == "gradient_descent"
        method = GradientDescent(;
            manifold,
            method_kwargs...
        )
    else
        error("Invalid method type: '$(method_type)'")
    end
    options = Options(;
        opt_kwargs...
    )

    # Run optimization.
    @time s = optimize(objective, objective_gradient!, x0, method, options)
    observer(s.minimizer)
    x = reshape(s.minimizer, n)
    return x
end

