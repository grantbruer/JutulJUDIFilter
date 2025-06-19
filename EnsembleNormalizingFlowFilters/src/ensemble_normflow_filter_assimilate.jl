import EnsembleFilters

using BSON
using Distributed
using DrWatson
using Images
using JLD2
using LogExpFunctions
using PyCall
using PyPlot
using Random
using Statistics
using MyUtils

using .Flux
using .InvertibleNetworks
using .UNet
using .MLUtils

function get_device(device_type)
    if device_type == "gpu"
        return gpu
    end
    error("Invalid device type: '$(device_type)'")
end

function initialize_network(; params)
    network_params = params["filter"]["network"]
    device = get_device(network_params["device_type"])

    chan_x = network_params["chan_x"]
    chan_y = network_params["chan_y"]
    L = network_params["L"]
    K = network_params["K"]
    n_hidden = network_params["n_hidden"]
    n_c = network_params["n_c"]
    n_in = network_params["n_in"]
    sum_net = network_params["sum_net"]
    unet_lev = network_params["unet_lev"]

    h2 = nothing
    if sum_net
        h2 = Chain(Unet(n_c,n_in,unet_lev))
        trainmode!(h2, true)
        h2 = FluxBlock(h2)|> device
    end

    # Network and input
    flow = NetworkConditionalGlow(chan_x, chan_y, n_hidden, L, K; split_scales=true, ndims=2)
    G = SummarizedNet(flow, h2)|> device
end

function EnsembleFilters.assimilate_data(prior::EnsembleNormFlowFilter, obs_filter::EnsembleNormFlowFilter, y_obs, job_dir, step_index; params, save_update=true)
    network_params = params["filter"]["network"]
    ensemble = prior.ensemble
    obs_ensemble = obs_filter.ensemble

    work_dir = get_filter_work_dir(params)
    work_path = joinpath(job_dir, work_dir)

    # # Read test data.
    # gt_step = step_index * params["filter"]["update_interval"]
    # baseline, extra_data = read_ground_truth_seismic_baseline(stem; state_keys = [:rtm_born, :rtm_born_noisy])
    # gt_plume, extra_data = read_ground_truth_plume(params, job_dir, gt_step)
    # gt_seismic, extra_data = read_ground_truth_seismic(params, job_dir, gt_step)

    data_preprocessing_type = params["filter"]["data_preprocessing"]
    if data_preprocessing_type == "standard_normalizing_flow"
        n_total = 250
        nx = 320
        ny = 320
        N = (nx, ny)
        preprocess_x = x -> imresize(x[1:320, 22:end], N)
        postprocess_x = (x_post, target) -> (target[1:320, 22:end] .= x_post).parent
        preprocess_y = y -> y[1:320, 22:end]

        # Training data setup.
        Xs = zeros(Float32, nx, ny, 1, n_total);
        for (i, em) in enumerate(ensemble[1:n_total])
            Xs[:,:,:,i] = preprocess_x(em.state);
        end
        Ys = zeros(Float32, nx, ny, 1, n_total);
        for (i, em) in enumerate(obs_ensemble[1:n_total])
            Ys[:,:,:,i] = preprocess_y(em.state[2]) # TODO: This is supposed to be the noisy data, right?
        end

        # # Test data setup.
        # Xstest = zeros(Float32, nx, ny, 1, 1)
        # Xstest[:, :, :, 1] = imresize(gt_plume[1:320, 22:end], N);
        # Ystest = zeros(Float32, nx, ny, 1, 1)
        # Ystest[:, :, :, 1] = gt_seismic[:rtm_born_noisy][1:320, 22:end] .- baseline[:rtm_born_noisy]

        # Scale the training and test data.
        # rtm_min_val = minimum([minimum(Ys) minimum(Ystest)])
        # rtm_max_val = maximum([maximum(Ys) maximum(Ystest)])
        rtm_min_val, rtm_max_val = (0, 1e15) # extrema(Ys)
        @show rtm_min_val rtm_max_val
        preprocess_y_more = y -> (y .- rtm_min_val) ./ (rtm_max_val - rtm_min_val)
        # Xs = permutedims(Xs, (4, 1, 2, 3))
        # Ys = permutedims(Ys, (4, 1, 2, 3))
        for i in 1:n_total
            Ys[:,:,:,i] = preprocess_y_more(Ys[:,:,:,i])
        end
        @show extrema(Ys)
        # for i in 1:n_total
        #     Ystest[:,:,:,i] = preprocess_y_more(Ystest[i,:,:,:])
        # end

    else
        error("Unknown data preprocessing type: '$(data_preprocessing_type)'")
    end

    @show extrema(y_obs)
    y_processed = preprocess_y_more(preprocess_y(y_obs))
    @show extrema(y_processed)

    if params["filter"]["make_assimilation_figures"]
        # Plot the situation before assimilating.
        work_dir = get_filter_work_dir(params)
        save_dir = joinpath(job_dir, "figs", work_dir, "assimilation")
        mkpath(save_dir)

        n = size(Ys[:, :, 1, 1])
        d = Tuple(params["observation"]["d"])

        # Use mesh in kilometers instead of meters.
        grid = CartesianMesh(n, d .* n ./ 1000.0)

        # Plot rtm.
        colormap = :balance
        mean_y = reshape(mean(Ys, dims=4), n)
        fig, ax = plot_heatmap_from_grid(mean_y, grid; colormap, make_divergent=true)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        ax.title = "mean obs"
        file_path = joinpath(save_dir, "mean_obs_$(step_index).png")
        save(file_path, fig)

        try
            mean_err = mean_y - y_processed
            fig, ax = plot_heatmap_from_grid(mean_err, grid; colormap, make_divergent=true)
            ax.xlabel = "Length (km)"
            ax.ylabel = "Depth (km)"
            ax.title = "mean error"
            file_path = joinpath(save_dir, "mean_obs_error_$(step_index).png")
            save(file_path, fig)
        catch e
            println(e)
        end
    end


    println("Initializing network...")
    G = initialize_network(; params)

    println("Training network...")
    G = train_network(G; Xs, Ys, tl_step = step_index, rtm_min_val, rtm_max_val, params, job_dir)

    y_network = reshape(y_processed, nx, ny, 1, 1)
    device = get_device(network_params["device_type"])
    X_post = posterior_sampler(G,  y_network, size(y_network); device, num_samples=256)
    X_post = X_post |> cpu
    @show extrema(X_post)

    if params["filter"]["make_assimilation_figures"]
        # Plot posterior mean.
        colormap = parula
        mean_x = reshape(mean(X_post, dims=4), n)
        fig, ax = plot_heatmap_from_grid(mean_x, grid; colormap)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        ax.title = "mean posterior"
        file_path = joinpath(save_dir, "mean_posterior_$(step_index).png")
        save(file_path, fig)
    end

    X_post[X_post .< 0.0] .= 0.0;
    X_post[X_post .> 0.9] .= 0.9;
    if params["filter"]["make_assimilation_figures"]
        # Plot posterior mean.
        colormap = parula
        mean_x = reshape(mean(X_post, dims=4), n)
        fig, ax = plot_heatmap_from_grid(mean_x, grid; colormap)
        ax.xlabel = "Length (km)"
        ax.ylabel = "Depth (km)"
        ax.title = "mean posterior"
        file_path = joinpath(save_dir, "mean_posterior_clamped_$(step_index).png")
        save(file_path, fig)
    end


    # Save ensemble.
    println("  - Updating ensemble members...")
    ensemble = prior.ensemble
    # X_post_train = zeros(Float32, size_x[1:end-1]..., num_samples)
    for (i, em) in enumerate(ensemble)
        println("      - Updating member $(i)")
        postprocess_x(X_post[:, :, 1, i], em.state)
    end


    if save_update
        println("  - Creating posterior filter object")
        posterior_states = [typeof(em)(em.state, nothing, nothing) for em in ensemble]
        posterior = EnsembleNormFlowFilter(posterior_states, prior.params, prior.work_dir)
        filepath = joinpath(work_path, "filter_$(step_index)_state_update")
        save_filter(filepath, posterior)
    else
        posterior = EnsembleNormFlowFilter(ensemble, prior.params, prior.work_dir)
    end
    return posterior
end

function train_network(G;  Xs, Ys, tl_step, rtm_min_val, rtm_max_val, params, job_dir)
    training_params = params["filter"]["training"]
    network_params = params["filter"]["network"]

    # Use MLutils to split into training and validation set
    validation_perc = 0.96
    (X_train, Y_train), (X_test, Y_test) = splitobs((Xs, Ys); at=validation_perc, shuffle=true);

    lr = training_params["lr"]
    clipnorm_val = training_params["clipnorm_val"]
    n_epochs = training_params["n_epochs"]
    batch_size = training_params["batch_size"]
    noise_lev_x = training_params["noise_lev_x"]
    noise_lev_y = training_params["noise_lev_y"]
    low = training_params["low"]
    num_post_samples = training_params["num_post_samples"]
    save_every = training_params["save_every"]
    plot_every = training_params["plot_every"]
    n_condmean = training_params["n_condmean"]
    noisy = training_params["noisy"]
    vmin = training_params["vmin"]
    vmax = training_params["vmax"]

    chan_x = network_params["chan_x"]
    chan_y = network_params["chan_y"]
    L = network_params["L"]
    K = network_params["K"]
    n_hidden = network_params["n_hidden"]
    n_c = network_params["n_c"]
    n_in = network_params["n_in"]
    sum_net = network_params["sum_net"]
    unet_lev = network_params["unet_lev"]


    device = get_device(network_params["device_type"])
    n_train = size(X_train)[end]
    n_test = size(X_test)[end]
    n_batches = cld(n_train, batch_size)
    N = size(X_train)[1:2]
    nx, ny = N

    # Set up save information.
    work_dir = get_filter_work_dir(params)
    save_dir_fig = joinpath(job_dir, "figs", work_dir, "assimilation", "training_$(tl_step)")
    save_dir_data = joinpath(job_dir, work_dir, "training_data_$(tl_step)")
    mkpath(save_dir_fig)
    save_training_dict = @strdict nx clipnorm_val noise_lev_x n_train lr batch_size tl_step
    save_network_dict = @strdict n_c unet_lev sum_net n_hidden L K

    # Create optimizer.
    opt = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr))

    # Create batch logs.
    batch_loss   = [];
    batch_logdet = [];

    # Create epoch logs.
    ssim   = [];
    l2_cm  = [];

    loss_test   = [];
    logdet_test = [];
    ssim_test   = [];
    l2_cm_test  = [];
    X_post = nothing

    font_size = 8
    PyPlot.rc("font", family="serif", size=font_size)
    PyPlot.rc("xtick", labelsize=font_size)
    PyPlot.rc("ytick", labelsize=font_size)
    PyPlot.rc("axes", labelsize=font_size) # fontsize of the x and y labels
    PyPlot.rc("text", usetex=false)

    # Run training.
    for e=1:n_epochs
        idx_e = reshape(randperm(n_train), batch_size, n_batches)
        for b = 1:n_batches
            @time begin
                X = X_train[:, :, :, idx_e[:,b]];
                Y = Y_train[:, :, :, idx_e[:,b]];
                X .+= noise_lev_x*randn(Float32, size(X));
                Y .+= noise_lev_y*randn(Float32, size(Y));

                # Forward pass of normalizing flow
                Zx, Zy, lgdet = G.forward(X |> device, Y |> device)

                # Loss function is l2 norm
                append!(batch_loss, norm(Zx)^2 / (prod(N)*batch_size))  # normalize by image size and batch size
                append!(batch_logdet, -lgdet / prod(N)) # logdet is internally normalized by batch size

                # Set gradients of flow and summary network
                G.backward(Zx / batch_size, Zx, Zy; Y_save=Y |> device)

                for p in InvertibleNetworks.get_params(G)
                  Flux.update!(opt,p.data,p.grad)
                end
                clear_grad!(G)

                print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches,
                    "; f l2 = ",  batch_loss[end],
                    "; lgdet = ", batch_logdet[end], "; f = ", batch_loss[end] + batch_logdet[end], "\n")
                Base.flush(Base.stdout)
            end
        end

        @time begin
            println("Getting metrics")
            # get objective mean metrics over testing batch
            @time l2_test_val, lgdet_test_val  = get_loss(G, X_test, Y_test; device, batch_size, N, noise_lev_x, noise_lev_y)
            append!(logdet_test, -lgdet_test_val)
            append!(loss_test, l2_test_val)

            # get conditional mean metrics over training batch
            @time cm_l2_train, cm_ssim_train = get_cm_l2_ssim(G, X_train[:,:,:,1:n_condmean], Y_train[:,:,:,1:n_condmean]; device, batch_size, num_samples=num_post_samples)
            append!(ssim, cm_ssim_train)
            append!(l2_cm, cm_l2_train)

            # get conditional mean metrics over testing batch
            @time cm_l2_test, cm_ssim_test = get_cm_l2_ssim(G, X_test[:,:,:,1:n_condmean], Y_test[:,:,:,1:n_condmean]; device, batch_size, num_samples=num_post_samples)
            append!(ssim_test, cm_ssim_test)
            append!(l2_cm_test, cm_l2_test)
        end
        if mod(e, plot_every) == 0
            testmode!(G.sum_net, true)
            save_dynamic_dict = merge((@strdict e), save_training_dict, save_network_dict)
            file_prefix = savename(save_dynamic_dict; digits=6)

            file_path = joinpath(save_dir_fig, file_prefix * "_train.png")
            plot_posterior_samples(G, X_train, Y_train, file_path; device, num_post_samples, params, rtm_min_val, rtm_max_val)

            file_path = joinpath(save_dir_fig, file_prefix * "_test.png")
            X_post = plot_posterior_samples(G, X_test, Y_test, file_path; device, num_post_samples, params, rtm_min_val, rtm_max_val)

            ############# Training metric logs
            sum_train = batch_loss + batch_logdet
            sum_test = loss_test + logdet_test

            fig = figure("training logs ", figsize=(10,12))
            subplot(5,1,1); title("L2 Term: train="*string(batch_loss[end])*" test="*string(loss_test[end]))
            plot(batch_loss, label="train");
            plot(n_batches:n_batches:n_batches*e, loss_test, label="test");
            axhline(y=1, color="red", linestyle="--", label="Normal Noise")
            ylim(bottom=0.,top=1.5)
            xlabel("Parameter Update"); legend();

            subplot(5,1,2); title("Logdet Term: train="*string(batch_logdet[end])*" test="*string(logdet_test[end]))
            plot(batch_logdet);
            plot(n_batches:n_batches:n_batches*e, logdet_test);
            xlabel("Parameter Update") ;

            subplot(5,1,3); title("Total Objective: train="*string(sum_train[end])*" test="*string(sum_test[end]))
            plot(sum_train);
            plot(n_batches:n_batches:n_batches*e, sum_test);
            xlabel("Parameter Update") ;

            subplot(5,1,4); title("SSIM train $(ssim[end]) test $(ssim_test[end])")
            plot(1:n_batches:n_batches*e, ssim);
            plot(1:n_batches:n_batches*e, ssim_test);
            xlabel("Parameter Update")

            subplot(5,1,5); title("l2 train $(l2_cm[end]) test $(l2_cm_test[end])")
            plot(1:n_batches:n_batches*e, l2_cm);
            plot(1:n_batches:n_batches*e, l2_cm_test);
            xlabel("Parameter Update")

            tight_layout()
            file_path = joinpath(save_dir_fig, file_prefix * "_log.png")
            safesave(file_path, fig)
            close(fig)
        end

        if e == n_epochs

        end

        if mod(e,save_every) == 0
            # Saving parameters and logs
            G_save = deepcopy(G);
            if sum_net
                reset!(G_save.sum_net); # clear params to not save twice
            end
            G_params = InvertibleNetworks.get_params(G_save) |> cpu;
            y = Y_test[:, :, :, 1:1] 
            X_post = posterior_sampler(G,  y, size(X_test[:, :, :, 1:1]); device, num_samples=num_post_samples)
            X_post = X_post |> cpu
            save_dynamic_dict = @strdict e n_in G_params batch_loss batch_logdet l2_cm ssim loss_test logdet_test l2_cm_test ssim_test X_post
            if sum_net
                unet_model = G.sum_net.model |> cpu;
                save_dynamic_dict = merge((@strdict unet_model), save_dynamic_dict)
            end

            save_dict = merge(save_dynamic_dict, save_training_dict, save_network_dict)
            file_name = savename(save_dict, "jld2"; digits=6)

            @tagsave(
                joinpath(save_dir_data, file_name),
                save_dict;
                safe=true
            );
            G = G |> device;
        end
    end
    return G
end

py"""
from scipy.signal import hilbert
import numpy as np
def normalize_std(mu, sigma):
    analytic_mu = hilbert(mu, axis=1)
    return sigma*np.abs(analytic_mu)/(np.abs(analytic_mu)**2 + 5e-1), analytic_mu
"""
normalize_std(mu, sigma) = py"normalize_std"(mu, sigma)

function posterior_sampler(G, y, size_x; device=gpu, num_samples=1, batch_size=16)
    # Make samples from posterior for train sample
    X_forward = randn(Float32, size_x[1:end-1]..., batch_size) |> device
    Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
    _, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat); # needs to set the proper sizes here

    X_post_train = zeros(Float32, size_x[1:end-1]..., num_samples)
    for i in 1:div(num_samples, batch_size)
        ZX_noise_i = randn(Float32, size_x[1:end-1]..., batch_size)|> device
        X_post_train[:,:,:, (i-1)*batch_size+1 : i*batch_size] = G.inverse(
            ZX_noise_i,
            Zy_fixed_train
            ) |> cpu;
    end
    X_post_train
end

function get_cm_l2_ssim(G, X_batch, Y_batch; device=gpu, num_samples=1, batch_size)
        num_test = size(Y_batch)[end]
        l2_total = 0
        ssim_total = 0
        # get cm for each element in batch
        for i in 1:num_test
            y_i = Y_batch[:,:,:,i:i]
            x_i = X_batch[:,:,:,i:i]
            X_post_test = posterior_sampler(G, y_i, size(x_i); device, num_samples, batch_size)
            X_post_mean_test = mean(X_post_test; dims=4)
            ssim_total += assess_ssim(X_post_mean_test[:,:,1,1], x_i[:,:,1,1] |> cpu)
            l2_total   += norm(X_post_mean_test[:,:,1,1] - (x_i[:,:,1,1] |> cpu))^2
        end
    return l2_total / num_test, ssim_total / num_test
end

function get_loss(G, X_batch, Y_batch; device=gpu, batch_size=16, N, noise_lev_x, noise_lev_y)
    num_test = size(Y_batch)[end]
    l2_total = 0
    logdet_total = 0
    num_batches = div(num_test, batch_size)
    for i in 1:num_batches
        x_i = X_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size]
        y_i = Y_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size]

        x_i .+= noise_lev_x * randn(Float32, size(x_i));
        y_i .+= noise_lev_y * randn(Float32, size(y_i));

        Zx, Zy, lgdet = G.forward(x_i |> device, y_i |> device) |> cpu;
        l2_total     += norm(Zx)^2 / (prod(N)*batch_size)
        logdet_total += lgdet / prod(N)
    end

    return l2_total / (num_batches), logdet_total / (num_batches)
end

function plot_posterior_samples(G, latent_test_x, latent_test_y, file_path; device, num_post_samples, params, rtm_min_val, rtm_max_val)
    num_cols = 7
    vmax_error = nothing
    vmax_std = nothing
    fig = figure(figsize=(20, 5));
    plot_len = 3

    extent = (0, 12.5f0*320, 6.26f0*320, 0)

    training_params = params["filter"]["training"]
    vmin = training_params["vmin"]
    vmax = training_params["vmax"]

    for i in 1:plot_len
        x = latent_test_x[:,:,:,i:i]
        y = latent_test_y[:,:,:,i:i]

        # make samples from posterior for train sample
        X_post = posterior_sampler(G,  y, size(x); device, num_samples=num_post_samples)
        X_post = X_post |> cpu

        X_post_mean = mean(X_post,dims=4);
        X_post_std  = std(X_post, dims=4);
        normalized_std, _ = normalize_std(X_post_mean[:, :, 1, 1], X_post_std[:, :, 1, 1])
        error_mean = abs.(X_post_mean[:,:,1,1] - x[:,:,1,1]);
        #error_mean, _ = normalize_std(abs.(X_post_mean[:,:,1,1]-x[:,:,1,1]), X_post_std[:, :, 1, 1]);
        ssim_i = round(assess_ssim(X_post_mean[:,:,1,1], x[:,:,1,1]), digits=2);
        rmse_i = round(sqrt(mean(error_mean.^2)), digits=4);
        mean_var_i = round(sqrt(mean(X_post_std.^2)), digits=4);

        var_fact = 2*rmse_i / mean_var_i
        vmax_err = 0.25

        # X_post_mean = mean(X_post,dims=4)
        # X_post_std  = std(X_post, dims=4)
        # error_mean = abs.(X_post_mean[:,:,1,1]-x[:,:,1,1])
        # ssim_i = round(assess_ssim(X_post_mean[:,:,1,1], x[:,:,1,1]),digits=2)
        # mse_i = round(mean(error_mean.^2),digits=2)

        y_plot = y[:,:,1,1] .* (rtm_max_val - rtm_min_val) .+ rtm_min_val;
        a = quantile(abs.(vec(y_plot)), 98/100)

        subplot(plot_len,num_cols,(i-1)*num_cols+1)
        imshow(y_plot'; extent, vmin=-a, vmax=a, interpolation="none", cmap="gray")
        colorbar(fraction=0.023, pad=0.04)
        title(L"$y$")

        subplot(plot_len,num_cols,(i-1)*num_cols+2)
        imshow(X_post[:,:,1,1]'; extent, vmin, vmax, interpolation="none", cmap="plasma")
        axis("off")
        colorbar(fraction=0.023, pad=0.04)
        title("Posterior sample")

        subplot(plot_len,num_cols,(i-1)*num_cols+3)
        imshow(X_post[:,:,1,2]'; extent, vmin, vmax, interpolation="none", cmap="plasma")
        axis("off")
        colorbar(fraction=0.023, pad=0.04)
        title("Posterior sample")

        x_plot = x[:,:,1,1]
        subplot(plot_len,num_cols,(i-1)*num_cols+4)
        imshow(x_plot'; extent, vmin, vmax, interpolation="none", cmap="plasma")
        axis("off")
        title(L"$\mathbf{x_{gt}}$")
        colorbar(fraction=0.023, pad=0.04)

        subplot(plot_len,num_cols,(i-1)*num_cols+5)
        imshow(X_post_mean[:,:,1,1]'; extent, vmin, vmax,  interpolation="none", cmap="plasma")
        axis("off")
        title("Conditional mean SSIM="*string(ssim_i))
        colorbar(fraction=0.023, pad=0.04)

        subplot(plot_len,num_cols,(i-1)*num_cols+6)
        imshow(error_mean'; extent, vmin=0, vmax=vmax_err, interpolation="none", cmap="plasma")
        axis("off")
        title("RMSE="*string(rmse_i))
        cb = colorbar(fraction=0.023, pad=0.04)
        if i == 1
            vmax_error = cb.vmax
        end

        subplot(plot_len, num_cols, (i-1)*num_cols+7)
        imshow(var_fact.*normalized_std'; extent, vmin=0, vmax=vmax_err, interpolation="none", cmap="plasma")
        axis("off")
        title("Normalized Std")
        cb =colorbar(fraction=0.023, pad=0.04)
        if i == 1
            vmax_std = cb.vmax
        end
    end
    tight_layout()
    safesave(file_path, fig)
    close(fig)
end

