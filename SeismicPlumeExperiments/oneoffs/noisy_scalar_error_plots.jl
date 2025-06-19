using JLD2
using CairoMakie
include("utils.jl")

let cache = Dict{Any, Any}()
    global function load_error_data(error_file)
        if ! (error_file in keys(cache))
            cache[error_file] = load(error_file, "times", "ys")
        end
        return cache[error_file]
    end
end

common_kwargs = (;
    linewidth = 6,
    markersize = 15,
)

show_mean_error = false

text_size_normal = 28
text_size_smaller = 24
ORANGE = "#fc8d62"
BLUE = "#8da0cb"
GREEN = "#66c2a5"
MAGENTA = "#e7298a"

time_scale = 1 # years per step

# Plot l2 errors.
error_file = "run/figs/enkf_N256/mean_ensemble_saturation_l2_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
ys = (ys ./ (325 * 341)) .^ 0.5
my_kwargs = (;
    label="EnKF",
    color=GREEN,
    common_kwargs...,
)
fig, ax, sc = plot_disjoint_lines(times, ys; my_kwargs...)

error_file = "run/figs/noobs_N256/mean_ensemble_saturation_l2_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
ys = (ys ./ (325 * 341)) .^ 0.5
my_kwargs = (;
    label="NoObs",
    color=BLUE,
    linestyle=:dashdot,
    common_kwargs...,
)
plot_disjoint_lines!(ax, times, ys; my_kwargs...)

error_file = "run/figs/justobs_N256/mean_ensemble_saturation_l2_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
ys = (ys ./ (325 * 341)) .^ 0.5
my_kwargs = (;
    label="JustObs",
    color=ORANGE,
    linestyle=:dot,
    common_kwargs...,
)
plot_disjoint_lines!(ax, times, ys; my_kwargs...)


error_file = "run/figs/justobs_N256_noisy/mean_ensemble_saturation_l2_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
ys = (ys ./ (325 * 341)) .^ 0.5
my_kwargs = (;
    label="JustObs noisy",
    color=MAGENTA,
    linestyle=:dot,
    common_kwargs...,
)
plot_disjoint_lines!(ax, times, ys; my_kwargs...)


ax.xlabel = "Time since injection (years)"
ax.ylabel = L"\text{root mean squared error}"
ax.xticks = 0:5
ylims!(ax, 0, 0.12)
ax.yticks = 0:0.02:0.12

axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
hidespines!(ax)
ax.xticklabelsize = text_size_normal
ax.xlabelsize = text_size_normal
ax.yticklabelsize = text_size_normal
ax.ylabelsize = text_size_normal

save_dir = "figs"
mkpath(save_dir)

save(joinpath(save_dir, "noisy_l2_errors.png"), fig)


# Plot ssim errors.
error_file = "run/figs/enkf_N256/mean_ensemble_saturation_ssim_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
my_kwargs = (;
    label="EnKF",
    color=GREEN,
    common_kwargs...,
)
fig, ax, sc = plot_disjoint_lines(times, ys; my_kwargs...)

error_file = "run/figs/noobs_N256/mean_ensemble_saturation_ssim_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
my_kwargs = (;
    label="NoObs",
    color=BLUE,
    linestyle=:dashdot,
    common_kwargs...,
)
plot_disjoint_lines!(ax, times, ys; my_kwargs...)

error_file = "run/figs/justobs_N256/mean_ensemble_saturation_ssim_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
my_kwargs = (;
    label="JustObs",
    color=ORANGE,
    linestyle=:dot,
    common_kwargs...,
)
plot_disjoint_lines!(ax, times, ys; my_kwargs...)

error_file = "run/figs/justobs_N256_noisy/mean_ensemble_saturation_ssim_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
my_kwargs = (;
    label="JustObs noisy",
    color=MAGENTA,
    linestyle=:dot,
    common_kwargs...,
)
plot_disjoint_lines!(ax, times, ys; my_kwargs...)


ax.xlabel = "Time since injection (years)"
ax.ylabel = L"SSIM error (i.e., $1 - SSIM$)"
ax.xticks = 0:5
ylims!(ax, 0, 0.12)
ax.yticks = 0:0.02:0.12

axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
hidespines!(ax)
ax.xticklabelsize = text_size_normal
ax.xlabelsize = text_size_normal
ax.yticklabelsize = text_size_normal
ax.ylabelsize = text_size_normal

save_dir = "figs"
mkpath(save_dir)
save(joinpath(save_dir, "noisy_ssim_errors.png"), fig)
