using JLD2
using CairoMakie
using LaTeXStrings
using Format
include("utils.jl")

let cache = Dict{Any, Any}()
    global function load_error_data(error_file)
        if ! (error_file in keys(cache))
            times, ys = load(error_file, "times", "ys")
            cache[error_file] = (times[3:end], ys[3:end])
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

time_scale = 1 # years per step

# Plot l2 errors.
error_file = "run/figs/enkf_N256/mean_saturation_l2_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
ys = (ys ./ (325 * 341)) .^ 0.5
my_kwargs = (;
    label=L"\text{EnKF}",
    color=GREEN,
    common_kwargs...,
)
fig, ax, sc = plot_disjoint_lines(times, ys; my_kwargs...)

error_file = "run/figs/noobs_N256/mean_saturation_l2_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
ys = (ys ./ (325 * 341)) .^ 0.5
my_kwargs = (;
    label=L"\text{NoObs}",
    color=BLUE,
    linestyle=:dashdot,
    markersize=15,
    common_kwargs...,
)
plot_disjoint_lines!(ax, times, ys; my_kwargs...)

error_file = "run/figs/justobs_N256/mean_saturation_l2_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
ys = (ys ./ (325 * 341)) .^ 0.5
my_kwargs = (;
    label=L"\text{JustObs}",
    color=ORANGE,
    # linestyle=:dot,
    markersize = 18,
    marker=:star8,
    common_kwargs...,
)
scatter!(ax, times[1:2:end], ys[1:2:end]; my_kwargs...)
# plot_disjoint_lines!(ax, times, ys; my_kwargs...)

if show_mean_error
    error_file = "run/figs/enkf_N256/mean_ensemble_saturation_l2_error.jld2"
    times, ys = load_error_data(error_file)
    times .*= time_scale
    ys = ys .^ 0.5
    my_kwargs = (;
        label="mean(EnKF members' errors)",
        color=ORANGE,
        common_kwargs...,
    )
    plot_disjoint_lines!(ax, times, ys; my_kwargs...)

    error_file = "run/figs/noobs_N256/mean_ensemble_saturation_l2_error.jld2"
    times, ys = load_error_data(error_file)
    times .*= time_scale
    ys = ys .^ 0.5
    my_kwargs = (;
        label="mean(NoObs members' errors)",
        color=ORANGE,
        linestyle=:dashdot,
        common_kwargs...,
    )
    plot_disjoint_lines!(ax, times, ys; my_kwargs...)
end

ax.xlabel = "Time since injection (years)"
ax.ylabel = L"\text{root mean squared error}"
ax.xticks = 1:5
ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
axis_setup(ax; xtickformat="%.0f", ytickformat="%.2f")

save_dir = "figs"
mkpath(save_dir)

save(joinpath(save_dir, "l2_errors_star.png"), fig)

# Plot ssim errors.
error_file = "run/figs/enkf_N256/mean_saturation_ssim_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
my_kwargs = (;
    label=L"\text{EnKF}",
    color=GREEN,
    common_kwargs...,
)
fig, ax, sc = plot_disjoint_lines(times, ys; my_kwargs...)

error_file = "run/figs/noobs_N256/mean_saturation_ssim_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
my_kwargs = (;
    label=L"\text{NoObs}",
    color=BLUE,
    linestyle=:dashdot,
    common_kwargs...,
)
plot_disjoint_lines!(ax, times, ys; my_kwargs...)

error_file = "run/figs/justobs_N256/mean_saturation_ssim_error.jld2"
times, ys = load_error_data(error_file)
times .*= time_scale
my_kwargs = (;
    label=L"\text{JustObs}",
    color=ORANGE,
    # linestyle=:dot,
    markersize = 18,
    marker=:star8,
    common_kwargs...,
)
scatter!(ax, times[1:2:end], ys[1:2:end]; my_kwargs...)
# plot_disjoint_lines!(ax, times, ys; my_kwargs...)

if show_mean_error
    error_file = "run/figs/enkf_N256/mean_ensemble_saturation_ssim_error.jld2"
    times, ys = load_error_data(error_file)
    times .*= time_scale
    my_kwargs = (;
        label="mean(EnKF members' errors)",
        color=ORANGE,
        common_kwargs...,
    )
    plot_disjoint_lines!(ax, times, ys; my_kwargs...)

    error_file = "run/figs/noobs_N256/mean_ensemble_saturation_ssim_error.jld2"
    times, ys = load_error_data(error_file)
    times .*= time_scale
    my_kwargs = (;
        label="mean(NoObs members' errors)",
        color=ORANGE,
        linestyle=:dashdot,
        common_kwargs...,
    )
    plot_disjoint_lines!(ax, times, ys; my_kwargs...)
end

ax.xlabel = "Time since injection (years)"
ax.ylabel = L"SSIM error (i.e., $1 - SSIM$)"
ax.xticks = 1:5
ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
axis_setup(ax; xtickformat="%.0f", ytickformat="%.2f")

save_dir = "figs"
mkpath(save_dir)

save(joinpath(save_dir, "ssim_errors_star.png"), fig)
