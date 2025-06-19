using JLD2
using CairoMakie
using LaTeXStrings
using Printf
include("utils.jl")

if ! @isdefined load_error_data
    let cache = Dict{Any, Any}()
        global function load_error_data(error_file)
            if ! (error_file in keys(cache))
                cache[error_file] = load(error_file, "times", "ys")
            end
            return cache[error_file]
        end
    end
end

common_kwargs = (;
    linewidth = 3,
    markersize = 15,
    linestyle=:dot,
    marker=:vline,
)

text_size_normal = 28
text_size_smaller = 24
ORANGE = "#fc8d62"
BLUE = "#8da0cb"
GREEN = "#66c2a5"

time_scale = 1 # years per step

root_dir = "pace-04.03.2024/figs"

true_noise = 1.29e15
cases = [
    (dir="enkf_noise0_5-singlenoise", noise=0.75e14, plot_kwargs=(;)),
    (dir="enkf_noise1-singlenoise", noise=0.75e15, plot_kwargs=(;)),
    (dir="enkf_noise3-singlenoise", noise=3e15, plot_kwargs=(;)),
    (dir="enkf_noise5-singlenoise", noise=12e15, plot_kwargs=(;)),
]
colors = cgrad(:RdBu, length(cases), categorical=true, rev=true)

function get_noise_scale(case)
    return case.noise / true_noise
end

function get_label(case)
    # return @sprintf("%.1e", get_noise_scale(case))
    return latexstring(@sprintf("10^{%.1f}", log10(case.noise)))
end

# Plot l2 errors.
fig = Figure()
ax = Axis(fig)
fig[1, 1] = ax

for (i, case) in enumerate(cases)
    error_file = joinpath(root_dir, case.dir, "mean_saturation_l2_error.jld2")
    times, ys = load_error_data(error_file)
    times .*= time_scale
    ys = (ys ./ (325 * 341)) .^ 0.5
    label = get_label(case)
    my_kwargs = (;
        label=label,
        # linestyle=:dashdot,
        color = (colors[i], 0.5),
        common_kwargs...,
        case.plot_kwargs...,
    )
    plot_disjoint_lines!(ax, times[2:end], ys[2:end]; my_kwargs...)
end

ax.xlabel = "Time since injection (years)"
ax.ylabel = L"\text{root mean squared error}"
ax.xticks = 0:5
# ax.yscale = log10
# ylims!(ax, 0, 0.12)
# xlims!(ax, 1, 5)
# ax.yticks = 0:0.02:0.12

# axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
# fig[1, 2] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)
cb = add_colorbar_from_labels(ax, ticklabelsize=text_size_smaller)

hidespines!(ax)
ax.xticklabelsize = text_size_normal
ax.xlabelsize = text_size_normal
ax.yticklabelsize = text_size_normal
ax.ylabelsize = text_size_normal

save_dir = "figs"
mkpath(save_dir)

save(joinpath(save_dir, "noise_l2_errors-singlenoise.png"), fig)


# Plot ssim errors.
fig = Figure()
ax = Axis(fig)
fig[1, 1] = ax

for (i, case) in enumerate(cases)
    error_file = joinpath(root_dir, case.dir, "mean_saturation_ssim_error.jld2")
    times, ys = load_error_data(error_file)
    times .*= time_scale
    label = get_label(case)
    my_kwargs = (;
        label=label,
        # linestyle=:dashdot,
        color = (colors[i], 0.5),
        common_kwargs...,
        case.plot_kwargs...,
    )
    plot_disjoint_lines!(ax, times[2:end], ys[2:end]; my_kwargs...)
end

ax.xlabel = "Time since injection (years)"
ax.ylabel = L"SSIM error (i.e., $1 - SSIM$)"
ax.xticks = 0:5
# ylims!(ax, 0, 0.12)
# ax.yscale = log10
# xlims!(ax, 1, 5)
# ax.yticks = 0:0.02:0.12

# axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
# fig[1, 2] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)
cb = add_colorbar_from_labels(ax, ticklabelsize=text_size_smaller)
cb.labelsize = text_size_normal

hidespines!(ax)
ax.xticklabelsize = text_size_normal
ax.xlabelsize = text_size_normal
ax.yticklabelsize = text_size_normal
ax.ylabelsize = text_size_normal

save_dir = "figs"
mkpath(save_dir)
save(joinpath(save_dir, "noise_ssim_errors-singlenoise.png"), fig)
