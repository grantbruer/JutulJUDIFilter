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
)

text_size_normal = 28
text_size_smaller = 24
ORANGE = "#fc8d62"
BLUE = "#8da0cb"
GREEN = "#66c2a5"

time_scale = 1 # years per step

root_dir = "run/figs"

true_noise = 1.29e15
true_snr = 8
cases = [
    (dir="enkf_noise0_01", noise=0.75e5, plot_kwargs=(;)),
    (dir="enkf_noise0_02", noise=0.75e6, plot_kwargs=(;)),
    (dir="enkf_noise0_03", noise=0.75e7, plot_kwargs=(;)),
    (dir="enkf_noise0_04", noise=0.75e8, plot_kwargs=(;)),
    (dir="enkf_noise0_05", noise=0.75e9, plot_kwargs=(;)),
    (dir="enkf_noise0_1", noise=0.75e10, plot_kwargs=(;)),
    (dir="enkf_noise0_3", noise=0.75e12, plot_kwargs=(;)),
    (dir="enkf_noise0_5", noise=0.75e14, plot_kwargs=(;)),
    (dir="enkf_noise1", noise=0.75e15, plot_kwargs=(;)),
    # (dir="enkf_noise2", noise=1.5e15, plot_kwargs=(;)),
    (dir="enkf_N256", noise=3e15, plot_kwargs=(;)),
    # (dir="enkf_noise4", noise=6e15, plot_kwargs=(;)),
    (dir="enkf_noise5", noise=12e15, plot_kwargs=(;)),
    # (dir="enkf_noise6", noise=24e15, plot_kwargs=(;)),
    (dir="enkf_noise7", noise=48e15, plot_kwargs=(;)),
]
noobs_case = (dir="noobs_N256", noise=-1, plot_kwargs=(;))

colors = cgrad(:RdBu, length(cases), categorical=true, rev=true)
true_nu = 10^(-true_snr / 20)

function get_noise_scale(case)
    return case.noise / true_noise
end

get_beta(case::NamedTuple) = case.noise / true_nu / 1e9 * ORIGINAL_PA_TO_NEW_PA

function get_label(case)
    # return @sprintf("%.1e", get_noise_scale(case))
    return latexstring(@sprintf("10^{%.1f}", log10(get_beta(case))))
end

# shrink_stuff(a) = [a[i] for i in vcat(1:3, 6:7, 10:11)]
shrink_stuff(a) = [a[i] for i in vcat(1, 3, 7, 11)]

function read_data_l2_error(case)
    error_file = joinpath(root_dir, case.dir, "mean_saturation_l2_error.jld2")
    times, ys = load_error_data(error_file)

    times = shrink_stuff(times)
    ys = shrink_stuff(ys)

    times .*= time_scale
    ys = (ys ./ (325 * 341)) .^ 0.5
    return times, ys
end

function read_data_ssim_error(case)
    error_file = joinpath(root_dir, case.dir, "mean_saturation_ssim_error.jld2")
    times, ys = load_error_data(error_file)

    times = shrink_stuff(times)
    ys = shrink_stuff(ys)

    times .*= time_scale
    return times, ys
end

# Plot l2 errors.
fig = Figure()
ax = Axis(fig)
fig[1, 1] = ax

# noises = get_noise_scale.(cases)
# c_values = log10.(noises)

let
    for (i, case) in enumerate(cases)
        times, ys = read_data_l2_error(case)
        # error_file = joinpath(root_dir, case.dir, "mean_saturation_l2_error.jld2")
        # times, ys = load_error_data(error_file)
        # times .*= time_scale
        # ys = (ys ./ (325 * 341)) .^ 0.5
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
end

ax.xlabel = L"\text{Time since injection (years)}"
ax.ylabel = L"\text{root mean squared error}"
ax.xticks = 0:5
# ax.yscale = log10
# ylims!(ax, 0, 0.12)
# xlims!(ax, 1, 5)
# ax.yticks = 0:0.02:0.12

# axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
# fig[1, end+1] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)

ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axis_setup(ax; xtickformat=nothing, ytickformat="%.2f")

cb = add_colorbar_from_labels(ax, ticklabelsize=text_size_smaller)

save_dir = "figs"
mkpath(save_dir)

save(joinpath(save_dir, "beta_l2_errors_vs_time.png"), fig)

# Read all the l2 error data.
noises = get_beta.(cases)
noobs_noise = 10 ^ (log10(maximum(noises)) + 1.5e-1 * log10(maximum(noises)/minimum(noises)))
# push!(noises, noobs_noise)

times, noobs_ys = read_data_l2_error(noobs_case)

ys = zeros(length(times), length(cases))
# ys = zeros(length(times), length(cases)+1)
for (i, case) in enumerate(cases)
    case_times, case_ys = read_data_l2_error(case)
    # error_file = joinpath(root_dir, case.dir, "mean_saturation_l2_error.jld2")
    # case_times, case_ys = load_error_data(error_file)
    # case_times .*= time_scale
    # case_ys = (case_ys ./ (325 * 341)) .^ 0.5

    ys[:, i] .= case_ys

    # Make sure the times are the same as for the first case.
    for j in 1:length(times)
        if abs(times[j] - case_times[j]) > 1e-6 * times[j]
            error("""
                For case $i, time at step $j does not match NoObs case.
                NoObs has time = $(times[j]) and case $i has time $(case_times[j])
                abs(times[j] - case_times[j]) = $(abs(times[j] - case_times[j]))
            """)
        end
    end
end
# ys[:, end] .= noobs_ys

# Plot l2 errors vs noise
fig = Figure()
ax = Axis(fig)
fig[1, 1] = ax

time_colors = cgrad(:Dark2_8, length(times), categorical=true, rev=true)

for (i, t) in enumerate(times)
    if i == 1
        continue
    end
    label = latexstring(@sprintf("t = %.1f", t))
    my_kwargs = (;
        color = time_colors[i],
        label,
        marker = :circle,
    )
    scatterlines!(ax, noises, ys[i, :]; my_kwargs...)
end

my_kwargs = (;
    color = :black,
    label = L"\text{NoObs}",
    marker = :star4,
    markersize = 15,
)
# scatter!(ax, noobs_noise, ys[2, end]; my_kwargs...)
for (i, t) in enumerate(times)
    if i == 1 || times[i] == times[i - 1]
        continue
    end
    my_kwargs = (;
        color = time_colors[i],
        label = L"\text{NoObs}" => (color = time_colors[2:end],
            points = Point2f[(0.2, 0.50), (0.5, 0.50), (0.8, 0.50)]),
        marker = :star4,
        markersize = 15,
    )
    # scatter!(ax, noobs_noise, noobs_ys[i]; my_kwargs..., color=:black)
    scatter!(ax, noobs_noise, noobs_ys[i]; my_kwargs...)
end

ax.xscale = log10
ylims!(ax, 0, 1.1 * maximum(noobs_ys))

fig[1, end+1] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)

ax.xlabel = L"\beta"
ax.ylabel = L"\text{root mean squared error}"

# ticks = Makie.get_ticks(ax.xticks[], ax.xscale, ax.xtickformat[], minimum(noises), maximum(noises))
# ax.xticks = (push!(ax.xaxis.tickvalues[], noobs_noise), push!(ax.xaxis.ticklabels[], "NoObs"))

ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axis_setup(ax; xtickformat=nothing, ytickformat="%.2f")


save(joinpath(save_dir, "beta_l2_errors_vs_noise.png"), fig)

# Plot ssim errors.
fig = Figure()
ax = Axis(fig)
fig[1, 1] = ax

for (i, case) in enumerate(cases)
    times, ys = read_data_ssim_error(case)
    # error_file = joinpath(root_dir, case.dir, "mean_saturation_ssim_error.jld2")
    # times, ys = load_error_data(error_file)
    # times .*= time_scale
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

ax.xlabel = L"\text{Time since injection (years)}"
ax.ylabel = L"SSIM error (i.e., $1 - SSIM$)"
ax.xticks = 0:5
# ax.yscale = log10
# xlims!(ax, 1, 5)
# ax.yticks = 0:0.02:0.12

# axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
# fig[1, 2] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)
cb = add_colorbar_from_labels(ax, ticklabelsize=text_size_smaller)


ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axis_setup(ax; xtickformat=nothing, ytickformat="%.2f")


save_dir = "figs"
mkpath(save_dir)
save(joinpath(save_dir, "beta_ssim_errors_vs_time.png"), fig)



# Read all the ssim error data.
noises = get_beta.(cases)
noobs_noise = 10 ^ (log10(maximum(noises)) + 1.5e-1 * log10(maximum(noises)/minimum(noises)))
# push!(noises, noobs_noise)

times, noobs_ys = read_data_ssim_error(noobs_case)

ys = zeros(length(times), length(cases))
# ys = zeros(length(times), length(cases)+1)
for (i, case) in enumerate(cases)
    case_times, case_ys = read_data_ssim_error(case)
    ys[:, i] .= case_ys

    # Make sure the times are the same as for the first case.
    for j in 1:length(times)
        if abs(times[j] - case_times[j]) > 1e-6 * times[j]
            error("""
                For case $i, time at step $j does not match NoObs case.
                NoObs has time = $(times[j]) and case $i has time $(case_times[j])
                abs(times[j] - case_times[j]) = $(abs(times[j] - case_times[j]))
            """)
        end
    end
end
# ys[:, end] .= noobs_ys

# Plot ssim errors vs noise
fig = Figure()
ax = Axis(fig)
fig[1, 1] = ax

time_colors = cgrad(:Dark2_8, length(times), categorical=true, rev=true)

for (i, t) in enumerate(times)
    if i == 1
        continue
    end
    label = latexstring(@sprintf("t = %.1f", t))
    my_kwargs = (;
        color = time_colors[i],
        label,
        marker = :circle,
    )
    scatterlines!(ax, noises, ys[i, :]; my_kwargs...)
end

my_kwargs = (;
    color = :black,
    label = L"\text{NoObs}",
    marker = :star4,
    markersize = 15,
)
# scatter!(ax, noobs_noise, ys[2, end]; my_kwargs...)
for (i, t) in enumerate(times)
    if i == 1 || times[i] == times[i - 1]
        continue
    end
    my_kwargs = (;
        color = time_colors[i],
        label = L"\text{NoObs}" => (color = time_colors[2:end],
            points = Point2f[(0.2, 0.50), (0.5, 0.50), (0.8, 0.50)]),
        marker = :star4,
        markersize = 15,
    )
    # scatter!(ax, noobs_noise, noobs_ys[i]; my_kwargs..., color=:black)
    scatter!(ax, noobs_noise, noobs_ys[i]; my_kwargs...)
end

ax.xscale = log10
ylims!(ax, 0, 1.1 * maximum(noobs_ys))
fig[1, end+1] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)

ax.xlabel = L"\beta"
ax.ylabel = L"SSIM error (i.e., $1 - SSIM$)"

# ticks = Makie.get_ticks(ax.xticks[], ax.xscale, ax.xtickformat[], minimum(noises), maximum(noises))
# ax.xticks = (push!(ax.xaxis.tickvalues[], noobs_noise), push!(ax.xaxis.ticklabels[], "NoObs"))

ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axis_setup(ax; xtickformat=nothing, ytickformat="%.2f")


save(joinpath(save_dir, "beta_ssim_errors_vs_noise.png"), fig)
