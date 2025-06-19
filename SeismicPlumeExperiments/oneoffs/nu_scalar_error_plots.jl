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

true_snr = 8 # dB
cases = [
    (dir="enkf_noise3-nu64", snr_db=-12, plot_kwargs=(;)),
    (dir="enkf_noise3-nu48", snr_db=-2, plot_kwargs=(;)),
    (dir="enkf_noise3-nu40", snr_db=3, plot_kwargs=(;)),
    (dir="enkf_N256", snr_db=8, plot_kwargs=(;)),
    (dir="enkf_noise3-nu28", snr_db=13, plot_kwargs=(;)),
    (dir="enkf_noise3-nu24", snr_db=18, plot_kwargs=(;)),
    (dir="enkf_noise3-nu20", snr_db=23, plot_kwargs=(;)),
    (dir="enkf_noise3-nu16", snr_db=28, plot_kwargs=(;)),
]
noobs_case = (dir="noobs_N256", snr_db=-1, plot_kwargs=(;))

colors = cgrad(:RdBu, length(cases), categorical=true, rev=true)

get_snr(case) = case.snr_db
get_noise_factor(case) = 10 ^ ((true_snr - case.snr_db)/20)
function get_label(case)
    return latexstring(@sprintf("%.2g", get_snr(case)))
end

shrink_stuff(a) = [a[i] for i in vcat(1, 3, 7, 11)]
# shrink_stuff(a) = [a[i] for i in vcat(1:3, 6:7, 10:11)]

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

for (i, case) in enumerate(cases)
    times, ys = read_data_l2_error(case)
    label = get_label(case)
    my_kwargs = (;
        label=label,
        color = (colors[i], 0.5),
        common_kwargs...,
        case.plot_kwargs...,
    )
    plot_disjoint_lines!(ax, times[2:end], ys[2:end]; my_kwargs...)
end

ax.xlabel = L"\text{Time since injection (years)}"
ax.ylabel = L"\text{root mean squared error}"
ax.xticks = 0:5
ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axis_setup(ax; xtickformat=nothing, ytickformat="%.2f")

cb = add_colorbar_from_labels(ax, ticklabelsize=text_size_smaller)

save_dir = "figs"
mkpath(save_dir)

save(joinpath(save_dir, "nu_l2_errors_vs_time.png"), fig)

# Read all the l2 error data.
case_snrs = [get_snr(case) for case in cases]
noobs_snr = minimum(case_snrs) - 1.5e-1 * (maximum(case_snrs) - minimum(case_snrs))
snrs = vcat([noobs_snr], case_snrs)
snrs = case_snrs

times, noobs_ys = read_data_l2_error(noobs_case)

ys = zeros(length(times), length(cases))
for (i, case) in enumerate(cases)
    case_times, case_ys = read_data_l2_error(case)

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
    scatterlines!(ax, snrs, ys[i, :]; my_kwargs...)
end


my_kwargs = (;
    color = :black,
    label = L"\text{NoObs}",
    marker = :star4,
    markersize = 15,
)
# scatter!(ax, noobs_snr, ys[2, 1]; my_kwargs...)
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
    # scatter!(ax, noobs_snr, noobs_ys[i]; my_kwargs..., color=:black)
    scatter!(ax, noobs_snr, noobs_ys[i]; my_kwargs...)
end
fig[1, end+1] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)

ax.xlabel = L"\text{SNR}"
ax.ylabel = L"\text{root mean squared error}"

function add_new_xtick!(ax, val, label)
    tickvalues = ax.xaxis.tickvalues[]
    ticklabels = ax.xaxis.ticklabels[]
    @show tickvalues ticklabels
    push!(ax.xaxis.tickvalues[], val)
    push!(ax.xaxis.ticklabels[], label)
    ax.xticks = (tickvalues, ticklabels)
end
# add_new_xtick!(ax, true_snr, L"\gamma^*")
ax.xticks = (
    [-20, 0, true_snr, 20],
    ["-20", "0", L"\gamma^*", "20"]
)

ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axis_setup(ax; xtickformat=nothing, ytickformat="%.2f")

save(joinpath(save_dir, "nu_l2_errors_vs_snr.png"), fig)


# Plot ssim errors.
fig = Figure()
ax = Axis(fig)
fig[1, 1] = ax

for (i, case) in enumerate(cases)
    times, ys = read_data_ssim_error(case)
    label = get_label(case)
    my_kwargs = (;
        label=label,
        color = (colors[i], 0.5),
        common_kwargs...,
        case.plot_kwargs...,
    )
    plot_disjoint_lines!(ax, times[2:end], ys[2:end]; my_kwargs...)
end

ax.xlabel = L"\text{Time since injection (years)}"
ax.ylabel = L"SSIM error (i.e., $1 - SSIM$)"
ax.xticks = 0:5
ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axis_setup(ax; xtickformat=nothing, ytickformat="%.2f")

cb = add_colorbar_from_labels(ax, ticklabelsize=text_size_smaller)


hidespines!(ax)
ax.xticklabelsize = text_size_normal
ax.xlabelsize = text_size_normal
ax.yticklabelsize = text_size_normal
ax.ylabelsize = text_size_normal

save_dir = "figs"
mkpath(save_dir)
save(joinpath(save_dir, "nu_ssim_errors_vs_time.png"), fig)



# Read all the ssim error data.
case_snrs = [get_snr(case) for case in cases]
noobs_snr = minimum(case_snrs) - 1.5e-1 * (maximum(case_snrs) - minimum(case_snrs))
snrs = vcat([noobs_snr], case_snrs)
snrs = case_snrs

times, noobs_ys = read_data_ssim_error(noobs_case)

ys = zeros(length(times), length(cases))
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
    scatterlines!(ax, snrs, ys[i, :]; my_kwargs...)
end

my_kwargs = (;
    color = :black,
    label = L"\text{NoObs}",
    marker = :star4,
    markersize = 15,
)
# scatter!(ax, noobs_snr, ys[2, 1]; my_kwargs...)
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
    # scatter!(ax, noobs_snr, noobs_ys[i]; my_kwargs..., color=:black)
    scatter!(ax, noobs_snr, noobs_ys[i]; my_kwargs...)
end
fig[1, end+1] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)

ax.xlabel = L"\text{SNR}"
ax.ylabel = L"SSIM error (i.e., $1 - SSIM$)"

ax.xticks = (
    [-20, 0, true_snr, 20],
    ["-20", "0", L"\gamma^*", "20"]
)
ylims!(ax, 0, 0.09)
ax.yticks = 0:0.02:0.09

axis_setup(ax; xtickformat=nothing, ytickformat="%.2f")

save(joinpath(save_dir, "nu_ssim_errors_vs_snr.png"), fig)
