import TOML

include("../lib/params.jl")

using JLD2
using CairoMakie
using Printf
using LinearAlgebra
using Statistics
include("utils.jl")

if ! @isdefined load_svd_data
    let cache = Dict{Any, Any}()
        global function load_svd_data(filter_file, preprocess=nothing)
            key = isnothing(preprocess) ? filter_file : (filter_file, preprocess)
            if ! (key in keys(cache))
                preprocess = isnothing(preprocess) ? x->x : preprocess
                filter = load_filter(params, filter_file);

                X = hcat([vec(preprocess(em.state)) for em in filter.ensemble]...);
                mean_X = mean(X, dims=2)
                X .-= mean_X

                F = svd(X);
                info = (;
                    S = F.S,
                    U = F.U,
                    mean = mean_X,
                )
                cache[key] = info
            end
            return cache[key]
        end
    end
end

common_kwargs = (;
    linewidth = 6,
    markersize = 15,
)

text_size_normal = 28
text_size_smaller = 24
ORANGE = "#fc8d62"
BLUE = "#8da0cb"
GREEN = "#66c2a5"

time_scale = 1 # years per step

root_dir = "pace-04.03.2024/"

params = TOML.parsefile("params/enkf/base.toml")
work_dir = get_filter_work_dir(params)


fig = Figure()
ax = Axis(fig[1,1])

for i = 1:5
    filepath = joinpath(root_dir, "enkf_N256", "filter_$(i)_prior")
    F = load_svd_data(filepath)
    label = @sprintf("%d", i)
    lines!(ax, F.S; label)
    break
end

ax.xlabel = "Singular value index"
ax.ylabel = "Singular value"
ax.yscale = log10
ylims!(ax, 1e0, nothing)
fig[1, 2] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)

hidespines!(ax)
ax.xticklabelsize = text_size_normal
ax.xlabelsize = text_size_normal
ax.yticklabelsize = text_size_normal
ax.ylabelsize = text_size_normal

save_dir = "figs"
mkpath(save_dir)

save(joinpath(save_dir, "covariance_state_enkf.png"), fig)

display(fig)
# xlims!(ax, 1, 5)
# ax.yticks = 0:0.02:0.12



fig = Figure()
ax = Axis(fig[1,1])

for i = 1:5
    filepath = joinpath(root_dir, "enkf_N256", "filter_obs_$(i)_prior")
    F = load_svd_data(filepath, x -> x[1])
    label = @sprintf("%d", i)
    lines!(ax, F.S; label)
    break
end

ax.xlabel = "Singular value index"
ax.ylabel = "Singular value"
ax.yscale = log10
# ylims!(ax, 1e0, nothing)
fig[1, 2] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)

hidespines!(ax)
ax.xticklabelsize = text_size_normal
ax.xlabelsize = text_size_normal
ax.yticklabelsize = text_size_normal
ax.ylabelsize = text_size_normal

save_dir = "figs"
mkpath(save_dir)

save(joinpath(save_dir, "covariance_obs_enkf.png"), fig)




true_noise = 1.29e15
cases = [
    (dir="enkf_noise5-singlenoise", noise=12e15, plot_kwargs=(;linestyle=:dot)),
    (dir="enkf_noise3-singlenoise", noise=3e15, plot_kwargs=(;linestyle=:dot)),
    (dir="enkf_noise1-singlenoise", noise=0.75e15, plot_kwargs=(;linestyle=:dot)),
    (dir="enkf_noise0_5-singlenoise", noise=0.75e14, plot_kwargs=(;linestyle=:dot)),
    (dir="enkf_noise2", noise=1.5e15, plot_kwargs=(;)),
    (dir="enkf_noise0_1", noise=0.75e10, plot_kwargs=(;)),
    (dir="enkf_noise0_05", noise=0.75e9, plot_kwargs=(;)),
    (dir="enkf_noise0_01", noise=0.75e5, plot_kwargs=(;)),
]

function get_noise_scale(case)
    return case.noise / true_noise
end

function get_label(case)
    return @sprintf("%.1e", get_noise_scale(case))
end


fig = Figure()
ax = Axis(fig[1,1])

for i = 1:5
    filepath = joinpath(root_dir, "enkf_N256", "filter_obs_$(i)_prior")
    F = load_svd_data(filepath, x -> x[1])
    label = @sprintf("%d", i)
    lines!(ax, F.S; label)
    break
end

lines!(ax, [1, 256], [true_noise, true_noise]; label="true", color=:black)
for (i, case) in enumerate(cases)
    label = get_label(case)
    my_kwargs = case.plot_kwargs
    lines!(ax, [1, 256], [case.noise, case.noise]; label, my_kwargs...)
end

ax.xlabel = "Singular value index"
ax.ylabel = "Singular value"
ylims!(ax, 1e13, nothing)
ax.yscale = log10
fig[1, 2] = Legend(fig, ax, labelsize=text_size_smaller, unique=true)

hidespines!(ax)
ax.xticklabelsize = text_size_normal
ax.xlabelsize = text_size_normal
ax.yticklabelsize = text_size_normal
ax.ylabelsize = text_size_normal

save(joinpath(save_dir, "covariance_obs_enkf-markers.png"), fig)
