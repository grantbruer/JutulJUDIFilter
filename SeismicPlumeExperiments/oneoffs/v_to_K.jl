using CairoMakie
using LaTeXStrings
using Format
include("utils.jl")

common_kwargs = (;
    linewidth = 6,
    # markersize = 15,
)

text_size_normal = 28
text_size_smaller = 24
ORANGE = "#fc8d62"
BLUE = "#8da0cb"
GREEN = "#66c2a5"

time_scale = 1 # years per step

xs = range(0, 1, 100)
# ys = 1.25^2 .* clamp.((xs .- 0.1).^2, 0, 1)
ys = clamp.((xs .- 0.1) .* 1.25, 0, 1) .^ 2


my_kwargs = (;
    color=:black,
    # linestyle=:dashdot,
    common_kwargs...,
)

fig = Figure()
ax = Axis(fig[1, 1])

v_0 = range(1, 3.6, 50)
K_0 = log(1e-2) .+ v_0 .- 3.5
lines!(ax, v_0, K_0; my_kwargs...)


v_1 = range(3.6, 3.85, 20)
K_1_lower = -4 .* (3.85 .- v_1) .* log(1e1) .+ 2 .* (v_1 .- 3.35) .* log(1.2)
K_1_upper = -4 .* (3.85 .- v_1) .* log(1e1) .+ 2 .* (v_1 .- 3.35) .* log(7.2e3)
band!(ax, v_1, K_1_lower, K_1_upper; my_kwargs...)

v_2 = range(3.85, 5, 20)
K_2_upper = log(1.2e3 .- 6e3) .+ v_2 .- 3.7
K_2_upper = log(1.2e3) .+ v_2 .- 3.7
band!(ax, v_2, K_2_lower, K_2_upper; my_kwargs...)

ax.xlabel = "velocity (km/s)"
ax.ylabel = "log permeability"
# ax.xticks = [1, 3.6, 3.85, 5]
# ylims!(ax, 0, 0.09)
# ax.yticks = 0:0.02:0.09

# axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
axis_setup(ax; xtickformat="%.1f", ytickformat="%.1f")

# save_dir = "figs"
# mkpath(save_dir)

display(fig)

# save(joinpath(save_dir, "relative_permeability.png"), fig)
