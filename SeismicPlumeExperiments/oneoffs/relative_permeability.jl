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
lines!(ax, xs, ys; my_kwargs...)

ax.xlabel = "Saturation"
# ax.ylabel = "Relative permeability"
ax.xticks = [0.0, 0.1, 0.5, 0.9, 1.0]
# ylims!(ax, 0, 0.09)
# ax.yticks = 0:0.02:0.09

# axislegend(position = :lt, labelsize=text_size_smaller, margin=(30, 0, 0, 0), unique=true)
axis_setup(ax; xtickformat="%.1f", ytickformat="%.1f")

save_dir = "figs"
mkpath(save_dir)

save(joinpath(save_dir, "relative_permeability.png"), fig)
