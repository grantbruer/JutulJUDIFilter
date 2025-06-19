using JLD2
using CairoMakie
using Printf
using Images: imresize

# This is compatible with Zygote by not mutating array.
function find_water_bottom_immutable(m::AbstractArray{avDT, N};eps = 1e-4) where {avDT, N}
    #return the indices of the water bottom of a seismic image
    wbfunc(x) = abs(x) > eps
    m_offset = m .- m[:, 1]
    idx = findfirst.(x->wbfunc(x), eachrow(m_offset))
    return idx
end

function get_divergent_colorrange(data)
    m1, m2 = extrema(data)
    return -m1 > m2 ? (m1, -m1) : (-m2, m2)
end

function get_extrema_string(data)
    ex = extrema(data)
    return @sprintf("extrema = (%6.2e, %6.2e)", ex[1], ex[2])
end

a = load("compass/BGCompass_tti_625m.jld2")

# Get parameters.

ρ_sat1 = a["rho"] .* 1e3 # Convert from g/cm³ to kg/m³.
vₚ_sat1 = 1e3 ./ sqrt.(a["m"]) # Convert from s²/km² to m/s.
constant_phi = false
if constant_phi
    save_dir = "patchy_figs_constant_phi"
    n = size(vₚ_sat1)
    ϕ = 0.25 * ones(n)
else
    save_dir = "patchy_figs"
    b = load("compass/broad&narrow_perm_models_new.jld2")
    ϕ = b["phi"]
    n = size(ϕ)
    vₚ_sat1 = imresize(vₚ_sat1, n)
    ρ_sat1 = imresize(ρ_sat1, n)
end
idx_wb = maximum(find_water_bottom_immutable(log.(vₚ_sat1) .- log(vₚ_sat1[1,1])))
ϕ[:, 1:idx_wb] .= 1
mkpath(save_dir)


d = a["d"]
B_fl1 = 2.735f9 # Bulk modulus of H₂O in Pa.
B_fl2 = 0.125f9 # Bulk modulus of CO₂ in Pa.
ρ_fl1 = 1.053f3 # Density of H₂O in kg/m³.
ρ_fl2 = 7.766f2 # Density of CO₂ in kg/m³ 
cap_speed_threshold = 3500f0 # m/s
cap_depth = 50f0 # meters
cap_bulk_modulus = 5f10 # Pascals

# Plot porosity.
fig, ax, hm = heatmap(ϕ, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "Porosity (kg/m³)"
file_path = joinpath(save_dir, "porosity.png")
save(file_path, fig)

# Plot density.
fig, ax, hm = heatmap(ρ_sat1, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "Density (kg/m³)"
file_path = joinpath(save_dir, "density.png")
save(file_path, fig)

# Plot p-wave velocity.
data = vₚ_sat1
fig, ax, hm = heatmap(data, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "P-wave velocity (m/s) with 0% CO₂"
file_path = joinpath(save_dir, "pwave_velocity1.png")
save(file_path, fig)

# Plot p-wave modulus.
M_sat1 = ρ_sat1 .* (vₚ_sat1 .^ 2)
fig, ax, hm = heatmap(M_sat1, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "P-wave modulus (Pa) with 0% CO₂"
file_path = joinpath(save_dir, "pwave_modulus1.png")
save(file_path, fig)

# Plot s-wave velocity.
vₛ = vₚ_sat1 ./ √(3f0);
fig, ax, hm = heatmap(vₛ, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "S-wave velocity (m/s)"
file_path = joinpath(save_dir, "swave_velocity.png")
save(file_path, fig)

# Using the given ρ_sat1 and vₚ_sat1, we first compute the implied P-wave and shear
# moduli, as well as the bulk moduli of the rock saturated with water.

# Plot s-wave modulus.
μ_sat1 = ρ_sat1 .* (vₛ .^ 2)
fig, ax, hm = heatmap(μ_sat1, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "S-wave modulus (Pa)"
file_path = joinpath(save_dir, "swave_modulus.png")
save(file_path, fig)

# Plot bulk modulus of the rock saturated with water.
B_sat1 = M_sat1 .- 4f0/3f0 .* μ_sat1
# which equals 
#   ρ_sat1 .* (vₚ_sat1 .^ 2) - 4/3 ρ_sat1 .* (vₛ .^ 2)
# = ρ_sat1 (vₚ_sat1 .^ 2 - 4/3 .* (vₛ .^ 2))
# = ρ_sat1 (vₚ_sat1 .^ 2 - 4/3 .* (vₚ_sat1 .^ 2) /3)
# = ρ_sat1 (vₚ_sat1 .^ 2 - 4/9 .* vₚ_sat1 .^ 2)
# = 5/9 .* ρ_sat1 .* vₚ_sat1 .^ 2
fig, ax, hm = heatmap(B_sat1, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "Bulk modulus (Pa) with 0% CO₂"
file_path = joinpath(save_dir, "bulk_modulus1.png")
save(file_path, fig)

# Plot bulk modulus of the mineral.
slow_mask = vₚ_sat1 .< cap_speed_threshold
B_min = ifelse.(slow_mask, 1.2f0 .* B_sat1, cap_bulk_modulus);
data = B_min
fig, ax, hm = heatmap(data, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "Bulk modulus (Pa) of mineral"
file_path = joinpath(save_dir, "bulk_modulus_mineral.png")
save(file_path, fig)

# Then we solve Gassman's equation for the bulk modulus of the rock
# saturated with fluid 2.
# B_sat2 = B_min ./ (1 ./ ptemp .+ 1)
# 1 ./ptemp .+ 1 = B_min ./ B_sat2
# 1 ./ ptemp = B_min ./ B_sat2 .- 1
# 1 ./ ptemp = (B_min .- B_sat2) ./ B_sat2
# ptemp = B_sat2 ./ (B_min .- B_sat2)
# B_sat1 / (B_min - B_sat1) - B_fl1 / (ϕ(B_min - B_fl1)) = B_sat2 / (B_min - B_sat2) - B_fl2 / (ϕ(B_min - B_fl2))
ptemp = (
  B_sat1 ./(B_min .- B_sat1)
  .- B_fl1 ./ ϕ ./ (B_min .- B_fl1)
  .+ B_fl2 ./ ϕ ./ (B_min .- B_fl2)
);

# Copy the values from a depth below the uncomformity layer to the depth
# above the uncomformity layer. TBD: why?
capgrid = Int(round(cap_depth / d[2]))
idx = find_water_bottom_immutable((vₚ_sat1 .- cap_speed_threshold) .* (.!slow_mask))
rows = 1:n[end]
masks = [((rows .>= (idx[i] - capgrid)) .&& (rows .<= (idx[i]-1))) for i = 1:n[1]]
mask = hcat(masks...);
ptemp_shifted = circshift(ptemp, (0, -capgrid));
ptemp = ifelse.(mask', ptemp_shifted, ptemp);
B_sat2 = B_min ./ (1f0./ptemp .+ 1f0);

# Plot bulk modulus of the rock saturated with CO2.
data = B_sat2
fig, ax, hm = heatmap(data, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "Bulk modulus (Pa) with 100% CO₂"
file_path = joinpath(save_dir, "bulk_modulus2.png")
save(file_path, fig)

fig, ax, hm = heatmap(ifelse.(B_sat2 .< 0, NaN, B_sat2), axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "Bulk modulus (Pa) with 100% CO₂"
file_path = joinpath(save_dir, "bulk_modulus2_nanned.png")
save(file_path, fig)

fig, ax, hm = heatmap(ifelse.(B_sat2 .< 0, NaN, B_sat2), axis=(yreversed=true,), colorrange=extrema(B_sat1))
Colorbar(fig[:, end+1], hm)
ax.title = "Bulk modulus (Pa) with 100% CO₂"
file_path = joinpath(save_dir, "bulk_modulus2_nanned_ranged.png")
save(file_path, fig)

# Plot P-wave modulus of the rock saturated with CO2.
M_sat2 = B_sat2 + 4f0/3f0 * μ_sat1;
fig, ax, hm = heatmap(M_sat2, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "P-wave modulus (Pa) with 100% CO₂"
file_path = joinpath(save_dir, "pwave_modulus2.png")
save(file_path, fig)

fig, ax, hm = heatmap(ifelse.(M_sat2 .< 0, NaN, M_sat2), axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "P-wave modulus (Pa) with 100% CO₂"
file_path = joinpath(save_dir, "pwave_modulus2_nanned.png")
save(file_path, fig)

fig, ax, hm = heatmap(ifelse.(M_sat2 .< 0, NaN, M_sat2), axis=(yreversed=true,), colorrange=extrema(M_sat1))
Colorbar(fig[:, end+1], hm)
ax.title = "P-wave modulus (Pa) with 100% CO₂"
file_path = joinpath(save_dir, "pwave_modulus2_nanned_ranged.png")
save(file_path, fig)

# Post-process this bulk modulus.

# The bulk modulus of rock saturated with the second fluid should not be
# higher than with the first fluid
bigger_mask = B_sat2 .> B_sat1;
B_sat2_processed = ifelse.(bigger_mask, B_sat1, B_sat2);

# The bulk modulus of rock saturated with the second fluid should not be smaller than very small.
smaller_mask = B_sat2 .< 0;
B_sat2_processed = ifelse.(smaller_mask, B_sat1, B_sat2_processed);

# Plot bulk modulus of the rock saturated with CO2.
fig, ax, hm = heatmap(B_sat2_processed, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "Bulk modulus (Pa) with 100% CO₂ (processed)"
file_path = joinpath(save_dir, "bulk_modulus2_processed.png")
save(file_path, fig)

data = B_sat2_processed .- B_sat1
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
ax.title = "Bulk modulus (Pa) saturation difference (processed)"
file_path = joinpath(save_dir, "bulk_modulus_satdiff_processed.png")
save(file_path, fig)


# Plot P-wave modulus of the rock saturated with CO2.
M_sat2_processed = B_sat2_processed + 4f0/3f0 * μ_sat1;
fig, ax, hm = heatmap(M_sat2_processed, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
ax.title = "P-wave modulus (Pa) with 100% CO₂ (processed)"
file_path = joinpath(save_dir, "pwave_modulus2_processed.png")
save(file_path, fig)

data = M_sat2_processed .- M_sat1
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
ax.title = "P-wave modulus (Pa) saturation difference (processed)"
file_path = joinpath(save_dir, "pwave_modulus_satdiff_processed.png")
save(file_path, fig)


# Show how the values change with saturation at a few places.
idxs = (
    CartesianIndex((200, 100)),
    CartesianIndex((200, 240)),
    CartesianIndex((200, 300)),
    argmax(B_sat2),
    argmin(B_sat2),
)
fig, ax, hm = heatmap(ρ_sat1, axis=(yreversed=true,))
for idx in idxs
    scatter!(ax, idx[1], idx[2], marker='X', markersize=18)
end
Colorbar(fig[:, end+1], hm)
ax.title = "Density (kg/m³)"
file_path = joinpath(save_dir, "density_marked.png")
save(file_path, fig)

sat2 = range(0, 1, length=100)

figs = Dict(
    :M => Figure(),
    :ρ => Figure(),
    :vp => Figure(),
    :B => Figure(),
    :M_processed => Figure(),
    :vp_processed => Figure(),
    :B_processed => Figure(),
)
axes = Dict(k => Axis(f[1, 1]) for (k,f) in pairs(figs))

for idx in idxs
    println(idx)
    M_new = @. M_sat1[idx] * M_sat2[idx] / ((1f0 - sat2) * M_sat2[idx] + sat2 * M_sat1[idx]);
    ρ_new = @. ρ_sat1[idx] + ϕ[idx] * sat2 * (ρ_fl2 - ρ_fl1);
    lines!(figs[:M][1, 1], sat2, M_new)
    lines!(figs[:ρ][1, 1], sat2, ρ_new)

    vₚ_new = ifelse.(M_new .< 0, eltype(M_new)(NaN), sqrt.(abs.(M_new)./ρ_new))
    lines!(figs[:vp][1, 1], sat2, vₚ_new)

    B_new = M_new .- 4f0/3f0 .* μ_sat1[idx]
    lines!(figs[:B][1, 1], sat2, B_new)


    M_new = @. M_sat1[idx] * M_sat2_processed[idx] / ((1f0 - sat2) * M_sat2_processed[idx] + sat2 * M_sat1[idx]);
    lines!(figs[:M_processed][1, 1], sat2, M_new)

    vₚ_new = ifelse.(M_new .< 0, eltype(M_new)(NaN), sqrt.(abs.(M_new)./ρ_new))
    lines!(figs[:vp_processed][1, 1], sat2, vₚ_new)

    B_new = M_new .- 4f0/3f0 .* μ_sat1[idx]
    lines!(figs[:B_processed][1, 1], sat2, B_new)
end


fig = figs[:M]
ax = axes[:M]
ax.title = "P-wave modulus (Pa)"
file_path = joinpath(save_dir, "satfunc_pwave_modulus.png")
save(file_path, fig)

fig = figs[:ρ]
ax = axes[:ρ]
ax.title = "Density (kg/m³)"
file_path = joinpath(save_dir, "satfunc_density.png")
save(file_path, fig)

fig = figs[:vp]
ax = axes[:vp]
ax.title = "P-wave velocity (m/s)"
file_path = joinpath(save_dir, "satfunc_pwave_velocity.png")
save(file_path, fig)

fig = figs[:B]
ax = axes[:B]
ax.title = "Bulk modulus (Pa)"
file_path = joinpath(save_dir, "satfunc_bulk_modulus.png")
save(file_path, fig)


fig = figs[:M_processed]
ax = axes[:M_processed]
ax.title = "P-wave modulus (Pa) (processed)"
file_path = joinpath(save_dir, "satfunc_pwave_modulus_processed.png")
save(file_path, fig)

fig = figs[:vp_processed]
ax = axes[:vp_processed]
ax.title = "P-wave velocity (m/s) (processed)"
file_path = joinpath(save_dir, "satfunc_pwave_velocity_processed.png")
save(file_path, fig)

fig = figs[:B_processed]
ax = axes[:B_processed]
ax.title = "Bulk modulus (Pa) (processed)"
file_path = joinpath(save_dir, "satfunc_bulk_modulus_processed.png")
save(file_path, fig)




# Check Voigt bounds.

# For B_sat1, the mineral bulk modulus is one component. Water is the other component.
maximum_B_sat1 = ϕ .* B_fl1 .+ (1.0 .- ϕ) .* B_min
data = maximum_B_sat1
fig, ax, hm = heatmap(data, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "Maximum allowed B_sat1"
file_path = joinpath(save_dir, "bound_B_sat1_maximum.png")
save(file_path, fig)

data = maximum_B_sat1 .- B_sat1
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "Maximum allowed B_sat1 minus B_sat1"
file_path = joinpath(save_dir, "bound_B_sat1_maximum_diff.png")
save(file_path, fig)

data = ifelse.(data .> 0, 0, data)
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "Maximum allowed B_sat1 minus B_sat1, only negative"
file_path = joinpath(save_dir, "bound_B_sat1_maximum_diff_negative.png")
save(file_path, fig)


# For B_sat2, the mineral bulk modulus is one component. CO2 is the other component.
maximum_B_sat2 = ϕ .* B_fl2 .+ (1.0 .- ϕ) .* B_min
data = maximum_B_sat2
fig, ax, hm = heatmap(data, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "Maximum allowed B_sat2"
file_path = joinpath(save_dir, "bound_B_sat2_maximum.png")
save(file_path, fig)

data = maximum_B_sat2 .- B_sat2
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "Maximum allowed B_sat2 minus B_sat2"
file_path = joinpath(save_dir, "bound_B_sat2_maximum_diff.png")
save(file_path, fig)

data = ifelse.(data .> 0, 0, data)
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "Maximum allowed B_sat2 minus B_sat2, only negative"
file_path = joinpath(save_dir, "bound_B_sat2_maximum_diff_negative.png")
save(file_path, fig)

# We also get a minimum bound on the mineral bulk modulus.
minimum_B_min = (B_sat1 .- ϕ .* B_fl1) ./ (1 .- ϕ)
minimum_B_min = ifelse.(ϕ .== 1, B_min, minimum_B_min)
data = minimum_B_min
fig, ax, hm = heatmap(data, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "Minimum allowed B_min"
file_path = joinpath(save_dir, "bound_B_min_minimum.png")
save(file_path, fig)

data = B_min .- minimum_B_min
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "B_min minus minimum allowed"
file_path = joinpath(save_dir, "bound_B_min_minimum_diff.png")
save(file_path, fig)

data = ifelse.(data .> 0, 0, data)
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "B_min minus minimum allowed, only negative"
file_path = joinpath(save_dir, "bound_B_min_minimum_diff_negative.png")
save(file_path, fig)

# Check Reuss bound.

# The mineral bulk modulus is one component. Water is the other component.
minimum_B_sat1 = 1.0 ./ (ϕ ./ B_fl1 .+ (1.0 .- ϕ) ./ B_min)
data = minimum_B_sat1
fig, ax, hm = heatmap(data, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "Minimum allowed B_sat1"
file_path = joinpath(save_dir, "bound_B_sat1_minimum.png")
save(file_path, fig)

data = B_sat1 .- minimum_B_sat1
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "B_sat1 minus minimum allowed"
file_path = joinpath(save_dir, "bound_B_sat1_minimum_diff.png")
save(file_path, fig)

data = ifelse.(data .> 0, 0, data)
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "B_sat1 minus minimum allowed, only negative"
file_path = joinpath(save_dir, "bound_B_sat1_minimum_diff_negative.png")
save(file_path, fig)

# The mineral bulk modulus is one component. CO2 is the other component.
minimum_B_sat2 = 1.0 ./ (ϕ ./ B_fl2 .+ (1.0 .- ϕ) ./ B_min)
data = minimum_B_sat2
fig, ax, hm = heatmap(data, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "Minimum allowed B_sat2"
file_path = joinpath(save_dir, "bound_B_sat2_minimum.png")
save(file_path, fig)

data = B_sat2 .- minimum_B_sat2
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "B_sat2 minus minimum allowed"
file_path = joinpath(save_dir, "bound_B_sat2_minimum_diff.png")
save(file_path, fig)

data = ifelse.(data .> 0, 0, data)
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "B_sat2 minus minimum allowed, only negative"
file_path = joinpath(save_dir, "bound_B_sat2_minimum_diff_negative.png")
save(file_path, fig)

# We also get a maximum bound on the mineral bulk modulus.
maximum_B_min = (1 .- ϕ) ./ ( (1 ./ B_sat1) .- (ϕ ./ B_fl1))
maximum_B_min = ifelse.(maximum_B_min .< 0, B_min, maximum_B_min)
maximum_B_min = ifelse.(ϕ .== 1, B_min, maximum_B_min)
data = maximum_B_min
fig, ax, hm = heatmap(data, axis=(yreversed=true,))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "Maximum allowed B_min"
file_path = joinpath(save_dir, "bound_B_min_maximum.png")
save(file_path, fig)

fig, ax, hm = heatmap(data, axis=(yreversed=true,), colorrange=extrema(B_min))
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold, color = :white)
ax.title = "Maximum allowed B_min"
file_path = joinpath(save_dir, "bound_B_min_maximum_ranged.png")
save(file_path, fig)

data = maximum_B_min .- B_min
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "Maximum allowed B_min minus B_min"
file_path = joinpath(save_dir, "bound_B_min_maximum_diff.png")
save(file_path, fig)

data = ifelse.(data .> 0, 0, data)
colorrange = get_divergent_colorrange(data)
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange)
Colorbar(fig[:, end+1], hm)
text!(ax, (0, 0), text = get_extrema_string(data), align = (:left, :top), font = :bold)
ax.title = "Maximum allowed B_min minus B_min, only negative"
file_path = joinpath(save_dir, "bound_B_min_maximum_diff_negative.png")
save(file_path, fig)


# Plot bound breakers.
data = zeros(size(B_sat1))
data[B_sat1 .> maximum_B_sat1] .= 1.0
data[B_sat1 .< minimum_B_sat1] .= -1.0
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange=(-1,1))
Colorbar(fig[:, end+1], hm)
ax.title = "Bound-breaking points for B_sat1"
file_path = joinpath(save_dir, "bound_B_sat1_breakers.png")
save(file_path, fig)

data = zeros(size(B_sat2))
data[B_sat2 .> maximum_B_sat2] .= 1.0
data[B_sat2 .< minimum_B_sat2] .= -1.0
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange=(-1,1))
Colorbar(fig[:, end+1], hm)
ax.title = "Bound-breaking points for B_sat2"
file_path = joinpath(save_dir, "bound_B_sat2_breakers.png")
save(file_path, fig)

data = zeros(size(B_sat2))
data[B_sat2_processed .> maximum_B_sat2] .= 1.0
data[B_sat2_processed .< minimum_B_sat2] .= -1.0
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange=(-1,1))
Colorbar(fig[:, end+1], hm)
ax.title = "Bound-breaking points for B_sat2 (processed)"
file_path = joinpath(save_dir, "bound_B_sat2_breakers_processed.png")
save(file_path, fig)

data = zeros(size(B_min))
data[B_min .> maximum_B_min] .= 1.0
data[B_min .< minimum_B_min] .= -1.0
fig, ax, hm = heatmap(data, axis=(yreversed=true,); colormap=:balance, colorrange=(-1,1))
Colorbar(fig[:, end+1], hm)
ax.title = "Bound-breaking points for B_min (processed)"
file_path = joinpath(save_dir, "bound_B_min_breakers.png")
save(file_path, fig)
