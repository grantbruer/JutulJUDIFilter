module MyUtils

include("utils.jl")
export CartesianMesh
export get_colorrange
export get_shot_extrema
export set_circle
export get_circle
export RunningStats
export update_running_stats!
export get_sample_std
export get_connected_mask

include("parula.jl")
export parula

include("file_utils.jl")
export read_ground_truth_plume_all
export read_ground_truth_plume
export read_ground_truth_plume
export read_ground_truth_seismic_baseline
export read_ground_truth_seismic
export read_ground_truth_seismic_all

using Requires

function __init__()
    println("Loading MyUtils")
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("parallel_utils.jl")
        export distribute_gpus
    end

    @require CairoMakie="13f3f980-e62b-5c42-98c6-ff1f3baf88f0" begin
        include("makie_utils.jl")
        export plot_heatmap_from_grid
        export plot_shot_record
        export anim_reservoir_plotter
        export anim_shot_record_plotter
        export record_png
        export plot_anim
    end
end

end # module
