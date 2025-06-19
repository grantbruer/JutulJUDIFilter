__precompile__(false) # Requires.jl in MyUtils breaks precompilation.
module EnsembleKalmanFilters

include("ensemble_kalman_filter.jl")
export EnsembleKalmanMember
export EnsembleKalmanParams
export EnsembleKalmanFilter

using Requires

function __init__()
    @require KernelMatrices="b172bcd2-82ed-11e9-108e-f9294c34b9e6" begin
        include("ensemble_kalman_filter_assimilate.jl")
    end
end

end # module
