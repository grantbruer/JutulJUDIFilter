__precompile__(false) # Requires.jl in MyUtils breaks precompilation.
module EnsembleJustObsFilters

include("ensemble_justobs_filter.jl")
export EnsembleJustObsMember
export EnsembleJustObsParams
export EnsembleJustObsFilter

using Requires

function __init__()
    @require SlimOptim="e4c7bc62-5b23-4522-a1b9-71c2be45f1df" begin
        @require SeismicPlumeEnsembleFilter = "32f13ed4-4a67-48ae-a811-e7b1b7af3319" begin
            include("ensemble_justobs_filter_assimilate.jl")
        end
    end
end

end # module
