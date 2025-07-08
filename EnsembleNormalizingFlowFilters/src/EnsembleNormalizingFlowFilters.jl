__precompile__(false) # Requires.jl in MyUtils breaks precompilation.
module EnsembleNormalizingFlowFilters

include("ensemble_normflow_filter.jl")
export EnsembleNormFlowMember
export EnsembleNormFlowParams
export EnsembleNormFlowFilter

using Requires
using Statistics

function __init__()
    @require InvertibleNetworks="b7115f24-5f92-4794-81e8-23b0ddb121d3" begin
        @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
            @require UNet = "0d73aaa9-994a-4556-95d0-da67cb772a03" begin
                @require MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54" begin
                    include("ensemble_normflow_filter_assimilate.jl")
                end
            end
        end
    end
end

end # module
