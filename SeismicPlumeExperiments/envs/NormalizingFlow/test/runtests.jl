using Test

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.resolve()
Pkg.instantiate()

@testset "Normalizing Flow tests" begin
    include("../../../scripts/filter_assimilate.jl")
    @testset "Assimilate tests" begin
        include("assimilate_tests.jl")
    end
end
