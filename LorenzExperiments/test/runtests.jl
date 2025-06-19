using Test

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.resolve()
Pkg.instantiate()

@testset "LorenzExperiments tests" begin
    @testset "Simple tests" begin
        include("simple_tests.jl")
    end
end
