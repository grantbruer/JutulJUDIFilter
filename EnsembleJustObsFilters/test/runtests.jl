using Test

import Pkg
Pkg.develop(path=joinpath(@__DIR__, "..", "..", "SeismicPlumeEnsembleFilter"))

import SlimOptim
import SeismicPlumeEnsembleFilter


include("aqua_test.jl")
Pkg.rm("SeismicPlumeEnsembleFilter")
