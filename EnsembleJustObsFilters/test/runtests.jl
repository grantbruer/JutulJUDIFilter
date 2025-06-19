using Test

import Pkg
Pkg.develop(path=joinpath(@__DIR__, "..", "..", "SeismicPlumeEnsembleFilter"))

import SlimOptim
import SeismicPlumeEnsembleFilter

Pkg.rm("SeismicPlumeEnsembleFilter")

include("aqua_test.jl")
