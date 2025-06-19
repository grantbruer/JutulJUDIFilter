using Test

import Pkg
Pkg.add(url="https://bitbucket.org/cgeoga/kernelmatrices.jl")

import KernelMatrices
import EnsembleKalmanFilters

Pkg.rm("KernelMatrices")

include("aqua_test.jl")
