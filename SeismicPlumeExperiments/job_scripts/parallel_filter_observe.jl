using Distributed
@everywhere include("../scripts/filter_observe.jl")
using MyUtils: distribute_gpus

if abspath(PROGRAM_FILE) == @__FILE__
    distribute_gpus()
    filter_observe(ARGS)
end

