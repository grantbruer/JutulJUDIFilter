@everywhere include("../scripts/filter_transition.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    filter_transition(ARGS)
end

