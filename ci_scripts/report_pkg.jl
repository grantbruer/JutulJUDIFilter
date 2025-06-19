#!julia

using Test
using TestReports

ts = @testset ReportingTestSet "" begin
    pkg_path = ARGS[1]
    outputfilename = ARGS[2]
    pkg_path = realpath(pkg_path)
    push!(Base.LOAD_PATH, pkg_path)

    import Pkg

    Pkg.activate(pkg_path)
    Pkg.resolve()
    Pkg.instantiate()

    Pkg.activate(joinpath(pkg_path, "test"))
    Pkg.resolve()
    Pkg.instantiate()

    include(joinpath(pkg_path, "test", "runtests.jl"))
end

function TestReports.to_xml(v::TestReports.Error)
    message, type, ntest = TestReports.get_error_info(v)
    x_error = TestReports.error_xml(message, type, v.backtrace)
    x_testcase = TestReports.testcase_xml(v, [x_error])
    x_testcase, ntest, 0, 1  # Increment number of errors by 1
end

outputfilename = ARGS[2]
open(outputfilename,"w") do fh
    print(fh, report(ts))
end
exit(any_problems(ts))
