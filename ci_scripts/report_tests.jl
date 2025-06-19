#!julia

testfilename = popfirst!(ARGS)
outputfilename = popfirst!(ARGS)

testfilename = realpath(testfilename)
import Pkg

using Test
using TestReports

ts = @testset ReportingTestSet "" begin
    include(testfilename)
end

function TestReports.to_xml(v::TestReports.Error)
    message, type, ntest = TestReports.get_error_info(v)
    x_error = TestReports.error_xml(message, type, v.backtrace)
    x_testcase = TestReports.testcase_xml(v, [x_error])
    x_testcase, ntest, 0, 1  # Increment number of errors by 1
end

open(outputfilename,"w") do fh
    print(fh, report(ts))
end
exit(any_problems(ts))
