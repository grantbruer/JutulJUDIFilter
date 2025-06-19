.PHONY: test testreport

test:
	julia test/runtests.jl

testreport:
	julia --code-coverage=user --color=no ../ci_scripts/report_tests.jl test/runtests.jl report.xml
