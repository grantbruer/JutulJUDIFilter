ENVS := SeismicPlume NoObservations JustObservations EnKF NormalizingFlow

.PHONY: testall $(TEST_ENVS)

_test_prefix = test_
TEST_ENVS := $(foreach env,$(ENVS), $(_test_prefix)$(env))

testall: $(TEST_ENVS)

$(TEST_ENVS):
	julia envs/$(patsubst $(_test_prefix)%,%, $@)/test/runtests.jl


.PHONY: testreportall $(TESTREPORT_ENVS)

_test_report_prefix = testreport_
TESTREPORT_ENVS := $(foreach env,$(ENVS), $(_test_report_prefix)$(env))

testreportall: $(TESTREPORT_ENVS)

$(TESTREPORT_ENVS):
	env="envs/$(patsubst $(_test_report_prefix)%,%, $@)" && julia --code-coverage=user --color=no ../ci_scripts/report_tests.jl "$$env/test/runtests.jl" "$$env"/report.xml


.PHONY: build_html

report.html: ci_scripts/junit_to_html.py ci_scripts/junit_template.html envs/*/report.xml
	python ci_scripts/junit_to_html.py $@ envs/*/report.xml
