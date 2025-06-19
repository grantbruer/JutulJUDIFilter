PKGS := EnsembleFilters EnsembleJustObsFilters EnsembleKalmanFilters EnsembleNormalizingFlowFilters
PKGS += MyUtils SeismicPlumeEnsembleFilter

.PHONY: testall $(TEST_PKGS)

_test_prefix := test_
TEST_PKGS := $(foreach env, $(PKGS), $(_test_prefix)$(env))

testall: $(TEST_PKGS)

$(TEST_PKGS):
	pkg="$(patsubst $(_test_prefix)%,%, $@)" && julia -e "import Pkg; Pkg.activate(\"$$pkg\"); Pkg.resolve(); Pkg.instantiate(); Pkg.test()"


.PHONY: testreportall $(TESTREPORT_PKGS) clean_testreport

_testreport_prefix := testreport_
TESTREPORT_PKGS := $(foreach env,$(PKGS), $(_testreport_prefix)$(env))

testreportall: $(TESTREPORT_PKGS)

$(TESTREPORT_PKGS):
	pkg="$(patsubst $(_testreport_prefix)%,%, $@)" && julia --code-coverage=user --color=no ci_scripts/report_pkg.jl "$$pkg" "$$pkg"/report.xml

clean_testreport:
	rm -f ./*/report.xml

.PHONY: build_html

report.html: ci_scripts/junit_to_html.py ci_scripts/junit_template.html $(foreach env,$(PKGS), $(env)/report.xml)
	python ci_scripts/junit_to_html.py $@ $(foreach env,$(PKGS), $(env)/report.xml)


.PHONY: clean_coverage coverage-lcov.info coverage-lcov

PKGS_SRC_DIR := $(foreach env, $(PKGS), $(env)/src)
PKGS_TEST_DIR := $(foreach env, $(PKGS), $(env)/test)
EXTRA_COV_DIRS += SeismicPlumeExperiments/scripts SeismicPlumeExperiments/lib
EXTRA_COV_DIRS += LorenzExperiments/scripts LorenzExperiments/lib
coverage-lcov.info:
	julia ci_scripts/process_coverage.jl "coverage-lcov.info" $(PKGS_SRC_DIR) $(PKGS_TEST_DIR) $(EXTRA_COV_DIRS)

coverage-lcov: coverage-lcov.info
	rm -rf $@
	mkdir -p $@
	genhtml -o $@ $<

clean_coverage:
	julia -e "import Coverage; Coverage.clean_folder(\".\")"
