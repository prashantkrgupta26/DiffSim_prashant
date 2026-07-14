# OrgElMorph course — scientific-workflow targets (Phase 0).
#
# These wrap the course's standard harness so a student (or CI) drives the
# whole workflow with one command. Chapters adopt the harness incrementally;
# the *_quick / *_reference targets run the chapters already ported (00 smoke
# + P1) and are extended as later phases port more.
#
#   make course-doctor        env probe (doctor.py) — run this first
#   make tutorials-quick      quick-mode smoke of the ported chapters (<2 min)
#   make tutorials-reference  reference-mode runs (EXPECTED numbers; 5-30 min)
#   make course-figures       regenerate the LaTeX figures from saved data
#   make course-pdf           build the course PDF with tectonic
#   make course-tests         Warp-free unit tests (diagnostics + common)

COURSE      := tutorials/orgelmorph-course
SRC         := $(CURDIR)/src
PY          := PYTHONPATH=$(SRC) python
TECTONIC    := /home/bglab/Mojdeh/ENTER/envs/orgtex/bin/tectonic
LATEX_DIR   := $(COURSE)/latex
LATEX_MAIN  := main.tex

P1          := $(COURSE)/physics/01_ch_binary_energies
SETUP       := $(COURSE)/00_setup_and_smoke_test

.PHONY: course-doctor tutorials-quick tutorials-reference course-figures \
        course-pdf course-tests

course-doctor:
	cd $(SETUP) && $(PY) doctor.py

course-tests:
	$(PY) -m pytest tests/test_diagnostics.py tests/test_course_common.py -q

tutorials-quick:
	cd $(SETUP) && $(PY) run.py --config configs/smoke_cuda.yaml \
		--output outputs/smoke --overwrite
	cd $(P1) && $(PY) run_harness.py --config configs/p1.yaml \
		--mode quick --output outputs/p1_quick --overwrite

tutorials-reference:
	cd $(P1) && $(PY) run_harness.py --config configs/p1.yaml \
		--mode reference --output outputs/p1 --overwrite

course-figures:
	cd $(P1) && $(PY) gen_figures.py || true

course-pdf:
	@test -x $(TECTONIC) || { echo "tectonic not found at $(TECTONIC)"; exit 1; }
	cd $(LATEX_DIR) && $(TECTONIC) $(LATEX_MAIN)
	@echo "built $(LATEX_DIR)/$(basename $(LATEX_MAIN)).pdf"
