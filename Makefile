# Makefile for the flow-imbalance-market-making repository.
# The engines import one another by module name from their own directories and
# write their outputs to imagens_tex/<study>/ (see README "Repository layout").

PY ?= python

.PHONY: help paper test regen figures replot clean

help:
	@echo "Targets:"
	@echo "  make paper    - compile paper/main.tex -> paper/main.pdf (needs tectonic or latexmk)"
	@echo "  make test     - run the closed-form / cache regression tests"
	@echo "  make regen    - rerun the three RHJB engines and the Monte Carlo campaign (slow)"
	@echo "  make figures  - redraw the CSV-driven manuscript figures into paper/images_final/"
	@echo "  make replot   - redraw the summary figures of the engines from their *_summary.csv"
	@echo "  make clean    - remove LaTeX build artifacts and __pycache__"

paper:
	@cd paper && if command -v tectonic >/dev/null 2>&1; then \
		tectonic main.tex; \
	elif command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; \
	else \
		echo "error: install tectonic or latexmk to build paper/main.pdf" >&2; exit 1; \
	fi

test:
	$(PY) -m unittest -v \
		closed_form_tests.test_cara_aware_closed_forms \
		closed_form_tests.test_paired_cara_closed_form_mc \
		closed_form_tests.test_replot_caches \
		closed_form_tests.test_regime_sccubic

regen:
	bash regen_all.sh

figures:
	$(PY) repro/depth_figs_from_csv.py
	$(PY) repro/closed_form_anatomy_figs.py
	$(PY) repro/paired_ladder_fig.py
	$(PY) repro/regime_depth_figs.py
	$(PY) repro/inventory_by_regime_fig.py

replot:
	$(PY) replot_summary_figures.py

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg paper/*.pdf
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
