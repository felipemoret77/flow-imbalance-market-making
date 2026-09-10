#!/bin/bash
set -euo pipefail

ASOU5_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ASOU5_ROOT"

echo "Running toy_intro_diagnostics.py..."
python toy_intro_diagnostics.py

echo "Running cara_glft_yaware_study.py..."
python cara_glft_yaware_study.py

echo "Running alpha_studies/alpha_reduced_cole_hopf.py..."
cd alpha_studies
python alpha_reduced_cole_hopf.py
cd ..

echo "Running regime_studies/regime_fast_alpha_ergodic_cole_hopf.py..."
cd regime_studies
python regime_fast_alpha_ergodic_cole_hopf.py
cd ..

echo "Regenerating deterministic regime depth overlays..."
python regime_studies/plot_regime_depth_comparisons.py

echo "Regenerating companion diagnostics (robustness, domain convergence)..."
python alpha_studies/robustness_frozen_vs_resolvent.py
python domain_convergence_table.py

echo "All simulations finished successfully."
