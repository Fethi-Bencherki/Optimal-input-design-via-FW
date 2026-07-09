#!/bin/zsh
# Reproduce the final Experiment C Monte Carlo study and Figure 2.
# Requires numba: use the cyberrunner env (numpy 1.26, numba 0.65).
set -e
cd "$(dirname "$0")"
PY=/opt/miniconda3/envs/cyberrunner/bin/python

# Final tuned run: 40 seeds (17..56), stochastic protocol, probe 0.3.
$PY expc_run.py --mode tuned --trials 40 --workers 10 \
    --set sigma_probe=0.3 M_fw=400 --out mc_final.npz

# Regenerate the paper figure in place (width=8cm, height=5cm, 70% bands).
$PY expc_make_figure.py --npz mc_final.npz --band 70 \
    --out ../../adaptive_online_lqr_regret_simplified_sampling_band70.tex

# Optional verification (run from the repo root):
#   pdflatex -interaction=nonstopmode -jobname=expCcheck root4.tex  (twice)
#   pdftoppm -png -f 5 -l 5 -r 150 expCcheck.pdf page && open page-5.png
#   rm expCcheck.*
