# Runtime environment for the bootstrap runs — sourced by every PBS job
# (and handy to `source` interactively before a local test).
#
# WHY THIS EXISTS: Zeus's default `python3` is 3.6 (EOL). A fresh
# `pip install sympy` under 3.6 pulls an ancient sympy (~1.5) whose tensor
# module raises `ValueError: Repeated index` on dummy-index collisions that
# modern sympy auto-renames — so every run crashes immediately. Compute nodes
# start from the default environment, so each job must re-establish a modern
# Python (>= 3.9) + recent sympy here.
#
# EDIT THE TWO LINES BELOW for your Zeus setup, then leave the rest alone.
# (If `module` is not found in batch jobs, uncomment the modules-init line.)

# source /etc/profile.d/modules.sh           # uncomment if `module` is undefined in jobs
module load python/3.11                       # <-- set to `module avail python` output
source "$HOME/Venus_venv/bin/activate"        # <-- venv built with that Python (recent sympy)

# Sanity (printed into the job log): fail loudly if Python is too old.
# Also report sympy's GROUND_TYPES so the log confirms whether the fast
# rational backend is active. For the optimized runs, install it once into the
# venv:  pip install python-flint gmpy2   (Linux wheels; GROUND_TYPES -> flint,
# a ~2x constant factor on top of the #1/#2 redef speedups).
python - <<'PY'
import sys, sympy
from sympy.polys.domains import GROUND_TYPES
print(f"[env] python {sys.version.split()[0]}, sympy {sympy.__version__}, "
      f"GROUND_TYPES={GROUND_TYPES}", flush=True)
assert sys.version_info >= (3, 9), "Python >= 3.9 required (default python3 on Zeus is 3.6)"
PY
