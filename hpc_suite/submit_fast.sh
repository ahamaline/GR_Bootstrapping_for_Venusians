#!/bin/bash
# Submit OPTIMIZED-code parallel re-runs, alongside any old jobs still running.
#
# Why this exists: the field-redef substitution was the wall (HPC #5 stuck on
# order-2 redefs, #7 spent ~40h on a single order-3 redef step). Optimizations
# #1 (cached df/ddf templates) + #2 (order-bounded substitution) collapse that
# step (~48x from #2, ~1.5x more from #1 on the h-redef). These re-runs use the
# optimized `bootstrap/` (make sure Zeus has the `optimize-field-redef` branch
# checked out: `git pull` / `git checkout optimize-field-redef`).
#
# Non-destructive: every output name is tagged with $SUFFIX (default "_fast"),
# so old, still-running jobs writing to logs/<name>.log are NOT clobbered.
# Already-running jobs loaded the OLD code at start and are unaffected by a pull.
#
# Usage:
#   bash submit_fast.sh               # default: the redef-bound runs (5 and 7)
#   bash submit_fast.sh 5 7           # explicit run numbers
#   SUFFIX=_opt bash submit_fast.sh 1 4 5 6 7   # custom tag
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs pbs_generated

SUFFIX="${SUFFIX:-_fast}"

# Same run table as submit_all.sh (name / queue / walltime / mem).
RUNS=(
  "01_pure_gravity_hilbert                       zeus_long_q   168:00:00    360gb"
  "02_scalar_mass_potential_hilbert              zeus_new_q    72:00:00     960gb"
  "03_em_d4_hilbert                              zeus_new_q    72:00:00     960gb"
  "04_proca_mass_hilbert                         zeus_long_q   168:00:00    360gb"
  "05_scalar_optional_eom_tagged_belinfante      zeus_long_q   168:00:00    360gb"
  "06_em_d4_belinfante                           zeus_new_q    72:00:00     960gb"
  "07_conformal_scalar_injection_hilbert         zeus_new_q    72:00:00     960gb"
  "08_kitchen_sink_hilbert                       zeus_new_q    72:00:00     960gb"
  "09_traceless_kitchen_sink_hilbert             zeus_long_q   168:00:00    360gb"
  "10_traceless_kitchen_sink_injection_hilbert   zeus_new_q    72:00:00     960gb"
)

# Default selection: the two runs that were redef-bound on the old code.
want=("$@")
if [ "${#want[@]}" -eq 0 ]; then
  want=(5 7)
  echo "No run numbers given; defaulting to the redef-bound runs: 5 7"
fi

idx=0
for row in "${RUNS[@]}"; do
  idx=$((idx + 1))
  read -r name q wt mm <<< "$row"
  sel=0; for w in "${want[@]}"; do [ "$w" = "$idx" ] && sel=1; done
  [ "$sel" = "1" ] || continue
  tag="${name}${SUFFIX}"
  pbs="pbs_generated/${tag}.pbs"
  cat > "$pbs" <<EOF
#!/bin/bash
#PBS -N grb_${tag}
#PBS -q ${q}
#PBS -l select=1:ncpus=1:mem=${mm}
#PBS -l walltime=${wt}
#PBS -j oe
#PBS -o logs/${tag}.pbslog
cd \$PBS_O_WORKDIR
source env_setup.sh          # modern Python + recent sympy (+ flint if installed)
export PYTHONHASHSEED=0
python -u runs/${name}.py > logs/${tag}.log 2>&1
EOF
  echo "qsub #${idx} ${name}  ->  logs/${tag}.log   (${q}, walltime=${wt}, mem=${mm})"
  qsub "$pbs"
done
