#!/bin/bash
# Submit MEMORY-GATED re-runs (the chunking/folding fix).
#
# Background: always-on chunking/folding REGRESSED every run (run5 ~10x, run4
# ~4.4x, run9 ~3x) because `canon` cost is dominated by per-term Butler-Portugal
# (index/dummy permutations), not term count -> extra canons are pure overhead.
# The fix gates fold+chunk on memory pressure (jet._mem_pressure): they fire ONLY
# when process RSS exceeds GRB_MEM_BUDGET_GB. So by default these runs use the
# fast path (#1/#2 + Z-drop + X->Y, zero extra canons) and only trade canons for
# bounded peak if a run actually approaches its RAM ceiling.
#
# This script exports GRB_MEM_BUDGET_GB = 0.7 * requested mem per run, so the gate
# triggers at ~70% of the PBS allocation (PBS kills at 100%). Outputs are tagged
# with $SUFFIX (default "_mg") so they don't clobber old _chunk / _fast logs.
# Make sure Zeus has the latest commit (`git pull`) and flint installed.
#
# Usage:
#   bash submit_memgated.sh                 # default: the redef/build-heavy runs
#   bash submit_memgated.sh 1 4 5 6 9       # explicit
#   SUFFIX=_mg2 bash submit_memgated.sh 5    # custom tag
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs pbs_generated

SUFFIX="${SUFFIX:-_mg}"
BUDGET_FRAC_PCT="${BUDGET_FRAC_PCT:-70}"   # GRB_MEM_BUDGET_GB = this% of requested

# name / queue / walltime / mem  (same table as submit_all.sh)
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

# Default selection: the redef/build-heavy runs (the ones that regressed under
# always-on chunking; 7 already completed on _fast).
want=("$@")
if [ "${#want[@]}" -eq 0 ]; then
  want=(1 4 5 6 9)
  echo "No run numbers given; defaulting to the redef/build-heavy runs: ${want[*]}"
fi

idx=0
for row in "${RUNS[@]}"; do
  idx=$((idx + 1))
  read -r name q wt mm <<< "$row"
  sel=0; for w in "${want[@]}"; do [ "$w" = "$idx" ] && sel=1; done
  [ "$sel" = "1" ] || continue
  mm_num=${mm%gb}
  budget=$(( mm_num * BUDGET_FRAC_PCT / 100 ))
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
export GRB_MEM_BUDGET_GB=${budget}    # fold/chunk only above this RSS (~70% of req)
python -u runs/${name}.py > logs/${tag}.log 2>&1
EOF
  echo "qsub #${idx} ${name}  ->  logs/${tag}.log  (${q}, mem=${mm}, gate@${budget}gb)"
  qsub "$pbs"
done
