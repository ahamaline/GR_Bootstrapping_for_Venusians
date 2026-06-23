#!/bin/bash
# End-to-end builder-chunk parallelism measurement: run a FULL bootstrap case
# serial vs fork-parallel and report the real (Amdahl-limited) speedup.
#
# WHY a same-node A/B: cross-node Zeus walltime varies 2-3x with hardware, which
# is exactly what made the combine deep= "regression" a phantom. So this submits
# ONE PBS job (ncpus=$NCPUS) that runs the chosen case TWICE on the SAME node --
# serial (GRB_N_WORKERS=1) then parallel (GRB_N_WORKERS=$K) -- and prints both
# wall times. The ratio is on identical hardware => trustworthy. (apply_linear is
# inert at N_WORKERS=1, so the serial leg is the true baseline of this same code.)
#
# Size it with NMAX so both legs fit walltime: the serial leg dominates, so pick
# an order whose serial run is a few hours (e.g. proca NMAX=4, gravity NMAX=4).
#
# Usage:
#   bash submit_parallel_run.sh                         # RUN=4 (proca) NMAX=4 K=8 AB
#   RUN=1 NMAX=4 K=8 bash submit_parallel_run.sh        # pure gravity order 4
#   K=8 NMAX=4 MODE=parallel bash submit_parallel_run.sh   # parallel leg only
#   RUN=4 NMAX=4 K=8 NCPUS=8 bash submit_parallel_run.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs pbs_generated

RUN="${RUN:-4}"               # run number (see table below)
NMAX="${NMAX:-4}"             # bootstrap order to reach (sizes the A/B walltime)
K="${K:-8}"                   # parallel worker count (GRB_N_WORKERS)
NCPUS="${NCPUS:-8}"           # cores reserved on the node (>= K)
MODE="${MODE:-ab}"           # ab = serial+parallel same node; parallel = parallel only
QUEUE="${QUEUE:-zeus_long_q}"
WALLTIME="${WALLTIME:-72:00:00}"
MEM="${MEM:-360gb}"
SUFFIX="${SUFFIX:-_par}"
EXCL="${EXCL:-}"             # set EXCL=1 to reserve the whole node (no other tenants)
excl_line=""
[ -n "$EXCL" ] && excl_line="#PBS -l place=excl"

# run number -> script base name (matches submit_all.sh / runs/).
RUNS=(
  "01_pure_gravity_hilbert"
  "02_scalar_mass_potential_hilbert"
  "03_em_d4_hilbert"
  "04_proca_mass_hilbert"
  "05_scalar_optional_eom_tagged_belinfante"
  "06_em_d4_belinfante"
  "07_conformal_scalar_injection_hilbert"
  "08_kitchen_sink_hilbert"
  "09_traceless_kitchen_sink_hilbert"
  "10_traceless_kitchen_sink_injection_hilbert"
  "11_scalar_mass_potential_belinfante"
  "12_kitchen_sink_matter_optional_eom_hilbert"
)
name="${RUNS[$((RUN - 1))]}"
[ -n "${name:-}" ] || { echo "bad RUN=$RUN"; exit 1; }

tag="${name}_n${NMAX}_K${K}${SUFFIX}"
pbs="pbs_generated/${tag}.pbs"
cat > "$pbs" <<EOF
#!/bin/bash
#PBS -N grb_${tag}
#PBS -q ${QUEUE}
#PBS -l select=1:ncpus=${NCPUS}:mem=${MEM}
#PBS -l walltime=${WALLTIME}
${excl_line}
#PBS -j oe
#PBS -o logs/${tag}.pbslog
cd \$PBS_O_WORKDIR
source env_setup.sh
export PYTHONHASHSEED=0
export GRB_PARALLEL_MIN=128
{
  echo "[par-run] node \$(hostname)  nproc=\$(nproc)  RUN=${name}  NMAX=${NMAX}  K=${K}  MODE=${MODE}"
  if [ "${MODE}" = "ab" ]; then
    echo ""; echo "########## SERIAL (GRB_N_WORKERS=1) ##########"
    t0=\$SECONDS
    GRB_N_WORKERS=1 python -u runs/${name}.py ${NMAX}
    echo "[par-run] SERIAL wall: \$((SECONDS - t0))s"
  fi
  echo ""; echo "########## PARALLEL (GRB_N_WORKERS=${K}) ##########"
  t0=\$SECONDS
  GRB_N_WORKERS=${K} python -u runs/${name}.py ${NMAX}
  echo "[par-run] PARALLEL wall: \$((SECONDS - t0))s"
} > logs/${tag}.log 2>&1
EOF

echo "qsub ${tag}  ->  hpc_suite/logs/${tag}.log  (${QUEUE}, ncpus=${NCPUS}, mem=${MEM}, walltime=${WALLTIME})"
echo "  RUN=${name}  NMAX=${NMAX}  K=${K}  MODE=${MODE}"
qsub "$pbs"
