#!/bin/bash
# Quick CORRECTNESS check for the fork-parallel merge fix (canon-at-merge).
#
# Submits a SHORT job that forces the parallel path on at a low order
# (GRB_PARALLEL_MIN small) with GRB_PARALLEL_VERIFY=1, so every parallel merge is
# compared against F(whole) and prints:
#   [parallel VERIFY] canon-merge vs whole: <a> residual | combine-merge vs whole: <b> residual ...
# Interpret:
#   a==0 and the run reaches "Z = 0 (OK)"  -> FIX CONFIRMED (b>0 shows the old
#                                             combine-merge was the bug).
#   a>0  ("canon-merge STILL WRONG")        -> fix insufficient; the dumped residue
#                                             sample shows the real mechanism.
#
# Usage:
#   bash submit_parallel_verify.sh                     # RUN=1 NMAX=2 K=4 MIN=16
#   RUN=4 NMAX=2 K=4 bash submit_parallel_verify.sh    # proca (failed at order 2)
#   NMAX=3 bash submit_parallel_verify.sh              # heavier check (slower)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs pbs_generated

RUN="${RUN:-1}"
NMAX="${NMAX:-2}"
K="${K:-4}"
MIN="${MIN:-16}"          # GRB_PARALLEL_MIN: low so the parallel path triggers early
NCPUS="${NCPUS:-$K}"
QUEUE="${QUEUE:-zeus_new_q}"
WALLTIME="${WALLTIME:-04:00:00}"
MEM="${MEM:-64gb}"

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
)
name="${RUNS[$((RUN - 1))]}"
[ -n "${name:-}" ] || { echo "bad RUN=$RUN"; exit 1; }

tag="${name}_n${NMAX}_K${K}_verify"
pbs="pbs_generated/${tag}.pbs"
cat > "$pbs" <<EOF
#!/bin/bash
#PBS -N grb_${tag}
#PBS -q ${QUEUE}
#PBS -l select=1:ncpus=${NCPUS}:mem=${MEM}
#PBS -l walltime=${WALLTIME}
#PBS -j oe
#PBS -o logs/${tag}.pbslog
cd \$PBS_O_WORKDIR
source env_setup.sh
export PYTHONHASHSEED=0
{
  echo "[verify] node \$(hostname)  RUN=${name}  NMAX=${NMAX}  K=${K}  MIN=${MIN}"
  GRB_N_WORKERS=${K} GRB_PARALLEL_MIN=${MIN} GRB_PARALLEL_VERIFY=1 \
    python -u runs/${name}.py ${NMAX}
} > logs/${tag}.log 2>&1
EOF

echo "qsub ${tag}  ->  hpc_suite/logs/${tag}.log  (${QUEUE}, ncpus=${NCPUS}, mem=${MEM})"
echo "  RUN=${name}  NMAX=${NMAX}  K=${K}  GRB_PARALLEL_MIN=${MIN}  GRB_PARALLEL_VERIFY=1"
echo "  watch:  grep 'parallel VERIFY\\|Z = 0\\|residual' hpc_suite/logs/${tag}.log"
qsub "$pbs"
