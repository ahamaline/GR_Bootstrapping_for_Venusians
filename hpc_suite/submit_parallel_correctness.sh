#!/bin/bash
# Fork CORRECTNESS check for the parallel builders the original bench missed:
# the INDEX-RETURNING compute_h2_violation (the builder that had the Z!=0 fork bug,
# now canon-merge + substitute-only reindex) and the newly-parallelized field-redef
# substitution. Targeted builders, not a full loop -> fast. Real fork on Zeus.
#
# Expect both lines to end "== whole: True" and the final
#   *** parallel == whole for compute_h2_violation AND field-redef subst ***
# A False would mean a fork-merge regression (dump is in the log).
#
# Usage:
#   bash submit_parallel_correctness.sh                  # EH_ORDER=4 K=8
#   N=5 K=8 bash submit_parallel_correctness.sh          # heavier H2 input
#   K=16 NCPUS=16 bash submit_parallel_correctness.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs pbs_generated

N="${N:-4}"               # einstein_hilbert_lagrangian_order for the H2 input
K="${K:-8}"               # worker count passed straight to parallel_apply_linear
NCPUS="${NCPUS:-$K}"
QUEUE="${QUEUE:-zeus_new_q}"
WALLTIME="${WALLTIME:-02:00:00}"
MEM="${MEM:-64gb}"

tag="parallel_correctness_N${N}_K${K}"
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
  echo "[correctness] node \$(hostname)  nproc=\$(nproc)  N=${N}  K=${K}"
  python -u ../tests/bench_parallel_correctness.py ${N} ${K}
} > logs/${tag}.log 2>&1
EOF

echo "qsub ${tag}  ->  hpc_suite/logs/${tag}.log  (${QUEUE}, ncpus=${NCPUS}, mem=${MEM})"
echo "  watch:  grep '== whole\\|parallel == whole\\|FAIL' hpc_suite/logs/${tag}.log"
qsub "$pbs"
