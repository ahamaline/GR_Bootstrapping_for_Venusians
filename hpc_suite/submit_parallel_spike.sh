#!/bin/bash
# Submit the process-parallel canon BENCHMARK (tests/_parallel_spike.py) to Zeus.
#
# WHY: the parallelization go/no-go hinges on ONE number we cannot measure on the
# Windows dev box -- the fork+COW speedup. The spike's [fork/COW] path passes
# only index ranges to workers (chunks inherited copy-on-write, never pickled),
# which is the production-representative dispatch. It auto-activates on Linux.
#
# Already retired on Windows (so this run is purely about the speedup, not
# correctness): combine_canonical == canon, and the BP-free merge does NOT
# re-pay Butler-Portugal (serial-chunk/whole ~= 1.0). See tests/bench_parallel_canon.py.
#
# This requests a real multi-core node (ncpus=$NCPUS) and sweeps worker counts
# K in $KSWEEP at a fixed problem size $RVAL, so the log shows scaling +
# crossover. Read the [fork/COW] "speedup vs whole" line per K:
#   ~3x+ at K=8  -> build the real pool (50h run -> ~16h; algorithmic risk gone)
#   ~1.5x        -> IPC tax fundamental; pivot to algorithmic term-reduction
#
# Usage:
#   bash submit_parallel_spike.sh                 # defaults below
#   RVAL=3000 KSWEEP="1 4 8 16" NCPUS=16 bash submit_parallel_spike.sh
#   QUEUE=zeus_long_q bash submit_parallel_spike.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs pbs_generated

NCPUS="${NCPUS:-8}"                # cores reserved on the node (>= max K)
KSWEEP="${KSWEEP:-1 2 4 8}"        # worker counts to sweep
RVAL="${RVAL:-1500}"              # problem size (6*RVAL pre-canon terms)
QUEUE="${QUEUE:-zeus_new_q}"
WALLTIME="${WALLTIME:-02:00:00}"
MEM="${MEM:-32gb}"
SUFFIX="${SUFFIX:-}"

tag="parallel_spike${SUFFIX}"
pbs="pbs_generated/${tag}.pbs"
cat > "$pbs" <<EOF
#!/bin/bash
#PBS -N grb_${tag}
#PBS -q ${QUEUE}
#PBS -l select=1:ncpus=${NCPUS}:mem=${MEM}
#PBS -l walltime=${WALLTIME}
#PBS -j oe
#PBS -o logs/${tag}.pbslog
cd \$PBS_O_WORKDIR                      # hpc_suite (qsub launched from here)
source env_setup.sh                     # modern Python + recent sympy (+ flint)
export PYTHONHASHSEED=0
# Spike self-inserts the repo root on sys.path, so cwd here is irrelevant to
# imports; ../tests/ just locates the script file.
{
  echo "[spike] node \$(hostname), nproc=\$(nproc), RVAL=${RVAL}, KSWEEP='${KSWEEP}'"
  for K in ${KSWEEP}; do
    echo ""
    echo "########## K=\$K workers ##########"
    python -u ../tests/bench_parallel_canon.py ${RVAL} \$K
  done
} > logs/${tag}.log 2>&1
EOF

echo "qsub ${tag}  ->  hpc_suite/logs/${tag}.log  (${QUEUE}, ncpus=${NCPUS}, mem=${MEM})"
echo "  RVAL=${RVAL}  KSWEEP='${KSWEEP}'"
qsub "$pbs"
