#!/bin/bash
# Submit the GR-bootstrap heavy-run suite as one single-core PBS job per run.
# Each job writes UNBUFFERED output to logs/<name>.log (python -u), with
# PYTHONHASHSEED=0 for reproducibility. All runs are single-threaded, so the
# cluster runs them in parallel; suite wall-time = slowest single run.
#
# Cluster: Zeus (PBS Pro). Queues used:
#   zeus_new_q   72h walltime, nodes with 128 cores / 1 TB RAM   (big-RAM default)
#   zeus_long_q  168h walltime, nodes with  80 cores / 378 GB    (for >72h monsters)
#
# Usage:
#   bash submit_all.sh            # submit all runs
#   bash submit_all.sh 1 4 9      # submit only runs #1, #4, #9
#
# IMPORTANT: cross-process resume does NOT work (see README) — each run must
# finish within its walltime or it is lost. Walltimes below are the queue caps;
# estimates (~4x/order from laptop timings) put the four zeus_long_q runs near
# their limit, so watch #1/#4/#5/#9. #9 is also the RAM risk on 378 GB — if it
# OOMs, drop it to n=4 and move it to zeus_new_q (no queue offers >72h AND 1TB).
#
# If Zeus is Torque (not PBS Pro), replace the `-l select=...` line with
#   #PBS -l nodes=1:ppn=1
#   #PBS -l mem=<MEM>
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs pbs_generated

# Memory: requested NEAR each queue's node limit (zeus_new_q nodes = 1 TB,
# zeus_long_q nodes = 378 GB), leaving a little headroom so the job still
# schedules (a job can't claim a node's entire RAM; some is reserved for the OS).
# Rationale: PBS kills a job that exceeds its requested mem, and we never
# measured these runs' peak usage; since each is single-core and there are far
# more public nodes than jobs, there's no reason to under-request and risk a
# premature OOM-kill below the RAM the node actually has. Dial down only if you
# want jobs to pack onto shared nodes / start sooner.
#
# run name (without .py)                         queue         walltime     mem
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

want=("$@")  # optional list of 1-based run numbers

idx=0
for row in "${RUNS[@]}"; do
  idx=$((idx + 1))
  read -r name q wt mm <<< "$row"
  if [ "${#want[@]}" -gt 0 ]; then
    sel=0; for w in "${want[@]}"; do [ "$w" = "$idx" ] && sel=1; done
    [ "$sel" = "1" ] || continue
  fi
  pbs="pbs_generated/${name}.pbs"
  cat > "$pbs" <<EOF
#!/bin/bash
#PBS -N grb_${name}
#PBS -q ${q}
#PBS -l select=1:ncpus=1:mem=${mm}
#PBS -l walltime=${wt}
#PBS -j oe
#PBS -o logs/${name}.pbslog
cd \$PBS_O_WORKDIR
export PYTHONHASHSEED=0
python -u runs/${name}.py > logs/${name}.log 2>&1
EOF
  echo "qsub #${idx} ${name}  (${q}, walltime=${wt}, mem=${mm})"
  qsub "$pbs"
done
