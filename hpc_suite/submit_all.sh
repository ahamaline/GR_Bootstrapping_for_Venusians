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

# Memory: the requests below are deliberately generous (near each queue's node
# limit). We now have measured peaks from completed runs (trailing comment on
# each row; full table in README.md "Measured resource usage"): actual peak is
# 5.5-26.2 GB, i.e. 15-150x BELOW the request. So these requests are massively
# over-provisioned -- you could safely drop every one to mem=64gb (>2x headroom
# on the heaviest, #9 at 26 GB) to pack onto shared nodes / schedule sooner.
# They're left high here only as a no-think safety margin; PBS kills a job that
# exceeds its requested mem, and over-requesting a single-core job is cheap when
# nodes outnumber jobs. Queue choice is driven by WALLTIME, not mem (see README).
#
# run name (without .py)                         queue         walltime     mem      # measured peak
RUNS=(
  "01_pure_gravity_hilbert                       zeus_long_q   168:00:00    360gb"   # 13.6 GB
  "02_scalar_mass_potential_hilbert              zeus_new_q    72:00:00     960gb"   #  6.9 GB
  "03_em_d4_hilbert                              zeus_new_q    72:00:00     960gb"   #  6.7 GB
  "04_proca_mass_hilbert                         zeus_long_q   168:00:00    360gb"   # 24.7 GB
  "05_scalar_optional_eom_tagged_belinfante      zeus_long_q   168:00:00    360gb"   # 14.7 GB
  "06_em_d4_belinfante                           zeus_new_q    72:00:00     960gb"   # 22.1 GB
  "07_conformal_scalar_injection_hilbert         zeus_new_q    72:00:00     960gb"   #  5.5 GB
  "08_kitchen_sink_hilbert                       zeus_new_q    72:00:00     960gb"   # 21.6 GB
  "09_traceless_kitchen_sink_hilbert             zeus_long_q   168:00:00    360gb"   # 26.2 GB
  "10_traceless_kitchen_sink_injection_hilbert   zeus_new_q    72:00:00     960gb"   #  9.6 GB
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
source env_setup.sh          # modern Python (>=3.9) + recent sympy; see env_setup.sh
export PYTHONHASHSEED=0
python -u runs/${name}.py > logs/${name}.log 2>&1
EOF
  echo "qsub #${idx} ${name}  (${q}, walltime=${wt}, mem=${mm})"
  qsub "$pbs"
done
