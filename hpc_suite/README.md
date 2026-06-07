# GR-bootstrap HPC heavy-run suite

High-order closure runs for an HPC cluster (PBS). Each run is a self-contained
Python script under `runs/` that drives `BootstrapState` to a target order and
**raises on any closure failure** — so a run that finishes printing its `*** PASS`
line is a verified closure at every order it reached.

Each run takes an optional `n_max` argument (`python -u runs/NAME.py [n_max]`),
defaulting to the suite target. Use a small value (`1` or `2`) for a quick
local smoke; omit it for the full HPC run.

## The runs

| # | script | model / procedure | d | target n | notes |
|---|---|---|---|---|---|
| 1 | `01_pure_gravity_hilbert` | pure gravity, Hilbert | sym | 8 | grand-challenge; the core PsiForm+closure |
| 2 | `02_scalar_mass_potential_hilbert` | massive scalar (½∂φ²+m²φ²), Hilbert | sym | 7 | matter T_M + PsiForm + carryover |
| 3 | `03_em_d4_hilbert` | EM (−¼F·F), Hilbert | 4 | 7 | downstairs-A vector |
| 4 | `04_proca_mass_hilbert` | full Proca (upstairs V), Hilbert | sym | 6 | upstairs-V + g_down paths |
| 5 | `05_scalar_optional_eom_tagged_belinfante` | free scalar + tagged optional X_h/X_φ every order, Belinfante | sym | 6 | optional-EOM machinery under Belinfante |
| 6 | `06_em_d4_belinfante` | EM (−¼F·F), Belinfante | 4 | 6 | field-redef decompose/recover/apply chain |
| 7 | `07_conformal_scalar_injection_hilbert` | conformal scalar + dual injection, Hilbert | sym | 5 | BOTH traceless recoveries, rigorously (symbolic d) |
| 8 | `08_kitchen_sink_hilbert` | EM+charged scalar+charged vector+φ̄DV, Hilbert | sym | 5 | heaviest standard matter |
| 9 | `09_traceless_kitchen_sink_hilbert` | conformal charged scalar+massless charged vector+φ²V², Hilbert | 4 | 5 | traceless detection on rich complex matter |
| 10 | `10_traceless_kitchen_sink_injection_hilbert` | #9 + dual injection, Hilbert | 4 | 4 | traceless recovery on rich complex matter |

## Environment ⚠ (do this first)

The code needs **Python ≥ 3.9 with a recent sympy** (match your laptop's sympy
version for identical tensor-canon behavior). Zeus's default `python3` is **3.6**
(EOL) — a `pip install sympy` there pulls an *ancient* sympy (~1.5) whose tensor
module raises `ValueError: Repeated index` on dummy collisions that modern sympy
auto-renames, so every run crashes immediately.

```bash
module avail python                       # find a newer interpreter
module load python/3.11                   # (or whatever it's called)
python -m venv ~/Venus_venv
source ~/Venus_venv/bin/activate
pip install --upgrade pip
pip install sympy==<your-laptop-version>  # python -c "import sympy;print(sympy.__version__)" on the laptop
```

Then edit the two `module load` / `source ...venv...` lines in **`env_setup.sh`**
to match — every PBS job `source`s it, because compute nodes start from the
default (3.6) environment. `env_setup.sh` prints the python/sympy versions into
each log and asserts Python ≥ 3.9, so a misconfigured env fails loudly up front.

## Submitting (PBS — Zeus)

`submit_all.sh` qsubs one 1-core job per run (PBS Pro `select=1:ncpus=1`), each
writing **unbuffered** output to `logs/NAME.log` (`python -u`,
`PYTHONHASHSEED=0`). Every run is single-threaded, so they all run in parallel —
suite wall-time is the *slowest single run*, not the sum.

```bash
cd hpc_suite
mkdir -p logs
bash submit_all.sh            # qsub all 10
bash submit_all.sh 1 4 9      # qsub only runs 1, 4, 9
```

**Queue assignment** (in the `RUNS` table at the top of `submit_all.sh`):
- **`zeus_new_q`** (72 h, 1 TB RAM) — the big-RAM default; runs #2, #3, #6, #7,
  #8, #10, which fit in 72 h. The 1 TB helps the symbolic-d (#7) and
  kitchen-sink (#8) memory.
- **`zeus_long_q`** (168 h, 378 GB) — runs #1, #4, #5, #9, whose estimated
  wall-time exceeds 72 h. They trade 1 TB → 378 GB for the longer walltime.

(If Zeus turns out to be Torque rather than PBS Pro, swap the `-l select=...`
line for `-l nodes=1:ppn=1` + `-l mem=...` — noted in `submit_all.sh`.)

## No checkpoint-resume — but overrun only loses the *in-progress* order

Cross-process resume does **not** work here (module-global matter registry +
sympy tensor-head identity don't survive a pickle into a fresh process), so a
killed job cannot resume. **But this is not a disaster:** each run prints
`order N closed` per order with unbuffered output, so a walltime/OOM kill leaves
every *completed* order in `logs/NAME.log`. Overrun just caps how far a run got
— pick ambitious target orders; the only thing lost is the order in progress.

The real cost of a high `n_max` is therefore **not** overrun risk but the
up-front work it forces: `L_ref^(0..n_max+1)` is precomputed at `BootstrapState`
construction *before order 0 runs*, plus the per-order field-redef applications
to it. Set `n_max` absurdly high and the job can spend its whole walltime
building reference Lagrangians and log no bootstrap order at all — so keep
targets within a few orders of what's expected to complete (the suite defaults
are).

Rough laptop scaling is **~4× wall-time per order** (HPC per-core speed is
comparable, not faster), with measured anchors: #5 n=2 ≈ 16 min, #9 n=1 ≈ 15 min,
#8 n=4 ≈ 6.6 h, pure-gravity n=4 ≈ 20 min. The four **zeus_long_q monsters**
(#1 pure-gravity-8 ≈ 85 h, #4 Proca-6 ≈ 77 h, #5 ≈ 68 h, #9 ≈ 66–90 h) all land
near the 168 h cap, so watch them — an overrun is lost. **#9 is also the RAM
risk** on zeus_long_q's 378 GB (5 matter fields at n=5); if it OOMs, the only
fix is dropping it to n=4 (`python runs/09_*.py 4`) and moving it to zeus_new_q,
since **no single queue offers both >72 h and 1 TB**. To push a monster's order
down to fit a queue, just pass a smaller `n_max` (e.g. pure gravity n=7 ≈ 21 h
fits zeus_new_q). (Building the name-keyed serialization layer would enable both
resume and intra-run parallelism — see DEVELOPMENT_STATUS — but is not done.)
