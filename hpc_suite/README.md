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

## Measured resource usage (all 10 completed at target order)

From PBS `resources_used` on the completed runs (each ran to its `*** PASS` at
the suite-target order). **Peak mem is the stable, intrinsic number** — it's
set by the term counts / expression sizes at each order, so it's reproducible
and node-independent; size your `mem` request off this column.

**Walltime is NOT comparable across rows** and is shown only for order-of-
magnitude scale: these runs were submitted at different times with flint on or
off (a ~2× factor) and across intervening code changes, on whatever node PBS
picked, so the absolute hours are confounded. The one robust thing walltime
tells us is the **`cput/wall` ratio (98–99.8%)** — a within-run quantity that's
flint/hardware-independent and shows every run is CPU-bound, not memory-stalled.

| # | run (variant) | **peak mem** | vmem | walltime *(rough)* | cput/wall |
|---|---|---|---|---|---|
| 1 | pure_gravity_hilbert (_mg) | **13.6 GB** | 13.9 GB | ~74h | 99.4% |
| 2 | scalar_mass_potential_hilbert | **6.9 GB** | 7.0 GB | ~29h | 98.7% |
| 3 | em_d4_hilbert | **6.7 GB** | 6.8 GB | ~35h | 99.7% |
| 4 | proca_mass_hilbert (_mg) | **24.7 GB** | 24.9 GB | ~71h | 99.6% |
| 5 | scalar_optional_eom_tagged_belinfante (_mg) | **14.7 GB** | 14.9 GB | ~46h | 99.8% |
| 6 | em_d4_belinfante (_mg) | **22.1 GB** | 22.3 GB | ~64h | 98.3% |
| 7 | conformal_scalar_injection_hilbert (_fast) | **5.5 GB** | 5.7 GB | ~45h | 99.0% |
| 8 | kitchen_sink_hilbert | **21.6 GB** | 21.8 GB | ~39h | 99.8% |
| 9 | traceless_kitchen_sink_hilbert (_mg) | **26.2 GB** | 26.5 GB | ~41h | 99.8% |
| 10 | traceless_kitchen_sink_injection_hilbert | **9.6 GB** | 9.8 GB | ~44h | 99.8% |

Variant in parentheses: `_mg` = memory-gated build, `_fast` = optimized re-run,
otherwise the original first-go submission — peak mem is the same regardless
(the memory gate never fired; see takeaway 2).

**Two takeaways:**

1. **Compute-bound, not memory-bound.** `cput/walltime` is 98–99.8% everywhere
   and `vmem ≈ mem` (no swapping). The only lever for the multi-day runs is CPU
   (→ parallelism), not RAM.
2. **Memory requests are 15–150× over-provisioned.** Peak is **5.5–26.2 GB**
   against 360–960 GB requested. Right-sizing to e.g. `mem=64gb` would still
   leave >2× headroom on the heaviest run (#9, 26 GB) and schedule faster. It
   also means the memory-gated chunking (gate = `0.7×requested` = 252–672 GB)
   **never fires** — peak is 3–10% of the gate.

## Environment ⚠ (do this first)

The code needs **Python ≥ 3.9 with sympy 1.14** — the version this suite was
developed and validated against; match it for identical tensor-canon behavior.
Zeus's default `python3` is **3.6** (EOL) — a `pip install sympy` there pulls an
*ancient* sympy (~1.5) whose tensor module raises `ValueError: Repeated index`
on dummy collisions that modern sympy auto-renames, so every run crashes
immediately.

```bash
module avail python                       # find a newer interpreter
module load python/3.11                   # (or whatever it's called)
python -m venv ~/Venus_venv
source ~/Venus_venv/bin/activate
pip install --upgrade pip
pip install sympy==1.14
pip install python-flint gmpy2            # recommended — see below
```

**Recommended: `python-flint` + `gmpy2`.** With these installed sympy switches
its rational-arithmetic backend to FLINT (`GROUND_TYPES=flint`; verify with
`python -c "from sympy.polys.domains import GROUND_TYPES; print(GROUND_TYPES)"`),
giving a ~2× constant-factor speedup on the multivariate-rational coefficient
work (symbolic `d`, optional-EOM tags) that dominates these runs. On Linux
x86-64 both ship manylinux wheels that bundle FLINT/GMP — no system libs or
compiler needed. `env_setup.sh` echoes the active `GROUND_TYPES` into each job
log so you can confirm per run.

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

**Re-running on optimized code, in parallel with old jobs** — use
`submit_fast.sh`. It is identical to `submit_all.sh` but tags every output with
`$SUFFIX` (default `_fast`) so it does **not** clobber the `logs/NAME.log` of a
still-running old job, and it defaults to the two runs that were redef-bound on
the old code (#5, #7). Already-running jobs loaded the old code at launch and
are unaffected by a `git pull`, so the new fast jobs run safely alongside them.

```bash
bash submit_fast.sh                       # runs 5 and 7 -> logs/NAME_fast.log
bash submit_fast.sh 5 7 8                 # explicit selection
SUFFIX=_opt bash submit_fast.sh 1 4 5 6 7 # custom output tag
```

**Queue assignment** (in the `RUNS` table at the top of `submit_all.sh`):
- **`zeus_new_q`** (72 h, 1 TB RAM) — runs #2, #3, #6, #7, #8, #10, which fit in
  72 h.
- **`zeus_long_q`** (168 h, 378 GB) — runs #1, #4, #5, #9, whose estimated
  wall-time exceeds 72 h.

Queue choice here is driven entirely by **wall-time**, not memory: measured peak
is 5.5–26.2 GB (see "Measured resource usage"), so even zeus_long_q's 378 GB is
~15× more than the heaviest run needs. The 1 TB on zeus_new_q is irrelevant to
these runs; pick the queue solely on the >72 h question.

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
near the 168 h cap, so watch them — an overrun is lost. (Earlier notes flagged
#9 as a RAM risk on the 378 GB node; the completed run settles it — #9 peaked at
**26 GB**, so there is no memory risk on any of these runs. See "Measured
resource usage.") To push a monster's order down to fit a queue, just pass a
smaller `n_max` (e.g. pure gravity n=7 ≈ 21 h fits zeus_new_q). (Building the
name-keyed serialization layer would enable both resume and intra-run
parallelism — see DEVELOPMENT_STATUS — but is not done.)
