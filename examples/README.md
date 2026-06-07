# Examples

Five self-contained scripts that walk you through what the bootstrap
procedure does. Each runs from the command line with no arguments
(`python examples/<name>.py`), prints the steps as it goes, and
demonstrates one feature at a time. Read them in order if you're new.

| File | What it shows | Wall time |
|---|---|---|
| `01_pure_gravity_hilbert.py` | The simplest case: pure-gravity bootstrap with the Hilbert energy-momentum prescription, through order 2. You watch the 6-step loop fire at each order: `T̂[L^(n)]`, the wave operator, the Helmholtz checks, the superpotential, and the close-the-loop step. Verifies the result against the Einstein-Hilbert expansion. | ~2 min |
| `02_scalar_matter.py` | A free massless scalar coupled to gravity. Shows how matter changes the field equation (`E^(0) = κ T_M[L_φ]`) and how `T_M` then enters the higher-order bootstrap. Closes through order 2. | ~3 min |
| `03_belinfante_field_redef.py` | Same scalar, but using the symmetrized Belinfante energy-momentum prescription. Even though scalar matter has no spin at the bare T_M level, step 1 at every n ≥ 1 sees `T̂[L^(n)]` where `L^(n)` carries h × matter structure — and `h` itself has spin, so Belinfante's spin improvement makes T̂ differ from Hilbert. The bootstrap discovers and applies h-redefs to reconcile with the standard EH reference. Shows the integrability check, the recovery of `f_h^(n+1)`, and the substitution into `L_ref`. | ~10-15 min |
| `04_optional_eom.py` | The "voluntary path": user-supplied optional EOM terms (a form of field redefinition the user chooses up front). Pick any `X` that's derivative-free and Helmholtz-integrable; the bootstrap accepts, applies it, and produces a self-consistent set of equations. Demonstrates how the SAME free-scalar theory can be written in many equivalent (related by field redefinition) forms. | ~10-15 min |
| `05_electromagnetism_belinfante.py` | EM on Belinfante: spin contributions at both the bare matter level AND the h-spin level (see example 03), so this is the most active demonstration of the EOM-correction + field-redefinition machinery. Pure EM (Maxwell) bootstrapped through order 1 with `em_procedure='belinfante'`. Step 3's mandatory EOM correction fires, and the orchestrator's EOM-decomposition machinery handles it. | ~5 min |

For deeper validation (e.g., Hilbert through order 4, the full
multi-matter "kitchen sink" Lagrangian), see the `tests/` directory at
the repo root.

## What you'll see when you run an example

Each example prints output like:

```
============================================================
  BOOTSTRAP ORDER n = 1
============================================================
    Computing Hilbert energy-momentum tensor from L^(1) (2 terms)
    Adding wave operator W^(mu nu) (6 terms)
  Step 1 (E_1): 12 terms (6 h-only, 6 phi)
  Step 2 (E_2): 12 terms (6 h-only, 6 phi)
    H2 check: Z = 0 (OK)
  Step 3 (E_3): 12 terms (6 h-only, 6 phi)
  Step 4 (E_4): 12 terms (6 h-only, 6 phi) (no optional terms applied)
    Computing superpotential Psi^(1) via integral formula (n=1)
    Psi^(1) = 0 (no superpotential term)
  Step 5 (E^(1)): 12 terms (6 h-only, 6 phi)
  Step 6 (L^(2)): 8 terms (4 h-only, 4 phi)
  Verify EL(L^(2)) == E^(1): OK
  Verify vs L_ref^(2): EL-equivalent (OK)
```

Term counts are broken down by field content (`h-only`, per matter
field, or matter-matter cross terms) — useful for tracking what kind of
structure each step produces.
