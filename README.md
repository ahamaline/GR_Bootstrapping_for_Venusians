# Venusian Bootstrap of General Relativity

A computer-algebra implementation of the constructive procedure described in
the paper *GR Bootstrapping for Venusians*. Starting from the linear spin-2
wave equation on Minkowski space, the code iteratively builds the full
nonlinear Einstein Field Equation (and its coupling to matter) order by order
in the metric perturbation `h_{μν}`, using only the Helmholtz integrability
conditions and energy-momentum conservation — no input geometry, no
diffeomorphism invariance assumed.

The procedure produces, at every order, both the gravitational field equation
`E^{μν(n)}` and the matter-coupled Lagrangian `L^{(n+1)}` that produces it, in
the strong sense that `δL^{(n+1)}/δh_{μν} = E^{μν(n)}`. Comparing the
result against the standard Einstein-Hilbert expansion verifies the
"bootstrapping claim": the procedure is essentially unique, up to local field
redefinitions.

## What's here

- **`bootstrap/`** — the algorithm: tensor algebra, jet calculus,
  Euler-Lagrange operator, Hilbert and symmetrized-Belinfante
  energy-momentum tensors, Helmholtz-condition machinery (superpotential
  construction + H2 violation + EOM decomposition), and the 6-step
  bootstrap driver (`BootstrapState`).
- **`examples/`** — a series of progressively richer scripts that an
  interested visitor can run to see the algorithm in action. Start with
  `examples/README.md` for a guided tour.
- **`tests/`** — exhaustive validation and regression tests covering pure
  gravity through order 4, plus various matter scenarios (free scalar,
  electromagnetism, Proca, scalar QED, and a "kitchen sink" combining
  EM + charged scalar + charged vector).
- **`DEVELOPMENT_STATUS.md`** — running engineering log: what works, what's
  open, recent debugging lessons. Useful if you're poking at the internals.

## What the code does (one-page summary)

For each order `n = 0, 1, 2, …`, the bootstrap loop in
[`bootstrap/bootstrap_loop.py`](bootstrap/bootstrap_loop.py) runs the
6-step procedure from §4 of the paper:

1. **`E_1`** = `κ T̂[L^{(n)}] + δ_{1,n} W^{μν}`. Energy-momentum tensor of
   the current Lagrangian, plus the leading wave operator at `n=1`. `T̂`
   is either Hilbert (vary `√|g| L̃` by `g_{μν}`) or symmetrized
   Belinfante (canonical Noether + spin improvement, then `(μν)`-symmetrize).
2. **`E_2 = E_1 + Σ_{m<n} X^{(m)} · E^{(n-m)}`**. Carryover: EOM
   corrections chosen at earlier orders propagate forward.
3. **`E_3 = E_2 + X^{(n)} · E^{(0)}`**. The unique EOM correction
   demanded by Helmholtz condition H2 (the EOM must come from a
   Lagrangian). When the H2 violation `Z` is nonzero, it is decomposed
   into pieces proportional to the zeroth-order field equations of `h`
   and the matter fields (the algorithm in
   [`bootstrap/eom_decompose.py`](bootstrap/eom_decompose.py)).
4. **`E_4 = E_3 + X'^{(n)} · E^{(0)}`**. Optional / voluntary EOM
   additions (the "field-redefinition" path). The user supplies arbitrary
   `X'` coefficients; the code checks each one is derivative-free AND
   satisfies the Helmholtz integrability `∂X/∂h` symmetric in `(μν↔αβ)`
   before adding.
5. **`E^{(n)} = E_4 + Δ^{(n)}`** with `Δ^{(n)} = Ψ^{(n)}_{,ρσ}`. The
   superpotential correction demanded by Helmholtz condition H3 (the
   EOM's divergence must agree with conservation). `Ψ` is built from the
   constructive formula in
   [`bootstrap/helmholtz.py`](bootstrap/helmholtz.py).
6. **`L^{(n+1)} = (1/(n+1)) E^{(n)} h + boundary terms`**. Close the loop:
   the new Lagrangian whose EL derivative reproduces `E^{(n)}` exactly.

Each order finishes with two consistency checks: an internal one
(`EL(L^{(n+1)}) == E^{(n)}`), and an external one against the standard
covariant Einstein-Hilbert expansion (`L_ref^{(n+1)}`). When the two
disagree, the code recovers the field redefinition that bridges them,
applies it, and re-verifies.

## Quick start

You only need Python ≥ 3.9 and **SymPy 1.14** — the version the code is
developed and validated against (other versions may canonicalize tensor
expressions differently):

```bash
pip install sympy==1.14
```

**Recommended (especially for high-order / HPC runs):** also install
`python-flint` and `gmpy2`. SymPy then uses the FLINT backend for its rational
arithmetic (`GROUND_TYPES=flint`; check with
`python -c "from sympy.polys.domains import GROUND_TYPES; print(GROUND_TYPES)"`),
a ~2× constant-factor speedup on the symbolic-coefficient work that dominates
the higher orders. On Linux/macOS/Windows x86-64 both ship prebuilt wheels:

```bash
pip install python-flint gmpy2
```

Then run one of the examples:

```bash
python examples/01_pure_gravity_hilbert.py    # ~2 minutes
python examples/02_scalar_matter.py           # ~3 minutes
python examples/03_belinfante_field_redef.py  # ~5 minutes
python examples/04_optional_eom.py            # ~5 minutes
```

Each script prints what it's doing step by step. See
[`examples/README.md`](examples/README.md) for the recommended reading
order and what each example demonstrates.

The bootstrap is symbolic, not numeric — wall times grow roughly 4× per
order. Pure-gravity orders 0..2 close in a few minutes; matter scenarios
through order 3 take 10–30 minutes; the full kitchen-sink Lagrangian
through order 4 takes several hours (see `DEVELOPMENT_STATUS.md` for the
empirical timing table).

## How it's organized (modules at a glance)

```
bootstrap/
├── tensor_algebra.py    — abstract-index tensor algebra (SymPy backend);
│                          TensorHeads for h, dh, ddh; matter-field
│                          registration; fresh-index allocator; canon().
├── jet.py               — jet-space calculus: jet_derivative (∂/∂jet-var,
│                          with required symmetrization), total_derivative
│                          (∂_τ via chain rule on jet variables).
├── euler_lagrange.py    — δL/δφ for any registered field; integration-
│                          by-parts to first-order form.
├── covariant.py         — η→g and ∂→∇ promotion; order-by-order
│                          expansion of the Einstein-Hilbert Lagrangian
│                          and the covariantized matter Lagrangian.
├── energy_momentum.py   — Hilbert energy-momentum tensor (vary √|g| L̃
│                          by g_{μν}); symmetrized Belinfante (canonical
│                          Noether + spin improvement, then symmetrize).
├── helmholtz.py         — H1/H2/H3 condition machinery; the
│                          superpotential constructive formula for n≥2;
│                          the n=1 integral formula; H2 violation tensor.
├── eom_decompose.py     — decomposes a rank-2 or rank-4 EOM-proportional
│                          expression into a linear combination of
│                          zeroth-order EOMs (X_h · E_h^(0) + X_φ · E_φ^(0)).
│                          The 2026-05-26 user algorithm: pick a unique
│                          kinetic / 2nd-derivative monomial signature,
│                          strip from each term, accumulate.
└── bootstrap_loop.py    — BootstrapState: holds L[n], E[n], EOM-term
                           history, L_ref reference; drives the 6-step
                           loop; verifies and applies field redefinitions.
```

## Reading the paper alongside

If you want to understand the *why*, the most direct route is to read §3
("The Helmholtz conditions") and §4 ("The bootstrap procedure") of the
paper, then walk through `examples/01_pure_gravity_hilbert.py` with the
paper open. Every step in the example corresponds to one labelled
equation.

## Status and known limitations

The bootstrap closes end-to-end (with full field-redefinition handling on
the Belinfante path) on:
pure gravity through order 4, free scalar matter, electromagnetism
through order 4, full Proca, and a kitchen-sink Lagrangian combining EM
+ charged complex scalar + charged complex vector matter through order 4.

A small number of corner cases are intentionally left as `RuntimeError`
when reached, with messages pointing at the relevant open-work item in
`DEVELOPMENT_STATUS.md`. The most notable: matter EOMs that have no
2nd-derivative term at all (e.g., a pure mass term for a vector field)
fail the orchestrator's "unique monomial" precondition.

## Citation

If you use this code in academic work, please cite the paper
*"GR Bootstrapping for Venusians"* (citation forthcoming) and link back
to this repository.

## License

(to be decided before public release)
