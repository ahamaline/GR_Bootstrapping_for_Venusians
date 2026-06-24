# Belinfante ⟺ exponential parametrization of the metric

A derivation, from raw bootstrap output, of the field redefinition that the
**Belinfante** energy–momentum procedure corresponds to. The result:

> **The Belinfante stress tensor is the metric stress tensor of the *exponential
> parametrization* of the metric.** As a (1,1) endomorphism (index raised with η),
>
> $$\;g^{\mu}{}_{\nu} = \big(e^{\,2\kappa H}\big)^{\mu}{}_{\nu}\;,\qquad\text{i.e.}\qquad H^{\mu}{}_{\nu} = \tfrac{1}{2\kappa}\,\log\!\big(g^{\mu}{}_{\nu}\big),$$
>
> where `H` is the redefined graviton and $g^{\mu}{}_{\nu}=g_{\mu\rho}\,\eta^{\rho\nu}$ is
> the metric as a linear map on the tangent space. The down-down metric is then
> $g_{\mu\nu}=\eta_{\mu\rho}\,(e^{2\kappa H})^{\rho}{}_{\nu}$.

**Index position matters:** the matrix `exp`/`log` act on the **mixed-index**
$g^{\mu}{}_{\nu}$ (a genuine endomorphism — positive-definite in Euclidean signature,
so the log is well-defined), **not** on the bilinear form $g_{\mu\nu}$.

This also explains the striking empirical fact that motivated the calculation: the
bootstrapped Einstein field equation is ~2–2.5× **more compact** in the exponential
(Belinfante) frame than in the standard linear split $g_{\mu\nu}=\eta_{\mu\nu}+2\kappa h_{\mu\nu}$
(see runs 2 vs 11). The exponential parametrization is a known device in background-field
/ asymptotic-safety gravity; here it falls out of the bootstrap on its own.

## One obvious mechanism for the compaction (not necessarily the whole story): √|g|

At least one identifiable contributor — we make no claim it accounts for the full
~2–2.5× effect. Using $\det(e^{M})=e^{\operatorname{Tr}M}$ on $g^{\mu}{}_{\nu}=e^{2\kappa H}$,

$$\sqrt{|g|} = e^{\,\kappa\,\operatorname{Tr}H} = e^{\,\kappa\,H^{\mu}{}_{\mu}},$$

which has **exactly one term per order** (a power of the single scalar $\operatorname{Tr}H$).
In the linear split, $\sqrt{|g|}=\sqrt{\det(\eta+2\kappa h)}$ is a determinant-square-root
expansion — at order $n$, a sum over all the contraction patterns the determinant
produces. Since $\sqrt{|g|}$ is the covariant Lagrangian measure, and its variation
$\delta\sqrt{|g|}=\tfrac12\sqrt{|g|}\,g^{\mu\nu}\delta g_{\mu\nu}$ is exactly the trace
term in the Hilbert stress tensor that multiplies the *whole* Lagrangian, collapsing it
to a single $e^{\kappa\operatorname{Tr}H}$ removes one clear source of term proliferation.
(The inverse metric, by contrast, is *not* a differentiator — it is one matrix-power per
order in both frames: $(\eta+2\kappa h)^{-1}=\sum(-2\kappa h)^k$ vs $e^{-2\kappa H}=\sum(-2\kappa H)^k/k!$.)

Whether $\sqrt{|g|}$ alone explains the measured compaction, or other structures in the
covariant expansion (Christoffels, the Ricci/curvature combinations, the matter
covariantization) also simplify in the exponential frame, is not settled here — it would
take a per-contribution term-count comparison to attribute it properly.

## Matter too: it's the exponential VIELBEIN frame

The metric result is the rank-2 case of a single statement. The vielbein
`E = (g^{\mu}{}_{\nu})^{1/2} = e^{\kappa H}` (the *symmetric*-gauge vierbein, `g = E^{\mathsf T}\eta E`,
self-adjoint w.r.t. `\eta`) is the matrix square root of the mixed-index metric, so
`H = (1/\kappa)\log E`. The Belinfante bootstrap reconstructs GR **in this exponential
vielbein frame**, and every field is redefined to its **local-frame component** — one
vielbein per covariant index (inverse for an upper index):

| field | index | redefinition `f_*` composes to | run |
|---|---|---|---|
| scalar `φ` | none | `e⁰` → no redef | 11 |
| `A_μ` | one lower | `A_orig = e^{+κH}A = √g·A` | 6 |
| `V^μ` | one upper | `V_orig = e^{-κH}V = (1/√g)·V` | 13 |
| `g_{μν}` | two lower | `g = e^{2κH}` | 11 |

The covariant `A_μ` and contravariant `V^μ` redefs are exact inverses
(`e^{+κH}` vs `e^{-κH}`), confirming the redefined fields are frame components
`A_a = e_a{}^μ A_μ`, `V^a = e^a{}_μ V^μ`. `compose_A_scalar.py` / `compose_V_scalar.py`
verify the `e^{±κH}` series from the run-6 / run-13 `f_A`/`f_V` redefs.

## Where the input comes from

`f_h_redefs.txt` — the graviton field redefinitions `f_h^(n)`, **recovered automatically**
by the bootstrap at each order while closing run 11 (massive scalar, Belinfante,
`hpc_suite/runs/11_scalar_mass_potential_belinfante.py`). Scraped verbatim from its log.
Each is a single, trace-free **matrix power of h**:

```
f_h^(n)_{μν} = c_n · κ^(n-1) · (h^n)_{μν},   c = {1, 2/3, -1, 32/15, -8/3, 404/63, -34/3, 14336/405, -6418/135}   (n=2..10)
```

The individual `c_n` have no obvious pattern — yet they compose to *exactly* the
exponential. The bootstrap had no notion of "exp"; it produced eight messy rationals
that conspire to `e^{2κH}` through ten orders. That is the evidence the result is a
genuine structural fact.

## Files

| file | what it does |
|---|---|
| `f_h_redefs.txt` | the 8 recovered `f_h^(n)`, scraped from the run-11 log (the input data). |
| `recover_exponential_metric.py` | **the rigorous derivation.** Scrapes → SymPy tensor objects; verifies each `f_h^(n) = c_n κ^(n-1) (h^n)` at the tensor level; composes `φ₂∘…∘φ₉` with real tensor matrix-multiplication (truncate @ order 9); checks the composed `h` equals the `g=exp(2κH)` expansion term-by-term. No scalar reduction, no hardcoded series. |
| `compose_scalar.py` | the same composition done as a fast scalar (per-eigenvalue) power series — valid because the redefs are commuting trace-free matrix powers. Quick illustration / cross-check; identifies the 1/k! → `exp` series directly. |

## Reproduce

```bash
python -u recover_exponential_metric.py     # rigorous tensor-level proof (CONFIRMED orders 1..10)
python -u compose_scalar.py                 # fast scalar version (shows the 1/k! coefficients)
```

Both import the project's `bootstrap` tensor algebra (run from anywhere inside the repo).
