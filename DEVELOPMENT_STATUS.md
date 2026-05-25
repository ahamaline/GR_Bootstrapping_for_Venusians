# GR Bootstrapping — Status & Continuation Guide

## What this is

This codebase implements the iterative procedure of the paper *GR Bootstrapping for Venusians* (in this folder). Starting from the linear spin-2 wave equation, it constructs the Einstein Field Equation order by order in h_{μν}, using energy-momentum conservation and the Helmholtz conditions. The eventual goal is to verify that the procedure uniquely produces the EFE (up to field redefinitions). The paper spells out the 6-step bootstrap loop in §4; a one-page map is at the bottom of this document.

## Headline result so far

The bootstrap closure has been verified **at n=2, n=3, and n=4 in pure gravity, with the Hilbert energy-momentum procedure**. For each order n, the identity

  EL(L_EH^{(n+1)}) − κ T_H[L^{(n)}]  =  κ Ψ^{(n)}_{,ρσ}

holds exactly, where Ψ^{(n)} is produced by the constructive formula (PsiForm) from the paper. Closure at three successive orders — each one tested independently with the same algorithm, no bookkeeping changes — is strong empirical evidence that the procedure is uniformly correct as n grows.

**Kitchen-sink matter result (2026-05-25).** The "everything bagel" Lagrangian — EM + charged complex scalar + charged complex vector (upstairs V) + the φ̄ D_μ V^μ interaction + H.C. — closes the bootstrap **end-to-end at orders 0..4**. At every order, EL(L^(n+1)) == E^(n) and EL(L_ref^(n+1)) matches E^(n) modulo boundary terms. Ψ symmetries (sym_mn, sym_rs, cyclic) all pass at n=1..4. Term counts: E_1 grows 82 → 363 → 1330 → 3560 → 8786, E^(n) (after Ψ) 82 → 289 → 764 → 1700 → 3449, L^(n+1) (after IBP) 65 → 166 → 339 → 663 → 1174. Total wall ≈ 6.6 hours (n=4 alone ≈ 4.8 h). The final piece was a missing Christoffel chain rule for ∇A in `energy_momentum.py`: the F-antisymmetry that cancels Γ inside F_μν at order 0 does NOT carry over to T_M at order ≥ 1, so each registered downstairs-A field needs a `∂_ρ A_σ → ∇_ρ A_σ = dA(σ, ρ) − Γ^τ_{ρσ} A_τ` substitution inside `_christoffel_via_substitution`. (Open puzzle: scalar QED — same `A` head, same path — passed orders 0..2 without this fix; either there's a genuine cancellation specific to scalar-QED structure or the test is succeeding accidentally and will fail at higher orders. See open-work item 11.)

| Quantity | n=2 | n=3 | n=4 |
|---|---:|---:|---:|
| L^{(n)} = L_EH^{(n)} (after IBP) | 4 | 14 | 38 |
| T_H[L^{(n)}] | 30 | 100 | 315 |
| Ψ^{(n)} | 15 | 30 | 48 |
| Ψ symmetries (sym_mn, sym_rs, cyclic) | all pass | all pass | all pass |
| Δ^{(n)} = Ψ_{,ρσ} | 29 | 98 | 218 |
| L_EH^{(n+1)} (after IBP) | 4 | 66 | 136 |
| EL(L_EH^{(n+1)}) | 37 | 128 | 346 |
| (EL(L_EH^{(n+1)}) − κ T_H) − κ Δ | **0** | **0** | **0** |
| Wall time | ~1 min | ~5 min | ~20.5 min |

Along the way, a latent bug in `remove_second_derivatives` was uncovered and fixed: at n≥3 the IBP code was reusing the ddh's contracted dummy as the spacetime-derivative index in `total_derivative`, producing two same-name same-sign indices on one factor (sympy: *"two equal covariant indices"*). The fix rebuilds the ddh factor and its unique dummy-partner factor with a fresh name *before* invoking `total_derivative`; `substitute_indices` is not enough here because it's a silent no-op on bound dummies. See [bootstrap/euler_lagrange.py:174-217](bootstrap/euler_lagrange.py#L174).

## Module status

| Module | State |
|---|---|
| `tensor_algebra.py` | Working. Abstract-index algebra over SymPy `tensor.tensor`; TensorHeads for h, dh, ddh with correct symmetries; matter-field registration (`register_scalar_field`, `register_vector_field` for naturally-down A, `register_upstairs_vector_field` for naturally-up V); `fresh_indices()`; `canon()` (includes `contract_metric()`); order-in-h filtering. `NATURAL_POSITIONS` lives here (used by both energy_momentum and covariant). |
| `jet.py` | Working. `jet_derivative()` with ½-symmetrization on symmetric pairs and Leibniz; `total_derivative()` via chain rule on jet variables. |
| `euler_lagrange.py` | Working. `euler_lagrange()` and `remove_second_derivatives()`. Recovers the 6-term linearized Einstein tensor from L_h^{(2)}; the IBP path is now order-3+ safe (see above). |
| `covariant.py` | Working through order 5 for EH; arbitrary order for `matter_lagrangian_order(L_M, n)` (scalar, downstairs-A gauge-invariant vector, AND upstairs-V vector matter). EH order 3 = 28 terms (~10 s), order 4 = 66 terms (~20 s), order 5 = 136 terms (~45 s). `matter_lagrangian_order` covariantizes (a) η^{μν} → g^{μν} via `inverse_metric_order`, (b) η_{μν} → g_{μν} via the new `metric_order` (only orders 0 and 1 are non-zero since g_{μν} = η + 2κh), and (c) ∂_ρ V^σ → ∇_ρ V^σ = dV(σ, ρ) + Γ^σ_{ρτ} V^τ for each matter field registered as upstairs (index_pos='up'). Christoffel-corrections to dV are dispatched per field at composition time so each upstairs-dV factor independently chooses its h-order. |
| `energy_momentum.py` | Working for `hilbert_energy_momentum()`. Sums four contributions to δ(√|g| L̃)/δg_{μν}: A from δ√|g|, B from δg^{αβ} (variation of explicit g^{...} factors in L_cov), D from δg_{αβ} (variation of explicit g_{...} factors in L_cov; the two are tracked as separate `ginv` and `g_down` jet heads, dispatched by index sign in `replace_metric_with_ginv`), and C from the Christoffel chain rule. `_christoffel_via_substitution` handles dh, the dV → ∇V substitution for upstairs-vector matter fields, AND the dA → ∇A substitution for downstairs-vector matter fields. The dA branch was added 2026-05-25 — without it, T_M[L^(n)] at n ≥ 1 silently dropped a Γ A piece that broke EL self-consistency for any Lagrangian where A couples directly to other matter (e.g. e A V_partner cross-coupling in charged Proca). F·F-only Lagrangians (L_EM, scalar QED at the orders we ran) escaped this because F_μν's antisymmetry cancels Γ inside F itself at order 0. Verified on L_h^{(2)}, L_EH^{(3)}, free scalar, L_EM, L_Proca-mass, full Proca, charged Proca, and the kitchen-sink Lagrangian through n=4. |
| `helmholtz.py` | `compute_superpotential_n2()` (PsiForm for n ≥ 2), `compute_superpotential_n1()` (paper eq. 23, integral form for n = 1, scalar AND non-scalar matter — the rank-r generalization sums each field's tensor indices via the natural extension; the wrt-indices for the field derivative are signed OPPOSITE to the field's natural positions so the resulting deltas are clean Kronecker pairs, and the velocity factor is the field in its natural form). `superpotential_divergence()`, `verify_psi_symmetries()` — all working. `compute_h2_violation()` and `compute_eom_correction()` are placeholders. **Three latent bugs fixed during the upstairs-V work:** (i) for naturally-up matter fields, the previous code passed `field_indices` always UP, producing raising-η factors instead of Kronecker deltas (Ψ ended up with the wrong index structure); (ii) the multiplication `velocity × h × scaled` sometimes left a TensAdd as one arg of an outer TensMul; `_decompose_tensmul` (used by `_integrate_lambda`) silently dropped any TensAdd args, collapsing Ψ to a phantom 3-index expression — fixed by `.expand()` before integration; (iii) the **bracket itself** sometimes had top-level args of shape `TensMul(Rational, TensAdd)` (e.g. when ``Rational(1/2) * d2M_B`` didn't distribute), and `_scale_matter_fields` — also using `_decompose_tensmul` — silently dropped the inner TensAdd, leaving its matter-field factors unscaled and giving the wrong λ-power → non-cyclic-symmetric Ψ. Fixed by `.expand()` on the bracket before `_scale_matter_fields`. |
| `bootstrap_loop.py` | **Working end-to-end for pure gravity (orders 0–3), massive scalar matter (orders 0–3), complex scalar (two real components, orders 0–2), pure EM L_M = −¼ F·F (orders 0–2), AND scalar QED (charged scalar coupled to EM, orders 0–2).** `BootstrapState(L_matter, n_max=N).run_order(n)` drives the 6-step loop, performs EL self-consistency at each order (EL(L^{(n+1)}) == E^{(n)}), and runs the verification cycle against a pre-computed `L_ref^{(k)}` for k=0..n_max+1 (gravity + matter contributions). Step 3 (mandatory H2-EOM) currently assumes Z=0 (Butcher's claim for pure-gravity-Hilbert); the field-redefinition decomposition (called when L^{(n+1)} differs from L_ref^{(n+1)} by more than boundary terms) and Step 4 (optional EOM) are intentional `NotImplementedError` stubs so we trip loudly when Belinfante or non-trivial field-redef enters the picture. |

## Test inventory

| Test | Coverage |
|---|---|
| `test_smoke.py` | Tensor algebra basics. |
| `test_W.py` | EL(L_h^{(2)}) = W^{μν}. |
| `test_covariant.py` | L_EH expansion through order 3. |
| `test_sign_consistency.py` | EL(L_EH^{(2)}) matches the hand-written L_h^{(2)} (sign convention). |
| `test_hilbert.py`, `test_hilbert_debug.py` | Internals of the Hilbert procedure. |
| `test_hilbert_scalar.py` | T_H of a free scalar matches ∂_μφ ∂_νφ − ½ η_{μν} (∂φ)². |
| `test_hilbert_gravity.py` | κ⁻¹ EL(L_EH^{(3)}) − T_H[L_h^{(2)}] is identically conserved. |
| `test_bootstrap_pure_gravity.py` | Step 6 (close-the-loop): (1/(n+1)) h E^{(n)} after IBP recovers L_EH^{(n+1)} at n=1, 2. |
| `test_superpotential_pure_gravity.py` | **End-to-end closure at n=2, pure gravity.** |
| `test_superpotential_pure_gravity_n3.py` | **End-to-end closure at n=3, pure gravity.** |
| `test_superpotential_pure_gravity_n4.py` | **End-to-end closure at n=4, pure gravity.** Latest closure milestone. |
| `test_bootstrap_loop_pure_gravity.py` | **Standardized BootstrapState runner (pure gravity)**. Pass `n_max` on the command line (default 3); drives the loop end-to-end with EL self-consistency at each step and verification cycle against L_ref. Passes at n_max=3 in ~8 min. |
| `test_bootstrap_loop_scalar.py` | **BootstrapState with massless scalar matter, orders 0–1**. Verifies that `E^{(0)} == κ T_M` (textbook), L_ref expansion of √|g| L̃_M produces the right shape at orders 1 and 2, and the n=1 superpotential integral formula closes the loop. Ψ^{(1)} = 0 here, which is the expected outcome — see open-work item 7. |
| `test_bootstrap_loop_scalar_potential.py` | **BootstrapState with massive scalar (kinetic + m² φ² potential), orders 0–3.** Exercises matter L_ref through order 4 (83 terms), the n=1 integral formula, AND PsiForm at n=2 and n=3 with matter contributions inside M. All Ψ symmetries pass; EL and L_ref verification pass at every order. ~7 min total wall (n=3 alone is ~6 min). |
| `test_bootstrap_loop_complex_scalar.py` | **BootstrapState with a complex scalar (two real components, same mass) at orders 0–2.** Same form as the massive-scalar test but with two registered scalar fields whose sum is L = −∂φ*·∂φ − m² φ*φ in real-component form. Confirms multiple matter fields work with zero new code. ~80 s wall. |
| `test_hilbert_em.py` | T_H[L_EM = −¼ F_{μν} F^{μν}] reproduces the textbook Maxwell stress-energy tensor exactly (6 terms, 0.6 s). |
| `test_bootstrap_loop_em.py` | **BootstrapState with pure EM (L = −¼ F·F), orders 0–2.** First non-scalar matter case, confirming gauge-invariant vector matter threads through `matter_lagrangian_order`, `hilbert_energy_momentum`, PsiForm, and the verification cycle. ~120 s wall. |
| `test_bootstrap_loop_charged_scalar.py` | **BootstrapState with full scalar QED (charged complex scalar coupled to EM), orders 0–2.** L_M contains kinetic + mass for two real components of φ, F·F for the EM field, the minimal-coupling current −eA^μJ_μ, and the seagull −½e² A² φφ. All three matter fields (φ₁, φ₂, A) run through every step including `compute_superpotential_n1`; Ψ^(1) = 0 as expected on the default path, Ψ^(2) symmetries verified, EL and L_ref closures pass. ~3 min wall. |
| `test_hilbert_proca_mass.py` | T_H[−½ m² V^μ V_μ] with V upstairs (`register_upstairs_vector_field`) reproduces the expected −m² V^μ V^ν − ½ m² η^{μν} V·V (2 terms, ~0.1 s). Minimal sanity check for the new `g_down` jet head and `term_D` path in `hilbert_energy_momentum`. |
| `test_bootstrap_loop_proca_mass.py` | **BootstrapState with the Proca mass term only (no F·F), V upstairs, orders 0–2.** First test that exercises the downstairs-metric expansion (`metric_order`, term_D) end-to-end. All closures pass; ~90 s wall. |
| `test_bootstrap_loop_proca.py` | **BootstrapState with the full Proca Lagrangian (−¼ F·F − ½ m² V^μ V_μ, V upstairs), orders 0–2.** Exercises both the downstairs-metric path AND the upstairs-vector covariant-derivative path (`christoffel_order` in `matter_lagrangian_order`, the dV-branch of `_christoffel_via_substitution`). Passes orders 0..2 end-to-end after the bracket-expand fix in `compute_superpotential_n1`. Ψ^(1) is nonzero (12 terms — allowed because sympy's ddV index canonicalization breaks the strict-zero derivation) and cyclic-symmetric; Ψ^(2) symmetries also OK; EL self-consistency and L_ref verification pass at every order. Wall: ~21 min total (n=2 alone is ~18 min). |
| `test_bootstrap_loop_charged_vector_qed.py` | **Kitchen-sink Lagrangian — EM + charged complex scalar + charged complex vector + φ̄ D_μ V^μ + H.C., orders 0–4.** 28-term L_M after canon, every order closes (EL self-consistency AND L_ref verification). Term counts climb steeply: E_1 82 → 363 → 1330 → 3560 → 8786 terms across n=0..4. Ψ symmetries OK at n=1..4. Wall: ~6.6 h total (n=4 alone ≈ 4.8 h on the dev machine). Largest matter milestone so far; relies on the 2026-05-25 ∇A Christoffel fix in `energy_momentum.py`. |

## Open work, in priority order

1. **n=5 pure-gravity closure.** Same script at the next order. Wall-time projection from the n=2 → n=3 → n=4 progression (1 min → 5 min → 20.5 min, ~4× per order): roughly 80–120 min for n=5, dominated by `compute_superpotential_n2` (already 10 min at n=4 on a 315-term M) and EL of L_EH^{(6)}. Without further optimization, n=5 is the largest tractable order on this machine.
2. **Further performance work.** Two rounds of optimization landed so far: (a) one-shot `TensAdd(*terms)` via the new `_sum_terms` helper across `jet.py` / `energy_momentum.py` / `euler_lagrange.py` (sympy's `TensAdd.__add__` re-normalizes on every call, so the loop variant paid N× the bookkeeping; net ~45 % at n=3); (b) consolidated the three per-term canon calls in `compute_superpotential_n2`'s `_wrap` helper into a single canon over the combined bracket (sub-linear canon-on-canon vs three medium canons). Two attempted optimizations REGRESSED and were reverted: a canon-result cache (10 % hit rate didn't pay for hash overhead on big tensors) and deferring canon to the outermost recursive call (single huge uncanon'd TensAdd was slower than many small canons). The cProfile data now points at `contract_metric` and `canon_bp` inside sympy itself, with `_expand_hint` called tens of millions of times for ~80 s self-time. Further wins likely require either patching sympy or restructuring the algorithm so canon is called on smaller pieces.
3. **Field-redefinition decomposition (skeleton needs filling).** The verification cycle in `BootstrapState` handles cases (a) "L^{(n+1)} exactly equals L_ref^{(n+1)}" and (b) "EL difference is zero, so they agree modulo boundary terms" — both pass in pure-gravity-Hilbert. Case (c) "neither holds; decompose into integrable EOM terms and apply the corresponding field redefinitions to L_ref" is intentionally `NotImplementedError` (see `_apply_field_redef`). The detailed 7-step flow lives in the docstring of that method. This is needed once matter or Belinfante is in scope, and shares an algorithm with item 5 (H2 machinery). NOTE: (i) **The ordering of redefinitions is mathematically substantive, not a stylistic convention.** h-redefs and matter-field redefs do NOT commute — applying them in different orders lands on different members of the same field-redef equivalence class (all valid completions, just distinct representatives). The convention to use here is: apply h-redef across all L_ref^{(k)} first, then re-expand and replace L_ref, then matter-field redefs one at a time. (ii) Separately, an implementation-hygiene point: use fresh dummies on every occurrence of φ when substituting, to avoid the index-clash pitfall when replacing φ inside φ^n.
4. **Explicit H2-violation check at n=2, n=3, n=4.** Compute Z^{μνρσ} (paper eq. for Z) for M = T_H[L^{(n)}] and verify Z = 0. The closure tests *imply* Z=0 already, but a direct verification would confirm Butcher's claim and provide regression coverage for the H2 machinery once item 5 lands.
5. **H2-violation machinery (`compute_h2_violation` and `compute_eom_correction` in `helmholtz.py`).** Decompose Z = Y · E^{(0)} into matter and gravity pieces, then set X = −(1/(2(n+1))) Y · h. USER COMMENT: probably the hardest part of the project — needs creativity in finding an algorithm for the Z decomposition. Algorithm sketch is in the chat history of this project (basis-driven parametric ansatz + linear solve over ℚ); first sub-task is wiring `compute_h2_violation` and using it for item 4. Load-bearing once matter is added.
6. **Higher orders / other matter examples.** Push EM and scalar QED to n=3 (extrapolating: probably 10–20 min wall for EM, 30+ min for scalar QED). φ⁴ self-interaction (no new code expected — same algorithm as φ², just write the L_M). Massive vector field (needs the full Christoffel-correction machinery in `matter_lagrangian_order`; not gauge-invariant, so the cancellation our EM case relies on doesn't hold).
7. **Stronger Ψ^{(1)} test (waits on Belinfante or optional EOM).** USER NOTE: as long as we use the Hilbert energy-momentum and don't add optional EOM terms, the only superpotential terms that need to appear are the same pure-gravity ones we've already seen at n=2,3,4 — so Ψ^{(1)} = 0 is the *expected* outcome of all the matter tests, not a sign of an under-tested formula. The integral formula will only get a real workout once we have the Belinfante procedure (item 8) and/or the optional-EOM-terms switch (item 9) implemented, since those introduce field-redef-equivalent corrections at n=1 that the integral must absorb.
8. **Symmetrized Belinfante procedure** as an alternative to Hilbert (paper §2). Should differ from Hilbert only by EOM terms. USER COMMENT: with Belinfante, the verification cycle (item 3) will have to apply field redefinitions even when no optional EOM terms are added.
9. **Optional EOM terms (field redefinitions, voluntary path).** Integrability check ∂X/∂h symmetric under (αβ ↔ μν); add a switch in `bootstrap_loop.py` to inject such terms by user choice (orthogonal to the field redefinitions forced by the verification cycle in item 3).
10. **RESOLVED — full Proca passes orders 0..2.** The cyclic-asymmetric Ψ^(1) was caused by the *bracket itself* (formed inside `compute_superpotential_n1`) sometimes coming out as a TensAdd with top-level args of shape `TensMul(Rational, TensAdd)` — i.e. a rational coefficient that didn't distribute across the inner TensAdd. `_scale_matter_fields`, which uses `_decompose_tensmul`, silently dropped the inner TensAdd because `_decompose_tensmul` only collects Tensor/TensMul args. That meant matter-field factors hidden inside got the wrong λ-power (effectively λ⁰), so the λ-integral gave them the wrong coefficient and Ψ^(1) lost its cyclic-symmetry. Fix: `.expand()` on the bracket before `_scale_matter_fields`. User-side note that motivated finding this: **both V AND dV (and ddV) should get a λ factor** for each occurrence — verifying that prompted inspecting `_scale_matter_fields` and discovering the inner TensAdd was being silently skipped. With the fix, full Proca's Ψ^(1) is 12 terms, cyclic-symmetric, and the bootstrap closes through n=2.

11. **RESOLVED — kitchen-sink Lagrangian passes orders 0..4 after adding the ∇A Christoffel chain rule.** The 52-term EL-self-consistency residual at n=1 for charged Proca / kitchen-sink was caused by `energy_momentum.py` missing the `dA → ∇A = dA(σ, ρ) − Γ^τ_{ρσ} A_τ` substitution inside `_christoffel_via_substitution`. The F-antisymmetry that cancels Γ inside F_μν only does so at order 0; once T_M[L^(n)] is computed at n ≥ 1, the Γ piece is needed for any Lagrangian where A couples directly to other matter (here: e A V_partner inside F^V_i). With the dA branch added (mirroring the existing dV branch), all closures pass through n=4 — see headline result. **Open puzzle still:** scalar QED has the same A head and the same code path, but its tests (orders 0..2) pass *without* the dA fix. Either there's a structural cancellation specific to scalar QED's coupling pattern (worth understanding to articulate the boundary), or the tests are succeeding accidentally at the orders we've run and will fail higher up. Worth a quick re-run of scalar QED with the new code (sanity, should still pass) and an extension to n=3 to see if any latent issue surfaces.

## Conventions and pitfalls

- **All indices are abstract.** No spacetime dimension is fixed; expressions are never expanded into components.
- **Order in h** counts h, dh, and ddh each as one power.
- **`canon()` must include `contract_metric()`.** SymPy's `canon_bp()` does not contract metrics, and Kronecker deltas from jet derivatives pile up and blow expressions up if they aren't contracted.
- **Fresh indices via the global counter** (`_i0, _i1, ...`). Use `fresh_indices()` generously when building products — never reuse a free index from one expression as a dummy in another expression you're multiplying it with.
- **Half-symmetrization in jet derivatives.** ∂h_{μν}/∂h_{αβ} = ½(δ^α_μ δ^β_ν + δ^α_ν δ^β_μ). The ½ is required because h is symmetric.
- **TensorSymmetry API.** Use `TensorSymmetry.direct_product(2, 1)` for a "symmetric pair + single index" head (e.g. dh's symmetry). `from_generators` is not the right tool here.
- **Free-index relabelling in (PsiForm).** The three terms of the superpotential formula use M with *different* free indices: terms A and B use M^{αβ}, term C uses M^{μν}. `compute_superpotential_n2` relabels M per term — if you tweak that function, mismatched free-index sets surface as `ValueError: all tensors must have the same indices`.
- **`substitute_indices` is silent on bound dummies.** It rewrites free indices only. To rename a dummy pair you must rebuild the affected factors with the new index. This subtlety bit us inside the IBP fix (see headline).
- **Performance.** Wall times: n=2 ~1 min, n=3 ~5 min, n=4 ~20.5 min. Dominant step at n=4 is `compute_superpotential_n2` (~10 min on a 315-term M); next-tier are `hilbert_energy_momentum` (~4.5 min) and EL of L_EH^{(5)} (~3.5 min). Empirical scaling is roughly 4× wall-time per order.
- **The right idiom for summing tensor terms is `TensAdd(*terms)`, not a `result = result + t` loop.** sympy's `TensAdd.__add__` re-runs the term-collection pass on the growing accumulator, so the loop pattern pays N× the per-call overhead. Helper `_sum_terms()` lives in `jet.py` — use it for new code in this codebase.
- **canon caching does NOT help.** Was tried; sympy tensors are hashable but hashing big tensors is expensive enough that a 10 % cache hit rate didn't break even. Deferring canon to the outermost recursive call also REGRESSED — sympy's canon is super-linear in term count, so one big canon is slower than several small canons. Per-step canonicalization is load-bearing; the cost is real and hard to dodge.
- **Run-to-run timing variance.** Same code, same machine, no other user load → up to ~2× variance on individual steps. The likely culprits are `PYTHONHASHSEED` randomization (changing sympy's traversal order in dict/set iteration) and background Windows activity (Defender, Search indexer, OneDrive). Set `PYTHONHASHSEED=0` for reproducible runs when comparing optimization candidates.

## File dependencies

```
tensor_algebra.py ── jet.py ──┬── euler_lagrange.py ──┐
                              ├── helmholtz.py        ├── bootstrap_loop.py
                              └── covariant.py        │
                                                      └── energy_momentum.py
```

## How to run

```
cd "c:\Users\ahamaline\OneDrive - Technion\Documents\Venusian_code"
python tests\test_smoke.py
python tests\test_W.py
python tests\test_hilbert_gravity.py
python tests\test_superpotential_pure_gravity.py       # ~1 min — n=2 closure (low-level)
python tests\test_superpotential_pure_gravity_n3.py    # ~5 min — n=3 closure (low-level)
python tests\test_superpotential_pure_gravity_n4.py    # ~20 min — n=4 closure (low-level)
python tests\test_bootstrap_loop_pure_gravity.py 3     # ~8 min — full BootstrapState driver, pure gravity orders 0..3
python tests\test_bootstrap_loop_scalar.py             # ~30 s — BootstrapState with massless scalar, orders 0..1
python tests\test_bootstrap_loop_scalar_potential.py   # ~7 min — BootstrapState with massive scalar (kinetic + m^2 phi^2), orders 0..3
python tests\test_bootstrap_loop_complex_scalar.py     # ~80 s — BootstrapState with complex scalar (two real components, same mass), orders 0..2
python tests\test_hilbert_em.py                        # ~1 s — verifies Maxwell T_EM
python tests\test_bootstrap_loop_em.py                 # ~2 min — BootstrapState with pure EM (L = -1/4 F.F), orders 0..2
python tests\test_bootstrap_loop_charged_scalar.py     # ~3 min — full scalar QED, orders 0..2
python tests\test_hilbert_proca_mass.py                # ~0.1 s — T_H of Proca mass term (smoke test for g_down + term_D)
python tests\test_bootstrap_loop_proca_mass.py         # ~90 s — BootstrapState with Proca mass term only (upstairs V), orders 0..2
python tests\test_bootstrap_loop_proca.py              # ~21 min — full Proca (upstairs V, F·F + mass), orders 0..2
python tests\test_bootstrap_loop_charged_vector_qed.py # ~6.6 h  — kitchen sink (EM + charged scalar + charged vector + interaction), orders 0..4
```

The two superpotential tests are the headline tests.

## The 6-step bootstrap (paper §4, quick reference)

For each order n = 0, 1, 2, ...:

1. **E_1^{μν(n)}** = κ T̂[L^{(n)}] + δ_{1,n} W^{μν}. T̂ is Hilbert or symmetrized Belinfante.
2. **E_2^{μν(n)}** = E_1 + Σ_{m<n} ( X_h^{(m)} · E^{(n−m)} + Σ_i X_{φ_i}^{(m)} · E_{φ_i}^{(n−m)} ). Carries over EOM choices from earlier orders.
3. **Mandatory EOM (H2)**. Compute Z = 2(∂E_2/∂h)_{antisym} − ((∂E_2/∂dh)_{antisym})_,γ. Decompose Z = Y · E^{(0)}. Set X = −1/(2(n+1)) Y · h. Add X · E^{(0)} → E_3.
4. **Optional EOM (field redefinitions)**. Any X' with ∂X'/∂h symmetric in (αβ ↔ μν). Zero on the default path → E_4.
5. **Superpotential (H3)**. n ≥ 2: Ψ from (PsiForm). n = 1: integral formula. Add Δ = Ψ_{,ρσ}.
6. **Close the loop**. L^{(n+1)} = (1/(n+1)) E^{(n)} h + boundary terms. EL(L^{(n+1)}) must reproduce E^{(n)}.

For **pure gravity with Hilbert** and no optional EOM terms, steps 2, 3, and 4 are trivial (Butcher's claim — verified empirically at n=2, n=3, and n=4 by the existence of the closure; item 4 of the open-work list would confirm it directly by checking Z=0). The only non-trivial work at each order is step 1 (T_H), step 5 (Ψ via PsiForm), and step 6 (closing the loop). This is precisely what the `test_superpotential_pure_gravity*.py` tests exercise.
