# GR Bootstrapping — Status & Continuation Guide

## What this is

This codebase implements the iterative procedure of the paper *GR Bootstrapping for Venusians* (in this folder). Starting from the linear spin-2 wave equation, it constructs the Einstein Field Equation order by order in h_{μν}, using energy-momentum conservation and the Helmholtz conditions. The eventual goal is to verify that the procedure uniquely produces the EFE (up to field redefinitions). The paper spells out the 6-step bootstrap loop in §4; a one-page map is at the bottom of this document.

## Headline result so far

The bootstrap closure has been verified **at n=2, n=3, and n=4 in pure gravity, with the Hilbert energy-momentum procedure**. For each order n, the identity

  EL(L_EH^{(n+1)}) − κ T_H[L^{(n)}]  =  κ Ψ^{(n)}_{,ρσ}

holds exactly, where Ψ^{(n)} is produced by the constructive formula (PsiForm) from the paper. Closure at three successive orders — each one tested independently with the same algorithm, no bookkeeping changes — is strong empirical evidence that the procedure is uniformly correct as n grows.

**Kitchen-sink matter result (2026-05-25).** The "everything bagel" Lagrangian — EM + charged complex scalar + charged complex vector (upstairs V) + the φ̄ D_μ V^μ interaction + H.C. — closes the bootstrap **end-to-end at orders 0..4**. At every order, EL(L^(n+1)) == E^(n) and EL(L_ref^(n+1)) matches E^(n) modulo boundary terms. Ψ symmetries (sym_mn, sym_rs, cyclic) all pass at n=1..4. Term counts: E_1 grows 82 → 363 → 1330 → 3560 → 8786, E^(n) (after Ψ) 82 → 289 → 764 → 1700 → 3449, L^(n+1) (after IBP) 65 → 166 → 339 → 663 → 1174. Total wall ≈ 6.6 hours (n=4 alone ≈ 4.8 h). The final piece was a missing Christoffel chain rule for ∇A in `energy_momentum.py`: the F-antisymmetry that cancels Γ inside F_μν at order 0 does NOT carry over to T_M at order ≥ 1, so each registered downstairs-A field needs a `∂_ρ A_σ → ∇_ρ A_σ = dA(σ, ρ) − Γ^τ_{ρσ} A_τ` substitution inside `_christoffel_via_substitution`. (Open puzzle: scalar QED — same `A` head, same path — passed orders 0..2 without this fix; either there's a genuine cancellation specific to scalar-QED structure or the test is succeeding accidentally and will fail at higher orders. See open-work item 11.)

**Step 4 optional-EOM machinery + Helmholtz integrability check (2026-05-27).** `BootstrapState.add_optional_eom_term(n, field_name, X_expr)` lets users register a derivative-free X coefficient to apply at step 4 of order n ("voluntary path" / field redefinition). `_step4_optional_eom` validates each X before adding: (i) `_is_derivative_free` (necessary - X must be a function of fields only); (ii) `_check_X_integrable` runs the actual Helmholtz integrability condition `\partial X^{\mu\nu...}/\partial h_{\alpha\beta}` symmetric in `(\mu\nu \leftrightarrow \alpha\beta)` via `compute_h2_violation(X)` (this reduces to the algebraic antisymmetric part automatically since X has no dh). Either check failing raises with a clear message; on success, `X . E_{field}^(0)` is added to E and X is recorded in `eom_terms_h/matter` so step 2 of higher orders carries it forward (and merges with any step-3 X at the same order). Note: the earlier `_check_eom_decomposition` (formerly mis-named `_integrability_check`) for the verification cycle only checks NECESSARY conditions (clean decomposition + derivative-free coefficients); the actual Helmholtz integrability is verified indirectly when EL(updated L_ref^(n+1)) = E^(n) closes after the redef. `tests/test_optional_eom_smoke.py` documents three cases: constructive accept (constant X_h^(1) -- trivially integrable since X is h-independent), reject-on-derivatives, and reject-on-non-integrable.

**Step 2 EOM carryover wired (2026-05-27).** `_step2_eom_carryover` adds Σ_{m<n} ( X_h^{μν(m)}_{κλ} · E^{κλ(n−m)} + Σ_i X_{φ_i}^{μν(m) ...} · E_{φ_i}^{(n−m) ...} ) to E_1 at every order n. The X^(m)'s come from `self.eom_terms_h` / `self.eom_terms_matter` populated by step 3. The h-EOM at order n−m is `self.E[n−m]`; the matter EOM at order n−m is `EL(self.L[n−m], φ)` (EL by φ preserves h-count). No-op on the pure-gravity Hilbert path (eom_terms dicts stay empty). Reached at n=2 of EM Belinfante (52 carryover terms added at n=2 from the X_A^(1) chosen at step 3 of n=1).

**EM Belinfante full closure through n_max=2 (2026-05-27).** All three orders close on the EM Belinfante test:
- **n=0:** L_ref^(1) diff 4 → 0 (X_A 2 terms, f_A^(1) = κ A^L h_{ρL}).
- **n=1:** L_ref^(2) diff 18 → 0 (X_h 4 + X_A 4; f_h^(2) = κ h_α^L h_{βL} + f_A^(2) = -½ κ² A h h). 471s.
- **n=2:** L_ref^(3) diff 26 → 0 (Y_h kinetic 6 terms, X_A absent at this order; f_h^(3) = (2/3) κ² h h h). 1453s.

At n=2, step 2 carryover added 52 terms from the X_A^(1) recorded at step 3 of n=1, then step 3's H2 violation Z = 64 terms decomposed cleanly into Y_h kinetic (16 terms), corrected, post-correction Z = 0. **Total wall ~33 min** (EM, n_max=2). First complete bootstrap-with-Belinfante closure through 3 orders.

**Full field-redef cycle on EM Belinfante through n=1 (2026-05-27).** End-to-end closure verified. `_recover_field_redefs` computes f^(n+1) = (1/(n+1)) · h_{αβ} · X^{αβ ...} from the X coefficients (paper formula). `_substitute_field` substitutes φ → φ + f in any Lagrangian (matter OR h, via the field_info abstraction), propagating to dφ, ddφ via a `_total_derivative_at` helper that dodges sympy's TensMul-auto-canon dummy-name-collision pitfall (sympy renames dummies to L_0, L_1, ... so a directly-computed `total_derivative(f, deriv_idx)` clashes when deriv_idx is itself an L_x dummy from the parent — fix: differentiate against a fresh deriv, then contract via metric, which canon collapses safely). `_apply_field_redefs_to_L_ref` orchestrates per the paper's ordering: h-redef first across L_ref^(k) for k = 0..n_max-n+1, then each matter-field redef one at a time for k = 0..n_max-n. **L_ref^(0) MUST be included** — the L_M chain-rule pieces (via df, which has h since f does) are exactly what propagates into new L_ref^(j>0) and makes EL(new L_ref^(n+1)) = E^(n) at order n. After each per-order pass, EL(updated L_ref^(n+1)) is re-checked against E^(n) and the verification raises if not 0.

**Test results on EM Belinfante, n_max=1:**
- **n=0:** L_ref^(1) diff = 4 → 0 (X_A 2 terms → f_A^(1) = κ A^L h_{ρL}). L_ref^(1): 5→6, L_ref^(2): 22→28.
- **n=1:** L_ref^(2) diff = 18 → 0 (X_h 4 + X_A 4 → f_h^(2) = κ h_α^L h_{βL} + f_A^(2) = -½ κ² A h h). L_ref^(2): 28→27.

**Key debugging lesson:** the orchestrator must use the bootstrap state's `E[0]` directly rather than recomputing T_M from L_M. Recomputing T_M loses the κ factor that E[0] = κ T_M carries, and Y/X come out scaled wrong — the f's then have one extra power of κ and the substitution produces no-op EL changes. Also, with optional EOM terms, E[0] ≠ κ T_M and recomputing would be flat wrong. The orchestrator now accepts `E0=` and `E0_indices=` kwargs (callers in `BootstrapState` pass `self.E[0]`).

**Integrability check on L_ref diff (2026-05-27).** `BootstrapState._integrability_check` wraps `decompose_against_eoms` to decide whether the verification-stage diff `EL(L_ref^(n+1)) − E^(n)` is a valid Lagrangian field-redefinition diff. Criterion (paper §3): residual = 0 AND every X coefficient is *derivative-free* (no dh, ddh, dφ, ddφ, dA, ddA, dV, ddV — only h, φ, A, V, η, deltas). Wired into `_verify_vs_L_ref`: when E_diff ≠ 0, the check runs; on success it logs the X breakdown and a **loud warning that L_ref has NOT been updated via field redef** (open-work item 3); on failure (residual or derivative-laden X) it raises with the explicit reason. Verified on the EM Belinfante diff at n=0 (4 terms → X_A 2 terms, derivative-free). The check at n=1 currently fails because the n=0 redef wasn't actually applied to L_ref, so by n=1 L_ref is in a different field convention from L_bootstrap and the diff isn't a clean single-order redef — the test deliberately disables the n=1 L_ref check (with a banner) until applying the redef lands. Also added a free-index-name filter to the orchestrator's matchers (`_extract_coeff_two_factor`, `_extract_coeff_from_trace_signature`, trace variant) so matches that release dummies as extra free indices on stripping (because the matched signature factors carried dummies paired with non-signature factors) are skipped — this was masked in the step-3 case (whose Z had only ddA·A terms) but surfaced as a TensAdd index-mismatch crash on the richer verification-stage Z.

**EOM-decomposition orchestrator + step-3 wiring (2026-05-27).** The H2-violation decomposition `Z = Y_h · E_h^(0) + Σ_i Y_{φ_i} · E_{φ_i}^(0)` now has a general orchestrator in [bootstrap/eom_decompose.py:decompose_against_eoms](bootstrap/eom_decompose.py) that handles arbitrary matter, following the user's 2026-05-26 algorithm: (1) pick a kinetic signature in T_M with μ, ν on distinct first-derivative matter factors (NOT contracted together — that's a trace term, would over-count); (2) match the signature in Z, strip and accumulate Y_h (skipping pairs whose free-role indices form a contracted dummy pair in Z's term, since those are trace contributions); (3) trace-term step for leftover contracted ∂φ·∂φ pieces; (4) per-matter-field, match a unique 2nd-deriv trace signature (`ddφ(α, L, -L)` or `ddφ(L, -L)` for scalars) and strip. Wired into `_step3_mandatory_eom`: when `compute_h2_violation` returns Z ≠ 0, the orchestrator runs, forms `X = -1/(2(n+1)) · Y · h_{αβ}` per paper eq. for X, and adds `X · E^(0)` to E. An H2 re-check after the correction verifies Z = 0 — the new code passes this check. **Step-3 wiring exercised on EM Belinfante at n=1**: Z = 16 terms, decomposes as X_A only (4 terms after `· h` contraction; Y_h is 0 because the EM Bel-H diff is purely matter-EOM-proportional), correction applied, post-correction H2 = 0, step 5 + step 6 run, and step 6's EL self-consistency (EL(L^(2)) == E^(1), internal) passes. Wall ~100 s for n=1. **Scope caveat:** this is NOT full Belinfante closure — the test runs with `n_max=None` to bypass L_ref verification, since on the Belinfante path L^(2) is expected to differ from the reference Einstein-Hilbert expansion by an integrable field-redefinition EOM combination, which the verification cycle (open-work item 3) has not yet been taught to absorb. So what's verified is "the new step-3 machinery does its job mechanically and produces a Z-self-consistent E", not "Belinfante reproduces the EH expansion." Known limitation: a pure-mass-only matter EOM (e.g. EOM_V = m² V^α with no kinetic term) violates the algorithm's "unique 2nd-derivative signature per EOM" precondition and the orchestrator returns a non-zero residual; this is exposed by the standalone Proca-mass-only smoke test and intentionally left as-is. Two signature changes worth noting: (i) `compute_h2_violation` now returns `(Z, (α, β))` so the caller knows which 2 of Z's 4 free indices are the h-style pair to contract with `h_{αβ}`; (ii) when wiring the X-correction into `E`, the contraction indices on X (and on EOM) must be read off from `.get_free_indices()` rather than assumed up — canon can rearrange the signs from the metric contractions inside the orchestrator.

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
| `jet.py` | Working. `jet_derivative()` with ½-symmetrization on symmetric pairs and Leibniz; `total_derivative()` via chain rule on jet variables. `total_derivative` now `.expand()`'s its input at the top so callers feeding raw `A − B` (sympy keeps that as `TensAdd(A, TensMul(NegativeOne, B))`) self-correct. `_decompose_tensmul` raises a clear `ValueError` if it sees a TensMul wrapping a TensAdd — the long-standing footgun (MEMORY.md / project-decompose-tensmul-tensadd-pitfall) that previously made it silently drop terms. With both guards, the H2-check (compute_h2_violation in helmholtz.py) is correct without explicit canon/.expand() at the call site. |
| `euler_lagrange.py` | Working. `euler_lagrange()` and `remove_second_derivatives()`. Recovers the 6-term linearized Einstein tensor from L_h^{(2)}; the IBP path is now order-3+ safe (see above). |
| `covariant.py` | Working through order 5 for EH; arbitrary order for `matter_lagrangian_order(L_M, n)` (scalar, downstairs-A gauge-invariant vector, AND upstairs-V vector matter). EH order 3 = 28 terms (~10 s), order 4 = 66 terms (~20 s), order 5 = 136 terms (~45 s). `matter_lagrangian_order` covariantizes (a) η^{μν} → g^{μν} via `inverse_metric_order`, (b) η_{μν} → g_{μν} via the new `metric_order` (only orders 0 and 1 are non-zero since g_{μν} = η + 2κh), and (c) ∂_ρ V^σ → ∇_ρ V^σ = dV(σ, ρ) + Γ^σ_{ρτ} V^τ for each matter field registered as upstairs (index_pos='up'). Christoffel-corrections to dV are dispatched per field at composition time so each upstairs-dV factor independently chooses its h-order. |
| `energy_momentum.py` | Working for `hilbert_energy_momentum()`. Sums four contributions to δ(√|g| L̃)/δg_{μν}: A from δ√|g|, B from δg^{αβ} (variation of explicit g^{...} factors in L_cov), D from δg_{αβ} (variation of explicit g_{...} factors in L_cov; the two are tracked as separate `ginv` and `g_down` jet heads, dispatched by index sign in `replace_metric_with_ginv`), and C from the Christoffel chain rule. `_christoffel_via_substitution` handles dh, the dV → ∇V substitution for upstairs-vector matter fields, AND the dA → ∇A substitution for downstairs-vector matter fields. The dA branch was added 2026-05-25 — without it, T_M[L^(n)] at n ≥ 1 silently dropped a Γ A piece that broke EL self-consistency for any Lagrangian where A couples directly to other matter (e.g. e A V_partner cross-coupling in charged Proca). F·F-only Lagrangians (L_EM, scalar QED at the orders we ran) escaped this because F_μν's antisymmetry cancels Γ inside F itself at order 0. Verified on L_h^{(2)}, L_EH^{(3)}, free scalar, L_EM, L_Proca-mass, full Proca, charged Proca, and the kitchen-sink Lagrangian through n=4. Also implements `symmetrized_belinfante()` (paper §2 alternative to Hilbert): builds the canonical Noether T_can^{μν} = η^{μν} L − π·∂^ν φ (Mostly Plus convention, summed over all matter fields and the graviton h), then the Belinfante improvement T_Bel = T_can + ∂_ρ B^{ρμν} with B = ½(S^{ρμν} − S^{νρμ} + S^{μνρ}) from the spin tensor S of each field, then explicit (μν)-symmetrization T_SymBel = ½(T_Bel + T_Bel^T). Spin matrices implemented for downstairs A_α, upstairs V^α (with the correct same-structure-as-A formula — the opposite-sign claim from a literature review failed a hand-check against a Lorentz boost on V^μ), and h_{αβ} (tensor product of two A-generators, collapsed via h's symmetry to a single factor-of-2 term). Verified T_SymBel − T_Hilbert is purely EOM-proportional for every matter type: free/massive scalar (0 terms), EM (4 terms = sym(EOM_A·A)), Proca mass (1 term = ε V·V), full Proca (5 terms = (∂F+m²V)·V), pure gravity h (17 ddh·h terms, no dh·dh residue). |
| `helmholtz.py` | `compute_superpotential_n2()` (PsiForm for n ≥ 2), `compute_superpotential_n1()` (paper eq. 23, integral form for n = 1, scalar AND non-scalar matter — the rank-r generalization sums each field's tensor indices via the natural extension; the wrt-indices for the field derivative are signed OPPOSITE to the field's natural positions so the resulting deltas are clean Kronecker pairs, and the velocity factor is the field in its natural form). `superpotential_divergence()`, `verify_psi_symmetries()` — all working. `compute_h2_violation(E, E_indices)` implements the paper's eq-Z formula and now returns `(Z, (α, β))` (the h-style index pair generated for Z) so callers can form `X = -1/(2(n+1)) · Y · h_{αβ}` per paper. Verified Z = 0 directly for pure gravity and scalar matter on the Hilbert default path (Butcher's claim confirmed). `compute_eom_correction()` is still a placeholder — superseded by `decompose_against_eoms` in `eom_decompose.py`. **Three latent bugs fixed during the upstairs-V work:** (i) for naturally-up matter fields, the previous code passed `field_indices` always UP, producing raising-η factors instead of Kronecker deltas (Ψ ended up with the wrong index structure); (ii) the multiplication `velocity × h × scaled` sometimes left a TensAdd as one arg of an outer TensMul; `_decompose_tensmul` (used by `_integrate_lambda`) silently dropped any TensAdd args, collapsing Ψ to a phantom 3-index expression — fixed by `.expand()` before integration; (iii) the **bracket itself** sometimes had top-level args of shape `TensMul(Rational, TensAdd)` (e.g. when ``Rational(1/2) * d2M_B`` didn't distribute), and `_scale_matter_fields` — also using `_decompose_tensmul` — silently dropped the inner TensAdd, leaving its matter-field factors unscaled and giving the wrong λ-power → non-cyclic-symmetric Ψ. Fixed by `.expand()` on the bracket before `_scale_matter_fields`. |
| `eom_decompose.py` | `decompose_against_eoms(Z, L, em_procedure, E0, E0_indices)` — orchestrator for `Z = Y_h · E_h^(0) + Σ_i Y_{φ_i} · E_{φ_i}^(0)` (per user's 2026-05-26 algorithm). Picks a kinetic signature in E_h^(0) with μ, ν on distinct first-derivative matter factors (and NOT contracted together), matches in Z (skipping pairs whose free-role indices form a contracted dummy pair in Z's term — those are trace contributions not kinetic), then subtracts; trace-term step for leftover contracted ∂φ·∂φ residuals; per-matter-field, matches a unique 2nd-deriv trace signature and strips. Callers should pass the BootstrapState's `E[0]` (which is κ T_M at order 0, but may differ if optional EOM terms have been added) rather than relying on the recompute-T_M fallback. Verified on the EM Belinfante diff (X_A only, residual 0), on a synthetic Z = Y_test × T_M[scalar] (Y_h recovered modulo (α↔β) symmetry), and inside both step 3 (H2 violation) and the L_ref verification step. |
| `bootstrap_loop.py` | **Working end-to-end for pure gravity (orders 0–3), massive scalar matter (orders 0–3), complex scalar (two real components, orders 0–2), pure EM L_M = −¼ F·F (orders 0–2), scalar QED (orders 0–2), full Proca (orders 0–2), and the kitchen-sink Lagrangian (EM + charged scalar + charged vector + φ̄DV interaction, orders 0–4).** `BootstrapState(L_matter, n_max=N).run_order(n)` drives the 6-step loop, performs EL self-consistency at each order (EL(L^{(n+1)}) == E^{(n)}), and runs the verification cycle against a pre-computed `L_ref^{(k)}` for k=0..n_max+1 (gravity + matter contributions). Step 3 (mandatory H2-EOM) calls `compute_h2_violation`; if Z ≠ 0 it invokes `decompose_against_eoms` from `eom_decompose.py` to compute Y_h and Y_{φ_i} coefficients, forms `X = -1/(2(n+1)) · Y · h_{αβ}` per paper, adds `X · E^(0)` to E, and re-checks H2 (must be 0 after). EM Belinfante at n=1 verified end-to-end (Z = 16 terms → X_A correction → H2 satisfied). When applying the correction, the EOM-side contraction indices on X are read via `.get_free_indices()` (canon's metric reductions can flip signs from the orchestrator's assumed up convention). `em_procedure='belinfante'` wires Step 1 to `symmetrized_belinfante` and Step 3's decomposer to `symmetrized_belinfante` for E_h^(0). The field-redefinition decomposition (called when L^{(n+1)} differs from L_ref^{(n+1)} by more than boundary terms) and Step 4 (optional EOM) are intentional `NotImplementedError` stubs — Belinfante's L_ref verification still trips here (item 3, next on the list). (2026-06-03: trimmed 2444→1588 lines by extracting `loop_helpers.py` and `traceless.py`.) |
| `loop_helpers.py` | General bootstrap helpers shared without an import cycle: `_count`, `_term_breakdown`, `_format_breakdown` (term counting / per-matter-field breakdown for verbose logs), `_reindex_tensor` (free-index relabel, both signs), `_is_derivative_free` / `_derivative_heads` (field-redef integrability predicate). Depends only on `tensor_algebra`. |
| `traceless.py` | Trace-free-T_M subsystem (DEVELOPMENT_STATUS item 0). Module-level `_extract_ddh_box_signature` (doubly-traced ddh, used by the mandatory step) and `_extract_ddh_deriv_signature` (ddh whose derivative pair = the free indices, used by the verification step). `TracelessRecoveryMixin` (inherited by `BootstrapState`): `_check_T_M_traceless` (on-shell-traceless detection + c_i), `_build_traceless_S1`, `_build_traceless_redef_operator` (the T2 operator, rank-0 + rank-1 c_i), `_split_const_matter`, `_poly_divide_by_m_plus_v`, and the two recoveries `_recover_missed_traceless_redef` (verification) + `_recover_traceless_mandatory_eom` (step 3). |

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
| `test_bootstrap_loop_scalar.py` | **BootstrapState with massless scalar matter, orders 0–1**. Verifies that `E^{(0)} == κ T_M` (textbook), L_ref expansion of √|g| L̃_M produces the right shape at orders 1 and 2, and the n=1 superpotential integral formula closes the loop. Ψ^{(1)} = 0 here — specific to the minimal scalar's structure, NOT a general rule (Proca and the conformal scalar give nonzero Ψ^{(1)} on the Hilbert path; see item 7). |
| `test_bootstrap_loop_scalar_potential.py` | **BootstrapState with massive scalar (kinetic + m² φ² potential), orders 0–3.** Exercises matter L_ref through order 4 (83 terms), the n=1 integral formula, AND PsiForm at n=2 and n=3 with matter contributions inside M. All Ψ symmetries pass; EL and L_ref verification pass at every order. ~7 min total wall (n=3 alone is ~6 min). |
| `test_bootstrap_loop_complex_scalar.py` | **BootstrapState with a complex scalar (two real components, same mass) at orders 0–2.** Same form as the massive-scalar test but with two registered scalar fields whose sum is L = −∂φ*·∂φ − m² φ*φ in real-component form. Confirms multiple matter fields work with zero new code. ~80 s wall. |
| `test_hilbert_em.py` | T_H[L_EM = −¼ F_{μν} F^{μν}] reproduces the textbook Maxwell stress-energy tensor exactly (6 terms, 0.6 s). |
| `test_bootstrap_loop_em.py` | **BootstrapState with pure EM (L = −¼ F·F), orders 0–2.** First non-scalar matter case, confirming gauge-invariant vector matter threads through `matter_lagrangian_order`, `hilbert_energy_momentum`, PsiForm, and the verification cycle. ~120 s wall. |
| `test_bootstrap_loop_charged_scalar.py` | **BootstrapState with full scalar QED (charged complex scalar coupled to EM), orders 0–2.** L_M contains kinetic + mass for two real components of φ, F·F for the EM field, the minimal-coupling current −eA^μJ_μ, and the seagull −½e² A² φφ. All three matter fields (φ₁, φ₂, A) run through every step including `compute_superpotential_n1`; Ψ^(1) = 0 as expected on the default path, Ψ^(2) symmetries verified, EL and L_ref closures pass. ~3 min wall. |
| `test_hilbert_proca_mass.py` | T_H[−½ m² V^μ V_μ] with V upstairs (`register_upstairs_vector_field`) reproduces the expected −m² V^μ V^ν − ½ m² η^{μν} V·V (2 terms, ~0.1 s). Minimal sanity check for the new `g_down` jet head and `term_D` path in `hilbert_energy_momentum`. |
| `test_bootstrap_loop_proca_mass.py` | **BootstrapState with the Proca mass term only (no F·F), V upstairs, orders 0–2.** First test that exercises the downstairs-metric expansion (`metric_order`, term_D) end-to-end. All closures pass; ~90 s wall. |
| `test_bootstrap_loop_proca.py` | **BootstrapState with the full Proca Lagrangian (−¼ F·F − ½ m² V^μ V_μ, V upstairs), orders 0–2.** Exercises both the downstairs-metric path AND the upstairs-vector covariant-derivative path (`christoffel_order` in `matter_lagrangian_order`, the dV-branch of `_christoffel_via_substitution`). Passes orders 0..2 end-to-end after the bracket-expand fix in `compute_superpotential_n1`. Ψ^(1) is nonzero (12 terms — allowed because sympy's ddV index canonicalization breaks the strict-zero derivation) and cyclic-symmetric; Ψ^(2) symmetries also OK; EL self-consistency and L_ref verification pass at every order. Wall: ~21 min total (n=2 alone is ~18 min). |
| `test_bootstrap_loop_charged_vector_qed.py` | **Kitchen-sink Lagrangian — EM + charged complex scalar + charged complex vector + φ̄ D_μ V^μ + H.C., orders 0–4.** 28-term L_M after canon, every order closes (EL self-consistency AND L_ref verification). Term counts climb steeply: E_1 82 → 363 → 1330 → 3560 → 8786 terms across n=0..4. Ψ symmetries OK at n=1..4. Wall: ~6.6 h total (n=4 alone ≈ 4.8 h on the dev machine). Largest matter milestone so far; relies on the 2026-05-25 ∇A Christoffel fix in `energy_momentum.py`. |
| `test_belinfante_smoke.py` | **Smoke test for the symmetrized Belinfante tensor.** Verifies T_SymBel = T_Hilbert exactly for scalar matter (no spin), and that T_SymBel − T_Hilbert is purely EOM-proportional for EM, Proca (mass-only and full), and the pure-gravity L_h^{(2)}. No dA·dA / dV·dV / dh·dh residue surviving in any case. Wall: ~1 min. |
| `test_bootstrap_loop_scalar_belinfante.py` | **BootstrapState with em_procedure='belinfante', free scalar matter.** First bootstrap run using Belinfante T̂. Step 6 closes (EL(L^(2)) = E^(1)) and the H2 check passes (Z = 0) at n=1. L_ref verification raises NotImplementedError at n=1 because Belinfante differs from Hilbert by EOM terms that the field-redefinition decomposition (open-work item 3) is supposed to handle — expected behavior, confirms the plumbing works end-to-end. Wall: ~30 s. |
| `test_eom_decompose_orchestrator.py` | **Sanity test for `decompose_against_eoms`** on the Bel-H diffs: free scalar (trivial, diff = 0), EM (Y_h = 0, X_A 2 terms, residual 0), Full Proca (Y_h = 0, X_V 2 terms, residual 0). Proca-mass-only intentionally returns nonzero residual (algorithm precondition not met — pure-mass EOM has no 2nd-deriv signature). |
| `test_eom_decompose_synthetic.py` | **Synthetic Y_h test for the new T_M kinetic-extraction path.** Builds Y_test (derivative-free h × η × η) and Z := Y_test × T_M[free scalar], runs the orchestrator, verifies Y_h × T_M = Z exactly. Y_h is the (α↔β)-symmetric representative of Y_test (algorithm's only ambiguity). |
| `test_bootstrap_loop_em_belinfante.py` | **Full end-to-end EM Belinfante through n_max=2.** Exercises every piece: step 1 (κ T_Bel + W), step 2 carryover (52 terms at n=2 from X_A^(1)), step 3 H2 correction (Z = 64 at n=2 → 0 after), step 5 superpotential, step 6 closure, EL self-consistency, L_ref integrability check, redef recovery (`f_A^(1)`, `f_h^(2)+f_A^(2)`, `f_h^(3)`), and full substitution-into-L_ref application (h-redef + matter-redef per paper ordering). All three orders close: L_ref diff goes to 0 at n=0, 1, 2. **First complete Belinfante closure through 3 orders.** ~33 min wall. |
| `test_traceless_detect_em_d4.py` | **EM at d=4, full round-trip traceless detection.** Uses `set_dimension(4)` to build a d-dependent environment. Verifies (a) traceless_T_M is detected at order 0, (b) EL self-consistency holds at order 1, (c) off-shell-traceless-ness is correctly recorded. Validates the `_check_T_M_traceless()` machinery and the `set_dimension` import-order contract. ~10 s. |
| `test_traceless_verify_em_d4.py` | **EM d=4, verification-path traceless recovery at order 2.** Injects a traceless-shape optional h-EOM term at order 1 (invisible there, surfaces as ddh in E_diff at n=2). Recovery mechanism detects it, recovers the injected X exactly, applies the field redef, and drives E_diff to 0. Validates that the missed-redef rollback+augment path works correctly. **EM d=4 closure verified via this path.** ~80 s. |
| `test_traceless_mandatory_order3.py` | **Conformal scalar at order 3, mandatory-step traceless absorber (isolated).** Runs orders 0–2 cleanly, then exercises step 3's new mandatory-step absorber: H2 violation Z (24 terms, 4 ddh-bearing) is decomposed to recover order-2 X, X·S⁽¹⁾ is added to E, H2 is rechecked and verified zero. Recovered X stored and validated. **Both recoveries (verification and mandatory paths) working in isolation.** ~40 min wall. |
| `test_conformal_scalar_symbolic_d_n3.py` | **Conformal scalar (symbolic d), no injection, full closure orders 0–3.** The clean validation of the symbolic-d machinery: ξ(d) = (d−2)/(4(d−1)) via `dimension()`, on-shell traceless for all d, closes against L_ref at every order with NO field redef. H2 Z=0 at every order; Ψ = 9/42/84 terms at n=1/2/3; E^(3) = 367 terms; `c_φ = −(d−2)/2·κφ`. All d-rational coefficients kept clean by the canon-level `_simplify_d_coeffs` pass. **First end-to-end conformal-scalar (nonminimal-coupling) closure through order 3 with general d.** ~25 min wall. |
| `test_nonminimal_vector_ricci_n3.py` | **Nonminimal vector–Ricci coupling `−¼F·F + ξ V^a V^b R_{ab}`, upstairs V, Hilbert, orders 0–3 (d=4).** First exercise of the nonminimal-coupling machinery on a VECTOR–Ricci-tensor coupling (only the scalar `φ²R` had been run). Surfaced + fixed the missing 3rd-derivative head: vector registrations now define `dddV`/`dddA` (field index + symmetric derivative triple) so `_check_improvement_conserved` can take the divergence of the `ddV`-bearing improvement. Closes 0–3; E^(3) = 626 terms. |
| `test_canon_zero_index_ordering.py` | **Regression for the canon 0-index-tensor ordering fix.** `canon(φ1·φ2 − φ2·φ1) == 0` and related: canon must canonically order products of 0-index Tensors (matter fields) on recombine, else commuting products in different order don't cancel — a phantom-term bug that broke the traceless-T_M check for ≥2-scalar sectors (found via the traceless kitchen sink). ~1 s. |
| `test_conformal_order3_lref_closure.py` | **Conformal scalar, full orders 0–2 pipeline to checkpoint for order-3 debugging.** Symbolic d with `ξ = (d−2)/(4(d−1))` via `dimension()` (pickle-safe — no `set_dimension` rebuild). Runs the verification-path recovery at order 2 (missed order-1 traceless redef on nonzero c_i), saves `BootstrapState` checkpoint for resumption. ~10 min wall (orders 0–2 only). |
| `test_conformal_order3_lref_closure_resume.py` | **Conformal scalar, order 3 only (from checkpoint).** Loads the post-order-2 checkpoint and runs order 3 with full diagnostics. Fast iteration on order-3 closure debugging: ~20 min per run instead of 30+ if re-running orders 0–2. |

## Open work, in priority order

0. **Trace-free T_M loophole — DONE. Dual-injection conformal scalar CLOSES against L_ref at every order 0–3** (2026-06-03), exercising both the verification-path and mandatory-step traceless recoveries together with rollback+augment.

   **COMPLETED (2026-06-03):**
   - **`_recover_traceless_mandatory_eom(n, E, Z, h_indices)`** (lines 1524–1582): new method that fires in step 3 when `traceless_T_M=True` AND `Z` contains ddh. Recovers the order-(n−1) mandatory traceless-shape X from the ddh-box signature via polynomial division, computes the FULL H2 of `X·S⁽¹⁾`, subtracts from Z, asserts Z is now ddh-free, stores X for carryover. Tested at order 3 on conformal scalar with dual injection — recovered X matches the predicted structure (−⅓ eps² κ⁴ h·trh + ⅓ eps² κ⁴ η·(h·h)).
   - **Field-redef rollback+augment mechanism** (lines 1537–1552 in `_recover_missed_traceless_redef`, new helpers `_snapshot_L_ref_for_rollback`, `_add_one_redef`, `_add_redefs`): when order-n verification detects a missed order-(n−1) traceless redef, the mechanism now (1) rolls L_ref back to the snapshot taken just before order-(n−1)'s normal redef, (2) augments the two redefs by summing their f-expressions, and (3) re-applies the total once (avoiding double-count of chain-rule cross-terms). Rollback snapshot moved out of `_apply_field_redefs_to_L_ref` into `_verify_vs_L_ref` and taken at exactly one point per loop order (after missed-redef recovery, before `E_diff == 0` check and normal redef). **EM d=4 regression test passes** — verified X recovery still works post-refactor.
   - **Residual diagnostics** (lines 1977–1988 in `_verify_vs_L_ref`): when a non-integrable residual is encountered at the verification step, the code now prints the first 12 terms before raising, enabling fast debugging of closure failures.
   - **Symbolic-d coefficient simplification in `canon()` (2026-06-03)** — the enabling fix for general-d conformal-scalar work. `canon` now runs `_simplify_d_coeffs` (cancel each term's `.coeff`, drop zeros) after structural canonicalization, gated on symbolic d (see Conventions). This makes traceless-T_M detection, H2 Z=0, EL self-consistency and E_diff=0 all recognize identically-zero d-rational coefficients. New `tensor_algebra.dimension()` accessor returns `Lorentz.dim` so couplings share the metric-trace `d` Symbol (a fresh `Symbol('d', real=True)` is a different object and breaks cancellation). Verified: conformal scalar detects on-shell-traceless T_M with symbolic d, `c_phi = −(d−2)/2·κ·φ` (clean); orders 0–1 close against L_ref with d-coefficients; d-free regressions (scalar, EM d=4) unchanged. This supersedes the earlier d=4-substitution / skip-assertion workarounds.
   - **Checkpoint+resume infrastructure** (test files `test_conformal_order3_lref_closure.py` + `_resume.py`, uses pickle): orders 0–2 save a `BootstrapState` checkpoint, resume test loads it and runs order 3. Now uses **symbolic d** (no `set_dimension`) — pickle-safe because the Lorentz/h/dh/ddh heads are never rebuilt. `test_conformal_scalar_symbolic_d_n3.py` is the clean no-injection symbolic-d closure test (should close orders 0–3 with NO field redef).

   **Closure at order 3 — SOLVED (2026-06-03).** The dual-injection conformal scalar (`test_conformal_order3_lref_closure.py`, d=4) now closes at every order 0–3: order 2 via the verification-path recovery (recovers the injected `X = eps·κ²·h` exactly), order 3 via the mandatory-step absorber + the verification-path recovery + rollback-and-augment, with the order-3 residual going **31 → 19 → 0** (was a 12-term non-integrable residual before the fix). The recovered order-3 X is `−2 eps κ³ h(i3,L)h(i4,-L) + ⅔ eps²κ⁴ h(i3,i4)h(L,-L) + ⅓ eps²κ⁴ h(L,L1)h(-L,-L1)η(i3,i4)`.

   **The fix: corrected verification-path recovery template.** The old `_recover_missed_traceless_redef` used `E_diff = X^{ab}·S^(1)` (the "T1" template) and divided the doubly-traced ddh box signature by `(m+v)`. That is INCOMPLETE: per the paper, the missed redef's effect on `E_h^{mn(n)}` is `X^{ab} S_{ab}^{mn}` with `S = T1 + T2`, where
   - `T1 = X^{mn} S^(1)` (the old template), and
   - `T2 = [ (∂E_h^{mn(1)}/∂h_{jk,cd}) η_{jk} + Σ_i (∂E_h^{mn(0)}/∂φ_{i,cd}) c_i ] h_{ab,cd}` (new), from substituting `f_h = f η` into `E_h^(1)` and `f_{φ_i} = f c_i` into `E_h^(0)` and keeping the ddh part of `f_{,cd} = X^{ab}h_{ab,cd}`.
   T1 mis-weights the **trace sector** of X (it omits T2's `η`-piece and the `c_i` contributions). Concretely the trace coefficient came out 2/3 (wrong) vs 1/3 (correct); that single bad coefficient was the entire 12-term residual.
   New recovery: build `T2 = _build_traceless_redef_operator()`; extract `m = _extract_ddh_deriv_signature(T2, μ, ν)` and `P = _extract_ddh_deriv_signature(E_diff, μ, ν)` on the signature T1 cannot reach (ddh **derivative** pair = the free indices μ,ν — T1 keeps ddh inside the scalar S^(1) so it never lands there); then `X^{ab} = P/m`. `m = d−2` is universal (= 2 at d=4). `_extract_ddh_deriv_signature` relabels the freed field-pair indices to a common (A,B) (needed so the many E_diff terms sum) and reinterprets a self-traced field pair as `η^{ab}` (the X-trace part). A non-scalar `m` would signal a `v_{ab}^{cd}` contamination term (the "work harder" linear-solve case) and raises loudly — **verified v=0** for scalar & upstairs-vector × Hilbert & Belinfante (see `_diag_T2_v_check.py`). `_build_traceless_redef_operator` handles rank-0 (scalar `c_φ`) and rank-1 (vector `c_V`, vector index contracted) `c_i`; rank≥2 raises.

   **Known working cases:**
   - **EM d=4, n=2, verification path** (`test_traceless_verify_em_d4.py`): recovers injected X exactly, E_diff → 0 (regression-passes after the T2 rewrite + index-remap).
   - **Conformal scalar dual injection, orders 0–3, d=4** (`test_conformal_order3_lref_closure.py`): full closure, both recoveries + rollback/augment.
   - **Conformal scalar, NO injection, symbolic d, orders 0–3** (`test_conformal_scalar_symbolic_d_n3.py`): closes with no field redef; rigorous general-d check of the symbolic-d machinery.

   **Independent confirmation (analytic).** The corrected trace coefficient `1/3` (vs the old `2/3`) was checked BY HAND to be exactly the value that makes the recovered X **integrable** (Helmholtz `∂X/∂h` symmetric under αβ↔μν). Integrability is the genuine criterion for X to be a real field redefinition, and it is dimension-independent — so the `1/3` is forced analytically, by a route independent of the T2-substitution derivation. This means the d=4 closure has analytic backing and is not a numeric coincidence.

   **Caveats / open:** the dual-injection closure run itself is at **d=4** (fast). A symbolic-d confirmation of the *injected* case is now optional (nice belt-and-suspenders) rather than needed-for-rigor, since the `1/3` is independently forced by integrability (above). Cross-process pickle **resume does NOT work** (module globals `_matter_fields` + tensor-head identity don't survive the pickle) — the checkpoint is inspection-only (`_dump_ckpt.py`); each order-3 test must run in-process from order 0.

   **What was in the original spec (all NOW DONE):**
   - Verification-step traceless path: detect ddh in E_diff → recover missed `X^{ab(n-1)}` ✅
   - Roll back order-(n-1) redefs → augment → re-apply ✅
   - Mandatory-EOM-step path: detect ddh in Z → recover X → subtract full H2 ✅
   - "Full H2 of a product" utility: reused `compute_h2_violation` (already computes full, non-truncated H2) ✅

   **Detailed spec (reference):**

   **The problem.** A realistic T_M may be **on-shell traceless**, meaning
   there exist derivative-free `c_i(matter)` such that `E^(0)^a_a + Σ_i c_i E_i^(0) = 0`
   (off-shell traceless is the special case `c_i = 0`). Maxwell in `d=4` is the
   archetype (`c_i = 0`, fully off-shell-traceless). We don't currently fail
   on this because the codebase carries `d` as a general (un-evaluated) symbol
   so the `d−4` in tr(T_M[EM]) doesn't cancel, but that's not a property to
   rely on. When the condition holds, any EOM combination
   `X_h^{ab}_{cd} ≡ X^{ab} η_{cd}`, `X_i^{ab} ≡ c_i X^{ab}` (X derivative-free,
   order n in h) gives a zero contribution at order n. Its first nonzero
   contribution arrives at order n+1, via
   `X^{ab(n-1)} · (η_{cd} E_h^{(1)cd} + Σ_i c_i E_i^{(1)})` (where `E_h^{(1)}`
   contains the wave operator W plus h × T_M-stuff, and `E_i^{(1)}` is the
   order-1 piece of the matter EOM; both contain ddh, dh, and h).

   **Detection (end of run_order(0)).** Compute `E^(0)^a_a` (the trace of the
   stored E[0]). Run a decomposition against the matter EOMs to look for
   derivative-free `c_i` satisfying `E^(0)^a_a + Σ_i c_i E_i^(0) = 0`. If
   solvable, save the `c_i` and set a `self.traceless_T_M = True` flag. Off-
   shell traceless ⇒ all `c_i = 0` (still flag).

   **Optional-EOM step (Step 4): no change.** User-supplied X's are inserted
   and recorded as usual; the bootstrap doesn't need to recognise their
   traceless-loophole shape.

   **Verification step modifications.**
   1. After computing `E_diff^{ab(n)} = E^{(n)} - EL(L_ref^{(n+1)})`, check
      for ddh content. If present → there was a missed traceless-shape redef
      at order n−1 whose contribution cancelled at that order and only
      surfaces now.
   2. Recover the missed `X^{ab(n-1)}`. Pick the fully-contracted scalar
      ddh signature `ddh(L_0, -L_0, L_1, -L_1)`. Find its coefficient in
      `(η_{cd} E_h^{(1)cd} + Σ_i c_i E_i^{(1)})` — this has form `m + v` where
      `m` is a constant (the W-trace contribution) and `v` is a derivative-
      free matter scalar (the matter contributions). Find the same
      signature's prefactor in E_diff^{ab} — this prefactor equals
      `X^{ab} · (m + v)`. Use polynomial division (treating `v` as the
      indeterminate) to recover `X^{ab}`. Implementing this requires a
      tensor-level "collect by powers of a scalar matter function and divide
      by a polynomial in that scalar" utility — new.
   3. **Roll back** the previously-applied f's at order n−1. This requires
      keeping the previous L_ref state in reserve one order back (a new
      `self.L_ref_history` snapshot taken before each `_apply_field_redefs_to_L_ref`).
      Augment the order-(n−1) redefs with the missed-traceless contribution:
      `f_h^{(n)}_{cd} += (1/n) h_{ab} X^{ab(n-1)} η_{cd}`,
      `f_i^{(n)} += (1/n) h_{ab} X^{ab(n-1)} c_i`. Re-apply, recompute
      `EL(L_ref^{(n+1)})`. The new E_diff should be ddh-free.
   4. Then do the normal decomposition for the order-n f's on the cleaned
      E_diff.

   **Mandatory-EOM step (Step 3) modifications.**
   1. The H2 violation `Z^{abcd}` may contain a `X^{ab(n-1)} (η_{cd} E_h^{(1)cd} + Σ_i c_i E_i^{(1)})`
      piece. The product itself **does** carry ddh (from E_h^{(1)} and E_i^{(1)}),
      so naively scanning Z for ddh content does not isolate X's piece.
   2. **Key observation:** when the H2 formula
      `Z[F]^{μν αβ} = 2(∂F^{μν}/∂h_{αβ})_anti − ∂_γ(∂F^{μν}/∂h_{αβ,γ})_anti`
      is applied to the product `F = X · (η E_h^{(1)} + Σ c_i E_i^{(1)})`, the
      ddh in `Z[F]` arises **only** from the `2(∂X^{ab}/∂h_{cd})_anti · (η E_h^{(1)} + Σ c_i E_i^{(1)})`
      term. The other H2 contributions (`(∂E^{(1)}/∂h)_anti · X` and the
      `∂_γ(∂F/∂dh)_anti` piece) carry no ddh, because E_h^{(1)} and E_i^{(1)}
      are by definition order 1 in h ⇒ their ∂/∂h and ∂/∂dh are order 0
      (functions of matter fields only).
   3. So: extract `Y^{abcd}` as the coefficient of
      `(η_{cd} E_h^{(1)cd} + Σ_i c_i E_i^{(1)})` in Z^{abcd} via the same
      ddh-signature polynomial-division trick described above. Recover
      `X^{ab(n-1)}` the usual way (paper's `X = -1/(2n) Y h` formula).
      Compute the FULL H2 contribution of `X · (η E_h^{(1)} + Σ c_i E_i^{(1)})`
      (not just the antisymmetric ∂X/∂h piece — must include the ∂E^{(1)}/∂h
      and dh-piece contributions). Subtract from Z. Residual has no ddh or
      dh; decompose as usual.

   **New shared utilities required.**
   - `_check_T_M_traceless(E_0)` → `(is_traceless, {field_name: c_i})`. Runs
     a decomposition of `eta_{ab} E_0^{ab}` against the matter EOMs.
   - A tensor-level polynomial-division helper: given numerator
     `N^{...}` and a scalar denominator `p = m + v` (with `v` a known
     derivative-free matter scalar), return `N / p` by collecting powers of
     `v` in `N` and dividing.
   - `self.L_ref_history`: snapshot of `self.L_ref` taken just before each
     `_apply_field_redefs_to_L_ref` call, so we can roll back one order.
   - A "full H2 of a product" utility that doesn't truncate at the
     antisymmetric ∂X/∂h piece.

   Pending implementation. The traceless flag detection + new utilities
   would land first, then the verification-step modifications, then the
   mandatory-EOM-step modifications.

1. **n=5 pure-gravity closure.** Same script at the next order. Wall-time projection from the n=2 → n=3 → n=4 progression (1 min → 5 min → 20.5 min, ~4× per order): roughly 80–120 min for n=5, dominated by `compute_superpotential_n2` (already 10 min at n=4 on a 315-term M) and EL of L_EH^{(6)}. Without further optimization, n=5 is the largest tractable order on this machine.
2. **Further performance work.** Optimizations landed: (a) one-shot `TensAdd(*terms)` via the `_sum_terms` helper across `jet.py` / `energy_momentum.py` / `euler_lagrange.py` (sympy's `TensAdd.__add__` re-normalizes on every call, so the loop variant paid N× the bookkeeping; net ~45 % at n=3); (b) consolidated the three per-term canon calls in `compute_superpotential_n2`'s `_wrap` helper into a single canon over the combined bracket; (c) cached `_symmetric_permutations` in `jet.py` by `(n_indices, sym_groups)` — same shape recomputed on every jet-derivative-of-factor call; (d) short-circuits in `uncontract_metrics`, `replace_metric_with_ginv`, `_strip_zero_index_recursive` that return the original expression unchanged when no factor needs adjustment, avoiding pointless TensMul rebuilds. Two attempted optimizations REGRESSED and were reverted: a canon-result cache (10 % hit rate didn't pay for hash overhead on big tensors) and deferring canon to the outermost recursive call (single huge uncanon'd TensAdd was slower than many small canons). The cProfile data still points at `contract_metric` and `canon_bp` inside sympy itself; further wins likely require either patching sympy or restructuring the algorithm so canon is called on smaller pieces.
3. **DONE — Field-redefinition decomposition end-to-end on EM Belinfante through n=1.** Integrability check (`_integrability_check`) runs `decompose_against_eoms` on E_diff and validates residual = 0 + derivative-free X's. f^(n+1) recovery via `_recover_field_redefs` (paper formula `(1/(n+1)) h X`). Substitution machinery (`_substitute_field` + `_total_derivative_at` + `_apply_field_redefs_to_L_ref`) propagates φ → φ + f through dφ, ddφ via chain rule, with the fresh-deriv + metric-contract trick that dodges sympy's auto-canon dummy clash. Ordering convention from the paper followed: h-redef first across L_ref^(k) for k = 0..n_max-n+1, then matter-field redefs one at a time for k = 0..n_max-n. After application, EL(updated L_ref^(n+1)) is re-checked and must equal E^(n) (raises if not). **Both n=0 and n=1 close cleanly** on the EM Belinfante test. Next: push to higher n_max (n_max=2 will exercise the propagation of n=0 and n=1 redefs through to L_ref^(3) and test whether the orchestrator's signature coverage holds up at more orders); also extend beyond EM (Proca, charged scalar, kitchen sink) to confirm the matter+h substitution machinery is general.
4. **DONE — Explicit H2-violation check** at every order via `compute_h2_violation` in `helmholtz.py`, wired into `_step3_mandatory_eom` in `bootstrap_loop.py`. Verified Z = 0 directly for pure gravity (n=1, 2) and scalar matter (n=1); the closure tests previously implied this, now confirmed empirically per Butcher's claim. The implementation pass uncovered the long-running `_decompose_tensmul drops TensAdd children` footgun (raw `A − B` of TensAdds → sympy wraps `-B` as `TensMul(NegativeOne, B)` → `_decompose_tensmul` silently drops the inner TensAdd → wrong output). Fixes: `total_derivative` now `.expand()`'s its input; `_decompose_tensmul` raises if it sees the bad pattern so future instances are loud.
5. **DONE — H2-violation machinery.** Algorithm from user 2026-05-26 fully implemented in [bootstrap/eom_decompose.py](bootstrap/eom_decompose.py) via `decompose_against_eoms(Z, L, em_procedure)`: kinetic-signature extraction from T_M with the (contracted-pair-skip) refinement, trace-term step, and per-matter-EOM 2nd-deriv-monomial matching. Wired into `_step3_mandatory_eom` in [bootstrap_loop.py](bootstrap/bootstrap_loop.py); when Z ≠ 0, the orchestrator runs, X = -1/(2(n+1)) Y · h is formed, and X · E^(0) is added. An H2 re-check after the correction passes (Z = 0). Verified end-to-end on EM Belinfante through n=1 (~100 s); also tested via a synthetic Z = Y_test · T_M[scalar] (Y_h × T_M = Z, residual 0). The pure-mass-only EOM precondition (no 2nd-deriv signature in EOM) is a known un-handled corner case — see Proca-mass-only smoke. Two related signature notes: `compute_h2_violation` now returns `(Z, (α, β))`, and the X-side contractions must read `.get_free_indices()` (signs depend on canon's metric work).
6. **Higher orders / other matter examples.** Push EM and scalar QED to n=3 (extrapolating: probably 10–20 min wall for EM, 30+ min for scalar QED). φ⁴ self-interaction (no new code expected — same algorithm as φ², just write the L_M). Massive vector field (needs the full Christoffel-correction machinery in `matter_lagrangian_order`; not gauge-invariant, so the cancellation our EM case relies on doesn't hold).
7. **The n=1 Ψ integral formula is well-exercised; Ψ^{(1)} = 0 holds only "in principle" on the pure path.** On the PURE default path (Hilbert, no optional EOM, no nonminimal coupling) the derivation gives Ψ^{(1)} = 0 — pure-gravity superpotentials only, from n=2 up. But that cancellation depends on preserving the ordering of the symmetric second-derivative index pairs (ddV/ddh), which sympy's canonicalization permutes — so e.g. Hilbert full Proca comes out with a nonzero, cyclic-symmetric Ψ^{(1)} = 12 terms (item 10): the *ordering residue*. OUTSIDE the pure path — optional EOM, Belinfante, or a nonminimal matter-curvature coupling — Ψ^{(1)} is *genuinely* nonzero, and all three are now confirmed in tests: the Belinfante and optional-EOM cases both produce n=1 superpotentials, and the conformally coupled scalar gives 9 terms. (So the original "waits on Belinfante or optional EOM" framing of this item is moot.) The cyclic-symmetry check (not vanishing) is the gate.
8. **DONE — Symmetrized Belinfante procedure** as an alternative to Hilbert (paper §2). Implemented in `symmetrized_belinfante` using the canonical Noether + Belinfante improvement (spin tensor) + explicit (μν)-symmetrization. Verified T_SymBel − T_Hilbert is purely EOM-proportional for free/massive scalar (0 terms), EM (4 ddA·A terms), Proca, and pure gravity h_{αβ}. `em_procedure='belinfante'` plumbing wired into BootstrapState. Free-scalar bootstrap closes step 6 at n=1; L_ref verification fails by 3 terms as expected — Belinfante triggers the field-redefinition machinery in item 3 (which is exactly the user's prediction).
9. **Optional EOM terms (field redefinitions, voluntary path).** Integrability check ∂X/∂h symmetric under (αβ ↔ μν); add a switch in `bootstrap_loop.py` to inject such terms by user choice (orthogonal to the field redefinitions forced by the verification cycle in item 3).
10. **RESOLVED — full Proca passes orders 0..2.** The cyclic-asymmetric Ψ^(1) was caused by the *bracket itself* (formed inside `compute_superpotential_n1`) sometimes coming out as a TensAdd with top-level args of shape `TensMul(Rational, TensAdd)` — i.e. a rational coefficient that didn't distribute across the inner TensAdd. `_scale_matter_fields`, which uses `_decompose_tensmul`, silently dropped the inner TensAdd because `_decompose_tensmul` only collects Tensor/TensMul args. That meant matter-field factors hidden inside got the wrong λ-power (effectively λ⁰), so the λ-integral gave them the wrong coefficient and Ψ^(1) lost its cyclic-symmetry. Fix: `.expand()` on the bracket before `_scale_matter_fields`. User-side note that motivated finding this: **both V AND dV (and ddV) should get a λ factor** for each occurrence — verifying that prompted inspecting `_scale_matter_fields` and discovering the inner TensAdd was being silently skipped. With the fix, full Proca's Ψ^(1) is 12 terms, cyclic-symmetric, and the bootstrap closes through n=2.

11. **RESOLVED — kitchen-sink Lagrangian passes orders 0..4 after adding the ∇A Christoffel chain rule.** The 52-term EL-self-consistency residual at n=1 for charged Proca / kitchen-sink was caused by `energy_momentum.py` missing the `dA → ∇A = dA(σ, ρ) − Γ^τ_{ρσ} A_τ` substitution inside `_christoffel_via_substitution`. The F-antisymmetry that cancels Γ inside F_μν only does so at order 0; once T_M[L^(n)] is computed at n ≥ 1, the Γ piece is needed for any Lagrangian where A couples directly to other matter (here: e A V_partner inside F^V_i). With the dA branch added (mirroring the existing dV branch), all closures pass through n=4 — see headline result. **Open puzzle still:** scalar QED has the same A head and the same code path, but its tests (orders 0..2) pass *without* the dA fix. Either there's a structural cancellation specific to scalar QED's coupling pattern (worth understanding to articulate the boundary), or the tests are succeeding accidentally at the orders we've run and will fail higher up. Worth a quick re-run of scalar QED with the new code (sanity, should still pass) and an extension to n=3 to see if any latent issue surfaces.

## Conventions and pitfalls

- **All indices are abstract.** No spacetime dimension is fixed; expressions are never expanded into components.
- **Order in h** counts h, dh, and ddh each as one power.
- **`canon()` must include `contract_metric()`.** SymPy's `canon_bp()` does not contract metrics, and Kronecker deltas from jet derivatives pile up and blow expressions up if they aren't contracted.
- **`canon()` canonically ORDERS products of 0-index Tensors (matter fields).** canon's 0-index workaround strips out 0-index `Tensor` factors (e.g. `phi1()`, `phi2()` — sympy's `canon_bp`/`contract_metric` crash on them), canonicalizes the rest, then re-multiplies the stripped factors back. That re-multiply MUST sort them (we use `default_sort_key`): sympy's `TensMul` does not reorder 0-index Tensors and the manual re-multiply otherwise preserved source order, so `phi1*phi2` and `phi2*phi1` stayed distinct and an identically-zero difference like `phi1*phi2 - phi2*phi1` did NOT cancel. This silently left phantom terms (and defeated `== S.Zero` gates) in ANY matter sector with ≥2 multiplied scalar fields — it surfaced as a spurious 2-term residual in the traceless-T_M check of the "traceless kitchen sink" (two charged scalars). Locked by `tests/test_canon_zero_index_ordering.py`.
- **Symbolic-d coefficients: `canon()` cancels them; build couplings with `dimension()`.** With a symbolic spacetime dimension `d` (the default) and a d-dependent coupling like the conformal ξ(d) = (d−2)/(4(d−1)), scalar coefficients become rational functions of d. `canon_bp` combines like tensor structures but does NOT simplify those coefficients, so an identically-zero coefficient (e.g. `8d²κ/(16d−16) − dκ/2 − …`) survives as an unsimplified sum, inflating term counts and defeating every `== S.Zero` gate (traceless-T_M, H2 Z=0, EL self-consistency, E_diff=0). `canon` therefore runs a `_simplify_d_coeffs` pass (`cancel` each term's `.coeff`, drop the zeros) **after** structural canonicalization — GATED so it's a no-op when `Lorentz.dim` is a concrete int (`set_dimension(N)`) or when `d ∉ expr.free_symbols` (pure gravity / d-free matter pay one membership check). When forming a d-dependent coupling, use `d = tensor_algebra.dimension()` (= `Lorentz.dim`) — a freshly built `Symbol('d', real=True, ...)` is a DIFFERENT object (assumptions are part of Symbol identity), so `d_yours − d_trace` never collapses and traceless detection silently fails. Symbolic d is also pickle-safe for checkpoints, unlike `set_dimension(4)` which rebuilds the Lorentz/h/dh/ddh heads.
- **Fresh indices via the global counter** (`_i0, _i1, ...`). Use `fresh_indices()` generously when building products — never reuse a free index from one expression as a dummy in another expression you're multiplying it with. `fresh_indices(0)` returns `()` (was a crash: `tensor_indices('')` raises "no symbols given") — this degenerate case is hit when relabelling a SCALAR field's empty EOM-direction index set, e.g. merging two scalar optional-EOM / carryover coefficients (`_merge_eom_coeff`) at order ≥ 2, which is why tagged-optional scalar runs only tripped it from n=2 up.
- **Half-symmetrization in jet derivatives.** ∂h_{μν}/∂h_{αβ} = ½(δ^α_μ δ^β_ν + δ^α_ν δ^β_μ). The ½ is required because h is symmetric.
- **TensorSymmetry API.** Use `TensorSymmetry.direct_product(2, 1)` for a "symmetric pair + single index" head (e.g. dh's symmetry). `from_generators` is not the right tool here.
- **Free-index relabelling in (PsiForm).** The three terms of the superpotential formula use M with *different* free indices: terms A and B use M^{αβ}, term C uses M^{μν}. `compute_superpotential_n2` relabels M per term — if you tweak that function, mismatched free-index sets surface as `ValueError: all tensors must have the same indices`.
- **`substitute_indices` is silent on bound dummies.** It rewrites free indices only. To rename a dummy pair you must rebuild the affected factors with the new index. This subtlety bit us inside the IBP fix (see headline).
- **`_decompose_tensmul` raises on `TensMul(scalar, TensAdd)`.** The most common trigger is raw `A − B` of two TensAdds: sympy keeps `-B` as `TensMul(NegativeOne, TensAdd)` rather than distributing, and any per-term Leibniz/integration loop that iterates `expr.args` of such a wrap would silently drop the inner sum's terms. Now caught at the source. If you hit the error: call `.expand()` (or `canon`) on the input first. `total_derivative` already does this defensively at the top. See `project-decompose-tensmul-tensadd-pitfall` memory for the historical bites.
- **Performance.** Wall times: n=2 ~1 min, n=3 ~5 min, n=4 ~20.5 min. Dominant step at n=4 is `compute_superpotential_n2` (~10 min on a 315-term M); next-tier are `hilbert_energy_momentum` (~4.5 min) and EL of L_EH^{(5)} (~3.5 min). Empirical scaling is roughly 4× wall-time per order.
- **The right idiom for summing tensor terms is `TensAdd(*terms)`, not a `result = result + t` loop.** sympy's `TensAdd.__add__` re-runs the term-collection pass on the growing accumulator, so the loop pattern pays N× the per-call overhead. Helper `_sum_terms()` lives in `jet.py` — use it for new code in this codebase.
- **canon caching does NOT help.** Was tried; sympy tensors are hashable but hashing big tensors is expensive enough that a 10 % cache hit rate didn't break even. Deferring canon to the outermost recursive call also REGRESSED — sympy's canon is super-linear in term count, so one big canon is slower than several small canons. Per-step canonicalization is load-bearing; the cost is real and hard to dodge.
- **Run-to-run timing variance.** Same code, same machine, no other user load → up to ~2× variance on individual steps. The likely culprits are `PYTHONHASHSEED` randomization (changing sympy's traversal order in dict/set iteration) and background Windows activity (Defender, Search indexer, OneDrive). Set `PYTHONHASHSEED=0` for reproducible runs when comparing optimization candidates.

## File dependencies

```
tensor_algebra.py ── jet.py ──┬── euler_lagrange.py ──┐
                              ├── helmholtz.py        ├── bootstrap_loop.py
                              ├── covariant.py        │
                              ├── energy_momentum.py  │
                              ├── loop_helpers.py ────┤
                              └── eom_decompose.py ──┬─┘
                                                     │
                                  traceless.py ──────┘   (imports loop_helpers,
                                                          eom_decompose, helmholtz)
```

`loop_helpers.py` holds general helpers (`_count`, `_term_breakdown`,
`_format_breakdown`, `_reindex_tensor`, `_is_derivative_free`,
`_derivative_heads`) shared by `bootstrap_loop`, `traceless`, and
`eom_decompose` — extracted so the traceless subsystem could move out without
an import cycle. `traceless.py` holds the trace-free-T_M machinery: the two
`_extract_ddh_*` signature functions and `TracelessRecoveryMixin` (detection +
both field-redef recoveries), which `BootstrapState` inherits. Dependencies
are strictly one-directional (no cycles); `eom_decompose`'s former lazy
`from bootstrap_loop import _reindex_tensor` now points at `loop_helpers`.

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
python tests\test_belinfante_smoke.py                  # ~1 min — T_SymBel vs T_Hilbert for scalar/EM/Proca/h (diff purely EOM-proportional)
python tests\test_bootstrap_loop_scalar_belinfante.py  # ~30 s — first BootstrapState run with em_procedure='belinfante' (closes step 6 at n=1)
```

The two superpotential tests are the headline tests.

## HPC heavy-run suite (`hpc_suite/`)

A set of 10 high-order closure runs for an HPC cluster (Zeus, PBS Pro), in
`hpc_suite/runs/` (one self-contained script each; closure = `*** PASS`). Each
takes an optional `n_max` arg (`python -u runs/NAME.py [n_max]`) so it can be
smoke-tested locally at low order; all 10 are smoke-validated. Coverage spans
the axes that exercise distinct code paths, each pushed past current orders:
pure gravity (n=8), massive scalar (n=7), EM d=4 (n=7), full Proca (n=6),
scalar + tagged optional-EOM under Belinfante (n=6), EM d=4 Belinfante (n=6),
conformal scalar + dual injection at symbolic d (n=5), kitchen sink (n=5),
the conformal "traceless kitchen sink" (n=5), and that + dual injection (n=4).
`submit_all.sh` qsubs one single-core job per run (`zeus_new_q` 72 h/1 TB for the
6 that fit, `zeus_long_q` 168 h/378 GB for the 4 monsters), unbuffered to
`logs/`, `PYTHONHASHSEED=0`. `hpc_suite/README.md` has the run table, queue
rationale, and the no-resume / partial-completion behavior (an overrun keeps the
completed orders in the log; the real high-`n_max` cost is the up-front
`L_ref^(0..n_max+1)` precompute). See `hpc_suite/README.md` to submit.

### Performance: the field-redef bottleneck (branch `optimize-field-redef`; #1+#2 done & validated, #3 pending)

The first real Zeus run exposed a bottleneck: run #5 (tagged-optional Belinfante
scalar) sat ~15 h on order 1. It is **not** a cluster/resource problem — same
sympy 1.14, mem fine, and the **laptop reproduces it identically**. Root cause is
the **h-field-redefinition substitution** in `_substitute_field`
(`bootstrap/bootstrap_loop.py`): substituting `h → h + f` builds
`Π_i (fac_i + rep_i)` over all factors, `.expand()`s the full **`2^p`** binomial
(p = number of h-factors), `canon`s every term, **then** truncates by order — so
the high-order `L_ref^(k)` (many h-factors) detonate. The φ-redef is cheap; only
h is the wall. (Eliminated along the way, all red herrings: sympy version,
`PYTHONHASHSEED=0`, and missing gmpy/flint — the laptop has none of those and is
still fast.)

Measured h-redef @order 1 of #5 (n_max = 2/3/4): pure-Python **280 / 1493 /
14194 s** (~9.5×/order, worse-than-exponential). `python-flint` + `gmpy2`
(`GROUND_TYPES=flint`) buy only a **~2× constant factor** (127 / 851 s) — worth
installing on Zeus, but not the cure.

Three fixes (see memory `project_field_redef_optimization.md` for the full
recipe). **#1 and #2 are implemented + validated; #3 pending:**
1. **Cache `df`/`ddf` once per redef** — *done*. `_build_deriv_cache` differentiates
   `f` ONCE against template indices; `_apply_one_field_redef` builds it once and
   passes `deriv_cache=` into the k-loop; `_replacement_for` re-indexes the cached
   template per (d/dd)field occurrence (fresh dummies + metric contraction)
   instead of re-running `_total_derivative_at`. Diff and free-index relabel
   commute → exact to canon. Adds **~1.5×** on the h-redef on top of #2
   (**12 / 54 / 198 s** at n_max = 2/3/4 under flint, vs #2-only 20/76/298).
2. **Order-bounded substitution** — *the cure, done*. Generate only subsets of the
   substitutable factors of size `0..j_max`, with
   `j_max = ⌊(target_order − c)/δ⌋`, `c = order_in_h(term)`,
   `δ = order_in_h(f) − (1 if field is h else 0)` (h-redef order-n f adds n−1;
   matter adds n; always ≥ 1 — truly-linear redefs are forbidden). Keep two
   per-term counts distinct: `c` = h-expansion order (cutoff) vs `p` =
   **occurrences of the substituted field** (matter multiplicity, ≠ c). Skip
   building reps when `j_max = 0`. **~48×** (20/76/298 s at n_max 2/3/4, scaling
   9.5×→3.8×/order). Combined #1+#2 vs the original pure-Python: **14194 s → 198 s
   (~72×)** at n_max = 4.
3. **Periodic `canon` while accumulating** big sums (EM, Z, Ψ, redef) — distinct
   dummies prevent merging until `canon`, so the term list swells RAM and the
   working set; fold every ~50–100 pending terms. Lowest priority, measured,
   after #2; helps the other builders too. **Not yet done.**

**Validation of #1+#2:** (a) exact A/B — cached/order-bounded result == direct
full-binomial reference, to canon, on both single-substitution
(`tests/_ab_deriv_cache.py`) and the **multi-substitution `j ≥ 2`** path
(`tests/_ab_multisub.py`; the regression suite alone barely drives `j_max > 1`,
so this bespoke check was needed); (b) 28 regression tests PASS covering every
redef-stress path (Belinfante, optional-EOM, conformal closure incl. order-3);
(c) `conformal_order3_lref_closure` went from a 40-min-cap timeout to a clean 2 h
pass. Profiling probes are gitignored scratch under `tests/_*` and `/_*`.

**Deploy:** `python-flint` + `gmpy2` recommended (`GROUND_TYPES=flint`, ~2× more);
`hpc_suite/submit_fast.sh` submits suffix-tagged parallel re-runs (defaults to
the redef-bound #5/#7) without clobbering still-running old jobs.

**Production validation (Zeus, old vs `_fast`; same `E^(n)` term counts, i.e.
identical results):**
- Run 5 (scalar optional-EOM Belinfante), order 1: **79,159 s (22.0 h) → 1,989 s
  (33 min), ≈40×**; old never reached order 2, `_fast` closes it in 41 min.
- Run 7 (conformal-scalar injection), order 2: **10,542 s (2.9 h) → 2,140 s
  (36 min), ≈4.9×**.
- These end-to-end factors fold in the flint backend (~2×) on top of #1/#2, since
  the old runs predate the flint install — so "#1/#2 alone" is roughly half each
  factor. The spread (40× vs 4.9×) tracks how *redef-dominated* the order was:
  run 5 order 1 is almost pure h-redef binomial; run 7 order 2 spends most time
  in Ψ/Δ/EM, which #1/#2 don't touch (that residual is the chunking target below).

**More production points (old vs `_fast`, same `E⁽ⁿ⁾` counts):**
- Run 7 order 3: **175,917 s (48.9 h) → 8,867 s (2.5 h) ≈ 20×** — the redef step that
  pinned old run 7 for ~2 days. Fast 5 and 7 both reached order 4 (new closures).
- **Run 6 speedup DECAYS with order: o1 11.5× / o2 6.8× / o3 1.6× / o4 1.1×.** By
  order 4 the fast code is barely faster — the redef win (#1/#2) **saturates** and
  the time moves into the `E_1`/Δ builders. This is the empirical case for the
  next optimization.

### Chunkwise-linear construction — DONE + DEPLOYED (committed/pushed; runs 1/4/5/9 `_chunk` live on zeus_long_q)

**Why chunking, not folding (#3).** The big `canon` inputs are `jet_derivative`
outputs carrying 2 free indices (μν); `canon` can't rename *free* indices, so the
terms can't merge (collapse ×1.00 — measured via the env-gated `GRB_CANON_PROFILE`
in `tensor_algebra.canon`). #3-folding has nothing to fold there. **But** these
builders are **linear** (`jet_derivative`, `total_derivative`, the EL operator, the
Hilbert/Belinfante EM, `compute_h2_violation`, `superpotential_divergence`), so
`F(L) = Σ_j F(chunk_j)`: run the whole inflate-then-shrink op per small input chunk
→ peak intermediate bounded to ~(chunk/|input|)×inflation, cross-chunk combinations
still happen in the final `canon`. (#3's `CanonAccumulator` is still the right tool
for the *redef substitution*, which DOES collapse heavily — both kept, different
regimes.)

**`jet.apply_linear_chunked(F, expr, chunk_size, fold_every)`** — handles bare
builders (fixed indices) AND index-returning ones (`(tensor, fresh idx)` → reindex
each chunk to the first chunk's indices, return `(result, idx)`). A/B-validated
`chunked == whole` (diff=0) for `total_derivative`, `jet_derivative`,
`euler_lagrange`, `hilbert_EM` (`tests/_ab_chunked.py`).

**Wired into 6 sites** (`_LINEAR_CHUNK`, default 64, env `GRB_LINEAR_CHUNK`): the
3 EL recomputes (verify `E_r` @~1335, re-verify-after-redef `E_r_new`, step-6
self-consistency `E_check`), `E_1` (Hilbert+Belinfante EM), `Z`
(`compute_h2_violation`), `Δ` (`superpotential_divergence` over Psi). Psi stays
whole (small; its symmetry check needs the whole — residual is global, not
chunkable). Rollout rule: chunk only inflate-then-shrink ops; keep **Z, Ψ, E_diff**
whole at their *global* step (decompose / symmetry / closure `==0`).

**Rejected: incremental E_diff re-verify** (`E_diff_new = E_diff − EL(Δ)`). It's
CIRCULAR — the full `EL(L_ref_new)` recompute IS the independent verification that
`E_h` is right; chunking keeps that recompute intact (identical math, bounded peak).

**Dropped the redundant post-step-3 Z re-check** (`_RECHECK_H2_AFTER_CORRECTION =
False`) — guaranteed by decomposition `residual==0` + the `EL(L^(n+1))==E^(n)`
check. Replaced by a **cheap always-on step-3 check**: `_verify_X_reproduces_Y` —
`compute_h2_violation(X) == −Y` (the antisym h-derivative of X reproduces −Y; the
(n+1) h-powers cancel the `1/(2(n+1))`, the antisym-2 cancels the 1/2). Runs on the
small X/Y, validated on Belinfante (sign + 4-index spectator handling confirmed).

**Correctness:** conformal(n2) + Belinfante(n3) close with FULL chunking forced on
(`GRB_LINEAR_CHUNK=8`). **Peak:** `euler_lagrange` L_EH⁵ whole vs chunked =
**70.3 → 44.7 MB (~36% lower)**, same 346-term result; time is *higher* at this
small scale (2047 vs 1469 s — pure overhead, no pressure to relieve at 70 MB).

**VERDICT (from the deployed `_chunk` runs): always-on chunking/folding REGRESSED
everywhere.** `05_chunk` o1 = 21402s vs `05_fast` o1 = 1989s (flint-controlled) =
**~10×** (tagged redef, #3 folding); `04_chunk` o3 = 9993s vs flint-expected ~2255s
= **~4.4×** (Proca, chunking the builds); run 9 similar. **Root cause (KEY INSIGHT,
user):** `canon` cost is dominated by per-term Butler-Portugal (index count + dummy
permutations), NOT term count — so folding/chunking ADD canon calls = pure time
overhead, and it AMPLIFIES with order (per-canon BP cost grows with index count).
Confirmed: `tests/_repro_redef_fold.py` shows folding re-canons a non-shrinking
total (n_max=3 +27%; Zeus n_max=6 ~10×). The original stalls were CPU/walltime-bound
(redef — fixed by #1/#2 — + inherent BP cost), NOT memory-bound, so chunking solved
a problem the runs didn't have. The ~36% peak win is real but only matters AT a RAM
barrier, which these runs never hit.

**FIX: memory-gate fold+chunk (`jet._mem_pressure`).** Fold (CanonAccumulator) and
chunk (`apply_linear_chunked`) ONLY when process RSS exceeds a budget
(`GRB_MEM_BUDGET_GB` if set — PBS script should export ~0.7×requested mem — else
`GRB_MEM_BUDGET_FRAC`×total RAM, default 0.7; `/proc` fallback so it works on Zeus
without psutil; no gate ⇒ never fold/chunk if RAM unmeasurable). So on a roomy node
/ low order it does ZERO extra canons (== fast code: #1/#2 + Z-drop + X→Y), and only
near the RAM wall does it trade canons to bound the intermediate and avoid swap.
Validated: no-pressure repro now `folds=1` (no regression); forced-pressure (tiny
budget) conformal still closes. `_REDEF_FOLD_EVERY=64` is now just the under-pressure
fold granularity. **TODO: recommit + redeploy (restart `_chunk` runs); the currently
deployed `_chunk` runs carry the always-on regression and should be restarted.**

### combine_canonical — skip redundant Butler-Portugal at recombination sites (committed)

`canon`'s cost is per-term Butler-Portugal (BP), not term count (see the chunking
verdict above). At recombination sites where every summand is ALREADY canonical
(produced by a prior `canon`), re-running `canon` re-pays BP on terms that cannot
change. `tensor_algebra.combine_canonical(expr)` is "canon minus BP": `.expand()`
(distribute) + `TensAdd.doit()` (collect like terms) + the gated `_simplify_d_coeffs`,
with NO per-term `canon_bp` and NO metric/delta contraction. **SAFETY CONTRACT:** only
valid where a static provenance audit proves every incoming term is canonical — a
non-canonical term won't collect, so a supposed-zero fails its `==0` gate (a LOUD
failure, not a silent wrong answer). Wired at the recombination sites (step 1/2/3/5,
field-redef substitution, `remove_second_derivatives`, `superpotential_divergence`,
the `CanonAccumulator` fold). Validated STRUCTURALLY (`combine == canon` as
expressions, not `canon(combine−canon)==0`, which hides form differences). Net: a
modest reliable win — combine is not the dominant cost (BP inside the builders is).

**deep= settled (2026-06-21).** combine's `TensAdd.doit()` runs at `deep=_COMBINE_DEEP`
(env `GRB_COMBINE_DEEP`, default **deep=False**). On post-BP terms the per-arg `.doit()`
(re-contraction) is redundant; deep=False skips it. An earlier reading — "deep=False
regressed Proca ~2.8x" — triggered a revert (35c07c8) but was a CONFOUND: that build
also carried the CanonAccumulator fold rework AND it was a cross-node comparison. A
controlled SAME-MACHINE A/B (pure gravity order 3, 3 alternating rounds) found
deep=True 454.0s vs deep=False 452.4s — statistically identical (0.35%, within noise).
deep=True and deep=False produce byte-identical output (verified including raw / mixed
/ cross-dummy inputs; `tests/_deep_probe.py`). Default deep=False (less redundant work,
matches `canon_bp`'s own deep=False); `GRB_COMBINE_DEEP=1` restores deep=True to
re-confirm on Zeus. **LESSON (recurred several times):** Zeus walltime varies ~2-3x by
node; judge a perf change only via a same-machine alternating A/B, never cross-node.

### Builder-chunk parallelism — wired, pending Zeus end-to-end (branch `parallelize-canon`)

Parallelize whole LINEAR builders over input chunks (NOT per-canon — that was shelved
at a 1.27x Amdahl ceiling). F linear ⇒ `Σ_j F(chunk_j) == F(expr)`; fork + copy-on-write
workers (input never pickled — inherited via `_PARALLEL_G`; only the small per-chunk
result is pickled back), disjoint fresh-index ranges per worker (`set_index_counter` +
`_PARALLEL_STRIDE`), `combine_canonical` merge (per-chunk results are canonical).
- **`jet.parallel_apply_linear(F, expr, n_workers, chunk_size)`** — fork executor.
  Zeus-de-risked: **4.22x @ K=8**, `== whole` at every K (K=1 = 0.84x quantifies fork
  overhead). Fork = Linux/Zeus only; Windows falls back to `F(expr)`.
- **`jet.apply_linear(F, expr)`** — dispatcher, DROP-IN for `apply_linear_chunked` (same
  signature + return shape). Gated on `GRB_N_WORKERS` (default 1 ⇒ serial
  `apply_linear_chunked`, fully INERT) and `GRB_PARALLEL_MIN` (default 128 terms);
  chunk = ceil(len/K) (one per worker, the bench-validated shape). Wired at the 7
  linear-builder sites in `bootstrap_loop.py` (Hilbert/Belinfante EM, Z, Δ, the 3 EL
  recomputes).
- Validated: complex_scalar passes serial AND with `GRB_N_WORKERS=4` (Windows
  fallback); `tests/bench_parallel_builder.py` serial/parallel `== whole`.

**PENDING (the arbiter):** same-node end-to-end serial-vs-parallel A/B on Zeus via
`hpc_suite/submit_parallel_run.sh` — one PBS job runs both legs on ONE node (identical
hardware, avoiding the cross-node confound above): `RUN=4 NMAX=4 K=8 bash
hpc_suite/submit_parallel_run.sh`. Gives the real Amdahl-limited speedup.

## The 6-step bootstrap (paper §4, quick reference)

For each order n = 0, 1, 2, ...:

1. **E_1^{μν(n)}** = κ T̂[L^{(n)}] + δ_{1,n} W^{μν}. T̂ is Hilbert or symmetrized Belinfante.
2. **E_2^{μν(n)}** = E_1 + Σ_{m<n} ( X_h^{(m)} · E^{(n−m)} + Σ_i X_{φ_i}^{(m)} · E_{φ_i}^{(n−m)} ). Carries over EOM choices from earlier orders.
3. **Mandatory EOM (H2)**. Compute Z = 2(∂E_2/∂h)_{antisym} − ((∂E_2/∂dh)_{antisym})_,γ. Decompose Z = Y · E^{(0)}. Set X = −1/(2(n+1)) Y · h. Add X · E^{(0)} → E_3.
4. **Optional EOM (field redefinitions)**. Any X' with ∂X'/∂h symmetric in (αβ ↔ μν). Zero on the default path → E_4.
5. **Superpotential (H3)**. n ≥ 2: Ψ from (PsiForm). n = 1: integral formula. Add Δ = Ψ_{,ρσ}.
6. **Close the loop**. L^{(n+1)} = (1/(n+1)) E^{(n)} h + boundary terms. EL(L^{(n+1)}) must reproduce E^{(n)}.

For **pure gravity with Hilbert** and no optional EOM terms, steps 2, 3, and 4 are trivial (Butcher's claim — verified empirically at n=2, n=3, and n=4 by the existence of the closure; item 4 of the open-work list would confirm it directly by checking Z=0). The only non-trivial work at each order is step 1 (T_H), step 5 (Ψ via PsiForm), and step 6 (closing the loop). This is precisely what the `test_superpotential_pure_gravity*.py` tests exercise.
