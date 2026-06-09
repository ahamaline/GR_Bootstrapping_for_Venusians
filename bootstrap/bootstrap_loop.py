"""
The bootstrap loop: the 6-step procedure from Section 4 of the paper.

Given an initial matter Lagrangian L_M (or None for pure gravity) and a
choice of energy-momentum procedure (Hilbert or Belinfante), this module
iteratively constructs the gravitational field equation order by order in
h_{μν}. Each call to `run_order(n)` advances the state by one order.

Steps per the paper:
    1. E_1 = κ T̂[L^{(n)}] + δ_{1,n} W^{μν}
    2. E_2 = E_1 + Σ_{m<n} X^{(m)} · E^{(n−m)}                    (carryover)
    3. E_3 = E_2 + X^{(n)} · E^{(0)}                                (mandatory EOM, from H2)
    4. E_4 = E_3 + X'^{(n)} · E^{(0)}                               (optional, field-redef)
    5. E   = E_4 + Δ^{(n)},     Δ^{(n)} = Ψ_{,ρσ}                 (superpotential, from H3)
    6. L^{(n+1)} = (1/(n+1)) E^{(n)} h + b.t.                       (close the loop)

If `n_max` is supplied at construction, a reference Lagrangian
L_ref^{(k)} (the standard covariant Einstein-Hilbert expansion + matter)
is pre-computed for all k=0..n_max+1, and after each `run_order(n)` the
bootstrap-derived L^{(n+1)} is checked against L_ref^{(n+1)} — see
`_verify_vs_L_ref` for the full flow.
"""

import itertools

from sympy import S, Rational, Symbol
from sympy.tensor.tensor import TensAdd, TensMul, TensExpr, Tensor

from bootstrap.tensor_algebra import (
    h, dh, ddh, fresh_indices, canon, metric, _matter_fields,
    get_tensors_in_expr, filter_by_order, order_in_h,
)
from bootstrap.euler_lagrange import euler_lagrange, euler_lagrange_scalar, remove_second_derivatives
from bootstrap.jet import (
    total_derivative, jet_derivative, _decompose_tensmul, _get_component,
    _get_indices, _sum_terms,
)
from bootstrap.eom_decompose import decompose_against_eoms
from bootstrap.helmholtz import (
    compute_superpotential_n2, compute_superpotential_n1,
    superpotential_divergence, verify_psi_symmetries,
    compute_h2_violation,
)
from bootstrap.energy_momentum import (
    hilbert_energy_momentum, symmetrized_belinfante,
)
from bootstrap.covariant import einstein_hilbert_lagrangian_order, matter_lagrangian_order
from bootstrap.loop_helpers import (
    _count, _term_breakdown, _format_breakdown, _reindex_tensor,
    _is_derivative_free, _derivative_heads,
)
from bootstrap.traceless import TracelessRecoveryMixin


kappa = Symbol('kappa')










def _total_derivative_at(expr, target_deriv_idx):
    """Compute ∂_{target_deriv_idx} of expr, returning a result with
    `target_deriv_idx` as the deriv free index.

    Why this wrapper? `total_derivative(expr, deriv_idx)` builds a Tensor
    with the same indices as the differentiated factor plus deriv_idx
    appended. If deriv_idx's NAME collides with any of those indices
    (which happens whenever expr was canonicalized — its dummies become
    L_0, L_1, ... and the parent term's deriv_idx may also be named L_0
    or L_1), sympy raises "two equal contravariant indices".

    Workaround: compute the derivative against a FRESH deriv index, then
    contract with a metric to relabel it onto `target_deriv_idx`. The
    metric contraction is a no-op semantically and canon collapses it,
    but it sidesteps the name-collision pitfall.
    """
    if expr == S.Zero:
        return S.Zero
    fresh_d = fresh_indices(1)[0]  # fresh UP index
    # Differentiate against the down version of fresh_d.
    deriv = total_derivative(expr, -fresh_d)
    if isinstance(deriv, TensExpr):
        deriv = canon(deriv)
    if deriv == S.Zero:
        return S.Zero
    # Relabel fresh_d's down free index onto target_deriv_idx.
    # If target_deriv_idx is DOWN, we want metric(fresh_d, target_idx)
    # = δ^{fresh_d}_{target_idx} after canon contracts -fresh_d.
    # If target_deriv_idx is UP, metric(fresh_d, target_idx) is η^{fresh_d target_idx}.
    relabel = deriv * metric(fresh_d, target_deriv_idx)
    if isinstance(relabel, TensExpr):
        relabel = canon(relabel)
    return relabel


def _build_deriv_cache(field_info, f_expr, f_indices):
    """Precompute the derivative templates df = ∂f and ddf = ∂∂f ONCE for a
    given redef (optimization #1).

    `_substitute_field` substitutes field → f and propagates to dfield/ddfield
    factors via the chain rule, so every (d/dd)field OCCURRENCE needs ∂f / ∂∂f
    re-indexed onto that factor's indices. The old code re-ran the (expensive)
    `_total_derivative_at` on every occurrence; here we differentiate f just
    once against fixed TEMPLATE indices and let `_replacement_for` *re-index*
    the result per occurrence (a cheap `substitute_indices` + canon) instead.

    Differentiation and free-index relabeling commute (the total derivative is
    index-equivariant), so reindexing the template is identical to
    differentiating the reindexed f, up to canon — verified by the A/B check in
    `_ab_check_deriv_cache.py`.

    Returns a dict with the template free indices and the canonicalized df/ddf,
    or None when f carries no order in h beyond what makes derivatives matter
    (callers treat None as "fall back to direct differentiation").
    """
    tmpl_fields = tuple(fresh_indices(1)[0] for _ in f_indices)
    tmpl_d1 = fresh_indices(1)[0]
    tmpl_d2 = fresh_indices(1)[0]
    f_tmpl = _reindex_tensor(f_expr, f_indices, tmpl_fields) if f_indices else f_expr
    df_tmpl = _total_derivative_at(f_tmpl, tmpl_d1)
    ddf_tmpl = _total_derivative_at(df_tmpl, tmpl_d2) if df_tmpl != S.Zero else S.Zero
    return {
        'tmpl_fields': tmpl_fields,
        'tmpl_d1': tmpl_d1,
        'tmpl_d2': tmpl_d2,
        'df': df_tmpl,
        'ddf': ddf_tmpl,
    }


def _substitute_field(L, field_info, f_expr, f_indices, target_order,
                      deriv_cache=None):
    """Substitute field (head, dhead, ddhead) -> field + f everywhere in L.
    Propagate via chain rule (dfield gets df = total_derivative(f);
    ddfield gets ddf). Re-expand and keep terms with order_in_h <= target_order.

    Works for both matter fields (via _matter_fields entries) and the
    graviton h (caller builds field_info = {field: h, dfield: dh, ddfield: ddh, rank: 2}).

    Args:
        L: the Lagrangian expression to substitute into.
        field_info: dict {field, dfield, ddfield, rank, ...} (for matter,
            entries from _matter_fields; for h, constructed by the caller).
        f_expr: the redef expression (derivative-free in fields, with some
            order in h, free indices = matter natural rank). Will be
            re-indexed per occurrence so it matches the substituted
            factor's actual indices.
        f_indices: tuple of free indices on f_expr (matter natural rank).
        target_order: keep only terms with order_in_h ≤ this value.
        deriv_cache: optional precomputed df/ddf templates from
            `_build_deriv_cache` (optimization #1). When None it is built once
            here, so every (d/dd)field occurrence re-indexes the cached template
            instead of re-differentiating f. Callers applying the SAME redef to
            several L's (e.g. `_apply_one_field_redef` over its k-loop) should
            build it once and pass it in to share the cache across calls.

    Returns: substituted L, expanded, canonicalized, and truncated.
    """
    head = field_info['field']
    dhead = field_info['dfield']
    ddhead = field_info['ddfield']
    rank = field_info.get('rank', 0)

    if L == S.Zero:
        return S.Zero

    if deriv_cache is None:
        deriv_cache = _build_deriv_cache(field_info, f_expr, f_indices)
    df_tmpl = deriv_cache['df']
    ddf_tmpl = deriv_cache['ddf']
    tmpl_fields = deriv_cache['tmpl_fields']
    tmpl_d1 = deriv_cache['tmpl_d1']
    tmpl_d2 = deriv_cache['tmpl_d2']

    def _replacement_for(fac):
        """Compute the replacement expression for a Tensor factor `fac`,
        re-indexed to match fac's actual indices. Uses fresh internal
        dummies (via metric-contraction trick for derivatives) so that
        sympy's auto-canonicalization of L_x dummies in TensMul doesn't
        collide with the parent term's dummies."""
        head_of_fac = _get_component(fac)
        if head_of_fac == head:
            # field → f reindexed.
            # Re-index f_expr's free indices onto fac's indices using a
            # fresh round-trip to dodge dummy clashes.
            fresh_targets = tuple(fresh_indices(1)[0] for _ in f_indices)
            # First substitute f's natural-position free indices onto fresh
            # targets (each fresh is up by default; match the sign of f's
            # original free index).
            f_with_fresh = _reindex_tensor(f_expr, f_indices, fresh_targets)
            # Then contract each fresh target onto the corresponding fac
            # index via metric (semantic no-op).
            fac_idxs = list(fac.get_indices())
            result = f_with_fresh
            for fresh_idx, fac_idx in zip(fresh_targets, fac_idxs):
                result = result * metric(-fresh_idx, fac_idx)
            if isinstance(result, TensExpr):
                result = canon(result)
            return result
        elif head_of_fac == dhead:
            # dfield(field..., deriv) → ∂_{deriv} f reindexed at field.
            # Re-index the CACHED df template (optimization #1) instead of
            # re-differentiating: map the template's free indices onto fresh
            # dummies, then metric-contract those onto fac's actual indices
            # (same fresh-dummy trick the direct path used, so no clash with
            # the parent term's L_x dummies).
            fac_idxs = list(fac.get_indices())
            field_idxs = fac_idxs[:rank]
            deriv_idx = fac_idxs[rank]
            if df_tmpl == S.Zero:
                return S.Zero
            fresh_fields = tuple(fresh_indices(1)[0] for _ in f_indices)
            fresh_d = fresh_indices(1)[0]
            df = _reindex_tensor(
                df_tmpl, tmpl_fields + (tmpl_d1,), fresh_fields + (fresh_d,))
            result = df
            for fresh_idx, fac_idx in zip(fresh_fields, field_idxs):
                result = result * metric(-fresh_idx, fac_idx)
            result = result * metric(-fresh_d, deriv_idx)
            if isinstance(result, TensExpr):
                result = canon(result)
            return result
        elif head_of_fac == ddhead:
            fac_idxs = list(fac.get_indices())
            field_idxs = fac_idxs[:rank]
            d1, d2 = fac_idxs[rank], fac_idxs[rank + 1]
            if ddf_tmpl == S.Zero:
                return S.Zero
            fresh_fields = tuple(fresh_indices(1)[0] for _ in f_indices)
            fresh_d1 = fresh_indices(1)[0]
            fresh_d2 = fresh_indices(1)[0]
            ddf = _reindex_tensor(
                ddf_tmpl,
                tmpl_fields + (tmpl_d1, tmpl_d2),
                fresh_fields + (fresh_d1, fresh_d2))
            result = ddf
            for fresh_idx, fac_idx in zip(fresh_fields, field_idxs):
                result = result * metric(-fresh_idx, fac_idx)
            result = result * metric(-fresh_d1, d1) * metric(-fresh_d2, d2)
            if isinstance(result, TensExpr):
                result = canon(result)
            return result
        return None

    # --- order-bounded substitution (optimization #2) -----------------------
    # Replacing one (d/dd)field factor by f raises the term's h-EXPANSION order
    # (order_in_h) by
    #     delta = order_in_h(f) - (1 if the field is h else 0)
    # since an h factor is itself order-1 while a matter factor is order-0, and
    # f's derivatives df/ddf keep f's h-order. With the term at order c and the
    # cutoff at target_order, at most
    #     j_max = floor((target_order - c) / delta)
    # factors may be replaced before the result exceeds the cutoff. We enumerate
    # only those subsets, replacing the old `prod_i (fac_i + rep_i)` binomial
    # (all 2^p terms canon'd, then truncated) by sum_{j<=j_max} C(p,j) terms --
    # exponential -> polynomial in p, the dominant cost in high-n_max redefs.
    #
    # Two distinct per-term counts, NOT to be conflated (see matter case):
    #   c = order_in_h(term)          -- the h-EXPANSION order (the cutoff axis)
    #   p = len(replaceable_facs)     -- OCCURRENCES of the substituted field
    #                                    (for matter, its own multiplicity, e.g.
    #                                     2 for a phi^2 / dphi.dphi term -- this
    #                                     is generally != c)
    if isinstance(f_expr, TensAdd):
        f_order = min(order_in_h(t) for t in f_expr.args)
    else:
        f_order = order_in_h(f_expr)
    delta = f_order - (1 if head is h else 0)
    if delta < 1:
        # Truly-linear (order-preserving) redef: nothing self-truncates, so the
        # substitution count is unbounded. We forbid these; fall back to the
        # full expansion if one ever slips through.
        delta = None

    heads_set = {head, dhead, ddhead}
    result_terms = []
    for term in (L.args if isinstance(L, TensAdd) else [L]):
        if isinstance(term, TensMul):
            coeff, factors = _decompose_tensmul(term)
        elif isinstance(term, Tensor):
            coeff, factors = S.One, [term]
        else:
            continue
        c = order_in_h(term)
        # Cheap pass: identify the substitutable factors (head match) by
        # COUNTING per-term occurrences -- do NOT yet build the (expensive,
        # derivative-bearing) replacements.
        replaceable_facs = []
        fixed = coeff
        for fac in factors:
            if _get_component(fac) in heads_set:
                replaceable_facs.append(fac)
            else:
                fixed = fixed * fac
        p = len(replaceable_facs)
        if delta is None:
            j_max = p
        else:
            j_max = max(0, min((target_order - c) // delta, p))
        # j = 0 is always the original term, already canonical -- keep as-is.
        result_terms.append(term)
        if p == 0 or j_max == 0:
            continue
        # Only now (we will actually use them) build the replacements.
        reps = [_replacement_for(fac) for fac in replaceable_facs]
        for j in range(1, j_max + 1):
            for subset in itertools.combinations(range(p), j):
                ss = set(subset)
                product = fixed
                for i in range(p):
                    product = product * (reps[i] if i in ss else replaceable_facs[i])
                if isinstance(product, TensExpr):
                    product = product.expand()
                    product = canon(product)
                if product != S.Zero:
                    result_terms.append(product)

    if not result_terms:
        return S.Zero
    result = TensAdd(*result_terms) if len(result_terms) > 1 else result_terms[0]
    if isinstance(result, TensExpr):
        result = canon(result)

    # Safety truncation: with a homogeneous f the subset bound is exact, but a
    # non-homogeneous f (f_order = MIN over its terms) can still emit a few
    # over-target stragglers; drop them here.
    truncated_terms = []
    for term in (result.args if isinstance(result, TensAdd) else [result]):
        if order_in_h(term) <= target_order:
            truncated_terms.append(term)
    if not truncated_terms:
        return S.Zero
    truncated = TensAdd(*truncated_terms) if len(truncated_terms) > 1 else truncated_terms[0]
    if isinstance(truncated, TensExpr):
        truncated = canon(truncated)
    return truncated










class BootstrapState(TracelessRecoveryMixin):
    """Holds the state of the bootstrap computation and drives it forward.

    Public attributes:
        L: dict {n: L^{(n)}}                     -- the bootstrap Lagrangians
        E: dict {n: E^{μν(n)}}                   -- the field equation terms
        mu_E, nu_E: the canonical (reserved) free indices for E^{μν}
        n_max: highest order we plan to reach (drives L_ref pre-computation)
        L_ref: dict {n: L_ref^{(n)}}              -- the reference Lagrangians
        eom_terms_h: dict {n: X_h^{μν}_{ρσ}^{(n)}}     -- accumulated mandatory + optional gravity EOM coefficients
        eom_terms_matter: dict {n: {field_name: X_φ^{μν(n)}}}
    """

    def __init__(self, L_matter=None, em_procedure='hilbert', n_max=None,
                 verbose=True, nonminimal_coupling=None):
        if em_procedure not in ('hilbert', 'belinfante'):
            raise ValueError(f"em_procedure must be 'hilbert' or 'belinfante', got {em_procedure!r}")

        self.em_procedure = em_procedure
        self.verbose = verbose
        self.n_max = n_max

        # Optional nonminimal matter-curvature coupling C(fields)*Riemann
        # (e.g. xi*phi**2 * R for a conformally coupled scalar). Expanded into
        # L_ref via covariant_coupling_order, and its order-1 EL derivative
        # (the identically-conserved "improvement") is added to E^(0).
        self.nonminimal_coupling = nonminimal_coupling

        # Reserve the canonical free indices for E^{μν(n)} — used at every
        # order, so all field equations are in a comparable form without
        # having to substitute_indices every time we want to combine them.
        self.mu_E, self.nu_E = fresh_indices(2)

        # Pre-compute the wave operator W^{μν} once (paper eq. for W).
        # Equivalent to EL(L_EH^{(2)}); reindexed to our canonical (μ, ν).
        Lh2 = einstein_hilbert_lagrangian_order(2)
        W, (w_mu, w_nu) = euler_lagrange(Lh2, h)
        self.W = _reindex_tensor(W, (w_mu, w_nu), (self.mu_E, self.nu_E))

        # Lagrangians and field equations from the bootstrap, by order.
        self.L = {0: L_matter if L_matter is not None else S.Zero}
        self.E = {}

        # EOM coefficients accumulated across orders (steps 3 and 4).
        self.eom_terms_h = {}
        self.eom_terms_matter = {}

        # User-chosen optional EOM terms (Step 4 / voluntary path).
        # Populate via `add_optional_eom_term(n, field_name, X_expr)`.
        # Structure: {n: {field_name: X_expr}}.
        self.optional_eom_terms = {}

        # Cache: matter EOMs at each order. self.L[k] for k = 0..n is set
        # exactly once during run_order(k); EL(L[k], field) is the same
        # every time it's needed (step 2 carryover at multiple n's).
        # Structure: {(k, field_name): (EOM_expr, EOM_idx)} for rank>=1,
        #            {(k, field_name): EOM_expr} for scalar (no idx).
        self._matter_eom_cache = {}

        # Traceless-T_M loophole state (open-work item 0 in DEVELOPMENT_STATUS).
        # Set by `_check_T_M_traceless` at end of run_order(0). When the
        # zeroth-order h-EOM trace is decomposable as -Σ c_i E_i^(0), an
        # EOM combination X·(η·E_h + Σ c_i·E_i) contributes zero at the
        # current order and only shows up at the next order via E^(1)
        # pieces. Verification + mandatory-EOM steps need extra handling.
        self.traceless_T_M = False
        self.traceless_c_i = {}      # {field_name: c_i (tensor expression)}
        # X^{ab(n-1)} coefficients recovered by the verification-step traceless
        # path, keyed by the order n at which they surfaced. Exposed so tests
        # (and debugging) can check the recovered X against a known injection.
        self.recovered_traceless_X = {}

        # L_ref snapshots: rollback support for the traceless case.
        # Before each `_apply_field_redefs_to_L_ref` call, the current
        # L_ref dict is deep-copied here so a later verification step can
        # roll back and augment the previous-order redef. Keyed by the
        # order n at which the redef was applied.
        self.L_ref_history = {}

        # Recovered field redefinitions, keyed by order n+1 (the f's order
        # in h). Stored by `_verify_vs_L_ref` so the traceless-T_M rollback
        # path can augment the previous order's f's with a newly-recovered
        # missed contribution.
        self.recovered_redefs = {}

        self.max_order_run = -1

        # Reference Lagrangian L_ref^{(k)}, pre-computed if n_max given.
        self.L_ref = {}
        if n_max is not None:
            self._init_L_ref(n_max)

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------

    def run_order(self, n):
        """Execute the bootstrap at order n. Updates self.E[n] and self.L[n+1]."""
        if n > 0 and (n - 1) not in self.L:
            raise ValueError(f"Must run order {n-1} before order {n}")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  BOOTSTRAP ORDER n = {n}")
            print(f"{'='*60}")

        # Step 1
        E = self._step1_energy_momentum(n)
        if self.verbose:
            print(f"  Step 1 (E_1): {_format_breakdown(E)}")

        # Step 2 — EOM carryover from earlier orders
        E = self._step2_eom_carryover(E, n)
        if self.verbose:
            print(f"  Step 2 (E_2): {_format_breakdown(E)}")

        # Step 3 — mandatory EOM correction (H2)
        E = self._step3_mandatory_eom(E, n)
        if self.verbose:
            print(f"  Step 3 (E_3): {_format_breakdown(E)}")

        # Step 4 — optional EOM (field redefinition, voluntary). Picks up
        # any user-supplied X' coefficients registered via
        # `add_optional_eom_term`; each is validated for derivative-freeness
        # and Helmholtz integrability before being added.
        E = self._step4_optional_eom(E, n)
        if self.verbose:
            tag = ("" if n in self.optional_eom_terms
                   else " (no optional terms applied)")
            print(f"  Step 4 (E_4): {_format_breakdown(E)}{tag}")

        # Step 5 — superpotential correction (H3)
        E = self._step5_superpotential(E, n)
        if self.verbose:
            print(f"  Step 5 (E^({n})): {_format_breakdown(E)}")

        # Order-0 nonminimal-coupling improvement (added after step 5, per the
        # spec): the EL derivative of the order-1 coupling term, which is the
        # identically-conserved curvature-coupling contribution to T_M.
        if n == 0 and self.nonminimal_coupling is not None:
            imp = self._nonminimal_improvement()
            if imp != S.Zero:
                E = canon(E + imp) if E != S.Zero else imp
                if self.verbose:
                    print(f"    Nonminimal-coupling improvement added to E^(0): "
                          f"{_format_breakdown(imp)}")

        self.E[n] = E

        # Step 6 — close the loop
        L_next = self._step6_close_loop(E, n)
        self.L[n + 1] = L_next
        if self.verbose:
            print(f"  Step 6 (L^({n+1})): {_format_breakdown(L_next)}")

        # Self-consistency: EL(L^{(n+1)}) should reproduce E^{(n)} on the nose.
        self._verify_el(n)

        # External consistency: bootstrap result vs reference EH expansion.
        if self.n_max is not None:
            self._verify_vs_L_ref(n)

        # After run_order(0), check for the on-shell-traceless-T_M condition
        # (open-work item 0 in DEVELOPMENT_STATUS). The result enables extra
        # handling in step 3 + verification step at higher orders.
        if n == 0:
            self._check_T_M_traceless()

        # After run_order(1), pre-compute and cache the order-1 traceless
        # operators (S^(1) and the two numerical m coefficients) once — they are
        # fixed for all higher orders and reused by both recoveries. Doing it
        # here fails fast if a v-contamination (field-dependent m) ever appears.
        if n == 1 and getattr(self, 'traceless_T_M', False):
            self._ensure_traceless_operators()

        self.max_order_run = n
        return E

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _step1_energy_momentum(self, n):
        """E_1^{μν(n)} = κ T̂[L^{(n)}] + δ_{1,n} W^{μν}."""
        L_n = self.L.get(n, S.Zero)

        if L_n != S.Zero:
            em_name = ('Hilbert' if self.em_procedure == 'hilbert'
                       else 'symmetrized Belinfante')
            if self.verbose:
                print(f"    Computing {em_name} energy-momentum tensor "
                      f"from L^({n}) ({_format_breakdown(L_n)})")
            if self.em_procedure == 'hilbert':
                T_mn, T_idx = hilbert_energy_momentum(L_n)
            elif self.em_procedure == 'belinfante':
                T_mn, T_idx = symmetrized_belinfante(L_n)
            else:
                raise NotImplementedError(self.em_procedure)
            T_mn = _reindex_tensor(T_mn, T_idx, (self.mu_E, self.nu_E))
            E = kappa * T_mn
        else:
            E = S.Zero

        if n == 1:
            # Add the wave operator W^{μν} = EL(L_h^{(2)}, h).
            if self.verbose:
                print(f"    Adding wave operator W^(mu nu) ({_count(self.W)} terms)")
            E = E + self.W if E != S.Zero else self.W

        if isinstance(E, TensExpr):
            E = canon(E)
        return E

    def _step2_eom_carryover(self, E, n):
        """Add the order-n carryover of EOM terms chosen at earlier orders:

            E_2^{μν(n)} = E_1^{μν(n)}
                        + Σ_{m<n} ( X_h^{μν(m)}_{κλ} · E^{κλ(n−m)}
                                  + Σ_i X_{φ_i}^{μν(m) ...} · E_{φ_i}^{(n−m) ...} )

        X_h^(m) and X_{φ_i}^(m) were stored at step 3 of order m. The
        h-EOM at order (n−m) is just self.E[n−m]. The matter EOM at order
        (n−m) is the order-(n−m) piece of EL(L, φ); since EL by φ does
        not change h-count, that equals EL(self.L[n−m], φ).

        No-op for paths where step 3 never fired (pure-gravity Hilbert).
        """
        if not self.eom_terms_h and not self.eom_terms_matter:
            return E

        correction = S.Zero
        contribs = []  # list of (label, term_count) for verbose logging

        # --- h-EOM carryover --------------------------------------------------
        for m, X_h_m in self.eom_terms_h.items():
            if m >= n:
                continue
            target = n - m
            if target not in self.E:
                raise ValueError(
                    f"Step 2: need E^({target}) for X_h^({m}) carryover, "
                    f"but order {target} hasn't been run yet."
                )
            E_at = self.E[target]
            if E_at == S.Zero or X_h_m == S.Zero:
                continue
            # X_h_m's free indices: (μ_E, ν_E) plus the EOM-side pair (signs
            # depend on canon's metric work — read them off).
            X_free = X_h_m.get_free_indices() if isinstance(X_h_m, TensExpr) else []
            extra_h = [idx for idx in X_free
                       if idx != self.mu_E and idx != self.nu_E]
            if len(extra_h) != 2:
                raise RuntimeError(
                    f"X_h^({m}) has unexpected free-index structure {X_free}; "
                    f"expected (μ_E, ν_E, κ, λ)."
                )
            E_renamed = _reindex_tensor(E_at, (self.mu_E, self.nu_E),
                                        (-extra_h[0], -extra_h[1]))
            term = X_h_m * E_renamed
            if isinstance(term, TensExpr):
                term = canon(term)
            contribs.append((f"X_h^({m}) . E^({target})", _count(term)))
            correction = correction + term if correction != S.Zero else term

        # --- per-matter-field EOM carryover ----------------------------------
        for m, X_phi_dict in self.eom_terms_matter.items():
            if m >= n:
                continue
            target = n - m
            L_at = self.L.get(target, S.Zero)
            if L_at == S.Zero:
                continue
            for name, X_phi_m in X_phi_dict.items():
                if X_phi_m == S.Zero:
                    continue
                info = _matter_fields[name]
                rank = info.get('rank', 0)
                field = info['field']
                # Cache: EL(self.L[target], field) is computed once per
                # (target, name) and reused across orders.
                cache_key = (target, name)
                if rank == 0:
                    if cache_key not in self._matter_eom_cache:
                        self._matter_eom_cache[cache_key] = (
                            euler_lagrange_scalar(L_at, field)
                        )
                    E_phi_at = self._matter_eom_cache[cache_key]
                    if E_phi_at == S.Zero or E_phi_at == 0:
                        continue
                    term = X_phi_m * E_phi_at
                else:
                    if cache_key not in self._matter_eom_cache:
                        self._matter_eom_cache[cache_key] = (
                            euler_lagrange(L_at, field)
                        )
                    E_phi_at, E_phi_idx = self._matter_eom_cache[cache_key]
                    if E_phi_at == S.Zero:
                        continue
                    X_free = X_phi_m.get_free_indices() if isinstance(X_phi_m, TensExpr) else []
                    extra_phi = [idx for idx in X_free
                                 if idx != self.mu_E and idx != self.nu_E]
                    if len(extra_phi) != rank:
                        raise RuntimeError(
                            f"X_{name}^({m}) has {len(extra_phi)} extra free "
                            f"indices {extra_phi}; expected {rank}."
                        )
                    E_phi_renamed = _reindex_tensor(
                        E_phi_at, E_phi_idx,
                        tuple(-idx for idx in extra_phi),
                    )
                    term = X_phi_m * E_phi_renamed
                if isinstance(term, TensExpr):
                    term = canon(term)
                contribs.append((f"X_{name}^({m}) . E_{name}^({target})", _count(term)))
                correction = correction + term if correction != S.Zero else term

        if correction == S.Zero:
            if self.verbose:
                print(f"    Step 2 carryover: no contribution (all X.E vanish)")
            return E
        if isinstance(correction, TensExpr):
            correction = canon(correction)
        if self.verbose:
            for label, count in contribs:
                print(f"    Step 2 carryover: {label}  ({count} terms)")
            print(f"    Step 2 carryover total: {_format_breakdown(correction)}")
        E_new = E + correction
        if isinstance(E_new, TensExpr):
            E_new = canon(E_new)
        return E_new

    def _step3_mandatory_eom(self, E, n):
        """Add the mandatory EOM correction required by H2 (Helmholtz #2).

        Compute Z^{μν αβ} (paper eq. for Z). If Z = 0, no correction is
        needed. If Z ≠ 0, decompose Z = Y_h · E^{(0)} + Σ_i Y_{φ_i} · E_{φ_i}^{(0)}
        via `decompose_against_eoms`, then form

            X_h^{μν}_{κλ}   = -1/(2(n+1)) · Y_h^{μν αβ}_{κλ} · h_{αβ}
            X_{φ_i}^{μν ..} = -1/(2(n+1)) · Y_{φ_i}^{μν αβ ..} · h_{αβ}

        and add  X_h · E^{(0)} + Σ_i X_{φ_i} · E_{φ_i}^{(0)}  to E.

        For the Hilbert procedure with no optional EOM terms, Butcher's
        claim is Z = 0 at every order — confirmed empirically. Z ≠ 0 is
        expected on the Belinfante path (paper §2) or once optional EOM
        terms are introduced.
        """
        if n == 0:
            return E
        if E == S.Zero:
            if self.verbose:
                print(f"    H2 check skipped (E is zero)")
            return E
        Z, h_indices = compute_h2_violation(E, (self.mu_E, self.nu_E))
        if Z == S.Zero:
            if self.verbose:
                print(f"    H2 check: Z = 0 (OK)")
            return E
        if self.verbose:
            print(f"    H2 violation Z: {_format_breakdown(Z)} - decomposing")

        # Traceless mandatory-step path (DEVELOPMENT_STATUS item 0): when T_M
        # is on-shell traceless, Z may carry ddh that is NOT a Y · E^(0)
        # combination (E^(0) has no ddh). Absorb it via a new order-(n-1)
        # traceless-shape X whose H2 cancels the ddh, leaving a ddh-free Z for
        # the normal decomposition. Strictly gated on traceless_T_M so the
        # default path is unchanged.
        if self.traceless_T_M and ddh in set(get_tensors_in_expr(Z)):
            recovered = self._recover_traceless_mandatory_eom(n, E, Z, h_indices)
            if recovered is not None:
                E, Z = recovered
                if Z == S.Zero:
                    if self.verbose:
                        print(f"    H2 after traceless recovery: Z = 0 (OK)")
                    return E

        L_M = self.L[0]  # zeroth-order Lagrangian for matter EOMs
        result = decompose_against_eoms(
            Z, L_M, em_procedure=self.em_procedure, verbose=self.verbose,
            E0=self.E[0], E0_indices=(self.mu_E, self.nu_E),
        )
        Y_h = result['Y_h']
        alphas_h_orch = result['alphas_h']  # (κ, λ) on Y_h's EOM side
        Y_phi_dict = result['X_phi']        # {name: (Y_φ, alphas_φ)}
        residual = result['residual']
        if residual != S.Zero:
            n_r = _count(residual)
            raise NotImplementedError(
                f"H2 decomposition incomplete at order n={n}: residual has "
                f"{n_r} terms. decompose_against_eoms did not fully decompose "
                f"Z. Either Z is not a clean Y · E^(0) combination, or the "
                f"orchestrator is missing a signature path (e.g. mass-only "
                f"matter EOM lacks a 2nd-derivative signature)."
            )

        coeff = Rational(-1, 2 * (n + 1))
        correction = S.Zero
        alpha_h, beta_h = h_indices

        # --- h-EOM correction ---
        if Y_h != S.Zero and alphas_h_orch[0] is not None:
            # X_h^{μν}_{κλ} = coeff · Y_h^{μν αβ}_{κλ} · h_{αβ}.
            X_h = coeff * Y_h * h(-alpha_h, -beta_h)
            if isinstance(X_h, TensExpr):
                X_h = canon(X_h)
            # Apply to E^(0): contract X_h's EOM-side indices with E^(0)'s.
            X_h_free = X_h.get_free_indices()
            extra_h = [idx for idx in X_h_free
                       if idx != self.mu_E and idx != self.nu_E]
            if len(extra_h) != 2:
                raise RuntimeError(
                    f"X_h has unexpected free-index structure {X_h_free}; "
                    f"expected (μ_E, ν_E, κ, λ)."
                )
            if self.verbose:
                print(f"    Y_h: {_format_breakdown(Y_h)};  "
                      f"X_h: {_format_breakdown(X_h)}")
                print(f"      X_h = {X_h}")
                print(f"    EOM term added: X_h . E_h^(0)")
            E0 = self.E[0]
            E0_renamed = _reindex_tensor(E0, (self.mu_E, self.nu_E),
                                         (-extra_h[0], -extra_h[1]))
            term_h = X_h * E0_renamed
            if isinstance(term_h, TensExpr):
                term_h = canon(term_h)
            correction = correction + term_h if correction != S.Zero else term_h
            # Remember X_h for step 2 (carryover) at higher orders.
            self.eom_terms_h[n] = X_h

        # --- per-matter-field EOM corrections ---
        for name, (Y_phi, _) in Y_phi_dict.items():
            info = _matter_fields[name]
            rank = info.get('rank', 0)
            field = info['field']
            X_phi = coeff * Y_phi * h(-alpha_h, -beta_h)
            if isinstance(X_phi, TensExpr):
                X_phi = canon(X_phi)
            if self.verbose:
                print(f"    Y_{name}: {_format_breakdown(Y_phi)};  "
                      f"X_{name}: {_format_breakdown(X_phi)}")
                print(f"      X_{name} = {X_phi}")
                print(f"    EOM term added: X_{name} . E_{name}^(0)")
            if rank == 0:
                EOM_phi = euler_lagrange_scalar(L_M, field)
                term_phi = X_phi * EOM_phi
            else:
                EOM_phi, EOM_idx = euler_lagrange(L_M, field)
                # Read X_phi's actual EOM-side free indices (signs depend on
                # canon), rename EOM into their negatives for direct contraction.
                X_phi_free = X_phi.get_free_indices()
                extra_phi = [idx for idx in X_phi_free
                             if idx != self.mu_E and idx != self.nu_E]
                if len(extra_phi) != rank:
                    raise RuntimeError(
                        f"X_{name} has {len(extra_phi)} extra free indices "
                        f"{extra_phi}; expected {rank}."
                    )
                EOM_renamed = _reindex_tensor(EOM_phi, EOM_idx,
                                              tuple(-idx for idx in extra_phi))
                term_phi = X_phi * EOM_renamed
            if isinstance(term_phi, TensExpr):
                term_phi = canon(term_phi)
            correction = correction + term_phi if correction != S.Zero else term_phi
            self.eom_terms_matter.setdefault(n, {})[name] = X_phi

        E_new = E + correction
        if isinstance(E_new, TensExpr):
            E_new = canon(E_new)

        # Sanity: the new Z (after correction) should be zero.
        if self.verbose:
            print(f"    E + correction: {_format_breakdown(E_new)} - re-checking H2")
        Z_after, _ = compute_h2_violation(E_new, (self.mu_E, self.nu_E))
        if Z_after != S.Zero:
            n_z = _count(Z_after)
            raise RuntimeError(
                f"H2 correction at order n={n} failed: re-computed Z has "
                f"{n_z} terms after applying X · E^(0). Check the sign/contraction "
                f"convention in step 3 or the decomposition output."
            )
        if self.verbose:
            print(f"    H2 check after correction: Z = 0 (OK)")
        return E_new

    def _merge_eom_coeff(self, prior, X_new):
        """Sum two EOM coefficients that share the (mu_E, nu_E) field-equation
        indices but whose EOM-side free indices canon may have named
        differently. A direct `prior + X_new` then trips sympy's "all tensors
        must have the same indices" check. We reindex BOTH onto a common set
        of FRESH EOM-side indices (matched by sign, positionally within each
        sign class), which both makes the TensAdd legal and dodges any clash
        with existing dummy names.

        Used by step 4 when an optional EOM term lands at an order where step 3
        already recorded a mandatory coefficient for the same field. Only the
        stored carryover coefficient is affected — the current order's E
        contribution is built from X_new directly, before this merge.
        """
        if prior == S.Zero or prior == 0:
            return X_new
        if X_new == S.Zero or X_new == 0:
            return prior
        reserved = (self.mu_E, self.nu_E)
        p_extra = [i for i in prior.get_free_indices() if i not in reserved]
        x_extra = [i for i in X_new.get_free_indices() if i not in reserved]
        if len(p_extra) != len(x_extra):
            raise RuntimeError(
                f"_merge_eom_coeff: EOM-side index-count mismatch "
                f"(prior {p_extra} vs new {x_extra}); cannot merge.")
        # Pair same-sign indices positionally within each sign class.
        matched = []
        for up in (True, False):
            ps = [i for i in p_extra if i.is_up == up]
            xs = [i for i in x_extra if i.is_up == up]
            if len(ps) != len(xs):
                raise RuntimeError(
                    f"_merge_eom_coeff: EOM-side sign mismatch "
                    f"(prior {p_extra} vs new {x_extra}); cannot merge.")
            matched.extend(zip(xs, ps))
        fresh = fresh_indices(len(matched))
        x_old, x_tgt, p_old, p_tgt = [], [], [], []
        for (xi, pi), fr in zip(matched, fresh):
            signed = fr if xi.is_up else -fr  # xi and pi share sign by pairing
            x_old.append(xi); x_tgt.append(signed)
            p_old.append(pi); p_tgt.append(signed)
        X_a = _reindex_tensor(X_new, tuple(x_old), tuple(x_tgt))
        P_a = _reindex_tensor(prior, tuple(p_old), tuple(p_tgt))
        merged = P_a + X_a
        return canon(merged) if isinstance(merged, TensExpr) else merged

    def _step4_optional_eom(self, E, n):
        """Add user-chosen optional EOM terms (paper's "voluntary path" /
        field-redefinition step). Each X term registered via
        `add_optional_eom_term(n, field_name, X)` is validated then added
        as X . E_{field}^(0).

        Validation:
          1. X must be derivative-free (function of fields only, no dphi).
          2. X must be Helmholtz-integrable: \\partial X^{ab}/\\partial h_{cd}
             symmetric in (ab <-> cd), equivalent to compute_h2_violation(X)
             == 0. This is the strict criterion that X corresponds to a
             genuine Lagrangian shift; we check it up front because the
             user might supply anything.

        Either check failing raises immediately. On success, the X is added
        to E and recorded in self.eom_terms_h / self.eom_terms_matter so
        that step 2 of higher orders carries it forward.
        """
        user_terms = self.optional_eom_terms.get(n, {})
        if not user_terms:
            return E

        correction = S.Zero
        for field_name, X_expr in user_terms.items():
            # Validate the order in h: every term of X must have order n
            # (since X^(n) . E^(0) must contribute at order n to E^(n)).
            # An off-by-order X breaks step 6's (1/(n+1)) E h closure.
            for term in (X_expr.args if isinstance(X_expr, TensAdd)
                         else [X_expr] if X_expr != S.Zero else []):
                term_order = order_in_h(term)
                if term_order != n:
                    raise ValueError(
                        f"Optional EOM X_{field_name} at order n={n} has a "
                        f"term of order {term_order} in h; every term must be "
                        f"order {n}. (X^(n) . E^(0) contributes at order n; "
                        f"step 6's closure breaks if X carries the wrong order.) "
                        f"Offending term: {term}"
                    )
            if not _is_derivative_free(X_expr):
                raise ValueError(
                    f"Optional EOM X_{field_name} at order n={n} contains "
                    f"field derivatives; not a valid field-redefinition "
                    f"coefficient."
                )
            if not self._check_X_integrable(X_expr):
                raise ValueError(
                    f"Optional EOM X_{field_name} at order n={n} fails the "
                    f"Helmholtz integrability check (partial X/partial h "
                    f"is not symmetric in (mu nu <-> alpha beta)); refusing "
                    f"to apply -- it does not correspond to a Lagrangian "
                    f"field redefinition."
                )

            if field_name == 'h':
                X_free = X_expr.get_free_indices() if isinstance(X_expr, TensExpr) else []
                extra = [idx for idx in X_free
                         if idx != self.mu_E and idx != self.nu_E]
                if len(extra) != 2:
                    raise ValueError(
                        f"Optional X_h at n={n} has free indices {X_free}; "
                        f"expected (self.mu_E, self.nu_E, kappa, lambda)."
                    )
                # h-EOM optional terms are only valid for n >= 1 (enforced
                # in add_optional_eom_term), so self.E[0] is always set.
                E0 = self.E[0]
                E0_renamed = _reindex_tensor(E0, (self.mu_E, self.nu_E),
                                             (-extra[0], -extra[1]))
                term = X_expr * E0_renamed
                # Record X for step 2 carryover; accumulate if step 3 also
                # produced an X_h at this order.
                prior = self.eom_terms_h.get(n, S.Zero)
                merged = self._merge_eom_coeff(prior, X_expr)
                if isinstance(merged, TensExpr):
                    merged = canon(merged)
                self.eom_terms_h[n] = merged
            else:
                if field_name not in _matter_fields:
                    raise ValueError(
                        f"Optional EOM specifies unknown matter field "
                        f"{field_name!r}; expected one of "
                        f"{list(_matter_fields.keys())} or 'h'."
                    )
                info = _matter_fields[field_name]
                rank = info.get('rank', 0)
                field = info['field']
                if rank == 0:
                    EOM_phi = euler_lagrange_scalar(self.L[0], field)
                    term = X_expr * EOM_phi
                else:
                    EOM_phi, EOM_idx = euler_lagrange(self.L[0], field)
                    X_free = X_expr.get_free_indices() if isinstance(X_expr, TensExpr) else []
                    extra = [idx for idx in X_free
                             if idx != self.mu_E and idx != self.nu_E]
                    if len(extra) != rank:
                        raise ValueError(
                            f"Optional X_{field_name} at n={n} has "
                            f"{len(extra)} extra free indices {extra}; "
                            f"expected {rank}."
                        )
                    EOM_renamed = _reindex_tensor(
                        EOM_phi, EOM_idx, tuple(-i for i in extra)
                    )
                    term = X_expr * EOM_renamed
                # Record for step 2 carryover (merge with step 3's matter X
                # if any at this order).
                phi_dict = self.eom_terms_matter.setdefault(n, {})
                prior = phi_dict.get(field_name, S.Zero)
                merged = self._merge_eom_coeff(prior, X_expr)
                if isinstance(merged, TensExpr):
                    merged = canon(merged)
                phi_dict[field_name] = merged

            if isinstance(term, TensExpr):
                term = canon(term)
            correction = correction + term if correction != S.Zero else term
            if self.verbose:
                print(f"    Step 4 optional EOM: X_{field_name} "
                      f"({_format_breakdown(X_expr)}) applied as "
                      f"X_{field_name} . E_{field_name}^(0) -> "
                      f"{_format_breakdown(term)}")
                print(f"      X_{field_name} = {X_expr}")

        if correction == S.Zero:
            return E
        if isinstance(correction, TensExpr):
            correction = canon(correction)
        E_new = E + correction
        if isinstance(E_new, TensExpr):
            E_new = canon(E_new)
        return E_new

    # ------------------------------------------------------------------
    # User-facing setters for optional EOM terms (Step 4 inputs)
    # ------------------------------------------------------------------

    def add_optional_eom_term(self, n, field_name, X_expr):
        """Register a user-chosen optional EOM coefficient X to be applied
        at order n in step 4 ("voluntary path" / field redefinition).

        Args:
            n: bootstrap order at which to apply. For matter fields, n >= 0
                is valid (n=0 corresponds to choosing a different matter
                field convention up front). For the graviton h, the rule
                is n >= 1: at n=0 the only h-related thing is E^(0) = kappa
                T_M itself; there's no separate X_h^(0) E_h^(0) piece to add
                (it would be self-referential).
            field_name: 'h' (graviton) or a matter-field name from
                `tensor_algebra._matter_fields`.
            X_expr: derivative-free tensor coefficient. Free indices must
                be (self.mu_E, self.nu_E) plus the EOM-direction pair
                (rank-2 for h, matter natural rank for matter fields).

        Validation (derivative-free + Helmholtz integrability) runs lazily
        at step 4 of order n inside `run_order`. On failure, that step
        raises with a clear message.
        """
        if field_name == 'h':
            if n < 1:
                raise ValueError(
                    f"Optional X_h terms apply at order n >= 1, got {n}. "
                    f"(At n=0 the only h piece in the field equation is "
                    f"E^(0) = kappa T_M, with no separate X_h E_h^(0) to add.)"
                )
        else:
            if n < 0:
                raise ValueError(
                    f"Optional EOM terms apply at order n >= 0, got {n}"
                )
        if field_name != 'h' and field_name not in _matter_fields:
            raise ValueError(
                f"field_name must be 'h' or a registered matter field; "
                f"got {field_name!r}. Known matter: {list(_matter_fields.keys())}."
            )
        self.optional_eom_terms.setdefault(n, {})[field_name] = X_expr



    def _nonminimal_improvement(self):
        """Order-0 contribution of the nonminimal coupling to the field
        equation: EL_h of the order-1 coupling term. This is the curvature-
        coupling 'improvement' to the stress tensor (e.g. for a conformally
        coupled scalar, ~ -2 xi (d_mu d_nu - eta_mu_nu box) phi^2). It is
        identically conserved (see `_check_improvement_conserved`). Returns it
        reindexed to (mu_E, nu_E), or S.Zero if there is no coupling.
        """
        if self.nonminimal_coupling is None:
            return S.Zero
        from bootstrap.covariant import covariant_coupling_order
        coupling1 = covariant_coupling_order(self.nonminimal_coupling, 1)
        if coupling1 == S.Zero:
            return S.Zero
        imp, imp_idx = euler_lagrange(coupling1, h)
        if imp == S.Zero:
            return S.Zero
        imp = _reindex_tensor(imp, imp_idx, (self.mu_E, self.nu_E))
        if isinstance(imp, TensExpr):
            imp = canon(imp)
        self._check_improvement_conserved(imp)
        return imp

    def _check_improvement_conserved(self, imp):
        """Assert the nonminimal-coupling improvement is identically conserved,
        ∂_μ imp^{μν} = 0. A valid curvature-coupling improvement is a
        superpotential (divergence of an identically-conserved object), so its
        own divergence must vanish off-shell; a nonzero result means the
        coupling expansion or its EL derivative is wrong.
        """
        if imp == S.Zero:
            return
        d, = fresh_indices(1)
        div = total_derivative(imp, -d) * metric(d, -self.mu_E)
        if isinstance(div, TensExpr):
            div = canon(div)
        if div != S.Zero:
            raise RuntimeError(
                f"Nonminimal-coupling improvement is NOT identically conserved: "
                f"∂_μ imp^{{μν}} has {_count(div)} terms (must be 0). The "
                f"coupling expansion or its EL derivative is inconsistent.")

    # ------------------------------------------------------------------
    # Traceless-T_M verification-step recovery (DEVELOPMENT_STATUS item 0)
    # ------------------------------------------------------------------






    def _check_X_integrable(self, X):
        """Helmholtz integrability check on a derivative-free X^{mu nu ...}:

            \\partial X^{mu nu ...}/\\partial h_{alpha beta}
              symmetric in (mu nu <-> alpha beta).

        Equivalent to compute_h2_violation(X, (mu_E, nu_E)) returning 0.
        For derivative-free X the dh-derivative piece of H2 vanishes
        automatically, so this reduces to the algebraic antisymmetric part.

        Extra "EOM-side" indices on X (the (kappa, lambda) on X_h, or the
        matter-rank indices on X_phi) are treated as spectators throughout.

        Returns True iff X is integrable.
        """
        if X == S.Zero:
            return True
        Z, _ = compute_h2_violation(X, (self.mu_E, self.nu_E))
        return Z == S.Zero

    def _step5_superpotential(self, E, n):
        """Add Δ^{(n)} = Ψ^{(n)}_{,ρσ}.

        Ψ comes from the paper's PsiForm formula at n≥2, or the integral
        formula (paper eq. 23) at n=1 with matter. For n=1 pure gravity,
        E_4 = W already satisfies H3 with Ψ=0.
        """
        if n == 0 or E == S.Zero:
            # n=0: any superpotential here is a physical modification, skip.
            return E

        if self.verbose:
            formula = "integral formula (n=1)" if n == 1 else f"PsiForm (n={n})"
            print(f"    Computing superpotential Psi^({n}) via {formula}")

        if n == 1:
            if self.L[0] == S.Zero:
                return E  # pure gravity at n=1: no superpotential
            Psi, psi_idx = compute_superpotential_n1(
                E, (self.mu_E, self.nu_E)
            )
        else:
            Psi, psi_idx = compute_superpotential_n2(
                E, n, (self.mu_E, self.nu_E)
            )

        if Psi == S.Zero or psi_idx is None:
            if self.verbose:
                print(f"    Psi^({n}) = 0 (no superpotential term)")
            return E

        # Sanity-check Ψ's three symmetries; the paper guarantees they hold
        # when the bootstrap has a valid continuation.
        sym = verify_psi_symmetries(Psi, psi_idx)
        ok = all(sym.get(k, True) for k in ('sym_mn', 'sym_rs', 'cyclic'))
        if not ok:
            raise RuntimeError(
                f"Psi^({n}) failed symmetry checks: {sym}. Either the "
                "input E is not a valid bootstrap field equation or the "
                "implementation is broken."
            )
        if self.verbose:
            print(f"    Psi^({n}) symmetries: OK; Psi: {_format_breakdown(Psi)}")

        Delta = superpotential_divergence(Psi, psi_idx)
        Delta = _reindex_tensor(Delta, (psi_idx[0], psi_idx[1]),
                                (self.mu_E, self.nu_E))

        if self.verbose:
            print(f"    Delta = Psi_(,rho sigma): {_format_breakdown(Delta)}")
            print(f"    Superpotential term added")
        return canon(E + Delta) if Delta != S.Zero else E

    def _step6_close_loop(self, E, n):
        """L^{(n+1)} = (1/(n+1)) · E^{μν(n)} · h_{μν} + boundary terms.

        Boundary terms are absorbed by `remove_second_derivatives` so the
        Lagrangian comes out in standard first-order form.
        """
        if E == S.Zero:
            return S.Zero
        L_raw = Rational(1, n + 1) * E * h(-self.mu_E, -self.nu_E)
        if isinstance(L_raw, TensExpr):
            # .expand() to distribute Rational * TensAdd * Tensor — without it,
            # sympy can leave L_raw as a TensMul wrapping a TensAdd, and
            # remove_second_derivatives (which uses _decompose_tensmul) would
            # silently skip the wrapped TensAdd's terms. Same dropped-TensAdd
            # footgun as in compute_superpotential_n1 (see
            # project-decompose-tensmul-tensadd-pitfall memory).
            L_raw = L_raw.expand()
            L_raw = canon(L_raw)
        return remove_second_derivatives(L_raw)

    # ------------------------------------------------------------------
    # Verifications
    # ------------------------------------------------------------------

    def _verify_el(self, n):
        """Self-consistency: EL(L^{(n+1)}) must reproduce E^{(n)}.

        This is automatic in theory (paper §3 identity), but it catches
        bugs in `remove_second_derivatives` (IBP), `euler_lagrange`, and
        `_step6_close_loop` working in concert.
        """
        L_next = self.L[n + 1]
        E_target = self.E[n]
        if L_next == S.Zero:
            if E_target != S.Zero:
                raise RuntimeError(
                    f"L^({n+1}) is zero but E^({n}) is not — Step 6 dropped content"
                )
            if self.verbose:
                print(f"  Verify EL(L^({n+1})) == E^({n}): both zero, OK")
            return True

        E_check, E_check_idx = euler_lagrange(L_next, h)
        E_check = _reindex_tensor(E_check, E_check_idx, (self.mu_E, self.nu_E))

        diff = canon(E_check - E_target)
        if diff != S.Zero:
            n_diff = _count(diff)
            raise RuntimeError(
                f"EL(L^({n+1})) does not equal E^({n}): residual has {n_diff} terms"
            )
        if self.verbose:
            print(f"  Verify EL(L^({n+1})) == E^({n}): OK")
        return True

    # ------------------------------------------------------------------
    # Reference Lagrangian (verification cycle)
    # ------------------------------------------------------------------

    def _init_L_ref(self, n_max):
        """Pre-compute L_ref^{(k)} for k=0..n_max+1.

        L_ref is the "true" raw Lagrangian expansion — NOT IBP'd.
        For pure gravity: L_ref^{(k)} = L_EH^{(k)}. With matter, the
        matter Lagrangian's covariantized expansion (√|g| L̃_M)^{(k)} is
        added to each order; L_ref^{(0)} = L_M and L_ref^{(1)} is now
        non-zero (= κ h_{μν} T_M^{μν}) when matter is present.

        The bootstrap's own L^{(n+1)} comes out of `_step6_close_loop`
        with IBP applied (per the paper's recipe), so it's always going
        to differ from L_ref by boundary terms; the verification cycle
        uses the EL-equivalence check (case b) which sees through that.
        """
        L_M = self.L[0]
        has_matter = (L_M != S.Zero)

        if self.verbose:
            kind = "with matter" if has_matter else "pure gravity"
            print(f"  Pre-computing L_ref^{{(0..{n_max+1})}} ({kind})...")

        self.L_ref[0] = L_M
        if has_matter:
            self.L_ref[1] = matter_lagrangian_order(L_M, 1)
            if self.verbose:
                print(f"    L_ref^(1): {_format_breakdown(self.L_ref[1])}")
        else:
            self.L_ref[1] = S.Zero

        for k in range(2, n_max + 2):
            L_EH = einstein_hilbert_lagrangian_order(k)
            if has_matter:
                L_mat = matter_lagrangian_order(L_M, k)
                L_k = L_EH + L_mat if L_mat != S.Zero else L_EH
                if isinstance(L_k, TensExpr):
                    L_k = canon(L_k)
            else:
                L_k = L_EH
            self.L_ref[k] = L_k
            if self.verbose:
                print(f"    L_ref^({k}): {_format_breakdown(L_k)}")

        # Nonminimal matter-curvature coupling: add its covariant expansion to
        # L_ref^(k) for k = 1..n_max+1 (order 0 vanishes, since R^(0)=0).
        if self.nonminimal_coupling is not None:
            from bootstrap.covariant import covariant_coupling_order
            for k in range(1, n_max + 2):
                C_k = covariant_coupling_order(self.nonminimal_coupling, k)
                if C_k == S.Zero:
                    continue
                L_k = self.L_ref.get(k, S.Zero)
                L_k = C_k if L_k == S.Zero else canon(L_k + C_k)
                self.L_ref[k] = L_k
                if self.verbose:
                    print(f"    L_ref^({k}) + coupling: "
                          f"{_format_breakdown(self.L_ref[k])}")

    def _verify_vs_L_ref(self, n):
        """Verification cycle (per the paper's roadmap).

        After running order n, check that L^{(n+1)} is equivalent (modulo
        boundary terms) to L_ref^{(n+1)} by comparing their EL derivatives:

          canon(EL(L_ref^{(n+1)}) − E^{(n)}) == 0

        We skip the "literal equality" check that would compare L's
        directly: L_ref is the raw EH expansion (with second derivatives)
        and our L^{(n+1)} comes out of `_step6_close_loop` after IBP, so
        literal equality essentially never holds. The EL comparison is
        the real test anyway.

        If the EL difference is nonzero, the paper's prescription is to
        decompose it into integrable EOM terms (yielding field
        redefinitions to apply to L_ref). That decomposition is stubbed
        — when we reach a setting where this fires we want a loud failure
        so we know we need to fill it in.
        """
        target_n = n + 1
        if target_n not in self.L_ref:
            if self.verbose:
                print(f"  Verify vs L_ref: no L_ref^({target_n}) (n_max too low?), skipped")
            return None

        L_r = self.L_ref[target_n]
        if L_r == S.Zero:
            E_r = S.Zero
        else:
            E_r, E_r_idx = euler_lagrange(L_r, h)
            E_r = _reindex_tensor(E_r, E_r_idx, (self.mu_E, self.nu_E))
        E_diff = self.E[n] - E_r
        if isinstance(E_diff, TensExpr):
            E_diff = canon(E_diff)

        # Traceless-T_M verification recovery (DEVELOPMENT_STATUS item 0).
        # On an on-shell-traceless trace, an order-(n-1) traceless-shape redef
        # can stay invisible at its own order and reappear here as ddh content
        # in E_diff. Recover it -- rolling L_ref back to this order's snapshot
        # and AUGMENTING order-(n-1)'s redef -- before the normal
        # decomposition. Gated on traceless_T_M; default path is unchanged.
        recovered = False
        if (E_diff != S.Zero and self.traceless_T_M
                and ddh in set(get_tensors_in_expr(E_diff))):
            cleaned = self._recover_missed_traceless_redef(n, E_diff)
            if cleaned is not None:
                E_diff = cleaned
                recovered = True

        # ONE snapshot per loop order: taken AFTER any missed (order n-1) redef
        # and BEFORE this order's normal redef. A later order that discovers a
        # missed order-n redef rolls back to here, then re-applies the
        # augmented h-redef and then the augmented matter redefs before
        # recomputing E_diff.
        self._snapshot_L_ref_for_rollback(n)

        if E_diff == S.Zero:
            if self.verbose:
                tag = " after traceless recovery" if recovered else ""
                print(f"  Verify vs L_ref^({target_n}): EL-equivalent{tag} (OK)")
            return True

        if recovered and self.verbose:
            print(f"  Verify vs L_ref^({target_n}): traceless recovery left "
                  f"{_count(E_diff)} terms; continuing with the normal "
                  f"decomposition")

        # Nonzero EL difference. Per the paper's flow: decompose E_diff
        # into EOM-proportional pieces, check that those X coefficients are
        # integrable (i.e. derivative-free functions of the fields — the
        # signature of a Lagrangian field redef). If yes, the diff is a
        # valid field-redef diff; we report it but do NOT apply the redef
        # yet (open-work item 3 — applying the redef updates L_ref to
        # match L^{(n+1)}, which is the next sub-step).
        n_E_diff = _count(E_diff)
        if self.verbose:
            print(f"  Verify vs L_ref^({target_n}): E_diff has {n_E_diff} terms - "
                  f"checking if it decomposes as a field-redef EOM combination")
        ok, info = self._check_eom_decomposition(E_diff)
        if ok:
            if self.verbose:
                self._print_eom_decomposition_summary(info, target_n)
            # Recover f^{(n+1)} from X and remember for possible traceless-
            # T_M augmentation later (see open-work item 0).
            redefs = self._recover_field_redefs(info, n_plus_one=n + 1)
            self.recovered_redefs[n + 1] = redefs
            self._log_field_redefs(redefs, n + 1)
            # Apply f^{(n+1)} to L_ref^(k) for k=0..n_max+1 (h-redef first
            # per the paper's ordering convention, then each matter-field
            # redef one at a time).
            if redefs.get('h') is not None or redefs.get('phi'):
                self._apply_field_redefs_to_L_ref(redefs, n)
                # Re-verify: after applying the redef, EL(L_ref^(n+1)_new)
                # MUST now equal E^(n). The redef applied to L_ref^(0) = L_M
                # generates higher-order pieces (via chain-rule terms in df,
                # since f has h) that become part of new L_ref^(n+1) and
                # exactly cancel the original diff. If not 0, the redef
                # didn't fully absorb the diff — either the substitution is
                # incomplete, or h-redef is needed in addition (X_h ≠ 0).
                E_r_new_expr, E_r_new_idx = euler_lagrange(self.L_ref[target_n], h)
                E_r_new = _reindex_tensor(E_r_new_expr, E_r_new_idx, (self.mu_E, self.nu_E))
                E_diff_new = self.E[n] - E_r_new
                if isinstance(E_diff_new, TensExpr):
                    E_diff_new = canon(E_diff_new)
                if E_diff_new == S.Zero:
                    if self.verbose:
                        print(f"  After redef: L_ref^({target_n}) diff vs "
                              f"E^({n}) = 0 (was {n_E_diff} before). "
                              f"Closure VERIFIED at this order.")
                else:
                    n_new = _count(E_diff_new)
                    if self.verbose:
                        print(f"  *** Diff after redef: {n_new} terms (was {n_E_diff}) ***")
                        # Diagnostic dump for debugging.
                        terms_to_show = E_diff_new.args if isinstance(E_diff_new, TensAdd) else [E_diff_new]
                        print(f"  Residual sample (first 8 of {n_new} terms):")
                        for i, t in enumerate(terms_to_show[:8]):
                            print(f"    [{i}] {t}")
                        print(f"  Integrability decomposition that was applied:")
                        if info.get('C_h', S.Zero) != S.Zero:
                            print(f"    C_h ({_count(info['C_h'])} terms): {info['C_h']}")
                        for name, (Cp, _) in info.get('C_phi', {}).items():
                            print(f"    C_{name} ({_count(Cp)} terms): {Cp}")
                    raise RuntimeError(
                        f"After applying redef at n={n}, L_ref^({target_n}) "
                        f"diff vs E^({n}) is still {n_new} terms (was {n_E_diff}). "
                        f"The redef substitution didn't absorb the diff."
                    )
            return True
        # Not integrable: structural failure, raise loudly.
        msg = info.get('reason', 'unknown')
        residual = info.get('residual', S.Zero)
        if self.verbose and residual != S.Zero:
            terms = residual.args if isinstance(residual, TensAdd) else [residual]
            print(f"\n  *** Non-integrable residual (first 12 of {len(terms)} terms) ***")
            for i, t in enumerate(terms[:12]):
                print(f"    [{i}] {t}")
            if len(terms) > 12:
                print(f"    ... and {len(terms) - 12} more terms")
        raise RuntimeError(
            f"EL(L_ref^({target_n})) - E^({n}) is NOT an integrable EOM "
            f"combination: {msg}. Residual after EOM decomposition has "
            f"{_count(residual)} terms; coefficients "
            f"that contain field derivatives: {info.get('with_derivatives', [])}."
        )

    def _check_eom_decomposition(self, D):
        """Check whether the rank-2 expression D decomposes as
            D = C_h . E_h^(0) + Sum_i C_{phi_i} . E_{phi_i}^(0)
        with every coefficient C derivative-free (the necessary conditions
        for a Lagrangian field redefinition; paper §3).

        Note: derivative-freeness + clean decomposition are NECESSARY but
        not by themselves sufficient for integrability. The actual
        Helmholtz integrability condition on each C (antisymmetric part of
        \\partial C / \\partial h vanishing) is checked indirectly when the
        full-cycle EL(updated L_ref^(n+1)) = E^(n) check runs after the
        redef is applied; a violation there would catch a non-integrable C.
        For up-front integrability validation of arbitrary user-supplied
        EOM coefficients (Step 4 optional path), use `_check_X_integrable`.

        Returns (ok, info) where info is a dict with keys:
          'C_h', 'C_phi' (decomposition coefficients, indexed by field)
          'residual' (what's left after decomposition)
          'reason' (explanation if ok=False)
          'with_derivatives' (list of C names that contain field derivatives)
        """
        result = decompose_against_eoms(
            D, self.L[0], em_procedure=self.em_procedure, verbose=self.verbose,
            E0=self.E[0], E0_indices=(self.mu_E, self.nu_E),
        )
        residual = result['residual']
        # The orchestrator's API uses Y_h / X_phi (paper-aligned for Step 3
        # where Y is the natural name); here in the verification context we
        # call them C.
        C_h = result['Y_h']
        C_phi_dict = result['X_phi']
        if residual != S.Zero:
            return False, {
                'residual': residual,
                'reason': 'orchestrator left a nonzero residual',
                'C_h': C_h,
                'C_phi': C_phi_dict,
                'with_derivatives': [],
            }
        with_derivs = []
        if C_h != S.Zero and not _is_derivative_free(C_h):
            with_derivs.append('C_h')
        for name, (Cp, _) in C_phi_dict.items():
            if Cp != S.Zero and not _is_derivative_free(Cp):
                with_derivs.append(f'C_{name}')
        if with_derivs:
            return False, {
                'residual': residual,
                'reason': 'one or more coefficients contain field derivatives',
                'C_h': C_h,
                'C_phi': C_phi_dict,
                'with_derivatives': with_derivs,
            }
        return True, {
            'C_h': C_h,
            'alphas_h': result['alphas_h'],
            'C_phi': C_phi_dict,
            'residual': residual,
        }

    def _print_eom_decomposition_summary(self, info, target_n):
        C_h = info['C_h']
        print(f"  EOM decomposition OK at L_ref^({target_n}):")
        if C_h != S.Zero:
            print(f"    C_h: {_format_breakdown(C_h)}")
        for name, (Cp, _) in info['C_phi'].items():
            print(f"    C_{name}: {_format_breakdown(Cp)}")

    def _recover_field_redefs(self, info, n_plus_one):
        """Compute the field redefinitions f^(n+1) from the integrability-
        check coefficients (paper formula, around eq. 625):

          f_h^{(n+1)}_{κλ}     = (1/(n+1)) h_{αβ} C_h^{αβ}_{κλ}
          f_{φ_i}^{(n+1) ...}  = (1/(n+1)) h_{αβ} C_{φ_i}^{αβ ...}

        Returns dict:
          'h':   (f_h_expr, f_h_free_indices) or None if C_h = 0,
          'phi': {name: (f_phi_expr, f_phi_free_indices)} for matter fields.
        """
        coeff = Rational(1, n_plus_one)
        out = {'h': None, 'phi': {}}

        C_h = info.get('C_h', S.Zero)
        if C_h != S.Zero:
            # Contract C_h's (μ_E, ν_E) with h(-μ_E, -ν_E); after canon, f_h
            # carries C_h's κλ-pair as free indices.
            f_h = coeff * C_h * h(-self.mu_E, -self.nu_E)
            if isinstance(f_h, TensExpr):
                f_h = canon(f_h)
            f_h_free = list(f_h.get_free_indices()) if isinstance(f_h, TensExpr) else []
            out['h'] = (f_h, tuple(f_h_free))

        for name, (C_phi, _) in info.get('C_phi', {}).items():
            if C_phi == S.Zero:
                continue
            f_phi = coeff * C_phi * h(-self.mu_E, -self.nu_E)
            if isinstance(f_phi, TensExpr):
                f_phi = canon(f_phi)
            f_phi_free = list(f_phi.get_free_indices()) if isinstance(f_phi, TensExpr) else []
            out['phi'][name] = (f_phi, tuple(f_phi_free))

        return out

    def _snapshot_L_ref_for_rollback(self, n):
        """Snapshot L_ref for a possible later rollback of order-n's NORMAL
        redef. Called from _verify_vs_L_ref AFTER any missed redef belonging to
        order n-1 has been applied but BEFORE order-n's normal redef, so a
        higher order that discovers a missed order-n redef can restore this
        exact state and re-apply the augmented order-n redef once. Keeps only
        the latest two orders to bound memory."""
        self.L_ref_history[n] = dict(self.L_ref)
        for older in list(self.L_ref_history):
            if older < n - 1:
                del self.L_ref_history[older]

    def _add_one_redef(self, fa, fb):
        """Sum two (f_expr, free_indices) redef components, reindexing fb's
        free indices onto fa's so the addition is index-legal. Either may be
        None (then the other is returned)."""
        if fa is None:
            return fb
        if fb is None:
            return fa
        f_a, free_a = fa
        f_b, free_b = fb
        if (isinstance(f_b, TensExpr) and free_a and free_b
                and tuple(free_b) != tuple(free_a)):
            f_b = _reindex_tensor(f_b, tuple(free_b), tuple(free_a))
        total = f_a + f_b
        if isinstance(total, TensExpr):
            total = canon(total)
        free = (tuple(total.get_free_indices())
                if isinstance(total, TensExpr) else tuple(free_a))
        return (total, free)

    def _add_redefs(self, a, b):
        """Combine two field-redef dicts by summing their f-expressions.
        Used to AUGMENT an already-applied order-k redef `a` with a
        later-discovered missed contribution `b` of the same order: the total
        is applied once from a rolled-back L_ref, because applying `b` on top
        of `a` sequentially would double-count the chain-rule cross terms."""
        if not a:
            return b
        if not b:
            return a
        out = {'h': self._add_one_redef(a.get('h'), b.get('h')), 'phi': {}}
        for name in set(a.get('phi', {})) | set(b.get('phi', {})):
            merged = self._add_one_redef(
                a.get('phi', {}).get(name), b.get('phi', {}).get(name))
            if merged is not None:
                out['phi'][name] = merged
        return out

    def _apply_field_redefs_to_L_ref(self, redefs, n):
        """Apply field redefinitions (h and/or matter) to L_ref^{(k)} for
        k = 0..n_max+1, propagating each redef via chain rule, then
        aggregating the order-j piece of every substituted L_ref^{(k)}
        (for k ≤ j) into the updated L_ref^{(j)} for j = n+1..n_max+1.

        Per the paper's ordering convention: h-redef applied FIRST across
        all L_ref^{(k)} (replacing L_ref only AFTER substituting at every
        order — important), THEN each matter-field
        redef one at a time (each replacing L_ref before the next).

        **L_ref^(0) MUST be included** in the substitution range. It's the
        original L_M (matter side) / 0 (gravity side); substituting field →
        field + f in it produces higher-order pieces via the chain-rule
        terms in df, and those propagate into the new L_ref^{(n+1)} where
        they're exactly what makes EL(new L_ref^{(n+1)}) = E^{(n)}.
        """
        if self.n_max is None:
            return
        # Build the ordered list of (field_info, f_expr, f_indices, label).
        ordered_redefs = []
        if redefs.get('h') is not None:
            h_info = {'field': h, 'dfield': dh, 'ddfield': ddh,
                      'rank': 2, 'name': 'h'}
            f_h, f_h_idx = redefs['h']
            ordered_redefs.append((h_info, f_h, f_h_idx))
        for name, (f_phi, f_idx) in redefs.get('phi', {}).items():
            info = _matter_fields[name]
            ordered_redefs.append((info, f_phi, f_idx))

        # NOTE: the rollback snapshot is taken by `_verify_vs_L_ref` (via
        # `_snapshot_L_ref_for_rollback`) at the correct point -- AFTER any
        # missed redef belonging to order n-1, BEFORE this order's normal
        # redef -- not here, so it is not re-taken during a missed re-apply.
        for field_info, f_expr, f_indices in ordered_redefs:
            self._apply_one_field_redef(field_info, f_expr, f_indices, n)

    def _apply_one_field_redef(self, field_info, f_expr, f_indices, n):
        """Substitute one field → field + f in L_ref^(k) for k = 0..k_max_sub
        (where k_max_sub depends on field type — see below), aggregate by
        target order, replace self.L_ref^(j) for j ≥ n+1.

        Substitution range optimization: each replacement adds (n+1 - rank_h_of_field)
        to a term's order. For at least one new contribution at order ≤ n_max+1:
          - matter fields (rank_h = 0): k_max = n_max - n
          - h field (rank_h = 1):       k_max = n_max - n + 1
        For k > k_max, the substituted L_ref^(k) has no NEW contributions at
        order ≤ n_max+1; we just pass through the original.
        """
        if self.n_max is None:
            return
        is_h = (field_info['field'] is h)
        # +1 over the naive bound: we need L_ref through order n_max+1.
        k_max_sub = self.n_max - n + (2 if is_h else 1)
        # Optimization #1: differentiate f to df/ddf ONCE for this redef and
        # share the templates across every substituted L_ref^(k) below.
        deriv_cache = _build_deriv_cache(field_info, f_expr, f_indices)
        substituted = {}
        for k in range(0, k_max_sub + 1):
            L_k = self.L_ref.get(k, S.Zero)
            L_k = _substitute_field(
                L_k, field_info, f_expr, f_indices,
                target_order=self.n_max + 1,
                deriv_cache=deriv_cache,
            )
            substituted[k] = L_k

        label = field_info.get('name', '?')
        first_updated = n + 1
        for j in range(first_updated, self.n_max + 2):
            contributions = []
            for k in range(0, j + 1):
                if k in substituted:
                    src = substituted[k]
                else:
                    # k > k_max_sub: no new contributions, just passthrough.
                    src = self.L_ref.get(k, S.Zero)
                if src == S.Zero:
                    continue
                piece = filter_by_order(src, j)
                if piece != S.Zero:
                    contributions.append(piece)
            if not contributions:
                new_L = S.Zero
            elif len(contributions) == 1:
                new_L = contributions[0]
            else:
                new_L = TensAdd(*contributions)
            if isinstance(new_L, TensExpr):
                new_L = canon(new_L)
            old_L = self.L_ref.get(j, S.Zero)
            self.L_ref[j] = new_L
            if self.verbose:
                old_count = _count(old_L)
                new_count = _count(new_L)
                delta = new_count - old_count
                sign = '+' if delta >= 0 else ''
                print(f"    [{label}-redef] L_ref^({j}): "
                      f"{_format_breakdown(old_L)}  ->  "
                      f"{_format_breakdown(new_L)}  ({sign}{delta} terms)")

    def _log_field_redefs(self, redefs, n_plus_one):
        """Print the recovered field redefinitions in the form
            Applying field redefinition: phi -> phi + f_phi^(n+1)  where
              f_phi^(n+1) = <expression>
        with the field name spelled out (h for the graviton, the registered
        matter name otherwise). Derivative-freeness is checked silently in
        _integrability_check; not printed here."""
        if not self.verbose:
            return
        f_h_info = redefs.get('h')
        if f_h_info is not None:
            f_h, _ = f_h_info
            print(f"    Applying field redefinition: "
                  f"h -> h + f_h^({n_plus_one})  where")
            print(f"      f_h^({n_plus_one}) = {f_h}")
        for name, (f_phi, _) in redefs.get('phi', {}).items():
            print(f"    Applying field redefinition: "
                  f"{name} -> {name} + f_{name}^({n_plus_one})  where")
            print(f"      f_{name}^({n_plus_one}) = {f_phi}")

