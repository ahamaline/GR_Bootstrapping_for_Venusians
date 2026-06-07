"""Decompose tensor expressions into linear combinations of zeroth-order EOMs.

The public entry point is `decompose_against_eoms(Z, L, ..., E0, E0_indices)`,
which decomposes

  Z = Y_h^{cdef}_{ab} . E_h^{(0)ab}  +  Sum_i  X_{phi_i}^{cdef}_{a...} . E_{phi_i}^{(0) a...}
      + residual (must be 0 for a valid decomposition)

with E_h^{(0)} = kappa T_M (Hilbert or Belinfante; pass the bootstrap state's
E[0] explicitly to avoid recomputing T_M and to support optional EOM terms)
and E_{phi_i}^{(0)} = dL/dphi_i.

The Y_h / X_{phi_i} returned here is the decomposition coefficient. In step
3 of the bootstrap, the paper formula X^(n+1) = -1/(2(n+1)) Y · h converts
it into the X added to E. In the L_ref verification step, the returned
coefficient IS already the X for the field redef (the (1/(n+1)) factor is
applied separately in BootstrapState._recover_field_redefs).

Algorithm (user 2026-05-26): matter EOMs E_{phi_i}^{(0)} only contain second
derivatives; all first-derivative products live in T_M (so in Y_h × T_M).
Isolate Y_h by matching T_M's first-derivative-product signature in Z; after
subtraction, if a contracted dphi·dphi remains it indicates a Y_h trace
piece, extracted similarly against eta_{ab} T_M^{ab}; the final residual is
a sum over matter EOMs, each identified by a unique 2nd-derivative monomial.
"""

from sympy import S, Rational
from sympy.tensor.tensor import TensAdd, TensMul, Tensor, TensExpr
from bootstrap.tensor_algebra import (
    fresh_indices, canon, metric, _matter_fields,
)
from bootstrap.jet import (
    _decompose_tensmul, _get_component, _get_indices, _sum_terms,
)
from bootstrap.euler_lagrange import euler_lagrange, euler_lagrange_scalar
from bootstrap.energy_momentum import (
    hilbert_energy_momentum, symmetrized_belinfante,
)
from bootstrap.loop_helpers import _require_constant_coeff


def _is_dummy_pair(idx_a, idx_b):
    """Check if (idx_a, idx_b) is a contracted-dummy pair (one up, one down, same name)."""
    return idx_a.name == idx_b.name and (idx_a.is_up != idx_b.is_up)


def _find_trace_factor(factors, head, field_positions, trace_positions):
    """Find a factor in `factors` matching the signature head(...) with
    indices at `field_positions` being the "α-role" (extractable) and indices
    at `trace_positions` being a dummy pair (contracted).

    Args:
        factors: list of Tensor factors (decomposed from a term).
        head: TensorHead to match (e.g., ddA, ddV, ddh, ddphi).
        field_positions: tuple of positions in head's indices that play the
            α-role (e.g., (0,) for ddA, (0, 1) for ddh, () for ddphi-trace).
        trace_positions: tuple of 2 positions forming a dummy pair.

    Returns (factor_index_in_factors, tuple_of_field_index_values) or None.
    """
    for i, f in enumerate(factors):
        if _get_component(f) != head:
            continue
        indices = _get_indices(f)
        if max(*trace_positions, *field_positions, 0) >= len(indices):
            continue
        idx_t1 = indices[trace_positions[0]]
        idx_t2 = indices[trace_positions[1]]
        if not _is_dummy_pair(idx_t1, idx_t2):
            continue
        # field_positions must NOT also form a dummy pair within this factor
        # (e.g., ddh(L_0, -L_0, L_1, -L_1) matches trace_positions=(2,3) but
        # field_positions=(0,1) are ALSO a dummy pair, not free indices —
        # treating them as field indices would mis-attribute contributions).
        if len(field_positions) == 2:
            f_idx1, f_idx2 = indices[field_positions[0]], indices[field_positions[1]]
            if _is_dummy_pair(f_idx1, f_idx2):
                continue
        field_idx_values = tuple(indices[p] for p in field_positions)
        return i, field_idx_values
    return None


def _extract_coeff_from_trace_signature(Z, head, field_positions, trace_positions):
    """Extract C by matching each term in Z to a `head(field…, …, L, -L, …)` shape.

    For each matching term, strip the signature factor and contribute
    `stripped × Π_k metric(?_k, -α_k)` to C (where α_k are fresh up indices,
    one per field_position, introduced once for the whole expression).

    Args:
        Z: tensor expression.
        head: TensorHead to match (e.g., ddA, ddV, ddh, ddphi).
        field_positions: tuple of positions in head's indices that get
            extracted as α-role contraction indices.
        trace_positions: tuple of 2 positions forming a dummy pair.

    Returns: (C, alphas) — C has alphas (each as a free up index for
    contraction with EOM's α) as free indices, or (C, ()) for scalar
    (no field positions).
    """
    if Z == S.Zero:
        return S.Zero, ()

    if field_positions:
        # Generate fresh indices one at a time so we always get TensorIndex objects
        # (fresh_indices(n) returns a list for n>1 — not directly unpackable here).
        alphas = tuple(fresh_indices(1)[0] for _ in range(len(field_positions)))
    else:
        alphas = ()

    terms = Z.args if isinstance(Z, TensAdd) else [Z]
    contributions = []  # one-shot TensAdd at the end (see _sum_terms docstring)
    # Filter: each contribution must have free-index NAME set equal to
    # Z's free + alphas. Matches whose stripped factors leak dummies as
    # extra free indices are skipped (not a clean EOM-signature match).
    z_free_names = set(idx.name for idx in (
        Z.get_free_indices() if isinstance(Z, TensExpr) else []
    ))
    expected_names = z_free_names | {a.name for a in alphas}

    for term in terms:
        if isinstance(term, TensMul):
            coeff, factors = _decompose_tensmul(term)
        elif isinstance(term, Tensor):
            coeff, factors = S.One, [term]
        else:
            continue

        match = _find_trace_factor(factors, head, field_positions, trace_positions)
        if match is None:
            continue
        factor_idx, star_indices = match

        # Strip the signature factor.
        other_factors = factors[:factor_idx] + factors[factor_idx + 1:]
        stripped = coeff
        for f in other_factors:
            stripped = stripped * f

        contribution = stripped
        for star_idx, alpha in zip(star_indices, alphas):
            contribution = contribution * metric(star_idx, -alpha)
        contribution_free_names = (
            set(idx.name for idx in contribution.get_free_indices())
            if isinstance(contribution, TensExpr) else set()
        )
        if contribution_free_names != expected_names:
            continue
        contributions.append(contribution)

    C = _sum_terms(contributions)
    if isinstance(C, TensExpr):
        C = canon(C)
    return C, alphas


def _first_deriv_heads():
    """Set of first-derivative TensorHeads for all registered matter fields."""
    return {info['dfield'] for info in _matter_fields.values()}


def _second_deriv_heads():
    """Set of second-derivative TensorHeads for all registered matter fields."""
    return {info['ddfield'] for info in _matter_fields.values()}


def _term_factors(term):
    """Decompose a tensor term into (coeff, factor_list). Handles Tensor and TensMul."""
    if isinstance(term, TensMul):
        return _decompose_tensmul(term)
    if isinstance(term, Tensor):
        return S.One, [term]
    return None, None


def _terms_of(expr):
    """Iterator over the terms of a TensAdd (or [expr] if it's a single term)."""
    if expr == S.Zero or expr == 0:
        return []
    return expr.args if isinstance(expr, TensAdd) else [expr]


def _has_first_deriv_product(expr, contracted_only=False):
    """Return True if any term in `expr` contains at least two first-derivative
    factors (of any registered matter field). If contracted_only=True, only
    counts pairs whose free indices are contracted (i.e. NO free index
    appearing on either factor goes outside the pair)."""
    deriv_heads = _first_deriv_heads()
    for term in _terms_of(expr):
        coeff, factors = _term_factors(term)
        if factors is None:
            continue
        deriv_factors = [f for f in factors if _get_component(f) in deriv_heads]
        if len(deriv_factors) < 2:
            continue
        if not contracted_only:
            return True
        # Check whether any pair of derivative factors has all of its indices
        # contracted (i.e. NOT free in the whole term). Count appearances of
        # each index name across all factors: dummies appear twice (one up,
        # one down), free indices once.
        name_count = {}
        for f in factors:
            for idx in _get_indices(f):
                name_count[idx.name] = name_count.get(idx.name, 0) + 1
        # Indices appearing twice across the term are dummies (contracted).
        # If any pair of deriv factors has both of their indices as dummies,
        # they are "contracted" in our sense.
        for i in range(len(deriv_factors)):
            for j in range(i + 1, len(deriv_factors)):
                idx_i = _get_indices(deriv_factors[i])
                idx_j = _get_indices(deriv_factors[j])
                # Drop the field index (always at field-natural position).
                # For dphi rank-2: position 0 is the derivative-index; for dA/dV
                # we have 2 indices, the field-natural position is determined by
                # NATURAL_POSITIONS — but for the "contracted" check, we just
                # ask if all 4 indices (or 2+2 for vector) are dummies.
                all_dummy = all(name_count.get(idx.name, 0) >= 2
                                for idx in (idx_i + idx_j))
                if all_dummy:
                    return True
    return False


def _pick_kinetic_signature(T_M, mu_T, nu_T):
    """Pick a "good" kinetic signature term from T_M.

    A good term:
      - has mu_T and nu_T as free indices,
      - mu_T and nu_T sit on DIFFERENT first-derivative matter-field factors,
      - neither sits on a metric factor.

    Returns (coeff, factors, mu_loc, nu_loc) where mu_loc/nu_loc are
    (factor_index, index_position) tuples, or None if no good term exists.
    """
    deriv_heads = _first_deriv_heads()
    for term in _terms_of(T_M):
        coeff, factors = _term_factors(term)
        if factors is None:
            continue
        mu_loc = nu_loc = None
        for fi, f in enumerate(factors):
            head = _get_component(f)
            if head not in deriv_heads:
                continue
            for pi, idx in enumerate(_get_indices(f)):
                if idx == mu_T or idx == -mu_T:
                    mu_loc = (fi, pi)
                elif idx == nu_T or idx == -nu_T:
                    nu_loc = (fi, pi)
        if mu_loc is None or nu_loc is None:
            continue
        if mu_loc[0] == nu_loc[0]:
            continue  # both on the same factor (won't give a clean signature)
        return coeff, factors, mu_loc, nu_loc
    return None


def _extract_coeff_two_factor(Z, head_mu, pos_mu, head_nu, pos_nu, sig_coeff):
    """Match the signature `sig_coeff × head_mu(...,mu_T,...) × head_nu(...,nu_T,...)`
    in Z, where mu_T sits at index position `pos_mu` in head_mu's factor and
    nu_T at `pos_nu` in head_nu's factor.

    For each term in Z that contains a matching pair of factors, strip them
    and contribute `(stripped/sig_coeff) × metric(idx_mu, -α) × metric(idx_nu, -β)`
    to Y, where α, β are fresh up indices that will be contracted with the
    EOM's free indices (T_M^{αβ}).

    NOTE on multi-occurrence: if Z's term has more than one pair matching the
    signature, we sum over all ordered pairs. This is the correct algebraic
    contribution because T_M's monomial is taken once with these specific
    free indices, and each pair in Z's term arises from a distinct contraction
    choice in the Y_h × T_M product.

    Returns (Y, (alpha, beta)).
    """
    if Z == S.Zero:
        return S.Zero, (None, None)
    alpha = fresh_indices(1)[0]
    beta = fresh_indices(1)[0]
    same_head = (head_mu == head_nu and pos_mu == pos_nu)

    # Expected free indices on each contribution: Z's free + (alpha, beta).
    # If a match yields a contribution with a different free-index set,
    # the matched factors carried dummies linked to OTHER factors in the
    # term (so stripping released them as free), and this match doesn't
    # correspond to a clean Y · T_M decomposition — skip it.
    z_free_names = set(idx.name for idx in (
        Z.get_free_indices() if isinstance(Z, TensExpr) else []
    ))
    expected_names = z_free_names | {alpha.name, beta.name}

    contributions = []  # one-shot TensAdd at end (avoid O(N^2) re-collect)
    for term in _terms_of(Z):
        coeff, factors = _term_factors(term)
        if factors is None:
            continue
        # Indexed lists of factor positions of each head.
        mu_candidates = [i for i, f in enumerate(factors)
                         if _get_component(f) == head_mu]
        nu_candidates = [i for i, f in enumerate(factors)
                         if _get_component(f) == head_nu]
        if not mu_candidates or not nu_candidates:
            continue
        for i in mu_candidates:
            for j in nu_candidates:
                if i == j:
                    continue
                idx_i = _get_indices(factors[i])
                idx_j = _get_indices(factors[j])
                if pos_mu >= len(idx_i) or pos_nu >= len(idx_j):
                    continue
                idx_mu = idx_i[pos_mu]
                idx_nu = idx_j[pos_nu]
                # Skip pairs where the two free-role indices are themselves
                # a contracted dummy pair within Z's term. Such matches arise
                # from the TRACE part of T_M (or from contracted matter-EOM
                # pieces) and contribute to the trace coefficient, not Y_kin.
                # The trace term is handled separately by
                # `_extract_trace_coeff_two_factor`.
                if _is_dummy_pair(idx_mu, idx_nu):
                    continue
                # Strip both factors (by position, not value).
                other_factors = [f for k, f in enumerate(factors)
                                 if k != i and k != j]
                stripped = coeff / sig_coeff
                for f in other_factors:
                    stripped = stripped * f
                contribution = (stripped * metric(idx_mu, -alpha)
                                * metric(idx_nu, -beta))
                # Filter: contribution's free-index NAMES must match
                # expected_names. Otherwise the matched factors carried
                # dummies paired with non-signature factors, releasing those
                # dummies as free upon stripping — this is not a clean
                # kinetic-signature match.
                contribution_free_names = (
                    set(idx.name for idx in contribution.get_free_indices())
                    if isinstance(contribution, TensExpr) else set()
                )
                if contribution_free_names != expected_names:
                    continue
                contributions.append(contribution)
        # When mu and nu sit on identical (head, pos), the double-loop above
        # double-counts each unordered pair. Halve to compensate.
        # (We can't just iterate i<j because the signature has a definite
        # mu/nu assignment that we recover via the metric contraction.)
    Y = _sum_terms(contributions)
    if same_head and Y != S.Zero:
        Y = Rational(1, 2) * Y

    if isinstance(Y, TensExpr):
        Y = canon(Y)
    return Y, (alpha, beta)


def _trace_T_M(T_M, mu_T, nu_T):
    """Compute η_{αβ} T_M^{αβ} — the trace scalar of T_M."""
    if T_M == S.Zero:
        return S.Zero
    traced = T_M * metric(-mu_T, -nu_T)
    if isinstance(traced, TensExpr):
        traced = canon(traced)
    return traced


def _extract_trace_coeff_two_factor(Z_residual, head_mu, pos_mu, head_nu, pos_nu, sig_coeff):
    """Like _extract_coeff_two_factor but with NO alpha/beta extraction —
    matches the same kinetic-signature pattern but contracts the two stripped
    indices into a dummy pair (so the contribution is a scalar coefficient,
    representing the η_{αβ} trace contraction).

    Returns Y_trace (the coefficient that multiplies η_{αβ} T_M^{αβ}).
    """
    if Z_residual == S.Zero:
        return S.Zero
    same_head = (head_mu == head_nu and pos_mu == pos_nu)
    z_free_names = set(idx.name for idx in (
        Z_residual.get_free_indices() if isinstance(Z_residual, TensExpr) else []
    ))
    contributions = []  # one-shot TensAdd at end (avoid O(N^2) re-collect)
    for term in _terms_of(Z_residual):
        coeff, factors = _term_factors(term)
        if factors is None:
            continue
        mu_candidates = [i for i, f in enumerate(factors)
                         if _get_component(f) == head_mu]
        nu_candidates = [i for i, f in enumerate(factors)
                         if _get_component(f) == head_nu]
        if not mu_candidates or not nu_candidates:
            continue
        for i in mu_candidates:
            for j in nu_candidates:
                if i == j:
                    continue
                idx_i = _get_indices(factors[i])
                idx_j = _get_indices(factors[j])
                if pos_mu >= len(idx_i) or pos_nu >= len(idx_j):
                    continue
                idx_mu = idx_i[pos_mu]
                idx_nu = idx_j[pos_nu]
                other_factors = [f for k, f in enumerate(factors)
                                 if k != i and k != j]
                stripped = coeff / sig_coeff
                for f in other_factors:
                    stripped = stripped * f
                # Contract the two stripped indices via metric.
                contribution = stripped * metric(idx_mu, idx_nu)
                contribution_free_names = (
                    set(idx.name for idx in contribution.get_free_indices())
                    if isinstance(contribution, TensExpr) else set()
                )
                if contribution_free_names != z_free_names:
                    continue
                contributions.append(contribution)
    Y_trace = _sum_terms(contributions)
    if same_head and Y_trace != S.Zero:
        Y_trace = Rational(1, 2) * Y_trace
    if isinstance(Y_trace, TensExpr):
        Y_trace = canon(Y_trace)
    return Y_trace


def decompose_against_eoms(Z, L, em_procedure='hilbert', verbose=False,
                           E0=None, E0_indices=None):
    """Orchestrator: decompose Z = Y_h × E_h^(0) + Σ X_{φ_i} × E_{φ_i}^(0) + residual.

    Algorithm (user 2026-05-26):
      1. Pick a kinetic signature in E_h^(0) (one with μ, ν on distinct first-
         derivative matter factors, not contracted together).
      2. Extract Y_h^{cdef}_{αβ} by matching the signature in Z and stripping.
      3. Subtract Y_h × E_h^(0) from Z → residual.
      4. If residual still has first-derivative products with their indices
         contracted (a scalar built from ∂φ·∂φ), compute the trace and
         extract a trace coefficient → add η_{αβ} · Y_trace to Y_h, subtract.
      5. The remaining residual should be a linear combination of matter
         EOMs. For each matter field, pick a unique 2nd-derivative monomial,
         extract X_{φ_i}, subtract.
      6. Final residual should be 0 for a valid decomposition.

    Args:
        Z: tensor expression to decompose (typically rank-4: μνρσ).
        L: the matter Lagrangian (used for matter EOMs only).
        em_procedure: 'hilbert' or 'belinfante'. Used only as a fallback
            when E0 is not provided.
        verbose: print per-step term counts.
        E0: optional. The actual zeroth-order field equation E^(0)^{μν}
            from the bootstrap state. **Preferred over recomputing T_M**:
            if optional EOM terms have been added at order 0, E^(0) ≠ κ T_M
            and recomputing from L gives a wrong target. Caller should pass
            self.E[0] from BootstrapState.
        E0_indices: (μ, ν) — the free indices of E0. Required if E0 given.

    Returns:
        result dict:
          'Y_h': the Y_h coefficient (with free indices = Z's + (α, β)).
          'alphas_h': (α, β) tuple of fresh up indices on Y_h's EOM side.
          'X_phi': dict {field_name: (X_phi_expr, field_indices_tuple)}.
          'residual': what's left after the decomposition. Should be 0.
    """
    _n = lambda x: 0 if x == S.Zero else (len(x.args) if isinstance(x, TensAdd) else 1)

    # --- Step 1: get E_h^(0) for signature picking + subtraction.
    if E0 is not None:
        if E0_indices is None:
            raise ValueError("E0_indices must be provided when E0 is given")
        T_M = E0
        em_idx = E0_indices
    else:
        # Fallback: recompute via the energy-momentum procedure. This is
        # WRONG if optional EOM terms have been added at order 0; callers
        # in BootstrapState should pass self.E[0] explicitly.
        if em_procedure == 'hilbert':
            T_M, em_idx = hilbert_energy_momentum(L)
        elif em_procedure == 'belinfante':
            T_M, em_idx = symmetrized_belinfante(L)
        else:
            raise ValueError(f"em_procedure must be 'hilbert' or 'belinfante', got {em_procedure!r}")
    mu_T, nu_T = em_idx

    residual = Z
    Y_h_total = S.Zero
    alphas_h = (None, None)

    # --- T_M kinetic signature extraction. ------------------------
    sig = _pick_kinetic_signature(T_M, mu_T, nu_T)
    if sig is None:
        if verbose:
            print("    T_M has no kinetic signature (skipping Y_h)")
    else:
        sig_coeff, sig_factors, mu_loc, nu_loc = sig
        # The decomposition divides by this kinetic-signature coefficient of
        # T_M = E_h^(0); it must be a field-independent constant (canonical
        # kinetic normalization) — same requirement as the traceless m.
        _require_constant_coeff(sig_coeff, "the kinetic signature in T_M (E_h^(0))")
        head_mu = _get_component(sig_factors[mu_loc[0]])
        head_nu = _get_component(sig_factors[nu_loc[0]])
        pos_mu = mu_loc[1]
        pos_nu = nu_loc[1]
        Y_kin, alphas_h = _extract_coeff_two_factor(
            residual, head_mu, pos_mu, head_nu, pos_nu, sig_coeff
        )
        if verbose and Y_kin != S.Zero:
            print(f"    h-EOM kinetic coeff: {_n(Y_kin)} terms")
        if Y_kin != S.Zero:
            Y_h_total = Y_kin
            # Subtract Y_kin * T_M^{ab} (with T_M renamed onto alphas_h).
            from bootstrap.loop_helpers import _reindex_tensor
            T_M_renamed = _reindex_tensor(T_M, em_idx, alphas_h)
            contribution = Y_kin * T_M_renamed
            if isinstance(contribution, TensExpr):
                contribution = canon(contribution)
            residual = residual - contribution
            if isinstance(residual, TensExpr):
                residual = canon(residual)
            if verbose:
                print(f"    residual after h-EOM kinetic subtraction: {_n(residual)} terms")

        # --- Trace term: handles leftover contracted-pair first-deriv
        # products in the residual (which arise when Y_h × T_M-trace is
        # part of the decomposition, e.g. for matter with a non-zero T_M
        # trace like a massive scalar). For EM and other conformally-
        # invariant matter, T_M is traceless and this is a no-op.
        if sig is not None and _has_first_deriv_product(residual, contracted_only=True):
            T_M_tr = _trace_T_M(T_M, mu_T, nu_T)
            trace_sig_coeff = _pick_trace_signature_coeff(
                T_M_tr, head_mu, pos_mu, head_nu, pos_nu
            )
            if trace_sig_coeff is not None:
                # Also divided by below; require a constant coefficient.
                _require_constant_coeff(
                    trace_sig_coeff, "the trace signature in T_M (E_h^(0))")
                Y_trace = _extract_trace_coeff_two_factor(
                    residual, head_mu, pos_mu, head_nu, pos_nu, trace_sig_coeff
                )
                if Y_trace != S.Zero:
                    # Add eta_{ab} * Y_trace to Y_h (so Y_h * T_M now also
                    # accounts for the T_M trace piece).
                    if alphas_h[0] is None:
                        alphas_h = (fresh_indices(1)[0], fresh_indices(1)[0])
                    alpha, beta = alphas_h
                    Y_trace_term = Y_trace * metric(-alpha, -beta)
                    if isinstance(Y_trace_term, TensExpr):
                        Y_trace_term = canon(Y_trace_term)
                    Y_h_total = (Y_h_total + Y_trace_term
                                 if Y_h_total != S.Zero else Y_trace_term)
                    if isinstance(Y_h_total, TensExpr):
                        Y_h_total = canon(Y_h_total)
                    # Re-subtract: the trace piece × T_M absorbs the
                    # contracted-pair leftover.
                    from bootstrap.loop_helpers import _reindex_tensor
                    T_M_renamed = _reindex_tensor(T_M, em_idx, alphas_h)
                    contribution_tr = Y_trace_term * T_M_renamed
                    if isinstance(contribution_tr, TensExpr):
                        contribution_tr = canon(contribution_tr)
                    residual = residual - contribution_tr
                    if isinstance(residual, TensExpr):
                        residual = canon(residual)
                    if verbose:
                        print(f"    h-EOM trace coeff: {_n(Y_trace)} terms; "
                              f"residual now: {_n(residual)} terms")

    # --- Step 6: per-matter-field EOM extraction. ---------------------------
    X_phi = {}
    for name, info in _matter_fields.items():
        if residual == S.Zero:
            break
        rank = info.get('rank', 0)
        ddhead = info['ddfield']
        field = info['field']
        if rank == 0:
            field_positions = ()
            trace_positions = (0, 1)
        elif rank == 1:
            field_positions = (0,)
            trace_positions = (1, 2)
        else:
            continue  # unsupported rank for the trace signature
        C_phi, alphas_phi = _extract_coeff_from_trace_signature(
            residual, ddhead, field_positions, trace_positions
        )
        if C_phi == S.Zero:
            # No 2nd-derivative trace signature for this field in the residual.
            # Caller should check the final residual for any leftover terms
            # to detect EOMs that are pure mass terms (where the trace-
            # signature trick doesn't apply because EOM_φ has no 2nd-deriv).
            continue
        if rank == 0:
            EOM_phi = euler_lagrange_scalar(L, field)
        else:
            EOM_phi, EOM_idx = euler_lagrange(L, field)
        # The matter EOM's own 2nd-derivative normalization is the coefficient
        # implicitly divided out when the signature is stripped from the
        # residual; require it field-independent (canonical kinetic term) — the
        # E_phi^(0) analogue of the kinetic-signature guard above.
        eom_dd_coeff, _ = _extract_coeff_from_trace_signature(
            EOM_phi, ddhead, field_positions, trace_positions)
        _require_constant_coeff(
            eom_dd_coeff,
            f"the 2nd-derivative signature in the {name} EOM (E_{name}^(0))")
        if rank == 0:
            contribution = C_phi * EOM_phi
        else:
            from bootstrap.loop_helpers import _reindex_tensor
            EOM_phi_renamed = _reindex_tensor(EOM_phi, EOM_idx, alphas_phi)
            contribution = C_phi * EOM_phi_renamed
        if isinstance(contribution, TensExpr):
            contribution = canon(contribution)
        residual = residual - contribution
        if isinstance(residual, TensExpr):
            residual = canon(residual)
        X_phi[name] = (C_phi, alphas_phi)
        if verbose:
            print(f"    matched coeff_{name}: {_n(C_phi)} terms; "
                  f"residual now {_n(residual)} terms")

    return {
        'Y_h': Y_h_total,
        'alphas_h': alphas_h,
        'X_phi': X_phi,
        'residual': residual,
    }


def _pick_trace_signature_coeff(T_M_tr, head_mu, pos_mu, head_nu, pos_nu):
    """In T_M_tr (the trace scalar of T_M), find the coefficient of the same
    two-factor monomial that defined the kinetic signature, but with the two
    "free" positions now contracted (a dummy pair across the two factors).

    Returns the scalar coefficient (a sympy expression), or None if not found.
    """
    same_head = (head_mu == head_nu and pos_mu == pos_nu)
    for term in _terms_of(T_M_tr):
        coeff, factors = _term_factors(term)
        if factors is None:
            continue
        mu_candidates = [i for i, f in enumerate(factors)
                         if _get_component(f) == head_mu]
        nu_candidates = [i for i, f in enumerate(factors)
                         if _get_component(f) == head_nu]
        if not mu_candidates or not nu_candidates:
            continue
        for i in mu_candidates:
            for j in nu_candidates:
                if i == j:
                    continue
                idx_i = _get_indices(factors[i])
                idx_j = _get_indices(factors[j])
                if pos_mu >= len(idx_i) or pos_nu >= len(idx_j):
                    continue
                if _is_dummy_pair(idx_i[pos_mu], idx_j[pos_nu]):
                    # Found the trace monomial. Its coefficient (relative
                    # to "just these two factors with all other indices
                    # forming their natural dummies") is `coeff`.
                    # Symmetry factor: if same_head and same position,
                    # the two unordered orderings give the same monomial.
                    return coeff
    return None
