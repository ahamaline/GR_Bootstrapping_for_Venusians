"""
energy_momentum.py — Hilbert energy-momentum tensor.

Computes T_H^{mn}[L] by the Hilbert procedure:
1. Covariantize L: replace η → g, ∂ → ∇
2. Multiply by √|g|
3. T_H^{mn} = (2/√|g|) δ(√|g| L̃)/δg_{mn}
4. Evaluate at g = η (Γ→0, √|g|→1, g^{-1}→η)

Key: we can only set g=η AFTER the functional derivative is done.
"""

from sympy import S, Rational, Symbol
from sympy.tensor.tensor import (
    TensAdd, TensMul, TensExpr, Tensor, TensorHead, TensorSymmetry
)
from bootstrap.tensor_algebra import (
    Lorentz, metric, h, dh, ddh,
    fresh_indices, canon, _JET_HIERARCHY,
    NATURAL_POSITIONS,
)
from bootstrap.jet import (
    jet_derivative, total_derivative,
    _decompose_tensmul, _decompose_tensadd, _rebuild_tensmul,
    _get_component, _get_indices, _sum_terms,
)

kappa = Symbol('kappa')

# --- Tensor head for g^{-1} (the inverse metric, treated as a jet variable) ---
_sym2 = TensorSymmetry.fully_symmetric(2)
ginv = TensorHead('ginv', [Lorentz, Lorentz], _sym2)

# Register ginv in jet hierarchy so jet_derivative works on it
_JET_HIERARCHY[ginv] = {'n_field_indices': 2}

# --- Tensor head for g (downstairs metric: g_{mu nu}) ---
# Symmetric; treated as a jet variable distinct from ginv so that the Hilbert
# variation by g_{alpha beta} can be taken via jet_derivative.
g_down = TensorHead('g_down', [Lorentz, Lorentz], _sym2)
_JET_HIERARCHY[g_down] = {'n_field_indices': 2}

# --- Tensor head for dg (derivative of g_{mn}) ---
# dg_{mn,r} has symmetry: symmetric in first 2 (from g), no sym with 3rd
_sym_dg = TensorSymmetry.direct_product(2, 1)
dg = TensorHead('dg', [Lorentz, Lorentz, Lorentz], _sym_dg)
_JET_HIERARCHY[dg] = {'parent': None, 'n_field_indices': 2}


# ========================================================================
# Step 1: Uncontract metrics
# ========================================================================

def uncontract_metrics(expr):
    """Make implicit metric factors explicit.

    For each tensor factor, if an index is in the "wrong" position
    relative to its natural position, insert a metric factor to fix it.

    E.g., dh(a, b, -c) with natural=[down,down,down]:
    - a is UP but should be DOWN → insert metric(a, d) * dh(-d, b, -c)
    - b is UP but should be DOWN → insert metric(b, e) * dh(-d, -e, -c)
    - c is already DOWN → no change

    Returns expression with all fields having natural index positions
    and explicit metric(...) factors for each raising/lowering. Returns
    `expr` unchanged when no factor needs adjustment (common for
    already-natural products like dh(-a,-b,-c)).
    """
    if expr == S.Zero:
        return S.Zero
    if isinstance(expr, TensAdd):
        terms = [uncontract_metrics(t) for t in expr.args]
        if all(t is o for t, o in zip(terms, expr.args)):
            return expr
        return _sum_terms(terms)
    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)
        new_factors = []
        extra_metrics = []
        any_change = False
        for f in factors:
            nf, ms = _uncontract_factor(f)
            if ms or nf is not f:
                any_change = True
            new_factors.append(nf)
            extra_metrics.extend(ms)
        if not any_change:
            return expr
        return _rebuild_tensmul(coeff, extra_metrics + new_factors)
    if isinstance(expr, Tensor):
        nf, ms = _uncontract_factor(expr)
        if not ms and nf is expr:
            return expr
        if ms:
            result = S.One
            for m in ms:
                result = result * m
            return result * nf
        return nf
    return expr


def _uncontract_factor(factor):
    """Uncontract a single tensor factor.
    
    Returns (new_factor, [metric_factors]) where new_factor has all
    indices in their natural positions, and metric_factors are the 
    explicit metric tensors needed.
    """
    comp = _get_component(factor)
    if comp is None or comp not in NATURAL_POSITIONS:
        return factor, []
    
    natural = NATURAL_POSITIONS[comp]
    indices = _get_indices(factor)
    
    new_indices = list(indices)
    metrics = []
    
    for i, (idx, nat) in enumerate(zip(indices, natural)):
        if nat == 'down' and idx.is_up:
            # Index is UP but should be DOWN: insert metric to lower it
            d, = fresh_indices(1)
            metrics.append(metric(idx, d))  # η^{idx, d}
            new_indices[i] = -d  # put the lowered index on the tensor
        elif nat == 'up' and not idx.is_up:
            # Index is DOWN but should be UP: insert metric to raise it
            d, = fresh_indices(1)
            metrics.append(metric(idx, -d))  # η_{idx, d}
            new_indices[i] = d
    
    new_factor = comp(*new_indices)
    return new_factor, metrics


# ========================================================================
# Step 2: Replace metric → g (covariantize the metric factors)
# ========================================================================

def _classify_metric_factor(factor):
    """Classify a `metric(...)` factor by the index-sign pattern.

    Returns 'up_up' for both contravariant (eta^{mu nu}),
            'down_down' for both covariant (eta_{mu nu}),
            'mixed' for one of each (Kronecker delta).
    """
    indices = _get_indices(factor)
    if all(idx.is_up for idx in indices):
        return 'up_up'
    if all(not idx.is_up for idx in indices):
        return 'down_down'
    return 'mixed'


def replace_metric_with_ginv(expr):
    """Replace explicit metric factors with their jet-variable counterparts.

    Dispatch on the index-sign pattern of each metric factor:
      - eta^{mu nu} (both up)   → ginv(mu, nu)     == g^{mu nu}
      - eta_{mu nu} (both down) → g_down(mu, nu)   == g_{mu nu}
      - delta^mu_nu (mixed)     → leave unchanged (background Kronecker delta)

    The two ARE separate jet variables: they enter the Hilbert variation
    differently (delta g^{ab}/delta g_{mn} contributes via term_B with a
    minus-half sign and two raised contractions, while delta g_{ab}/delta
    g_{mn} contributes directly to term_D).
    """
    if expr == S.Zero:
        return S.Zero
    if isinstance(expr, TensAdd):
        terms = [replace_metric_with_ginv(t) for t in expr.args]
        if all(t is o for t, o in zip(terms, expr.args)):
            return expr
        return _sum_terms(terms)
    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)
        new_factors = []
        any_change = False
        for f in factors:
            if _get_component(f) == metric:
                kind = _classify_metric_factor(f)
                if kind == 'up_up' or kind == 'down_down':
                    indices = _get_indices(f)
                    new_factors.append(
                        ginv(*indices) if kind == 'up_up' else g_down(*indices)
                    )
                    any_change = True
                else:
                    new_factors.append(f)
            else:
                new_factors.append(f)
        if not any_change:
            return expr
        return _rebuild_tensmul(coeff, new_factors)
    if isinstance(expr, Tensor):
        if _get_component(expr) == metric:
            kind = _classify_metric_factor(expr)
            indices = _get_indices(expr)
            if kind == 'up_up':
                return ginv(*indices)
            if kind == 'down_down':
                return g_down(*indices)
        return expr
    return expr


# ========================================================================
# Step 3: Functional derivative δ(√|g| L̃)/δg_{μν}
# ========================================================================

def hilbert_energy_momentum(L):
    """Compute T_H^{μν}[L] via the Hilbert procedure.
    
    Args:
        L: a scalar Lagrangian (function of h, dh, and η)
        
    Returns:
        (T_mn, (mu, nu)): the Hilbert tensor and its free indices
    """
    mu, nu = fresh_indices(2)
    
    # Step 1: Uncontract metrics to make η factors explicit
    L_unc = uncontract_metrics(L)

    # Step 2: Replace η → ginv (covariantize metric factors)
    L_cov = replace_metric_with_ginv(L_unc)

    # Step 2.5: Contract spurious g_down × ginv pairs that arose from the
    # uncontract+replace chain (raise-then-lower or vice versa through a
    # metric). Mathematically these pairs are deltas and their variation by g
    # vanishes; carrying them through term_B (varies ginv) + term_D (varies
    # g_down) separately would split the variation into two pieces that
    # SHOULD cancel but may not in sympy's algebra. Collapsing here keeps
    # the variation clean. .expand() first so the pair-detection sees flat
    # TensMul terms (not TensMul-wrapped TensAdds — see the
    # project-decompose-tensmul-tensadd-pitfall memory).
    if isinstance(L_cov, TensExpr):
        L_cov = L_cov.expand()
    L_cov = _contract_ginv_g_down_pairs(L_cov)
    if isinstance(L_cov, TensExpr):
        L_cov = canon(L_cov)
    
    # Step 3: Functional derivative δ(√|g| L̃)/δg_{μν}
    # Three contributions:
    
    # --- A: From δ√|g|/δg_{μν} = ½ g^{μν} √|g| ---
    # At g=η: ½ η^{μν} L = ½ metric(μ,ν) L
    # But we need to evaluate L̃ at g=η too, which means ginv→metric.
    # Since L̃ with ginv→metric is just L (original), this gives:
    # Contribution A to T_H = 2 × ½ metric(μ,ν) L = metric(μ,ν) L
    term_A = metric(mu, nu) * L
    
    # --- B: From δg^{αβ}/δg_{μν} = -½(g^{αμ}g^{βν} + g^{αν}g^{βμ}) --- 
    # ∂L̃/∂g^{αβ} via jet_derivative by ginv:
    alpha, beta = fresh_indices(2)
    pi_ab = jet_derivative(L_cov, ginv, [alpha, beta])
    
    if pi_ab != S.Zero:
        # Multiply by δg^{αβ}/δg_{μν} = -½(g^{αμ}g^{βν} + g^{αν}g^{βμ})
        # At g=η: g^{αμ} = ginv(α,μ) → metric(α,μ)
        # So the contribution is: -½(π^{αβ})(metric(α,μ)metric(β,ν) + ...)
        # Replace ginv→metric in pi_ab first, then contract
        pi_ab_eta = _replace_ginv_with_metric(pi_ab)
        variation_B = -Rational(1, 2) * (
            pi_ab_eta * metric(-alpha, mu) * metric(-beta, nu)
            + pi_ab_eta * metric(-alpha, nu) * metric(-beta, mu)
        )
        # T_H contribution = 2 × variation_B
        term_B = 2 * canon(variation_B)
    else:
        term_B = S.Zero
    
    # --- D: From δg_{αβ}/δg_{μν} = ½(δ^μ_α δ^ν_β + δ^ν_α δ^μ_β) ---
    # The variation of an explicit g_{αβ} factor in L̃ is direct: each
    # g_down(-α, -β) factor contributes its surrounding expression with the
    # (α, β) slots replaced by (μ, ν). Since g_down is symmetric, the
    # jet derivative with wrt indices (μ, ν) already produces the symmetric
    # combination; multiplying by 2 (the prefactor of the Hilbert formula)
    # gives the contribution.
    pi_dn = jet_derivative(L_cov, g_down, [mu, nu])
    if pi_dn != S.Zero:
        pi_dn_eta = _replace_g_with_metric(pi_dn)
        term_D = 2 * canon(pi_dn_eta) if isinstance(pi_dn_eta, TensExpr) else 2 * pi_dn_eta
    else:
        term_D = S.Zero

    # --- C: From Christoffel variation (∇h = ∂h - Γh, Γ depends on dg) ---
    term_C = _christoffel_contribution(L, mu, nu)

    T_mn = term_A + term_B + term_C + term_D
    T_mn = canon(T_mn) if isinstance(T_mn, TensExpr) else T_mn

    return T_mn, (mu, nu)


def _replace_g_with_metric(expr):
    """Replace ginv(a,b) and g_down(a,b) factors with metric(a,b).

    This implements "setting g = η" after the functional derivative: both
    g^{mu nu} and g_{mu nu} reduce to the Minkowski metric on the background.
    """
    if expr == S.Zero:
        return S.Zero
    if isinstance(expr, TensAdd):
        terms = [_replace_g_with_metric(t) for t in expr.args]
        result = _sum_terms(terms)
        return canon(result) if isinstance(result, TensExpr) else result
    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)
        new_factors = []
        for f in factors:
            comp = _get_component(f)
            if comp is ginv or comp is g_down:
                new_factors.append(metric(*_get_indices(f)))
            else:
                new_factors.append(f)
        result = _rebuild_tensmul(coeff, new_factors)
        return canon(result) if isinstance(result, TensExpr) else result
    if isinstance(expr, Tensor):
        comp = _get_component(expr)
        if comp is ginv or comp is g_down:
            return metric(*_get_indices(expr))
        return expr
    return expr


# Backwards-compatible alias — older code may still call this name.
_replace_ginv_with_metric = _replace_g_with_metric


def _contract_ginv_g_down_pairs(expr):
    """Collapse g_down(...) × ginv(...) pairs that share a contracted dummy.

    g_down(-a, -b) × ginv(b, c) = δ^c_a  (one shared dummy → Kronecker delta).
    g_down(-a, -b) × ginv(a, b) = d      (two shared dummies → dimension).

    These pairs come from the uncontract+replace step when an index is
    raised then lowered (or vice versa). Mathematically the pair is a
    constant (its variation by g vanishes), but carrying it through
    term_B (varies ginv) + term_D (varies g_down) separately can leave a
    numerically non-cancelling residue. Collapsing here means term_B and
    term_D never see the spurious pair in the first place.

    Iterates until no more pairs can be contracted. Returns the
    simplified expression.
    """
    if expr == S.Zero:
        return S.Zero
    if isinstance(expr, TensAdd):
        return _sum_terms([_contract_ginv_g_down_pairs(t) for t in expr.args])
    if not isinstance(expr, TensMul):
        return expr

    changed = True
    cur = expr
    # Loop until no pair can be contracted (each pass collapses one pair).
    while changed and isinstance(cur, TensMul):
        changed = False
        coeff, factors = _decompose_tensmul(cur)
        ginv_positions = [i for i, f in enumerate(factors) if _get_component(f) is ginv]
        gdown_positions = [i for i, f in enumerate(factors) if _get_component(f) is g_down]
        for i_inv in ginv_positions:
            f_inv = factors[i_inv]
            inv_inds = _get_indices(f_inv)  # both UP
            for i_dn in gdown_positions:
                if i_dn == i_inv:
                    continue
                f_dn = factors[i_dn]
                dn_inds = _get_indices(f_dn)  # both DOWN
                # Look for shared dummies: UP-in-ginv matches DOWN-in-gdown by name.
                for ii, inv_idx in enumerate(inv_inds):
                    for di, dn_idx in enumerate(dn_inds):
                        if inv_idx == -dn_idx:
                            # Found a shared dummy. Build replacement.
                            inv_other = inv_inds[1 - ii]    # UP, free now
                            dn_other = dn_inds[1 - di]      # DOWN, free now
                            # Replace the pair with metric(inv_other, dn_other)
                            # which is a Kronecker delta δ^{inv_other}_{−dn_other}.
                            new_factor = metric(inv_other, dn_other)
                            new_factors = [
                                f for k, f in enumerate(factors)
                                if k not in (i_inv, i_dn)
                            ]
                            new_factors.append(new_factor)
                            cur = _rebuild_tensmul(coeff, new_factors)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
    return cur


def _christoffel_contribution(L, mu, nu):
    """Christoffel chain-rule contribution to T_H from the dV → ∇V and
    dh → ∇h substitutions implicit in covariantization. Delegates to
    _christoffel_via_substitution, which handles both dh (always) and dV
    for upstairs-vector matter fields (per registration).
    """
    return _christoffel_via_substitution(L, mu, nu)


def _christoffel_via_substitution(L, mu, nu):
    """Compute Christoffel contribution by building ∇h (and ∇V for upstairs
    vector matter) and using jet_derivative.

    Strategy:
    1. Build ∇_γ h_{αβ} = dh_{αβγ} - Γ^λ_{γα} h_{λβ} - Γ^λ_{γβ} h_{αλ}
       where Γ^λ_{ρσ} = ½ metric^{λτ}(dg_{στρ} + dg_{ρτσ} - dg_{ρστ}).
       For each registered upstairs vector V also build:
         ∇_ρ V^σ = dV(σ, -ρ) + Γ^σ_{ρτ} V^τ.
    2. Substitute these covariant derivatives in L to get the O(dg)
       correction δL_Γ (sum of an h-piece and one V-piece per upstairs
       vector field).
    3. Take jet_derivative of δL_Γ by dg to get Q = ∂L̃/∂(dg_{μν,ρ}).
    4. Compute -2 total_derivative(Q, -ρ) and set dg→0.
    """
    from bootstrap.tensor_algebra import _matter_fields

    # --- (a) dh contribution: ∂L/∂(dh) × (∇h - dh) ---
    alpha, beta, gamma, lam, tau = fresh_indices(5)
    pi = jet_derivative(L, dh, [alpha, beta, gamma])

    delta_L = S.Zero
    if pi != S.Zero:
        # Γ^λ_{γα} written via dg (Christoffel evaluated at g=η).
        Gamma_ga = Rational(1, 2) * metric(lam, tau) * (
            dg(-alpha, -tau, -gamma) + dg(-gamma, -tau, -alpha) - dg(-gamma, -alpha, -tau)
        )
        lam2, tau2 = fresh_indices(2)
        Gamma_gb = Rational(1, 2) * metric(lam2, tau2) * (
            dg(-beta, -tau2, -gamma) + dg(-gamma, -tau2, -beta) - dg(-gamma, -beta, -tau2)
        )
        # δL_Γ = π^{αβγ} × (-Γ^λ_{γα} h(-λ,-β) - Γ^{λ2}_{γβ} h(-α,-λ2)).
        delta_L_h = pi * (-Gamma_ga * h(-lam, -beta) - Gamma_gb * h(-alpha, -lam2))
        delta_L = delta_L + delta_L_h

    # --- (b) Upstairs-vector contributions ---
    # For each registered upstairs-vector matter field V (index_pos == 'up',
    # rank 1) the covariantization of L promotes ∂_ρ V^σ to
    #   ∇_ρ V^σ = dV(σ, -ρ) + Γ^σ_{ρτ} V^τ
    # Chain-rule contribution at g=η (after dropping the literal dV piece,
    # which is the order-zero of the substitution and doesn't depend on dg):
    #   δL_V = (∂L/∂dV) × (Γ^σ_{ρτ} V^τ).
    # We build it with index conventions chosen so the pi_V indices give
    # clean Kronecker deltas: pass the wrt_indices to jet_derivative with the
    # OPPOSITE sign of the natural positions of dV (which are [up, down]).
    # That means pass [-alpha, beta] so pi_V emerges with (alpha DOWN, beta UP),
    # which then contracts cleanly with the natural Γ correction structure
    # (σ UP, ρ DOWN, τ contracted) below.
    for _name, info in _matter_fields.items():
        if info.get('index_pos') != 'up' or info.get('rank') != 1:
            continue
        V_head = info['field']
        dV_head = info['dfield']

        a_V, b_V = fresh_indices(2)  # both UP by construction
        pi_V = jet_derivative(L, dV_head, [-a_V, b_V])
        if pi_V == S.Zero:
            continue

        lam_V, tau_V = fresh_indices(2)
        # Γ^{a_V}_{b_V, τ_V} = ½ η^{a_V, λ}(dg_{b_V λ τ} + dg_{τ λ b_V} - dg_{b_V τ λ})
        Gamma_V = Rational(1, 2) * metric(a_V, lam_V) * (
            dg(-b_V, -lam_V, -tau_V)
            + dg(-tau_V, -lam_V, -b_V)
            - dg(-b_V, -tau_V, -lam_V)
        )
        # δL_V = π_V × Γ_V × V^τ — fully contracted scalar contribution.
        delta_L_V = pi_V * Gamma_V * V_head(tau_V)
        delta_L = delta_L + delta_L_V

#USER CODE: add covariant derivatives for A
    for _name, info in _matter_fields.items():
        if info.get('index_pos') != 'down' or info.get('rank') != 1:
            continue
        A_head = info['field']
        dA_head = info['dfield']

        a_A, b_A = fresh_indices(2)  # both UP by construction
        pi_A = jet_derivative(L, dA_head, [a_A, b_A])
        if pi_A == S.Zero:
            continue

        lam_A, tau_A = fresh_indices(2)
        # Γ^{\tau_A}_{a_A, b_A} = ½ η^{\tau_A, λ}(dg_{b_A λ a_A} + dg_{a_A λ b_A} - dg_{b_A a_A λ})
        Gamma_A = Rational(1, 2) * metric(tau_A, lam_A) * (
            dg(-b_A, -lam_A, -a_A)
            + dg(-a_A, -lam_A, -b_A)
            - dg(-b_A, -a_A, -lam_A)
        )
        # δL_A = -π_A × Γ_A × A_τ — fully contracted scalar contribution.
        delta_L_A = -pi_A * Gamma_A * A_head(-tau_A)
        delta_L = delta_L + delta_L_A
#END USER CODE

    if delta_L == S.Zero:
        return S.Zero

    delta_L = canon(delta_L)
    if delta_L == S.Zero:
        return S.Zero

    # Q_{μν}^{ρ} = ∂(δL_Γ)/∂(dg_{μν,ρ}). Since δL_Γ is linear in dg, the
    # jet_derivative removes dg entirely.
    rho, = fresh_indices(1)
    Q = jet_derivative(delta_L, dg, [mu, nu, rho])
    if Q == S.Zero:
        return S.Zero

    # T_C = -2 ∂_ρ Q^{ρ}_{μν}.
    dQ = total_derivative(Q, -rho)
    if dQ == S.Zero:
        return S.Zero

    result = -2 * dQ
    return canon(result) if isinstance(result, TensExpr) else result
