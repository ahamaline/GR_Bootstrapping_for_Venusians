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
    fresh_indices, canon, _JET_HIERARCHY
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

# --- Tensor head for dg (derivative of g_{mn}) ---
# dg_{mn,r} has symmetry: symmetric in first 2 (from g), no sym with 3rd
#USER COMMENT: I think you forgot to implement this symmetry in the jet derivative
_sym_dg = TensorSymmetry.direct_product(2, 1)
dg = TensorHead('dg', [Lorentz, Lorentz, Lorentz], _sym_dg)
_JET_HIERARCHY[dg] = {'parent': None, 'n_field_indices': 2}

# Natural index positions for each tensor head:
# 'down' means the index is naturally covariant (lower)
# 'up' means naturally contravariant (upper)
NATURAL_POSITIONS = {
    h: ['down', 'down'],
    dh: ['down', 'down', 'down'],
    ddh: ['down', 'down', 'down', 'down'],
    # ginv is naturally up (it's g^{-1}), but we don't uncontract it
    # matter fields will be added when registered
}


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
    and explicit metric(...) factors for each raising/lowering.
    """
    if expr == S.Zero:
        return S.Zero
    if isinstance(expr, TensAdd):
        terms = [uncontract_metrics(t) for t in expr.args]
        return _sum_terms(terms)
    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)
        new_factors = []
        extra_metrics = []
        for f in factors:
            nf, ms = _uncontract_factor(f)
            new_factors.append(nf)
            extra_metrics.extend(ms)
        result = _rebuild_tensmul(coeff, extra_metrics + new_factors)
        return result
    if isinstance(expr, Tensor):
        nf, ms = _uncontract_factor(expr)
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
# Step 2: Replace metric → ginv (covariantize the metric factors)
# ========================================================================

def replace_metric_with_ginv(expr):
    """Replace every explicit metric(a,b) factor with ginv(a,b).
    
    This is the "covariantization" of the metric factors: η^{ab} → g^{ab}.
    
    NOTE: This only replaces metric factors that appear as standalone 
    Tensor objects in TensMul products. It does NOT affect metrics that
    are part of index contractions (which have already been un-contracted).
    USER COMMENT: We also need to replace the metric (with downstairs indices) by g. Not relevant for L_h^{(2)} because all the indices are naturally downstairs (so g is never used) but it is relevant for the general case.
    """
    if expr == S.Zero:
        return S.Zero
    if isinstance(expr, TensAdd):
        terms = [replace_metric_with_ginv(t) for t in expr.args]
        return _sum_terms(terms)
    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)
        new_factors = []
        for f in factors:
            if _get_component(f) == metric:
                indices = _get_indices(f)
                new_factors.append(ginv(*indices))
            else:
                new_factors.append(f)
        return _rebuild_tensmul(coeff, new_factors)
    if isinstance(expr, Tensor):
        if _get_component(expr) == metric:
            return ginv(*_get_indices(expr))
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
    
    # Step 3: Functional derivative δ(√|g| L̃)/δg_{μν}
    # Three contributions:
    
    # --- A: From δ√|g|/δg_{μν} = ½ g^{μν} √|g| ---
    # At g=η: ½ η^{μν} L = ½ metric(μ,ν) L
    # But we need to evaluate L̃ at g=η too, which means ginv→metric.
    # Since L̃ with ginv→metric is just L (original), this gives:
    # Contribution A to T_H = 2 × ½ metric(μ,ν) L = metric(μ,ν) L
    term_A = metric(mu, nu) * L
    
    # --- B: From δg^{αβ}/δg_{μν} = -½(g^{αμ}g^{βν} + g^{αν}g^{βμ}) ---
    # USER COMMENT: If we have a downstairs g, we need to take the jet derivative by that too 
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
    
    # --- C: From Christoffel variation (∇h = ∂h - Γh, Γ depends on dg) ---
    term_C = _christoffel_contribution(L, mu, nu)
    
    T_mn = term_A + term_B + term_C
    T_mn = canon(T_mn) if isinstance(T_mn, TensExpr) else T_mn
    
    return T_mn, (mu, nu)


def _replace_ginv_with_metric(expr):
    """Replace all ginv(a,b) → metric(a,b) in an expression.
    
    This implements "setting g = η" after the functional derivative.
    """
    if expr == S.Zero:
        return S.Zero
    if isinstance(expr, TensAdd):
        terms = [_replace_ginv_with_metric(t) for t in expr.args]
        result = _sum_terms(terms)
        return canon(result) if isinstance(result, TensExpr) else result
    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)
        new_factors = []
        for f in factors:
            if _get_component(f) == ginv:
                new_factors.append(metric(*_get_indices(f)))
            else:
                new_factors.append(f)
        result = _rebuild_tensmul(coeff, new_factors)
        return canon(result) if isinstance(result, TensExpr) else result
    if isinstance(expr, Tensor):
        if _get_component(expr) == ginv:
            return metric(*_get_indices(expr))
        return expr
    return expr


def _christoffel_contribution(L, mu, nu):
    """Compute the Christoffel contribution to T_H.
    
    When covariantizing, ∂_γ h_{αβ} → ∇_γ h_{αβ} = dh_{αβγ} - Γ^λ_{γα} h_{λβ} - Γ^λ_{γβ} h_{αλ}
    
    At g=η, Γ=0 so ∇h=dh. But δΓ/δg ≠ 0, contributing to T_H.
    
    The contribution is:
    T_C^{μν} = 2 × [-∂_ρ(∂L̃/∂(dg_{μν,ρ}))]|_{g=η, dg=0}
    
    where ∂L̃/∂(dg_{μν,ρ}) comes from the chain rule through Γ:
    ∂L̃/∂(dg_{μν,ρ}) = ∂L/∂(dh_{αβγ}) × ∂(∇_γ h_{αβ})/∂(dg_{μν,ρ})
    
    At g=η, ∂L/∂(dh_{αβγ}) is just the jet derivative π^{αβγ}.
    And ∂(∇_γ h_{αβ})/∂(dg_{μν,ρ}) = -(∂Γ^λ_{γα}/∂(dg_{μν,ρ})) h_{λβ} - (α↔β)
    
    This is computed explicitly using the Christoffel formula.
    """
    # Compute π^{αβγ} = ∂L/∂(dh_{αβγ})
    alpha, beta, gamma = fresh_indices(3)
    pi = jet_derivative(L, dh, [alpha, beta, gamma])
    
    if pi == S.Zero:
        return S.Zero
    
    # The Christoffel symbol at g=η is:
    # Γ^λ_{ρσ} = ½ η^{λτ}(dg_{στ,ρ} + dg_{ρτ,σ} - dg_{ρσ,τ})
    #
    # ∂Γ^λ_{γα}/∂(dg_{μν,ρ}):
    # We need the derivative of each dg_{??,?} term w.r.t. dg_{μν,ρ}.
    # Since dg_{ab,c} is symmetric in (a,b):
    # ∂(dg_{ab,c})/∂(dg_{μν,ρ}) = ½(δ^μ_a δ^ν_b + δ^ν_a δ^μ_b) δ^ρ_c
    #
    # So ∂Γ^λ_{γα}/∂(dg_{μν,ρ}) = ½ η^{λτ} × [
    #   ½(δ^μ_α δ^ν_τ + δ^ν_α δ^μ_τ) δ^ρ_γ    (from dg_{ατ,γ})
    # + ½(δ^μ_γ δ^ν_τ + δ^ν_γ δ^μ_τ) δ^ρ_α    (from dg_{γτ,α})
    # - ½(δ^μ_γ δ^ν_α + δ^ν_γ δ^μ_α) δ^ρ_τ    (from dg_{γα,τ})
    # ]
    #
    # After contracting η^{λτ} with the τ-deltas:
    # = ¼ [(δ^μ_α η^{λν} + δ^ν_α η^{λμ}) δ^ρ_γ
    #    + (δ^μ_γ η^{λν} + δ^ν_γ η^{λμ}) δ^ρ_α
    #    - (δ^μ_γ δ^ν_α + δ^ν_γ δ^μ_α) η^{λρ}]
    #
    # The contribution to ∂L̃/∂(dg_{μν,ρ}) from ∇_γ h_{αβ}:
    # = -π^{αβγ} × [∂Γ^λ_{γα}/∂(dg_{μν,ρ}) × h_{λβ} + (α↔β)]
    #
    # Then T_C = 2 × [-∂_ρ(that)]|_{dg=0}
    #
    # Since the expression is already evaluated at dg=0 and g=η,
    # the ∂_ρ acts only on π and h (not on anything dg-related).
    #
    # Rather than expanding all the delta products, let me compute this
    # by constructing the coefficient of dg_{μν,ρ} explicitly.
    #
    # Define Q^{ρ}_{μν} = ∂L̃/∂(dg_{μν,ρ}) at g=η
    # From the formula above:
    # Q^{ρ}_{μν} = -π^{αβγ} [∂Γ^λ_{γα}/∂(dg_{μν,ρ})] h(-λ,-β) - (α↔β)
    #
    # After substituting and contracting deltas:
    # From the 3 sub-terms of ∂Γ, and the two h-contractions (λβ and αλ):
    
    lam = fresh_indices(1)[0]
    
    # I'll compute this term by term. Each sub-term of ∂Γ gives a 
    # specific index routing. Let me use the contracted result directly.
    #
    # Sub-term 1 (from dg_{ατ,γ}): δ^ρ_γ factor means ρ = γ.
    # ∂Γ^λ_{γα}|_1 = ¼(δ^μ_α η^{λν} + δ^ν_α η^{λμ}) δ^ρ_γ
    # Contracted with -π^{αβγ} h(-λ,-β):
    # -¼ π^{αβρ} (δ^μ_α h^ν_β + δ^ν_α h^μ_β)  [γ→ρ, then contract λ]
    # = -¼ (π^{μβρ} h(nu,-beta) + π^{νβρ} h(mu,-beta))
    #
    # Similarly for the h(-α,-λ) term: 
    # -¼ π^{αβρ} (h(-alpha,nu) δ^μ_... )... 
    # Actually since π is symmetric in α,β, the h(-α,-λ) term is the 
    # same as h(-λ,-β) with α↔β, giving the same result. So we double.
    
    rho = fresh_indices(1)[0]
    b2 = fresh_indices(1)[0]
    
    # Sub-term 1: coefficient of dg_{μν,ρ} from dg_{ατ,γ} piece
    # = -½ π^{μ,b2,ρ} h(ν,-b2)  +  (μ↔ν)
    # Factor ½ = 2 × ¼ (doubled from α↔β symmetry)
    st1 = -Rational(1,2) * (
        jet_derivative(L, dh, [mu, b2, rho]) * h(nu, -b2)
        + jet_derivative(L, dh, [nu, b2, rho]) * h(mu, -b2)
    )
    
    # Sub-term 2 (from dg_{γτ,α}): δ^ρ_α factor means ρ = α.
    b3 = fresh_indices(1)[0]
    g2 = fresh_indices(1)[0]
    st2 = -Rational(1,2) * (
        jet_derivative(L, dh, [rho, b3, g2]) * h(nu, -b3) * metric(mu, -g2)
        + jet_derivative(L, dh, [rho, b3, g2]) * h(mu, -b3) * metric(nu, -g2)
    )
    # Wait, I need γ free here. Let me redo.
    # δ^ρ_α means we set α=ρ in π^{αβγ}. Then η^{λν} contracts with h_{λβ}→h^ν_β.
    # And γ stays free (it becomes the index we'll differentiate by in -∂_ρ).
    # But wait, ∂_ρ acts on dg_{μν,ρ}, so ρ is the derivative index.
    # The Q function already has ρ as a free index.
    # Let me reconsider the structure.
    
    # Actually, the full computation is:
    # T_C^{μν} = -2 ∂_ρ Q^ρ_{μν}  where Q = ∂L̃/∂(dg_{μν,ρ})
    # But Q is a function of the dynamical fields (h, dh) only (since g=η, dg=0).
    # So ∂_ρ Q means total_derivative(Q, -ρ).
    
    # Let me compute Q directly. I'll build it from the full formula.
    # Due to the complexity, let me use a more systematic approach:
    # express Γ in terms of dg, substitute into ∇h, expand L̃, and 
    # take jet_derivative by dg.
    
    return _christoffel_via_substitution(L, mu, nu)


def _christoffel_via_substitution(L, mu, nu):
    """Compute Christoffel contribution by building ∇h and using jet_derivative.
    
    Strategy:
    1. Build ∇_γ h_{αβ} = dh_{αβγ} - Γ^λ_{γα} h_{λβ} - Γ^λ_{γβ} h_{αλ}
       where Γ^λ_{ρσ} = ½ metric^{λτ}(dg_{στρ} + dg_{ρτσ} - dg_{ρστ})
    2. Substitute ∇h for dh in L to get the O(dg) correction δL_Γ
    3. Take jet_derivative of δL_Γ by dg to get Q = ∂L̃/∂(dg_{μν,ρ})
    4. Compute -2 total_derivative(Q, -ρ) and set dg→0
    """
    # The correction to L from covariantizing derivatives is:
    # δL_Γ = ∂L/∂(dh_{αβγ}) × (∇_γ h_{αβ} - dh_{αβγ})
    #       = ∂L/∂(dh_{αβγ}) × (-Γ^λ_{γα} h_{λβ} - Γ^λ_{γβ} h_{αλ})
    # evaluated at g=η (so ginv factors in Γ become metric).
    
    alpha, beta, gamma, lam, tau = fresh_indices(5)
    
    pi = jet_derivative(L, dh, [alpha, beta, gamma])
    if pi == S.Zero:
        return S.Zero
    
    # Γ^λ_{γα} = ½ g^{λτ}(g_{ατ,γ} + g_{γτ,α} - g_{γα,τ})
    # At g=η: g^{λτ} = metric(λ,τ), and g_{ab,c} = dg_{ab,c} with all
    # three indices covariant. The inverse metric (ginv) also depends on g,
    # but at g=η where dg=0, the variation of ginv×(dg terms) vanishes
    # because it's second-order in the deviation from η. So only the 
    # dg-variation contributes to δΓ/δg at g=η.
    Gamma_ga = Rational(1,2) * metric(lam, tau) * (
        dg(-alpha, -tau, -gamma) + dg(-gamma, -tau, -alpha) - dg(-gamma, -alpha, -tau)
    )
    
    lam2, tau2 = fresh_indices(2)
    Gamma_gb = Rational(1,2) * metric(lam2, tau2) * (
        dg(-beta, -tau2, -gamma) + dg(-gamma, -tau2, -beta) - dg(-gamma, -beta, -tau2)
    )
    
    # δL_Γ = π^{αβγ} × (-Γ^λ_{γα} h(-λ,-β) - Γ^{λ2}_{γβ} h(-α,-λ2))
    delta_L = pi * (-Gamma_ga * h(-lam, -beta) - Gamma_gb * h(-alpha, -lam2))
    delta_L = canon(delta_L)
    
    if delta_L == S.Zero:
        return S.Zero
    
    # Now take jet_derivative of δL_Γ by dg with indices (μ, ν, ρ)
    # dg has 3 indices: dg_{μν,ρ} where μν are field indices and ρ is the
    # derivative index (the third/last index of the dg TensorHead).
    # This gives Q_{μν}^{ρ} = ∂(δL_Γ)/∂(dg_{μν,ρ})
    rho = fresh_indices(1)[0]
    Q = jet_derivative(delta_L, dg, [mu, nu, rho])
    
    if Q == S.Zero:
        return S.Zero
    
    # T_C = -2 ∂_ρ Q^{ρ}_{μν}
    # But wait: Q still contains dg factors (from the remaining dg in δL_Γ
    # after one jet derivative). At dg=0, those terms vanish.
    # Since δL_Γ is LINEAR in dg (Γ is linear in dg), the jet_derivative 
    # by dg removes the dg entirely. So Q should be dg-free.
    
    dQ = total_derivative(Q, -rho)
    
    if dQ == S.Zero:
        return S.Zero
    
    result = -2 * dQ
    return canon(result) if isinstance(result, TensExpr) else result
