"""
Helmholtz conditions and the formulas derived from them.

This module implements the key formulas from Section 3 of the paper:

1. Superpotential formula (eq. 22): for n >= 2, determines the unique 
   identically conserved tensor Psi^{mu nu rho sigma} that must be added
   to make the field equation satisfy the third Helmholtz condition.

2. Superpotential formula (eq. 23): for n = 1, an integral formula for
   Psi when matter fields are present.

3. EOM term determination: from the second Helmholtz condition, determines
   what equation-of-motion terms must be added at each order.

4. Psi symmetry verification: checks that the computed superpotential has
   the required symmetries (symmetric in first pair, last pair, and 
   cyclic antisymmetry).
"""

from sympy import S, Rational, Symbol, integrate
from sympy.tensor.tensor import TensAdd, TensMul, TensExpr, Tensor
from bootstrap.tensor_algebra import (
    h, dh, ddh,
    fresh_indices, canon, _matter_fields,
    NATURAL_POSITIONS, swap_free_indices,
)
from bootstrap.jet import (
    jet_derivative, _sum_terms, _decompose_tensmul,
    _get_component,
)


# ---------------------------------------------------------------------------
# Superpotential formula for n >= 2 (eq. 22 of the paper)
# ---------------------------------------------------------------------------

def compute_superpotential_n2(M_expr, n, M_indices):
    """Compute the superpotential Psi^{mu nu rho sigma} for order n >= 2.

    Given M^{mu nu} (the "unimproved" field equation contribution at order n,
    possibly after EOM corrections), computes:

    Psi^{mu nu rho sigma} = 1/(n(n-1)) * h_{alpha beta} * h_{kappa lambda} *
        (d^2 M^{alpha beta} / (dh_{kappa lambda} dh_{mu nu, rho sigma})
         - 1/2 d^2 M^{alpha beta} / (dh_{kappa lambda, sigma} dh_{mu nu, rho})
         - 1/2 d^2 M^{mu nu} / (dh_{kappa lambda, sigma} dh_{alpha beta, rho}))

    where all derivatives are jet derivatives.

    Args:
        M_expr: tensor expression for M^{mu nu} at order n in h.
        n: the order in h (must be >= 2)
        M_indices: tuple (mu_M, nu_M) of M_expr's two free indices.
            Terms A,B use M with free indices (alpha,beta); Term C uses M
            with free indices (mu_f,nu_f). This argument lets us relabel.

    Returns:
        (Psi, psi_indices):
            Psi is the tensor expression for the superpotential
            psi_indices is (mu, nu, rho, sigma) — the four free indices
    """
    if n < 2:
        raise ValueError(f"This formula requires n >= 2, got n={n}")

    mu_M, nu_M = M_indices

    # Generate fresh indices for the formula
    mu_f, nu_f, rho_f, sigma_f = fresh_indices(4)  # free indices of Psi
    alpha, beta = fresh_indices(2)  # contracted with first h
    kappa, lam = fresh_indices(2)   # contracted with second h

    # M^{alpha beta}: relabel M's free indices (mu_M, nu_M) -> (alpha, beta).
    M_ab = M_expr.substitute_indices(
        (mu_M, alpha), (-mu_M, -alpha), (nu_M, beta), (-nu_M, -beta)
    )
    # M^{mu_f nu_f}: relabel M's free indices (mu_M, nu_M) -> (mu_f, nu_f).
    M_mn = M_expr.substitute_indices(
        (mu_M, mu_f), (-mu_M, -mu_f), (nu_M, nu_f), (-nu_M, -nu_f)
    )

    # Term A: d^2 M^{ab} / (dh_{kl} dh_{mn,rs})
    dM_dddh = jet_derivative(M_ab, ddh, [mu_f, nu_f, rho_f, sigma_f])
    d2M_A = jet_derivative(dM_dddh, h, [kappa, lam])

    # Term B: -1/2 d^2 M^{ab} / (dh_{kl,s} dh_{mn,r})
    dM_ddh_B = jet_derivative(M_ab, dh, [mu_f, nu_f, rho_f])
    d2M_B = jet_derivative(dM_ddh_B, dh, [kappa, lam, sigma_f])

    # Term C: -1/2 d^2 M^{mn} / (dh_{kl,s} dh_{ab,r})
    dM_ddh_C = jet_derivative(M_mn, dh, [alpha, beta, rho_f])
    d2M_C = jet_derivative(dM_ddh_C, dh, [kappa, lam, sigma_f])

    # Combine the three uncontracted terms first (they all share the same
    # 8 free-index set: alpha, beta, kappa, lam, mu_f, nu_f, rho_f, sigma_f),
    # then contract with h h and canonicalize once. Previously we canon'd
    # each of A, B, C in a `_wrap` helper, then canon'd the prefactor-product
    # at the end — 4 canon calls instead of 1. canon scales super-linearly
    # in the term count, so one big canon beats several medium-sized ones.
    prefactor = Rational(1, n * (n - 1))

    bracket = d2M_A
    if d2M_B != S.Zero:
        bracket = bracket - Rational(1, 2) * d2M_B
    if d2M_C != S.Zero:
        bracket = bracket - Rational(1, 2) * d2M_C

    if bracket == S.Zero:
        Psi = S.Zero
    else:
        Psi = canon(prefactor * h(-alpha, -beta) * h(-kappa, -lam) * bracket)

    psi_indices = (mu_f, nu_f, rho_f, sigma_f)
    return Psi, psi_indices


# ---------------------------------------------------------------------------
# Superpotential formula for n = 1 (paper eq. 23, integral form)
# ---------------------------------------------------------------------------

def compute_superpotential_n1(M_expr, M_indices, matter_field_heads=None):
    """Compute Psi^{mu nu rho sigma (1)} via the integral formula (paper eq. 23).

    For n=1, the second derivative by h_{kappa lambda} (used in PsiForm at
    n>=2) is replaced by a derivative w.r.t. each matter field, and the
    pure-multiplication by h_{alpha beta} is replaced by multiplication by
    the matter field. The result is integrated over the linear path
    phi'_i(lambda) = lambda * phi_i, lambda in [0, 1].

    Concretely:
        Psi^{mu nu rho sigma (1)} = sum_i integral_0^1 dlambda phi_i h_{alpha beta} (
            d^2 M'^{alpha beta} / (d phi'_i  d h_{mu nu, rho sigma})
          - (1/2) d^2 M'^{mu nu} / (d phi'_{i,sigma} d h_{alpha beta, rho})
          - (1/2) d^2 M'^{alpha beta} / (d phi'_{i,sigma} d h_{mu nu, rho})
        )

    where M' = M with phi -> lambda * phi for every matter field, applied
    *after* the partial derivatives are taken. For polynomial matter (per
    the dev-status note), this lambda integral is just
    integral_0^1 lambda^d dlambda = 1/(d+1) for each power d.

    Args:
        M_expr: tensor expression for M^{mu nu} at order n=1.
        M_indices: (mu_M, nu_M) tuple of M_expr's free indices.
        matter_field_heads: dict {name: (field, dfield, ddfield)}. If None,
            uses the global _matter_fields registry. Handles BOTH scalar
            and non-scalar (e.g. vector) matter — for a rank-r field A_α₁…α_r,
            the formula naturally generalizes: the derivative ∂/∂A^α₁…α_r
            returns those indices as free, and the "velocity" factor A_α₁…α_r
            (indices lowered) contracts them.

    Returns:
        (Psi, psi_indices) where psi_indices = (mu_f, nu_f, rho_f, sigma_f).
        Returns (S.Zero, None) if there are no matter fields.
    """
    if matter_field_heads is None:
        matter_field_heads = {}
        for name, info in _matter_fields.items():
            matter_field_heads[name] = (info['field'], info['dfield'], info['ddfield'])

    if not matter_field_heads:
        return S.Zero, None

    if M_expr == S.Zero:
        return S.Zero, None

    mu_M, nu_M = M_indices
    mu_f, nu_f, rho_f, sigma_f = fresh_indices(4)
    alpha, beta = fresh_indices(2)
    lam = Symbol('lambda', positive=True)

    M_ab = M_expr.substitute_indices(
        (mu_M, alpha), (-mu_M, -alpha), (nu_M, beta), (-nu_M, -beta)
    )
    M_mn = M_expr.substitute_indices(
        (mu_M, mu_f), (-mu_M, -mu_f), (nu_M, nu_f), (-nu_M, -nu_f)
    )

    matter_heads = set()
    for (field_h, dfield_h, ddfield_h) in matter_field_heads.values():
        matter_heads.update([field_h, dfield_h, ddfield_h])

    bracket_terms = []
    for name, (field_h, dfield_h, ddfield_h) in matter_field_heads.items():
        # Field's own tensor indices (empty for scalar, one for vector, etc.).
        n_field_idx = _matter_fields[name].get('rank', 0)

        # For jet_derivative to produce clean Kronecker deltas (rather than
        # raising η factors), the wrt_indices must have the OPPOSITE sign of
        # the field's natural index positions. Downstairs A (natural DOWN):
        # pass UP. Upstairs V (natural UP): pass DOWN. The velocity factor
        # below uses the opposite sign again, putting V back into its natural
        # form (V^α for upstairs V, A_α for downstairs A) so the contraction
        # with the jet result is clean.
        field_naturals = NATURAL_POSITIONS.get(field_h, ['down'] * n_field_idx)
        if n_field_idx > 0:
            raw_field_indices = list(fresh_indices(n_field_idx))
            field_indices = [(-r if nat == 'up' else r)
                             for r, nat in zip(raw_field_indices, field_naturals)]
        else:
            field_indices = []

        # Term A: d^2 M^{alpha beta} / (d field^{α…}  d h_{mu nu, rho sigma})
        dM_dddh = jet_derivative(M_ab, ddh, [mu_f, nu_f, rho_f, sigma_f])
        d2M_A = jet_derivative(dM_dddh, field_h, field_indices)

        # Term B: d^2 M^{mu nu} / (d (dfield)^{α…,sigma}  d h_{alpha beta, rho})
        dM_dh_B = jet_derivative(M_mn, dh, [alpha, beta, rho_f])
        d2M_B = jet_derivative(dM_dh_B, dfield_h, field_indices + [sigma_f])

        # Term C: d^2 M^{alpha beta} / (d (dfield)^{α…,sigma}  d h_{mu nu, rho})
        dM_dh_C = jet_derivative(M_ab, dh, [mu_f, nu_f, rho_f])
        d2M_C = jet_derivative(dM_dh_C, dfield_h, field_indices + [sigma_f])

        bracket = d2M_A - Rational(1, 2) * d2M_B - Rational(1, 2) * d2M_C
        if bracket == S.Zero:
            continue

        # Distribute Tensor * TensAdd here too: sympy will sometimes leave the
        # bracket as a TensAdd whose top-level args are TensMul(Rational, TensAdd)
        # — i.e. the Rational coefficient never distributed across the inner
        # TensAdd. _scale_matter_fields uses _decompose_tensmul which collects
        # only Tensor/TensMul args, silently dropping the TensAdd. That hides
        # every matter-field factor inside it from the lambda-count, giving the
        # wrong integral coefficient and a Ψ that violates cyclic symmetry.
        if isinstance(bracket, TensExpr):
            bracket = bracket.expand()

        # Apply path substitution phi' = lambda phi to all matter fields in
        # bracket (paper: the derivatives are evaluated at phi' = lambda*phi).
        scaled = _scale_matter_fields(bracket, lam, matter_heads)

        # Velocity factor — dphi'_i/dlambda for the linear path phi'=lambda*phi
        # is just phi_i itself (NO lambda, despite the bracket's lambdas).
        # field_h is invoked with indices OPPOSITE in sign to field_indices, so
        # the result has indices in the field's natural positions — and the
        # contraction with the bracket's anti-natural free indices is a clean
        # delta-pair (one up vs one down on every paired slot).
        if n_field_idx == 0:
            velocity = field_h()
        else:
            velocity = field_h(*[-i for i in field_indices])

        contracted = velocity * h(-alpha, -beta) * scaled
        # Distribute Tensor * TensAdd -> TensAdd of TensMuls. Without this,
        # sympy can leave `contracted` as a TensMul whose args include the
        # bracket TensAdd as a single factor; _decompose_tensmul (which
        # collects only Tensor/TensMul args) would then silently drop the
        # TensAdd and lose every free index it carried, collapsing Psi to a
        # 3-index expression with phantom dummies.
        if isinstance(contracted, TensExpr):
            contracted = contracted.expand()

        # Integrate lambda over [0, 1] term-by-term on the coefficient.
        integrated = _integrate_lambda(contracted, lam)
        if integrated == S.Zero:
            continue
        bracket_terms.append(integrated)

    Psi = _sum_terms(bracket_terms)
    if isinstance(Psi, TensExpr):
        Psi = canon(Psi)

    return Psi, (mu_f, nu_f, rho_f, sigma_f)


def _scale_matter_fields(expr, lam, matter_heads):
    """Multiply each matter-field factor in `expr` by `lam`. Used to apply
    the path-substitution phi -> lambda*phi after differentiation in the
    n=1 integral formula. matter_heads is a set of TensorHead objects."""
    if isinstance(expr, TensAdd):
        return _sum_terms([_scale_matter_fields(t, lam, matter_heads) for t in expr.args])
    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)
        count = sum(1 for f in factors if _get_component(f) in matter_heads)
        if count == 0:
            return expr
        result = coeff * (lam ** count)
        for f in factors:
            result = result * f
        return result
    if isinstance(expr, Tensor):
        if _get_component(expr) in matter_heads:
            return lam * expr
        return expr
    return expr


def _integrate_lambda(expr, lam):
    """Integrate `expr` with respect to `lam` from 0 to 1.

    The scalar coefficients in TensMul terms are sympy expressions that can
    depend on `lam`; the tensor factors do not. We extract each coefficient,
    integrate it, and multiply back. For polynomial integrands this is the
    standard lam^d -> 1/(d+1) substitution.
    """
    if isinstance(expr, TensAdd):
        return _sum_terms([_integrate_lambda(t, lam) for t in expr.args])
    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)
        new_coeff = integrate(coeff, (lam, 0, 1))
        if new_coeff == 0:
            return S.Zero
        if not factors:
            return new_coeff
        result = new_coeff
        for f in factors:
            result = result * f
        return result
    if isinstance(expr, Tensor):
        return expr  # constant in lam, integral over [0,1] = expr * 1
    # Pure scalar (no tensor factors)
    if hasattr(expr, 'has') and expr.has(lam):
        return integrate(expr, (lam, 0, 1))
    return expr


# ---------------------------------------------------------------------------
# Superpotential double-divergence: Delta = Psi_{,rho sigma}
# ---------------------------------------------------------------------------

def superpotential_divergence(Psi, psi_indices):
    """Compute Delta^{mu nu} = Psi^{mu nu rho sigma}_{,rho sigma}.
    
    Takes two total derivatives of Psi w.r.t. its last two indices.
    
    Args:
        Psi: tensor expression for the superpotential
        psi_indices: (mu, nu, rho, sigma) free indices of Psi
        
    Returns:
        Delta: tensor expression for the double divergence
    """
    from bootstrap.jet import total_derivative
    
    if Psi == S.Zero:
        return S.Zero
    
    mu_f, nu_f, rho_f, sigma_f = psi_indices
    
    # First derivative: ∂_sigma Psi
    d1 = total_derivative(Psi, -sigma_f)
    if d1 == S.Zero:
        return S.Zero
    
    # Second derivative: ∂_rho (∂_sigma Psi)
    Delta = total_derivative(d1, -rho_f)
    
    return canon(Delta) if isinstance(Delta, TensExpr) else Delta


# ---------------------------------------------------------------------------
# Verify Psi symmetries
# ---------------------------------------------------------------------------

def verify_psi_symmetries(Psi, psi_indices):
    """Verify that Psi has the required symmetries.
    
    Checks:
    1. Symmetric in (mu, nu) — first pair
    2. Symmetric in (rho, sigma) — last pair
    3. Cyclic antisymmetry: Psi^{mu nu rho sigma} + Psi^{mu rho nu sigma} 
       + Psi^{mu sigma rho nu} = 0
    
    Args:
        Psi: tensor expression
        psi_indices: (mu, nu, rho, sigma) the four free indices
        
    Returns:
        dict with keys 'sym_mn', 'sym_rs', 'cyclic', each mapping to
        True/False and the residual expression if False.
    """
    if Psi == S.Zero:
        return {'sym_mn': True, 'sym_rs': True, 'cyclic': True}
    
    mu, nu, rho, sigma = psi_indices
    results = {}
    
    # 1. Symmetric in (mu, nu): Psi(mu,nu,rho,sigma) - Psi(nu,mu,rho,sigma) = 0
    Psi_swapped_mn = swap_free_indices(Psi, mu, nu)
    diff_mn = canon(Psi - Psi_swapped_mn)
    results['sym_mn'] = (diff_mn == S.Zero)
    if not results['sym_mn']:
        results['sym_mn_residual'] = diff_mn

    # 2. Symmetric in (rho, sigma): similar
    Psi_swapped_rs = swap_free_indices(Psi, rho, sigma)
    diff_rs = canon(Psi - Psi_swapped_rs)
    results['sym_rs'] = (diff_rs == S.Zero)
    if not results['sym_rs']:
        results['sym_rs_residual'] = diff_rs

    # 3. Cyclic antisymmetry in the last three indices (paper):
    #    Psi^{mu nu rho sigma} + Psi^{mu rho nu sigma} + Psi^{mu sigma rho nu} = 0.
    # Each permutation is a free-index swap of nu with rho / sigma respectively.
    Psi_perm1 = swap_free_indices(Psi, nu, rho)    # Psi^{mu rho nu sigma}
    Psi_perm2 = swap_free_indices(Psi, nu, sigma)  # Psi^{mu sigma rho nu}
    cyclic_sum = canon(Psi + Psi_perm1 + Psi_perm2)
    results['cyclic'] = (cyclic_sum == S.Zero)
    if not results['cyclic']:
        results['cyclic_residual'] = cyclic_sum
    
    return results


# ---------------------------------------------------------------------------
# H2 violation and EOM term computation (eq. 28 and related)
# ---------------------------------------------------------------------------

def compute_h2_violation(E_expr, E_indices):
    """Compute the H2 violation tensor Z^{mu nu alpha beta}.

    H2 (paper §3, eq. labeled `H2` and the explicit form at eq. labeled
    `Z`) is the integrability condition for E^{μν} to come from a
    Lagrangian's EL derivative:

        Z^{μν αβ} = 2 (∂E^{μν}/∂h_{αβ} − ∂E^{αβ}/∂h_{μν})
                  − ∂_γ (∂E^{μν}/∂h_{αβ,γ} − ∂E^{αβ}/∂h_{μν,γ}).

    The paper (§3 around the H2 derivation) assumes the Lagrangian — and
    therefore E — depends on h only through h and dh, no ddh. Our E_2 can
    still carry ddh (e.g. the wave operator W^{μν} at n=1, or Christoffel
    contributions inside T_H[L^{(n)}]). Such ddh pieces are in the kernel
    of THIS formula (both ∂/∂h and ∂/∂dh annihilate ddh), so they don't
    contribute and the formula stays correct as written.

    With the Hilbert procedure and no optional EOM terms, Butcher's claim
    is Z = 0 at every order — verified empirically by the closure tests.
    On other paths (Belinfante, optional EOM additions) Z can be nonzero,
    in which case `decompose_against_eoms` (bootstrap.eom_decompose)
    decomposes it.

    Args:
        E_expr: tensor expression for E^{μν(n)}.
        E_indices: (mu, nu) — the free indices of E_expr.

    Returns:
        (Z, (alpha, beta)):
            Z: tensor expression with four free indices (mu, nu, alpha, beta),
               or S.Zero if E_expr is zero.
            (alpha, beta): the h-style index pair generated for Z. Z's free
               indices are then (mu, nu, alpha, beta). Returned so callers
               can form the X = -1/(2(n+1)) Y · h_{alpha beta} correction
               without re-discovering them.
    """
    from bootstrap.jet import total_derivative

    if E_expr == S.Zero:
        return S.Zero, (None, None)

    mu, nu = E_indices
    alpha, beta = fresh_indices(2)
    gamma, = fresh_indices(1)

    # E^{αβ}: relabel E_expr's free (μ, ν) → (α, β) via substitute_indices.
    E_ab = E_expr.substitute_indices(
        (mu, alpha), (-mu, -alpha), (nu, beta), (-nu, -beta)
    )

    # Algebraic piece: 2 (∂E^{μν}/∂h_{αβ} − ∂E^{αβ}/∂h_{μν}).
    dE_mn_h = jet_derivative(E_expr, h, [alpha, beta])
    dE_ab_h = jet_derivative(E_ab, h, [mu, nu])

    # dh piece: −∂_γ (∂E^{μν}/∂h_{αβ,γ} − ∂E^{αβ}/∂h_{μν,γ}).
    dE_mn_dh = jet_derivative(E_expr, dh, [alpha, beta, gamma])
    dE_ab_dh = jet_derivative(E_ab, dh, [mu, nu, gamma])
    dh_inner = dE_mn_dh - dE_ab_dh
    dh_piece = total_derivative(dh_inner, -gamma) if dh_inner != S.Zero else S.Zero

    Z = 2 * (dE_mn_h - dE_ab_h) - dh_piece
    if isinstance(Z, TensExpr):
        Z = canon(Z)
    return Z, (alpha, beta)
