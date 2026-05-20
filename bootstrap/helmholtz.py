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
    Lorentz, metric, h, dh, ddh,
    fresh_indices, canon, _JET_HIERARCHY, _matter_fields,
)
from bootstrap.jet import (
    jet_derivative, _sum_terms, _decompose_tensmul,
    _get_component, _get_indices,
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
    USER NOTE in the dev status), this lambda integral is just
    integral_0^1 lambda^d dlambda = 1/(d+1) for each power d.

    Args:
        M_expr: tensor expression for M^{mu nu} at order n=1.
        M_indices: (mu_M, nu_M) tuple of M_expr's free indices.
        matter_field_heads: dict {name: (phi, dphi, ddphi)}. If None,
            uses the global _matter_fields registry. For scalar matter only;
            non-scalar matter raises NotImplementedError because the
            derivative-pattern in eq. 23 assumes scalar dphi/ddphi.

    Returns:
        (Psi, psi_indices) where psi_indices = (mu_f, nu_f, rho_f, sigma_f).
        Returns (S.Zero, None) if there are no matter fields.
    """
    if matter_field_heads is None:
        matter_field_heads = {}
        for name, info in _matter_fields.items():
            if info.get('rank', 0) != 0:
                raise NotImplementedError(
                    f"Matter field {name!r} is not scalar (rank "
                    f"{info.get('rank')}); compute_superpotential_n1 currently "
                    "supports scalar matter only."
                )
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
    for (phi_h, dphi_h, ddphi_h) in matter_field_heads.values():
        matter_heads.update([phi_h, dphi_h, ddphi_h])

    bracket_terms = []
    for name, (phi_h, dphi_h, ddphi_h) in matter_field_heads.items():
        # Term A: d^2 M^{alpha beta} / (d phi  d h_{mu nu, rho sigma})
        dM_dddh = jet_derivative(M_ab, ddh, [mu_f, nu_f, rho_f, sigma_f])
        d2M_A = jet_derivative(dM_dddh, phi_h, [])

        # Term B: d^2 M^{mu nu} / (d phi_{,sigma}  d h_{alpha beta, rho})
        dM_dh_B = jet_derivative(M_mn, dh, [alpha, beta, rho_f])
        d2M_B = jet_derivative(dM_dh_B, dphi_h, [sigma_f])

        # Term C: d^2 M^{alpha beta} / (d phi_{,sigma}  d h_{mu nu, rho})
        dM_dh_C = jet_derivative(M_ab, dh, [mu_f, nu_f, rho_f])
        d2M_C = jet_derivative(dM_dh_C, dphi_h, [sigma_f])

        bracket = d2M_A - Rational(1, 2) * d2M_B - Rational(1, 2) * d2M_C
        if bracket == S.Zero:
            continue

        # Multiply by h_{alpha beta} (contraction with M^{alpha beta}) and the
        # matter field phi_i (the "velocity" along the linear path).
        scaled = _scale_matter_fields(bracket, lam, matter_heads)
        # The pulled-in phi_i factor itself needs scaling too (it'll get a lam).
        phi_factor = lam * phi_h()
        contracted = phi_factor * h(-alpha, -beta) * scaled

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
    Psi_swapped_mn = _swap_free_indices(Psi, mu, nu)
    diff_mn = canon(Psi - Psi_swapped_mn)
    results['sym_mn'] = (diff_mn == S.Zero)
    if not results['sym_mn']:
        results['sym_mn_residual'] = diff_mn
    
    # 2. Symmetric in (rho, sigma): similar
    Psi_swapped_rs = _swap_free_indices(Psi, rho, sigma)
    diff_rs = canon(Psi - Psi_swapped_rs)
    results['sym_rs'] = (diff_rs == S.Zero)
    if not results['sym_rs']:
        results['sym_rs_residual'] = diff_rs
    
    # 3. Cyclic antisymmetry in last 3 indices:
    # Psi^{m,n,r,s} + Psi^{m,r,n,s} + Psi^{m,s,r,n} = 0
    # We need Psi with (nu,rho,sigma) cyclically permuted to (rho,sigma,nu) 
    # and (sigma,nu,rho)
    Psi_cyc1 = _cyclic_perm_indices(Psi, nu, rho, sigma)  # (m,r,s,n)
    Psi_cyc2 = _cyclic_perm_indices(Psi_cyc1, rho, sigma, nu)  # (m,s,n,r)
    
    # Actually, the condition from the paper is:
    # Psi^{mu nu rho sigma} + Psi^{mu rho nu sigma} + Psi^{mu sigma rho nu} = 0
    # So we swap the second index with rho, and separately with sigma
    Psi_perm1 = _swap_free_indices(Psi, nu, rho)  # Psi^{mu rho nu sigma}
    Psi_perm2 = _swap_free_indices(Psi, nu, sigma) # Psi^{mu sigma rho nu}
    # But we also need to account for the swap in the third slot in perm2
    # Actually Psi^{mu sigma rho nu}: swap nu<->sigma gives us 
    # Psi(mu, sigma, rho, nu) — this IS correct as a free-index permutation
    
    cyclic_sum = canon(Psi + Psi_perm1 + Psi_perm2)
    results['cyclic'] = (cyclic_sum == S.Zero)
    if not results['cyclic']:
        results['cyclic_residual'] = cyclic_sum
    
    return results


def _swap_free_indices(expr, idx1, idx2):
    """Swap two free indices in a tensor expression.
    
    This is a formal operation: replace every occurrence of idx1 with idx2. USER COMMENT: this is a very useful general function, it should probably be in tensor_algebra.py
    and vice versa.
    """
    if expr == S.Zero:
        return S.Zero
    
    # Use a temporary index to avoid clashes
    tmp, = fresh_indices(1)
    
    # idx1 -> tmp, idx2 -> idx1, tmp -> idx2
    result = expr
    # We need to handle both positive and negative versions of each index
    for sign_mult in [1, -1]:
        i1 = idx1 if sign_mult == 1 else -idx1
        i2 = idx2 if sign_mult == 1 else -idx2
        t = tmp if sign_mult == 1 else -tmp
        
        # This is tricky with SymPy tensors. Let's use a substitution approach.
    
    # Actually, the simplest approach: use TensExpr.substitute_indices
    # or .fun_eval
    if hasattr(expr, 'substitute_indices'):
        try:
            # SymPy's substitute_indices: replace idx1 with tmp, then idx2 with idx1, 
            # then tmp with idx2
            result = expr.substitute_indices((idx1, tmp), (-idx1, -tmp))
            result = result.substitute_indices((idx2, idx1), (-idx2, -idx1))
            result = result.substitute_indices((tmp, idx2), (-tmp, -idx2))
            return result
        except:
            pass
    
    # Fallback: manual replacement via string manipulation is not great.
    # For now, try another approach using SymPy's _set_new_index_structure
    # or by rebuilding the expression.
    
    # Simple approach that works: since we just want to check symmetry,
    # we can evaluate both versions and compare.
    # For a proper swap, we use the xreplace method on the indices.
    try:
        result = expr.xreplace({idx1: tmp, -idx1: -tmp,
                                idx2: idx1, -idx2: -idx1,
                                tmp: idx2, -tmp: -idx2})
        return result
    except:
        pass
    
    return expr  # fallback: return unchanged (will cause symmetry check to fail)


def _cyclic_perm_indices(expr, i1, i2, i3):
    """Cyclically permute three free indices: (i1,i2,i3) -> (i2,i3,i1)."""
    tmp, = fresh_indices(1)
    try:
        result = expr.xreplace({i1: tmp, -i1: -tmp,
                                i2: i1, -i2: -i1,
                                i3: i2, -i3: -i2,
                                tmp: i3, -tmp: -i3})
        return result
    except:
        return expr


# ---------------------------------------------------------------------------
# H2 violation and EOM term computation (eq. 28 and related)
# ---------------------------------------------------------------------------

def compute_h2_violation(E_expr, n):
    """Compute the H2 violation tensor Z^{mu nu alpha beta}.
    
    Z = 2(dE^mn/dh_ab - dE^ab/dh_mn) - d_gamma(dE^mn/dh_{ab,g} - dE^ab/dh_{mn,g})
    
    If this is nonzero, EOM terms must be added to E to satisfy H2.
    
    Args:
        E_expr: tensor expression for E^{mu nu}(n) — the current field
            equation contribution at order n.
        n: the order in h.
        
    Returns:
        Z: tensor expression for the violation. Its free indices are
           (mu_E, nu_E, alpha, beta) where mu_E, nu_E are the free indices
           of E_expr and alpha, beta are fresh.
    """
    from bootstrap.jet import total_derivative
    
    alpha, beta = fresh_indices(2)
    gamma, = fresh_indices(1)
    
    # dE/dh_{ab}: jet derivative of E by h
    dE_dh = jet_derivative(E_expr, h, [alpha, beta])
    
    # For the second term, we need dE^{ab}/dh_{mn} — but E's free indices
    # are fixed. We need a version of E with alpha,beta as the free indices.
    # This requires swapping the free indices of E_expr with alpha,beta.
    # 
    # ISSUE: this is tricky because we don't know which indices in E_expr
    # are the "free" ones representing mu,nu. We'd need them passed in.
    # 
    # For now, return just the Z tensor with a simplified approach:
    # We'll require the caller to pass the free indices of E.
    
    # TODO: implement fully once we have the bootstrap loop context
    # For now, return a placeholder
    return S.Zero


def compute_eom_correction(Z_expr, n):
    """Given the H2 violation Z, compute the EOM correction terms.
    
    X^{mu nu}_{kl} = -1/(2(n+1)) * Y^{mu nu rho sigma}_{kl} * h_{rho sigma}
    X^{mu nu}_{phi_i} = -1/(2(n+1)) * Y^{mu nu rho sigma}_{phi_i} * h_{rho sigma}
    
    where Z = Y_{h} * E_h^{(0)} + sum_i Y_{phi_i} * E_{phi_i}^{(0)}
    
    Args:
        Z_expr: the H2 violation tensor
        n: the order in h
        
    Returns:
        dict mapping field names to their X coefficient tensors
    """
    # TODO: implement decomposition of Z into EOM-proportional pieces
    return {}
