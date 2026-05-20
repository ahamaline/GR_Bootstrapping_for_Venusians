"""
Covariant expansion: expand geometric quantities in powers of h_{mu nu}.

Given g_{mu nu} = eta_{mu nu} + 2*kappa*h_{mu nu}, computes:
- g^{mu nu} (inverse metric) to desired order
- Christoffel symbols Gamma^lambda_{mu nu}
- Riemann tensor, Ricci tensor/scalar, Einstein tensor
- sqrt(|g|) to desired order
- The Einstein-Hilbert Lagrangian density sqrt(|g|) R

All in abstract index notation, using the jet-space variables.

Follows the approach of GR.fr: when computing products of power series, sme of which only begin at orser 1,
each factor only needs to be expanded to the required order, which may be less then the total order we're interested in.
"""

from sympy import S, Rational, Symbol
from sympy.tensor.tensor import TensAdd, TensMul, TensExpr, Tensor
from bootstrap.tensor_algebra import (
    Lorentz, metric, h, dh, ddh,
    fresh_indices, canon, filter_by_order
)

kappa = Symbol('kappa')


# ---------------------------------------------------------------------------
# Inverse metric expansion: g^{mu nu} = eta^{mu nu} - 2*kappa*h^{mu nu} + ...
# ---------------------------------------------------------------------------

def inverse_metric_order(n, mu, nu):
    """Compute g^{mu nu} at order n in h.
    
    g^{mu nu} = sum_{k=0}^{inf} (-2*kappa)^k * (h^k)^{mu nu}
    
    Order 0: eta^{mu nu} = metric(mu, nu)
    Order 1: -2*kappa * h^{mu nu}
    Order k: (-2*kappa)^k * h^{mu alpha1} h^{alpha1 alpha2} ... h^{alpha_{k-1} nu} 
    
    Args:
        n: order in h (n=0 gives eta^{mu nu})
        mu, nu: contravariant TensorIndex objects
        
    Returns:
        Tensor expression for the order-n part.
    """
    if n == 0:
        return metric(mu, nu)
    
    # (-2*kappa)^n * h^{mu a1} h^{a1 a2} ... h^{a_{n-1} nu}
    # We need n-1 dummy indices
    prefactor = (-2 * kappa) ** n
    
    if n == 1:
        return prefactor * h(mu, nu)
    
    # Chain of h multiplications
    dummies = fresh_indices(n - 1)
    
    # First factor: h(mu, dummies[0])
    result = h(mu, dummies[0])
    
    # Middle factors: h(dummies[i], dummies[i+1])
    for i in range(n - 2):
        result = result * h(-dummies[i], dummies[i + 1])
    
    # Last factor: h(dummies[-1], nu) 
    # Actually we need to lower the last dummy to contract
    result = result * h(-dummies[-1], nu)
    
    return canon(prefactor * result)


def inverse_metric_to_order(n, mu, nu):
    """Compute g^{mu nu} as a sum up to order n in h."""
    result = S.Zero
    for k in range(n + 1):
        term = inverse_metric_order(k, mu, nu)
        result = result + term
    return canon(result) if isinstance(result, TensExpr) else result


# ---------------------------------------------------------------------------
# Christoffel symbols
# ---------------------------------------------------------------------------

def christoffel_order(n, lam, mu_idx, nu_idx):
    """Compute Gamma^lambda_{mu nu} at order n in h.
    
    Gamma^lambda_{mu nu} = 1/2 g^{lambda sigma} (g_{sigma mu, nu} + g_{nu sigma, mu} - g_{mu nu, sigma})
    
    Since g_{mu nu} = eta_{mu nu} + 2*kappa*h_{mu nu}, we have
    g_{mu nu, rho} = 2*kappa * h_{mu nu, rho} (= 2*kappa * dh_{mu nu, rho})
    
    So Gamma = kappa * g^{lambda sigma} * (dh_{sigma mu, nu} + dh_{nu sigma, mu} - dh_{mu nu, sigma})
    
    The g^{-1} only needs up to order n-1 because dh is order 1.
    
    Args:
        n: order in h (n >= 1; Gamma is zero at order 0)
        lam: contravariant index (upper)
        mu_idx, nu_idx: covariant indices (lower)
        
    Returns:
        Tensor expression at order n.
    """
    if n < 1:
        return S.Zero
    
    sigma, = fresh_indices(1)
    
    # g^{lambda sigma} at order n-1. USER COMMENT: good that you noticed we only need one specific order here! 
    ginv = inverse_metric_order(n - 1, lam, sigma)
    
    # The Christoffel bracket: dh_{sigma mu, nu} + dh_{nu sigma, mu} - dh_{mu nu, sigma}
    bracket = (dh(-sigma, mu_idx, nu_idx) + dh(nu_idx, -sigma, mu_idx) 
               - dh(mu_idx, nu_idx, -sigma))
    
    result = kappa * ginv * bracket
    return canon(result) if isinstance(result, TensExpr) else result


def christoffel_to_order(n, lam, mu_idx, nu_idx):
    """Compute Gamma^lambda_{mu nu} summed up to order n."""
    result = S.Zero
    for k in range(1, n + 1):
        result = result + christoffel_order(k, lam, mu_idx, nu_idx)
    return canon(result) if isinstance(result, TensExpr) else result


# ---------------------------------------------------------------------------
# Riemann tensor
# ---------------------------------------------------------------------------

def riemann_order(n, lam, mu_idx, rho_idx, nu_idx):
    """Compute R^lambda_{mu rho nu} at order n in h.
    
    R^lambda_{mu rho nu} = dGamma^lambda_{mu nu}/dx^rho - dGamma^lambda_{mu rho}/dx^nu
                          + Gamma^lambda_{sigma rho} Gamma^sigma_{nu mu}
                          - Gamma^lambda_{sigma nu} Gamma^sigma_{rho mu}
    
    The derivative terms are order n (Gamma is order >= 1, derivative adds 0).
    The quadratic terms at order n need Gamma at orders k and n-k for k=1..n-1.
    
    Args:
        n: order in h (>= 1)
        lam: upper index
        mu_idx, rho_idx, nu_idx: lower indices
    """
    from bootstrap.jet import total_derivative
    
    if n < 1:
        return S.Zero
    
    # Derivative terms: d_rho Gamma^lam_{mu nu} - d_nu Gamma^lam_{mu rho}
    # These come from Christoffel at order n (derivatives don't change h-order
    # because dh is already order 1 in h)
    chris_mn = christoffel_order(n, lam, mu_idx, nu_idx)
    chris_mr = christoffel_order(n, lam, mu_idx, rho_idx)
    
    term_deriv = S.Zero
    if chris_mn != S.Zero:
        term_deriv = term_deriv + total_derivative(chris_mn, rho_idx)
    if chris_mr != S.Zero:
        term_deriv = term_deriv - total_derivative(chris_mr, nu_idx)
    
    # Quadratic terms: sum over k=1..n-1 of
    # Gamma^lam_{sig rho}(k) * Gamma^sig_{nu mu}(n-k)
    # - Gamma^lam_{sig nu}(k) * Gamma^sig_{rho mu}(n-k)
    term_quad = S.Zero
    for k in range(1, n):
        sig, = fresh_indices(1)
        c1 = christoffel_order(k, lam, -sig, rho_idx)
        c2 = christoffel_order(n - k, sig, nu_idx, mu_idx)
        c3 = christoffel_order(k, lam, -sig, nu_idx)
        c4 = christoffel_order(n - k, sig, rho_idx, mu_idx)
        
        if c1 != S.Zero and c2 != S.Zero:
            term_quad = term_quad + c1 * c2
        if c3 != S.Zero and c4 != S.Zero:
            term_quad = term_quad - c3 * c4
    
    result = term_deriv + term_quad
    return canon(result) if isinstance(result, TensExpr) else result


# ---------------------------------------------------------------------------
# Ricci scalar
# ---------------------------------------------------------------------------

def ricci_scalar_order(n):
    """Compute R (Ricci scalar) at order n in h.
    
    R = g^{mu nu} R^lambda_{mu lambda nu}
    
    = g^{mu nu}(k) * R^lam_{mu lam nu}(n-k) summed over k=0..n-1
    (g^{-1} at order 0 is eta, so it goes down to k=0;
     Riemann starts at order 1, so n-k >= 1 means k <= n-1)
    """
    if n < 1:
        return S.Zero  # R^{(0)} = 0 (flat space); R starts at order 1
    
    result = S.Zero
    for k in range(0, n):
        # g^{mu nu} at order k, R^lam_{mu lam nu} at order n-k
        mu, nu, lam = fresh_indices(3)
        ginv_k = inverse_metric_order(k, mu, nu)
        riem_nk = riemann_order(n - k, lam, -mu, -lam, -nu)  #USER COMMENT: here you are recomputing Riemann each time you run a new order of R. Seems inefficient? 
        if ginv_k != S.Zero and riem_nk != S.Zero:
            result = result + ginv_k * riem_nk
    
    return canon(result) if isinstance(result, TensExpr) else result


# ---------------------------------------------------------------------------
# sqrt(|g|) expansion
# ---------------------------------------------------------------------------

def sqrt_det_g_order(n):
    """Compute the order-n part of sqrt(|g|) in powers of h.
    
    We use: sqrt(|g|) = exp(1/2 * tr(log(g)))
    with g_{mu nu} = eta_{mu nu} + 2*kappa*h_{mu nu}.
    
    In terms of the trace h^mu_mu and traces of powers of h:
    
    log det(g) = tr(log(I + 2*kappa*h)) 
               = sum_{k=1}^inf (-1)^{k+1}/k * (2*kappa)^k * tr(h^k)
    
    sqrt(|g|) = exp(1/2 * log det(g))
    
    Order 0: 1
    Order 1: kappa * h^mu_mu  (= kappa * trace of h)
    Order 2: kappa^2 * (1/2 (h^mu_mu)^2 - 1/2 h^mu_nu h^nu_mu)  ... etc.
    
    This is a purely algebraic function of the field h (no derivatives).
    """
    if n == 0:
        return S.One
    
    # We know that log det(g) = sum_{k=1} (-1)^{k+1}/k * (2*kappa)^k * Tr(h^k)
    # where Tr(h^k) = h^{a1}_{a2} h^{a2}_{a3} ... h^{ak}_{a1}
    #
    # Then sqrt(|g|) = exp(1/2 * log det(g))
    # Expanding the exponential as a power series:
    # exp(x) = 1 + x + x^2/2 + ... 
    # where x = 1/2 * log det(g)
    
    # We'll compute log_det coefficients first, then exponentiate.
    
    # log_det at order k (in h): (-1)^{k+1}/k * (2*kappa)^k * Tr(h^k)
    def log_det_order(k):
        if k < 1:
            return S.Zero
        coeff = Rational((-1)**(k+1), k) * (2*kappa)**k
        return coeff * _trace_h_power(k)
    
    # half_log_det at order k
    def half_log_det_order(k):
        return Rational(1, 2) * log_det_order(k)
    
    # Now exponentiate: sqrt(|g|) = exp(sum_k half_log_det(k))
    # The order-n part of exp(x1 + x2 + ...) where x_k is order k
    # is computed by the standard power series multiplication.
    # 
    # s_0 = 1
    # s_n = 1/n * sum_{k=1}^{n} k * half_log_det(k) * s_{n-k}
    # (this follows from d/dt exp(f(t)) = f'(t) exp(f(t)))
    
    s = {0: S.One}
    for m in range(1, n + 1):
        s[m] = S.Zero
        for k in range(1, m + 1):
            hld_k = half_log_det_order(k)
            if hld_k != S.Zero and s.get(m - k, S.Zero) != S.Zero:
                s[m] = s[m] + Rational(k, m) * hld_k * s[m - k]
        if isinstance(s[m], TensExpr):
            s[m] = canon(s[m])
    
    return s[n]


def _trace_h_power(k):
    """Compute Tr(h^k) = h^{a1}_{a2} h^{a2}_{a3} ... h^{ak}_{a1}.
    
    This is a scalar (no free indices) of order k in h.
    """
    if k == 0:
        return S.One  # Tr(I) = d, but we don't need this case
    
    if k == 1:
        a, = fresh_indices(1)
        return h(a, -a)  # h^mu_mu = trace
    
    # For k >= 2: chain of h's with contracted adjacent indices
    indices = fresh_indices(k)
    result = S.One
    for i in range(k):
        next_i = (i + 1) % k
        result = result * h(indices[i], -indices[next_i])
    
    return canon(result)


# ---------------------------------------------------------------------------
# Einstein-Hilbert Lagrangian to arbitrary order
# ---------------------------------------------------------------------------

def matter_lagrangian_order(L_M, n):
    """Compute the order-n part of √|g| L̃_M.

    L_M is a matter Lagrangian using η-contractions on the matter fields
    (e.g. for a scalar: L_M = -1/2 ∂φ·∂φ). L̃_M is the covariantized form:
    every η^{μν} factor is promoted to g^{μν}. (For scalar fields, partial
    derivatives are already tensorial, so this is the only change. Vector
    or higher-rank matter fields would also need Christoffel corrections
    on their covariant derivatives — currently NOT handled here; a runtime
    check below traps that case.)

    The expansion is:
        √|g| L̃_M = (√|g|)(g^{μν} factors of L_M expanded in h)
    For an L_M with M implicit η-contractions, the order-n piece is a sum
    over compositions n = k_0 + k_1 + ... + k_M, where k_0 is the order of
    √|g| and k_i is the order of the i-th metric factor.

    Args:
        L_M: matter Lagrangian (scalar, using registered matter field heads).
        n: desired order in h.

    Returns:
        Tensor expression for the order-n piece.
    """
    from bootstrap.energy_momentum import uncontract_metrics
    from bootstrap.jet import (
        _decompose_tensmul, _decompose_tensadd, _get_component, _get_indices,
        _sum_terms,
    )
    from bootstrap.tensor_algebra import _matter_fields, _JET_HIERARCHY

    if L_M == S.Zero:
        return S.Zero
    if n < 0:
        return S.Zero

    # Guard against unsupported matter field types: this implementation
    # assumes covariant derivatives = partial derivatives, which only holds
    # for scalar fields. If any non-scalar matter field's derivatives appear
    # in L_M, refuse the computation (Christoffel corrections needed).
    for info in _matter_fields.values():
        if info.get('rank', 0) != 0:
            # Check whether this field's dfield/ddfield appears in L_M.
            d_head = info.get('dfield')
            dd_head = info.get('ddfield')
            for head in (d_head, dd_head):
                if head is None:
                    continue
                # Quick scan via _decompose_tensmul on each term.
                terms = L_M.args if isinstance(L_M, TensAdd) else [L_M]
                for t in terms:
                    if isinstance(t, TensMul):
                        _, factors = _decompose_tensmul(t)
                        if any(_get_component(f) is head for f in factors):
                            raise NotImplementedError(
                                f"Non-scalar matter field {info.get('name')!r} "
                                "appears with derivatives; matter_lagrangian_order "
                                "currently handles only scalar matter (covariant "
                                "derivative Christoffel corrections not implemented)."
                            )
                    elif isinstance(t, Tensor) and _get_component(t) is head:
                        raise NotImplementedError(
                            f"Non-scalar matter field {info.get('name')!r} "
                            "appears with derivatives; matter_lagrangian_order "
                            "currently handles only scalar matter."
                        )

    if n == 0:
        return canon(L_M)

    # Uncontract η factors so each implicit contraction becomes an explicit
    # metric(μ, ν) factor that we can then expand in h.
    L_unc = uncontract_metrics(L_M)

    # Process each term of L_unc independently; sum at the end.
    terms_in = L_unc.args if isinstance(L_unc, TensAdd) else [L_unc]
    out_terms = []
    for term in terms_in:
        if isinstance(term, TensMul):
            coeff, factors = _decompose_tensmul(term)
        elif isinstance(term, Tensor):
            coeff, factors = S.One, [term]
        else:
            # Pure scalar coefficient (no tensor factors) — at order n>0 this
            # has no h expansion (constant has no h dependence) unless
            # multiplied by √|g|, which contributes at order n via the
            # k_metric_i = 0 / k_sqrt = n composition. Handle below.
            coeff, factors = term, []

        metric_positions = [i for i, f in enumerate(factors)
                            if _get_component(f) is metric]
        non_metric_factors = [f for i, f in enumerate(factors)
                              if i not in metric_positions]
        M = len(metric_positions)
        metric_indices = [_get_indices(factors[i]) for i in metric_positions]

        # Sum over compositions (k_0, k_1, ..., k_M) with sum = n.
        for ks in _compositions(n, M + 1):
            k_sqrt = ks[0]
            k_metrics = ks[1:]
            sg = sqrt_det_g_order(k_sqrt)
            if sg == S.Zero and k_sqrt > 0:
                continue
            # Build the metric-expansion product.
            piece = coeff * sg
            for (mi, ni), k_i in zip(metric_indices, k_metrics):
                ginv_part = inverse_metric_order(k_i, mi, ni)
                if ginv_part == S.Zero:
                    piece = S.Zero
                    break
                piece = piece * ginv_part
            if piece == S.Zero:
                continue
            for f in non_metric_factors:
                piece = piece * f
            out_terms.append(piece)

    result = _sum_terms(out_terms)
    return canon(result) if isinstance(result, TensExpr) else result


def _compositions(n, k):
    """Yield all k-tuples of non-negative integers summing to n.

    Order matters (so (1,2) and (2,1) are distinct). k must be >= 1.
    """
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for tail in _compositions(n - i, k - 1):
            yield (i,) + tail


def einstein_hilbert_lagrangian_order(n):
    """Compute the order-n (in h) part of 1/(2*kappa^2) * sqrt(|g|) * R.
    
    The EH Lagrangian density is (mostly-plus signature):
        L_EH = 1/(2*kappa^2) * sqrt(|g|) * R
    
    We expand both sqrt(|g|) and R in powers of h and collect terms at
    each order. Since R starts at order 1 and sqrt(|g|) starts at order 0:
    
        L_EH^{(n)} = 1/(2*kappa^2) * sum_{k=0}^{n-1} sqrt_g^{(k)} * R^{(n-k)}
    
    Order 0: zero (Minkowski vacuum, R=0)
    Order 1: zero (linear in h, vanishes as a total derivative after IBP)
    Order 2: L_h^{(2)} -- the Fierz-Pauli kinetic Lagrangian
    Order n >= 3: gravitational self-interaction vertices
    """
    if n <= 0:
        return S.Zero
    
    prefactor = Rational(1, 2) * kappa**(-2)
    
    result = S.Zero
    for k in range(0, n):  # k from 0 to n-1 (R^{(n-k)} needs n-k >= 1)
        sg_k = sqrt_det_g_order(k)
        R_nk = ricci_scalar_order(n - k)
        
        if sg_k != S.Zero and R_nk != S.Zero:
            result = result + sg_k * R_nk
    
    if result == S.Zero:
        return S.Zero
    
    result = prefactor * result
    return canon(result) if isinstance(result, TensExpr) else result

