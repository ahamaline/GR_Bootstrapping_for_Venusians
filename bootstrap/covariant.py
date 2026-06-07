"""
Covariant expansion: expand geometric quantities in powers of h_{mu nu}.

Given g_{mu nu} = eta_{mu nu} + 2*kappa*h_{mu nu}, computes:
- g^{mu nu} (inverse metric) to desired order
- Christoffel symbols Gamma^lambda_{mu nu}
- Riemann tensor, Ricci tensor/scalar, Einstein tensor
- sqrt(|g|) to desired order
- The Einstein-Hilbert Lagrangian density sqrt(|g|) R

All in abstract index notation, using the jet-space variables.

Follows the approach of GR.fr: when computing products of power series, some of which only begin at order 1,
each factor only needs to be expanded to the required order, which may be less than the total order we're interested in.
"""

from sympy import S, Rational, Symbol
from sympy.tensor.tensor import (
    TensAdd, TensMul, TensExpr, Tensor, TensorHead, TensorSymmetry,
)
from bootstrap.tensor_algebra import (
    Lorentz, metric, h, dh, ddh,
    fresh_indices, canon, filter_by_order, NATURAL_POSITIONS,
)

kappa = Symbol('kappa')

# Riemann tensor head R^lambda_{mu rho nu}: a placeholder the user writes
# nonminimal couplings with (e.g. xi*phi**2 * Ricci, with Ricci built by
# contracting Riemann via ginv). `covariant_coupling_order` replaces each
# Riemann factor with `riemann_order(k)` during the h-expansion, so the head
# carries no symmetry of its own (the expansion supplies the real Riemann
# symmetries). Natural index positions: up, down, down, down.
Riemann = TensorHead('Riemann', [Lorentz] * 4, TensorSymmetry.no_symmetry(4))
NATURAL_POSITIONS[Riemann] = ['up', 'down', 'down', 'down']


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
# Downstairs metric expansion: g_{mu nu} = eta_{mu nu} + 2*kappa*h_{mu nu}
# ---------------------------------------------------------------------------

def metric_order(n, mu, nu):
    """Compute g_{mu nu} at order n in h.

    g_{mu nu} = eta_{mu nu} + 2*kappa*h_{mu nu}, so only orders 0 and 1 are
    nonzero. Used when matter_lagrangian_order encounters a metric factor
    with both indices DOWN (originally an eta_{mu nu} for lowering an
    upstairs field).

    Args:
        n: order in h.
        mu, nu: tensor indices (sign-preserving — pass them in whatever sign
            they have in the original factor).
    """
    if n == 0:
        return metric(mu, nu)
    if n == 1:
        return 2 * kappa * h(mu, nu)
    return S.Zero


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
    
    # g^{lambda sigma} only needs order n-1 (the dh factor is already order 1).
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
    (e.g. for a scalar: L_M = -1/2 ∂φ·∂φ). L̃_M is the covariantized form
    obtained by:
      (a) η^{μν} → g^{μν}    and    η_{μν} → g_{μν}    (metric promotion)
      (b) covariantize vector derivatives:
            upstairs V^σ:   ∂_ρ V^σ → ∇_ρ V^σ = ∂_ρ V^σ + Γ^σ_{ρτ} V^τ
            downstairs A_σ: ∂_ρ A_σ → ∇_ρ A_σ = ∂_ρ A_σ − Γ^τ_{ρσ} A_τ
      (c) overall √|g| volume factor

    On (b): the Christoffel correction applies to ANY rank-1 vector field
    and is implemented for both orientations. Whether it survives depends on
    the combination it sits in:
      - DOWNSTAIRS A: a gauge field enters only through F_{μν} = ∂_μ A_ν −
        ∂_ν A_μ, the exterior derivative of the one-form A. That is
        connection-independent — ∇_μ A_ν − ∇_ν A_μ = ∂_μ A_ν − ∂_ν A_μ
        because the symmetric Γ^τ_{μν} cancels between the two terms — so (b)
        is a no-op for pure F^2 matter (EM). This cancellation is special to
        the downstairs/exterior-derivative structure; in any
        NON-antisymmetric combination of A's derivatives it does NOT cancel,
        so we apply (b) unconditionally rather than assuming the F-structure.
      - UPSTAIRS V: there is no analogous cancellation. (Antisymmetrizing a
        field strength means putting both indices in the same position; that
        cancellation is the downstairs one-form's exterior-derivative
        property and has no upstairs counterpart.) The Christoffel term in
        ∇_ρ V^σ genuinely survives, so (b) always contributes for V.
    The cancellation, where it occurs, happens term-by-term across the
    expanded monomials under canon. (The Hilbert procedure carries the same
    corrections on the EM-tensor side — see
    energy_momentum._christoffel_via_substitution.)

    The order-n piece sums over compositions n = k_sqrt + Σ k_metric + Σ k_dvec
    where each metric factor and each vector d-field factor independently picks
    an h-order to contribute. Metric factors expand via inverse_metric_order
    (up-up) or metric_order (down-down); vector d-field factors contribute the
    raw ∂V/∂A at k=0 or the order-k Christoffel correction at k≥1.

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
    from bootstrap.tensor_algebra import (
        _matter_fields, _JET_HIERARCHY, NATURAL_POSITIONS,
    )

    if L_M == S.Zero:
        return S.Zero
    if n < 0:
        return S.Zero

    if n == 0:
        return canon(L_M)

    # Build {dfield_head: (field_head, orientation)} for rank-1 vector fields
    # whose covariant derivative carries a Christoffel correction. BOTH
    # upstairs V^σ and downstairs A_σ are included; they differ in the sign
    # and index placement of the correction (handled in the expansion below).
    dvec_dfields = {}
    for _nm, info in _matter_fields.items():
        if info.get('rank') == 1 and info.get('index_pos') in ('up', 'down'):
            dvec_dfields[info['dfield']] = (info['field'], info['index_pos'])

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

        # Classify factors into three buckets:
        #   metric_factors:  list of (indices, 'up_up'|'down_down')
        #   dvec_factors:    list of (field_idx, deriv_idx, field_head,
        #                             dfield_head, orientation)
        #   kept_factors:    everything else, passed through unchanged
        metric_factors = []
        dvec_factors = []
        kept_factors = []
        for f in factors:
            comp = _get_component(f)
            if comp is metric:
                inds = _get_indices(f)
                if all(idx.is_up for idx in inds):
                    metric_factors.append((inds, 'up_up'))
                elif all(not idx.is_up for idx in inds):
                    metric_factors.append((inds, 'down_down'))
                else:
                    # mixed → Kronecker delta; passes through unchanged.
                    kept_factors.append(f)
                continue
            if comp in dvec_dfields:
                inds = _get_indices(f)
                field_head, orientation = dvec_dfields[comp]
                # Natural d-field positions are ['up','down'] (upstairs V) or
                # ['down','down'] (downstairs A); in both, index 0 is the field
                # index and index 1 the derivative index. After
                # uncontract_metrics the factor is in this natural form.
                if NATURAL_POSITIONS.get(comp) in (['up', 'down'], ['down', 'down']):
                    dvec_factors.append(
                        (inds[0], inds[1], field_head, comp, orientation)
                    )
                    continue
            kept_factors.append(f)

        M = len(metric_factors)
        D = len(dvec_factors)

        # Sum over compositions (k_sqrt, k_metrics..., k_dvecs...) with total n.
        for ks in _compositions(n, M + D + 1):
            k_sqrt = ks[0]
            k_metrics = ks[1:1 + M]
            k_dvecs = ks[1 + M:]
            sg = sqrt_det_g_order(k_sqrt)
            if sg == S.Zero and k_sqrt > 0:
                continue
            piece = coeff * sg
            # Metric expansion.
            zeroed = False
            for ((mi, ni), kind), k_i in zip(metric_factors, k_metrics):
                if kind == 'up_up':
                    g_part = inverse_metric_order(k_i, mi, ni)
                else:
                    g_part = metric_order(k_i, mi, ni)
                if g_part == S.Zero:
                    zeroed = True
                    break
                piece = piece * g_part
            if zeroed:
                continue
            # Vector-dfield expansion: raw partial at k=0, Christoffel piece
            # at k≥1. Upstairs V^σ: ∇_ρ V^σ = ∂_ρ V^σ + Γ^σ_{ρτ} V^τ.
            # Downstairs A_σ: ∇_ρ A_σ = ∂_ρ A_σ − Γ^τ_{ρσ} A_τ. For a gauge
            # field the Γ cancels in the antisymmetric F_{ρσ}, but that
            # cancellation happens term-by-term across the expanded monomials
            # under canon — we apply the correction unconditionally and let it
            # cancel where it should.
            for (field_idx, deriv_idx, field_head, dfield_head, orientation), k_dv \
                    in zip(dvec_factors, k_dvecs):
                if k_dv == 0:
                    # Raw ∂_ρ V^σ / ∂_ρ A_σ piece.
                    dvec_part = dfield_head(field_idx, deriv_idx)
                elif orientation == 'up':
                    # +Γ^{field}_{deriv τ} V^τ. christoffel_order args are
                    # (k, lambda_up, mu_down, nu_down); field_idx is up,
                    # deriv_idx down.
                    tau, = fresh_indices(1)
                    Gamma_k = christoffel_order(k_dv, field_idx, deriv_idx, -tau)
                    dvec_part = (Gamma_k * field_head(tau)
                                 if Gamma_k != S.Zero else S.Zero)
                else:
                    # −Γ^τ_{deriv field} A_τ. field_idx and deriv_idx are both
                    # down; Γ is symmetric in its lower pair so their order is
                    # immaterial.
                    tau, = fresh_indices(1)
                    Gamma_k = christoffel_order(k_dv, tau, deriv_idx, field_idx)
                    dvec_part = (-Gamma_k * field_head(-tau)
                                 if Gamma_k != S.Zero else S.Zero)
                if dvec_part == S.Zero:
                    zeroed = True
                    break
                piece = piece * dvec_part
            if zeroed:
                continue
            for f in kept_factors:
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


# ---------------------------------------------------------------------------
# Nonminimal matter-curvature coupling expansion
# ---------------------------------------------------------------------------

def covariant_coupling_order(coupling, n):
    """Order-n (in h) part of sqrt|g| * a nonminimal coupling C(fields)*Riemann.

    `coupling` is a derivative-free function of the matter fields times one or
    more `Riemann` factors (e.g. xi*phi**2 * R, with the Ricci scalar R written
    as metric(mu,nu)*Riemann(lam,-mu,-lam,-nu)). The expansion:
      - each Riemann factor -> riemann_order(k) (contributes at order k>=1,
        since R^(0)=0 on the flat background; k=0 compositions self-prune);
      - metric factors (from contracting Riemann down to Ricci/scalar) ->
        inverse_metric_order (up-up) / metric_order (down-down);
      - matter factors C(fields) pass through unchanged;
      - overall sqrt|g| volume factor.
    Mirrors `matter_lagrangian_order`'s composition machinery. Riemann indices
    are assumed natural (up,down,down,down) after uncontract_metrics, which
    inserts explicit metric factors for any non-natural index.
    """
    from bootstrap.energy_momentum import uncontract_metrics
    from bootstrap.jet import (
        _decompose_tensmul, _get_component, _get_indices, _sum_terms,
    )

    if coupling == S.Zero or n < 0:
        return S.Zero

    L_unc = uncontract_metrics(coupling)
    terms_in = L_unc.args if isinstance(L_unc, TensAdd) else [L_unc]
    out_terms = []
    for term in terms_in:
        if isinstance(term, TensMul):
            coeff, factors = _decompose_tensmul(term)
        elif isinstance(term, Tensor):
            coeff, factors = S.One, [term]
        else:
            coeff, factors = term, []

        metric_factors = []   # ((mi, ni), 'up_up'|'down_down')
        riemann_factors = []  # index tuples (natural up,down,down,down)
        kept_factors = []     # matter C(fields), passed through
        for f in factors:
            comp = _get_component(f)
            if comp is metric:
                inds = _get_indices(f)
                if all(idx.is_up for idx in inds):
                    metric_factors.append((inds, 'up_up'))
                elif all(not idx.is_up for idx in inds):
                    metric_factors.append((inds, 'down_down'))
                else:
                    kept_factors.append(f)  # mixed -> Kronecker delta
                continue
            if comp is Riemann:
                riemann_factors.append(_get_indices(f))
                continue
            kept_factors.append(f)

        M = len(metric_factors)
        R = len(riemann_factors)
        for ks in _compositions(n, M + R + 1):
            k_sqrt = ks[0]
            k_metrics = ks[1:1 + M]
            k_riems = ks[1 + M:]
            sg = sqrt_det_g_order(k_sqrt)
            if sg == S.Zero and k_sqrt > 0:
                continue
            piece = coeff * sg
            zeroed = False
            for ((mi, ni), kind), k_i in zip(metric_factors, k_metrics):
                g_part = (inverse_metric_order(k_i, mi, ni) if kind == 'up_up'
                          else metric_order(k_i, mi, ni))
                if g_part == S.Zero:
                    zeroed = True
                    break
                piece = piece * g_part
            if zeroed:
                continue
            for inds, k_r in zip(riemann_factors, k_riems):
                # Riemann starts at order 1; riemann_order(0)=0 prunes k_r=0.
                # Re-index to FRESH '_i' indices before calling riemann_order:
                # it runs total_derivative/canon internally, which clashes with
                # canon'd 'L_n' names (dummy_name='L'); fresh names dodge it,
                # the same way ricci_scalar_order does. Internal contractions
                # (same index name in two slots, e.g. the Ricci self-trace) are
                # preserved by mapping equal names to the same fresh index.
                name_to_fresh = {}
                fresh_inds = []
                for idx in inds:
                    if idx.name not in name_to_fresh:
                        name_to_fresh[idx.name] = fresh_indices(1)[0]
                    fr = name_to_fresh[idx.name]
                    fresh_inds.append(fr if idx.is_up else -fr)
                r_part = riemann_order(k_r, *fresh_inds)
                if r_part == S.Zero:
                    zeroed = True
                    break
                # Relabel the surviving (external/free) fresh indices back to
                # the originals via delta-metric contraction + canon. A raw
                # substitute_indices collides with riemann_order's internal
                # 'L_n' dummies; canon renames the colliding dummy instead.
                fresh_to_orig = {}
                for orig, fr_s in zip(inds, fresh_inds):
                    fr_pos = fr_s if fr_s.is_up else -fr_s
                    fresh_to_orig[fr_pos] = orig if orig.is_up else -orig
                for fidx in list(r_part.get_free_indices()):
                    fr_pos = fidx if fidx.is_up else -fidx
                    orig_pos = fresh_to_orig.get(fr_pos)
                    if orig_pos is None or orig_pos == fr_pos:
                        continue
                    delta = (metric(-fr_pos, orig_pos) if fidx.is_up
                             else metric(fr_pos, -orig_pos))
                    r_part = canon(r_part * delta)
                piece = piece * r_part
            if zeroed:
                continue
            for f in kept_factors:
                piece = piece * f
            out_terms.append(piece)

    result = _sum_terms(out_terms)
    return canon(result) if isinstance(result, TensExpr) else result

