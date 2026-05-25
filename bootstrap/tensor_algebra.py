"""
Core tensor algebra for the GR bootstrapping computation.

Provides the abstract-index tensor infrastructure:
- Lorentz index type and fresh index generation
- TensorHead definitions for all jet-space variables (h, dh, ddh, matter fields)
- Helper functions for raising/lowering, symmetrization, traces
- Order-in-h extraction (filtering by power of h)

All indices are abstract — we never fix the spacetime dimension or expand
into components. SymPy's tensor module handles canonicalization and
dummy-index renaming.
"""

from sympy.tensor.tensor import (
    TensorIndexType, TensorHead, TensorIndex, TensorSymmetry,
    tensor_indices, TensAdd, TensMul, TensExpr, Tensor, TensorManager
)
from sympy import Rational, S, Symbol, Add, Mul, Number
from functools import lru_cache

# ---------------------------------------------------------------------------
# Index type and metric
# ---------------------------------------------------------------------------

# Dimension left symbolic (not fixed to 4). SymPy uses 'dim' for traces like
# eta^mu_mu = d. We use None to leave it unspecified (no automatic trace
# simplification involving d), which is fine since we rarely need such traces.
Lorentz = TensorIndexType('Lorentz', dummy_name='L', dim=Symbol('d'))

# The metric is Lorentz.metric — SymPy provides it automatically.
# metric(mu, -nu) acts as Kronecker delta, metric(-mu,-nu) as eta_munu, etc.
metric = Lorentz.metric

# ---------------------------------------------------------------------------
# Fresh index generation
# ---------------------------------------------------------------------------
# We maintain a global counter to produce unique index names, avoiding
# clashes when indices are created in different contexts (cf. the GR.fr
# approach of using Module[] for local scoping).

_index_counter = 0

def fresh_indices(n):
    """Generate n fresh, unique Lorentz tensor indices.
    
    Returns a tuple of TensorIndex objects with names like _i0, _i1, ...
    Each call produces indices that do not clash with any previously
    generated ones.
    """
    global _index_counter
    names = ' '.join(f'_i{_index_counter + k}' for k in range(n))
    _index_counter += n
    result = tensor_indices(names, Lorentz)
    if n == 1:
        return (result,)  # tensor_indices returns a single index for n=1
    return result

def named_indices(names):
    """Create Lorentz indices with specific names (for readability).
    
    Args:
        names: space-separated string of index names, e.g. 'mu nu rho'
    
    Returns:
        Tuple of TensorIndex objects.
    """
    result = tensor_indices(names, Lorentz)
    if isinstance(result, TensorIndex):
        return (result,)
    return result

# ---------------------------------------------------------------------------
# Symmetry specifications
# ---------------------------------------------------------------------------
# TensorSymmetry objects for the various jet variables.

# Fully symmetric pair (for h_munu)
_sym2 = TensorSymmetry.fully_symmetric(2)

# Symmetric in first 2 of 3 indices (for dh = h_{mu nu, rho})
# direct_product(2, 1) means: symmetric pair + single index
_sym_dh = TensorSymmetry.direct_product(2, 1)

# Symmetric in first 2 AND in last 2 of 4 indices (for ddh = h_{mu nu, rho sigma})
_sym_ddh = TensorSymmetry.direct_product(2, 2)

# ---------------------------------------------------------------------------
# Gravitational field jet variables
# ---------------------------------------------------------------------------

# h_{mu,nu}: the gravitational field perturbation (symmetric rank-2)
h = TensorHead('h', [Lorentz, Lorentz], _sym2)

# dh_{mu,nu,rho} = h_{mu nu, rho}: first derivative (symmetric in first 2)
dh = TensorHead('dh', [Lorentz, Lorentz, Lorentz], _sym_dh)

# ddh_{mu,nu,rho,sigma} = h_{mu nu, rho sigma}: second derivative
# (symmetric in first 2 and in last 2)
ddh = TensorHead('ddh', [Lorentz, Lorentz, Lorentz, Lorentz], _sym_ddh)

# dddh_{mu,nu,rho,sigma,tau} = h_{mu nu, rho sigma tau}: third derivative
# (symmetric in first 2 and in last 3). Only used for conservation checks.
_sym_dddh = TensorSymmetry.direct_product(2, 3)
dddh = TensorHead('dddh', [Lorentz]*5, _sym_dddh)

# ---------------------------------------------------------------------------
# Natural index positions
# ---------------------------------------------------------------------------
# 'down' = naturally covariant (lower), 'up' = naturally contravariant (upper).
# Used by covariantization (uncontract_metrics, replace_metric_with_g) and by
# the order-in-h expansion (matter_lagrangian_order) to decide between
# raising/lowering insertions and between g^{mu nu}/g_{mu nu} expansions.
NATURAL_POSITIONS = {
    h: ['down', 'down'],
    dh: ['down', 'down', 'down'],
    ddh: ['down', 'down', 'down', 'down'],
}

# ---------------------------------------------------------------------------
# Registry of jet variable relationships
# ---------------------------------------------------------------------------
# Maps each TensorHead to its "parent" (one fewer derivative) and "child"
# (one more derivative). This is used by total_derivative.

# For gravity:
_JET_HIERARCHY = {
    h: {'child': dh, 'n_field_indices': 2},
    dh: {'parent': h, 'child': ddh, 'n_field_indices': 2},
    ddh: {'parent': dh, 'child': dddh, 'n_field_indices': 2},
    dddh: {'parent': ddh, 'n_field_indices': 2},
}

# We'll add matter fields dynamically via register_matter_field()

# ---------------------------------------------------------------------------
# Matter field registration
# ---------------------------------------------------------------------------

_matter_fields = {}  # name -> dict with keys: field, dfield, ddfield, rank, ...

def register_scalar_field(name):
    """Register a scalar matter field (rank 0).

    Creates TensorHeads for the field, its first and second derivatives.
    Also populates NATURAL_POSITIONS for the d- and dd-heads.
    Returns (field, dfield, ddfield).
    """
    phi = TensorHead(name, [], TensorSymmetry.fully_symmetric(0))
    dphi = TensorHead(f'd{name}', [Lorentz], TensorSymmetry.no_symmetry(1))
    ddphi = TensorHead(f'dd{name}', [Lorentz, Lorentz], _sym2)

    _JET_HIERARCHY[phi] = {'child': dphi, 'n_field_indices': 0}
    _JET_HIERARCHY[dphi] = {'parent': phi, 'child': ddphi, 'n_field_indices': 0}
    _JET_HIERARCHY[ddphi] = {'parent': dphi, 'n_field_indices': 0}

    NATURAL_POSITIONS[dphi] = ['down']
    NATURAL_POSITIONS[ddphi] = ['down', 'down']

    info = {'field': phi, 'dfield': dphi, 'ddfield': ddphi, 'rank': 0, 'name': name}
    _matter_fields[name] = info
    return phi, dphi, ddphi

def register_vector_field(name):
    """Register a vector matter field (rank 1) with a naturally DOWN field index.

    The field index is conventionally covariant (A_μ). Use
    register_upstairs_vector_field for the V^μ convention.

    Creates TensorHeads for the field, its first and second derivatives.
    Also populates NATURAL_POSITIONS so callers don't need to set it manually.
    Returns (field, dfield, ddfield).
    """
    A = TensorHead(name, [Lorentz], TensorSymmetry.no_symmetry(1))
    dA = TensorHead(f'd{name}', [Lorentz, Lorentz], TensorSymmetry.no_symmetry(2))
    # ddA_{mu,rho sigma}: field index + symmetric pair of derivative indices
    # direct_product(1, 2) = single index + symmetric pair
    _sym_vec_dd = TensorSymmetry.direct_product(1, 2)
    ddA = TensorHead(f'dd{name}', [Lorentz, Lorentz, Lorentz], _sym_vec_dd)

    _JET_HIERARCHY[A] = {'child': dA, 'n_field_indices': 1}
    _JET_HIERARCHY[dA] = {'parent': A, 'child': ddA, 'n_field_indices': 1}
    _JET_HIERARCHY[ddA] = {'parent': dA, 'n_field_indices': 1}

    NATURAL_POSITIONS[A] = ['down']
    NATURAL_POSITIONS[dA] = ['down', 'down']
    NATURAL_POSITIONS[ddA] = ['down', 'down', 'down']

    info = {'field': A, 'dfield': dA, 'ddfield': ddA, 'rank': 1, 'name': name,
            'index_pos': 'down'}
    _matter_fields[name] = info
    return A, dA, ddA


def register_upstairs_vector_field(name):
    """Register a vector matter field with a naturally UP field index (V^mu).

    Mirrors register_vector_field but with NATURAL_POSITIONS set to
    ['up'] for V, ['up', 'down'] for dV (field up, derivative down), and
    ['up', 'down', 'down'] for ddV.

    The dV convention is dV(field_index, deriv_index) = ∂_{deriv} V^{field}.
    Covariantization of dV introduces a +Γ^{field}_{deriv σ} V^σ Christoffel
    correction (opposite sign from the downstairs case); this is wired up in
    matter_lagrangian_order and _christoffel_via_substitution.

    Returns (field, dfield, ddfield).
    """
    V = TensorHead(name, [Lorentz], TensorSymmetry.no_symmetry(1))
    dV = TensorHead(f'd{name}', [Lorentz, Lorentz], TensorSymmetry.no_symmetry(2))
    _sym_vec_dd = TensorSymmetry.direct_product(1, 2)
    ddV = TensorHead(f'dd{name}', [Lorentz, Lorentz, Lorentz], _sym_vec_dd)

    _JET_HIERARCHY[V] = {'child': dV, 'n_field_indices': 1}
    _JET_HIERARCHY[dV] = {'parent': V, 'child': ddV, 'n_field_indices': 1}
    _JET_HIERARCHY[ddV] = {'parent': dV, 'n_field_indices': 1}

    NATURAL_POSITIONS[V] = ['up']
    NATURAL_POSITIONS[dV] = ['up', 'down']
    NATURAL_POSITIONS[ddV] = ['up', 'down', 'down']

    info = {'field': V, 'dfield': dV, 'ddfield': ddV, 'rank': 1, 'name': name,
            'index_pos': 'up'}
    _matter_fields[name] = info
    return V, dV, ddV

def get_jet_hierarchy():
    """Return the full jet hierarchy dict (read-only view)."""
    return dict(_JET_HIERARCHY)

def get_matter_fields():
    """Return info about all registered matter fields."""
    return dict(_matter_fields)

# ---------------------------------------------------------------------------
# Expression manipulation helpers
# ---------------------------------------------------------------------------

def get_tensors_in_expr(expr):
    """Extract the list of TensorHead objects appearing in a tensor expression."""
    if isinstance(expr, TensMul):
        return [t.component for t in expr.args if hasattr(t, 'component')]
    elif isinstance(expr, TensAdd):
        heads = set()
        for term in expr.args:
            heads.update(get_tensors_in_expr(term))
        return list(heads)
    elif hasattr(expr, 'component'):
        return [expr.component]
    return []

def order_in_h(expr):
    """Determine the order (power of h) in a tensor monomial.
    
    Counts all h-type jet variables (h, dh, ddh) as contributing 1 each.
    Only works on a single monomial (TensMul), not a sum.
    """
    if isinstance(expr, (int, float, Number)):
        return 0
    if isinstance(expr, TensMul):
        count = 0
        for t in expr._tids if hasattr(expr, '_tids') else []:
            pass
        # Walk the tensor factors
        heads = _get_heads_from_tensmul(expr)
        h_heads = {h, dh, ddh}
        return sum(1 for hd in heads if hd in h_heads)
    if hasattr(expr, 'component'):
        if expr.component in {h, dh, ddh}:
            return 1
        return 0
    return 0

def _get_heads_from_tensmul(expr):
    """Get all TensorHead components from a TensMul expression."""
    result = []
    for arg in expr.args:
        if hasattr(arg, 'component'):
            result.append(arg.component)
        elif isinstance(arg, TensMul):
            result.extend(_get_heads_from_tensmul(arg))
    return result

def filter_by_order(expr, n):
    """Extract terms of order n in h from a tensor expression.
    
    Args:
        expr: a TensAdd (sum of terms) or single term
        n: desired order in h (counting h, dh, ddh each as 1)
        
    Returns:
        Sum of terms that have exactly n powers of h-type fields.
    """
    if isinstance(expr, TensAdd):
        terms = [t for t in expr.args if order_in_h(t) == n]
        if not terms:
            return S.Zero
        if len(terms) == 1:
            return terms[0]
        return TensAdd(*terms)
    else:
        if order_in_h(expr) == n:
            return expr
        return S.Zero

def canon(expr):
    """Canonicalize a tensor expression.

    Renames dummy indices and applies tensor symmetries so that
    algebraically identical terms are represented identically. Should be
    called frequently to keep expressions compact — especially after jet
    derivatives that produce symmetrized Kronecker deltas.

    Workaround: sympy's `canon_bp` and `contract_metric` crash with an
    `IndexError` deep in `combinatorics/tensor_can.py` when a TensMul
    contains a 0-index Tensor factor (e.g. `phi()` for a registered
    scalar field — appears in matter potential terms like phi^2 or
    phi^4). Sympy's canonicalizer assumes every Tensor argument carries
    at least one index. We detect this case and apply a strip/recombine
    workaround per-term; in the common case where there are no 0-index
    Tensor factors we fall through to the standard sympy path (which
    correctly does cross-term simplifications on a whole TensAdd).
    """
    if isinstance(expr, (int, float)) or expr == S.Zero:
        return expr
    if not isinstance(expr, TensExpr):
        return expr
    if not _has_zero_index_tensor(expr):
        return _canon_indexed(expr)

    # We have 0-index Tensors somewhere in expr. If expr is a TensMul that
    # contains a TensAdd factor (e.g. `metric * (A + B)` where one of the
    # B terms has phi^2), distribute first so our TensAdd-distribution
    # path below sees flat terms instead of a wrapped sum.
    if isinstance(expr, TensMul) and any(isinstance(a, TensAdd) for a in expr.args):
        expanded = expr.expand()
        if expanded is not expr:
            return canon(expanded)

    # Workaround path: 0-index Tensor factors are present somewhere.
    # Distribute over TensAdd. When a recursive canon call returns a
    # TensAdd, flatten it into new_terms — otherwise we end up with
    # nested TensAdds that sympy's collect_terms doesn't combine across,
    # so cancelable like-terms across different branches survive.
    if isinstance(expr, TensAdd):
        new_terms = []
        for t in expr.args:
            ct = canon(t)
            if ct is S.Zero or ct == 0:
                continue
            if isinstance(ct, TensAdd):
                new_terms.extend(ct.args)
            else:
                new_terms.append(ct)
        if not new_terms:
            return S.Zero
        if len(new_terms) == 1:
            return new_terms[0]
        # .doit() triggers TensAdd._tensAdd_collect_terms which combines
        # like terms (cancellable pairs etc.); the bare constructor doesn't.
        result = TensAdd(*new_terms).doit()
        return result

    if isinstance(expr, TensMul):
        # Recursively strip 0-index Tensor factors from anywhere in the
        # TensMul tree (they can be nested inside TensMul-of-TensMul or
        # appear after sympy operations that don't fully flatten).
        stripped, removed = _strip_zero_index_recursive(expr)
        if not removed:
            return _canon_indexed(expr)
        if isinstance(stripped, TensExpr):
            # If the stripped form still somehow contains 0-index tensors
            # (e.g. nested TensAdd survived), recurse via canon.
            if _has_zero_index_tensor(stripped):
                stripped_canon = canon(stripped)
            else:
                stripped_canon = _canon_indexed(stripped)
        else:
            stripped_canon = stripped
        result = stripped_canon
        for st in removed:
            result = result * st
        return result

    # Single 0-index Tensor — nothing to canonicalize on.
    return expr


def _strip_zero_index_recursive(expr):
    """Walk expr; return (expr_with_0index_Tensor_factors_replaced_by_1,
    list_of_those_tensors). Recurses into nested TensMul; treats TensAdd
    as a leaf (the outer canon() distributes over TensAdd first)."""
    if isinstance(expr, Tensor):
        if len(expr.get_indices()) == 0:
            return S.One, [expr]
        return expr, []
    if isinstance(expr, TensMul):
        new_args = []
        scalars = []
        for a in expr.args:
            sub_stripped, sub_scalars = _strip_zero_index_recursive(a)
            scalars.extend(sub_scalars)
            if sub_stripped is not S.One and sub_stripped != 1:
                new_args.append(sub_stripped)
        if not new_args:
            return S.One, scalars
        if len(new_args) == 1:
            return new_args[0], scalars
        try:
            return TensMul(*new_args).doit(), scalars
        except Exception:
            result = new_args[0]
            for a in new_args[1:]:
                result = result * a
            return result, scalars
    return expr, []


def _canon_indexed(expr):
    """Direct canonicalization; assumes no 0-index Tensor factors that would
    crash sympy's canon_bp / contract_metric."""
    if hasattr(expr, 'contract_metric'):
        expr = expr.contract_metric(metric)
    if hasattr(expr, 'canon_bp'):
        expr = expr.canon_bp()
    return expr


def _has_zero_index_tensor(expr):
    """Walk a TensExpr looking for any 0-index Tensor factor."""
    if isinstance(expr, Tensor):
        return len(expr.get_indices()) == 0
    if isinstance(expr, TensMul):
        return any(_has_zero_index_tensor(a) for a in expr.args)
    if isinstance(expr, TensAdd):
        return any(_has_zero_index_tensor(a) for a in expr.args)
    return False

# ---------------------------------------------------------------------------
# Kronecker delta helper
# ---------------------------------------------------------------------------

def kdelta(up_idx, dn_idx):
    """Create a Kronecker delta: delta^up_dn = metric(up, -dn).
    
    Both arguments should be TensorIndex. up_idx should be contravariant
    (positive), dn_idx should be covariant (negative). If signs are wrong,
    they will be adjusted.
    """
    # Ensure up is positive, dn is negative
    if up_idx.is_up and not dn_idx.is_up:
        return metric(up_idx, dn_idx)
    elif not up_idx.is_up and dn_idx.is_up:
        return metric(dn_idx, up_idx)
    else:
        # Both same sign — use metric to handle it
        return metric(up_idx, dn_idx)

# ---------------------------------------------------------------------------
# Symmetrized delta for h-type derivatives
# ---------------------------------------------------------------------------

def symmetrized_h_delta(free_indices, field_indices):
    """Produce the symmetrized Kronecker delta for differentiating by h.
    
    When computing ∂h_{μν}/∂h_{αβ}, the result is:
        ½(δ^α_μ δ^β_ν + δ^α_ν δ^β_μ)
    
    This function produces that expression.
    
    Args:
        free_indices: (alpha, beta) — the indices on the differentiation 
            variable (will appear as free upper indices in the result)
        field_indices: (mu, nu) — the indices on the h being differentiated
            (will be contracted away in the context of the full expression)
    
    Returns:
        Tensor expression for the symmetrized delta product.
    """
    a, b = free_indices
    m, n = field_indices
    # ½(δ^a_m δ^b_n + δ^a_n δ^b_m)
    term1 = metric(a, -m) * metric(b, -n)
    term2 = metric(a, -n) * metric(b, -m)
    return Rational(1, 2) * (term1 + term2)
