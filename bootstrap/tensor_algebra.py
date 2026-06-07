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
from sympy import Rational, S, Symbol, Add, Mul, Number, cancel, default_sort_key
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
    if n == 0:
        return ()  # zero fresh indices (e.g. relabelling a scalar's empty
                   # EOM-direction pair); tensor_indices('') would raise.
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

def swap_free_indices(expr, idx1, idx2):
    """Swap two free indices in a tensor expression (both signs).

    A formal relabelling: every occurrence of idx1 becomes idx2 and vice
    versa, routed through a fresh temporary index to avoid clashes. Does
    NOT canonicalize — callers that compare expressions should canon the
    result themselves.
    """
    if expr == S.Zero or not hasattr(expr, 'substitute_indices'):
        return expr
    tmp, = fresh_indices(1)
    return expr.substitute_indices(
        (idx1, tmp), (-idx1, -tmp)
    ).substitute_indices(
        (idx2, idx1), (-idx2, -idx1)
    ).substitute_indices(
        (tmp, idx2), (-tmp, -idx2)
    )

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
    # Third derivative: only needed so total_derivative(ddphi) is defined,
    # e.g. for the identically-conserved check on a nonminimal-coupling
    # improvement (the matter analogue of dddh for gravity conservation).
    dddphi = TensorHead(f'ddd{name}', [Lorentz] * 3,
                        TensorSymmetry.fully_symmetric(3))

    _JET_HIERARCHY[phi] = {'child': dphi, 'n_field_indices': 0}
    _JET_HIERARCHY[dphi] = {'parent': phi, 'child': ddphi, 'n_field_indices': 0}
    _JET_HIERARCHY[ddphi] = {'parent': dphi, 'child': dddphi, 'n_field_indices': 0}
    _JET_HIERARCHY[dddphi] = {'parent': ddphi, 'n_field_indices': 0}

    NATURAL_POSITIONS[dphi] = ['down']
    NATURAL_POSITIONS[ddphi] = ['down', 'down']
    NATURAL_POSITIONS[dddphi] = ['down', 'down', 'down']

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
    # Third derivative (field index + symmetric triple): only needed so
    # total_derivative(ddA) is defined for the conservation check on a
    # nonminimal-coupling improvement (matches dddphi / the upstairs dddV).
    dddA = TensorHead(f'ddd{name}', [Lorentz] * 4, TensorSymmetry.direct_product(1, 3))

    _JET_HIERARCHY[A] = {'child': dA, 'n_field_indices': 1}
    _JET_HIERARCHY[dA] = {'parent': A, 'child': ddA, 'n_field_indices': 1}
    _JET_HIERARCHY[ddA] = {'parent': dA, 'child': dddA, 'n_field_indices': 1}
    _JET_HIERARCHY[dddA] = {'parent': ddA, 'n_field_indices': 1}

    NATURAL_POSITIONS[A] = ['down']
    NATURAL_POSITIONS[dA] = ['down', 'down']
    NATURAL_POSITIONS[ddA] = ['down', 'down', 'down']
    NATURAL_POSITIONS[dddA] = ['down', 'down', 'down', 'down']

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
    # Third derivative: field index + symmetric triple of derivative indices.
    # Only needed so total_derivative(ddV) is defined, e.g. for the identically-
    # conserved check on a nonminimal-coupling improvement (the vector analogue
    # of dddphi / dddh).
    dddV = TensorHead(f'ddd{name}', [Lorentz] * 4, TensorSymmetry.direct_product(1, 3))

    _JET_HIERARCHY[V] = {'child': dV, 'n_field_indices': 1}
    _JET_HIERARCHY[dV] = {'parent': V, 'child': ddV, 'n_field_indices': 1}
    _JET_HIERARCHY[ddV] = {'parent': dV, 'child': dddV, 'n_field_indices': 1}
    _JET_HIERARCHY[dddV] = {'parent': ddV, 'n_field_indices': 1}

    NATURAL_POSITIONS[V] = ['up']
    NATURAL_POSITIONS[dV] = ['up', 'down']
    NATURAL_POSITIONS[ddV] = ['up', 'down', 'down']
    NATURAL_POSITIONS[dddV] = ['up', 'down', 'down', 'down']

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

# ---------------------------------------------------------------------------
# Spacetime dimension control
# ---------------------------------------------------------------------------
# The Lorentz TensorIndexType's `dim` attribute is what sympy fills in for
# fully-contracted metric traces (η^μ_μ → dim). By default dim = Symbol('d')
# so traces stay symbolic. To work at a concrete dimension (e.g. d=4 for
# the traceless-T_M tests on Maxwell), call `set_dimension(4)` BEFORE any
# matter-field registration or BootstrapState creation.


def set_dimension(d):
    """Globally fix the spacetime dimension by rebuilding the Lorentz
    TensorIndexType and the gravitational jet variables.

    Args:
        d: either a concrete int (e.g. 4) or None to restore the default
           symbolic Symbol('d').

    Rebuilds the module-level Lorentz, metric, h, dh, ddh, dddh with the
    requested dim, and refreshes _JET_HIERARCHY / NATURAL_POSITIONS to
    point at the new heads.

    **Import order matters.** Other bootstrap modules
    (bootstrap_loop, eom_decompose, ...) bind their own local h/dh/ddh
    via `from bootstrap.tensor_algebra import h, dh, ddh, ...` at
    import time. If they're imported BEFORE set_dimension runs, they
    will be bound to the original (Symbol('d')) heads and will then
    mismatch anything built after the rebuild.

    The required pattern for any script that needs a non-default dim:

        # 1) Import set_dimension alone -- this is the first thing.
        from bootstrap.tensor_algebra import set_dimension
        set_dimension(4)
        # 2) Only NOW import the rest. All these modules bind to the
        #    just-rebuilt heads.
        from bootstrap.tensor_algebra import h, register_vector_field
        from bootstrap.bootstrap_loop import BootstrapState
        ...

    set_dimension also raises RuntimeError if any matter field is already
    registered (those heads carry the old Lorentz and would silently
    mismatch).
    """
    if _matter_fields:
        raise RuntimeError(
            f"set_dimension({d!r}) called after matter fields were already "
            f"registered ({list(_matter_fields.keys())}). These fields carry "
            f"the OLD Lorentz type and would mismatch the rebuilt gravitational "
            f"heads. Call set_dimension before any register_*_field()."
        )
    new_dim = Symbol('d') if d is None else d

    new_Lorentz = TensorIndexType('Lorentz', dummy_name='L', dim=new_dim)
    new_metric = new_Lorentz.metric
    new_h = TensorHead('h', [new_Lorentz, new_Lorentz], _sym2)
    new_dh = TensorHead('dh', [new_Lorentz, new_Lorentz, new_Lorentz], _sym_dh)
    new_ddh = TensorHead('ddh',
                         [new_Lorentz, new_Lorentz, new_Lorentz, new_Lorentz],
                         _sym_ddh)
    new_dddh = TensorHead('dddh', [new_Lorentz] * 5, _sym_dddh)

    g = globals()
    g['Lorentz'] = new_Lorentz
    g['metric'] = new_metric
    g['h'] = new_h
    g['dh'] = new_dh
    g['ddh'] = new_ddh
    g['dddh'] = new_dddh

    _JET_HIERARCHY.clear()
    _JET_HIERARCHY.update({
        new_h: {'child': new_dh, 'n_field_indices': 2},
        new_dh: {'parent': new_h, 'child': new_ddh, 'n_field_indices': 2},
        new_ddh: {'parent': new_dh, 'child': new_dddh, 'n_field_indices': 2},
        new_dddh: {'parent': new_ddh, 'n_field_indices': 2},
    })
    NATURAL_POSITIONS.clear()
    NATURAL_POSITIONS.update({
        new_h: ['down', 'down'],
        new_dh: ['down', 'down', 'down'],
        new_ddh: ['down', 'down', 'down', 'down'],
    })


def dimension():
    """Return the current spacetime-dimension object used by the Lorentz
    TensorIndexType — either `Symbol('d')` (default, symbolic) or a concrete
    int if `set_dimension(N)` was called.

    Use this anywhere you need the SAME `d` that fully-contracted metric
    traces (η^μ_μ → dim) produce, e.g. when building a dimension-dependent
    coupling like the conformal ξ(d) = (d−2)/(4(d−1)). Reconstructing your
    own `Symbol('d', ...)` with assumptions will NOT compare equal to the
    bare `Symbol('d')` the metric produces — so `d_yours − d_trace` won't
    collapse to 0, and traceless-T_M detection silently fails.
    """
    return Lorentz.dim


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

    Scalar-coefficient simplification (symbolic d only): canon_bp combines
    like tensor structures but does NOT run rational-function simplification
    on the scalar coefficients. With a symbolic spacetime dimension d and a
    d-dependent coupling (e.g. the conformal ξ(d) = (d−2)/(4(d−1))), the
    coefficients become rational functions of d that may be identically zero
    yet survive as unsimplified sums — inflating term counts and defeating
    every `== S.Zero` gate (traceless-T_M, H2 Z=0, EL self-consistency,
    E_diff=0). After canonicalizing structure, we therefore run `cancel` on
    each term's coefficient and drop the ones that vanish. This pass is GATED
    on d being symbolic, so pure-gravity and set_dimension(N) runs (where the
    dimension is a concrete int) pay nothing.
    """
    result = _canon_impl(expr)
    return _simplify_d_coeffs(result)


def _canon_impl(expr):
    """Structural canonicalization (dummy renaming + symmetry + metric
    contraction), without the d-coefficient simplification pass. Recursion
    stays inside this function so the (gated) coefficient pass runs only once,
    on the fully-combined top-level result returned by `canon`."""
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
            return _canon_impl(expanded)

    # Workaround path: 0-index Tensor factors are present somewhere.
    # Distribute over TensAdd. When a recursive canon call returns a
    # TensAdd, flatten it into new_terms — otherwise we end up with
    # nested TensAdds that sympy's collect_terms doesn't combine across,
    # so cancelable like-terms across different branches survive.
    if isinstance(expr, TensAdd):
        new_terms = []
        for t in expr.args:
            ct = _canon_impl(t)
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
                stripped_canon = _canon_impl(stripped)
            else:
                stripped_canon = _canon_indexed(stripped)
        else:
            stripped_canon = stripped
        result = stripped_canon
        # Re-multiply the stripped 0-index Tensors in a CANONICAL ORDER. They
        # are commuting scalars (e.g. matter fields phi1, phi2), but sympy's
        # TensMul does not reorder 0-index Tensors, and this manual re-multiply
        # preserves source order — so phi1*phi2 and phi2*phi1 would stay
        # distinct and fail to combine/cancel in a later TensAdd (e.g. a trace
        # term phi1*phi2 - phi2*phi1 that is identically zero). Sorting by a
        # stable key normalizes both to the same form.
        for st in sorted(removed, key=default_sort_key):
            result = result * st
        return result

    # Single 0-index Tensor — nothing to canonicalize on.
    return expr


def _simplify_d_coeffs(expr):
    """Run `cancel` on each term's scalar coefficient and drop vanishing
    terms. Gated on the spacetime dimension being symbolic — when
    `set_dimension(N)` has fixed a concrete int dimension (or on the
    pure-gravity path), there is no d Symbol to simplify and we return
    `expr` untouched, so those hot paths pay nothing.

    See `canon` for why this is necessary. Per-term we only invoke `cancel`
    when the coefficient actually contains d, so d-free terms in an otherwise
    d-dependent expression are also cheap.
    """
    d_sym = Lorentz.dim
    if not isinstance(d_sym, Symbol):
        return expr  # concrete dimension → no rational-in-d coefficients
    if not isinstance(expr, TensExpr):
        # Bare scalar (rare at this layer): cancel if it mentions d.
        if isinstance(expr, (int, float)):
            return expr
        return cancel(expr) if d_sym in getattr(expr, 'free_symbols', set()) else expr
    # Fast path: if no coefficient anywhere mentions d, there is nothing to
    # simplify. This keeps pure-gravity / d-free matter runs (which use the
    # default symbolic d but never form d-rational coefficients) at the cost
    # of a single free_symbols membership check.
    if d_sym not in expr.free_symbols:
        return expr

    def _simp_coeff_term(term):
        """Return term with its coefficient cancelled, or None if it vanishes."""
        if isinstance(term, TensMul):
            c = term.coeff
            if d_sym not in c.free_symbols:
                return term
            cc = cancel(c)
            if cc == S.Zero:
                return None
            if cc == c:
                return term
            return cc * term.nocoeff
        # Bare Tensor (coeff 1) or other — no d-coefficient to simplify.
        return term

    if isinstance(expr, TensAdd):
        kept = []
        for term in expr.args:
            st = _simp_coeff_term(term)
            if st is None or st is S.Zero or st == 0:
                continue
            kept.append(st)
        if not kept:
            return S.Zero
        if len(kept) == 1:
            return kept[0]
        return TensAdd(*kept)

    st = _simp_coeff_term(expr)
    return S.Zero if st is None else st


def _strip_zero_index_recursive(expr):
    """Walk expr; return (expr_with_0index_Tensor_factors_replaced_by_1,
    list_of_those_tensors). Recurses into nested TensMul; treats TensAdd
    as a leaf (the outer canon() distributes over TensAdd first).

    Short-circuit: if nothing was stripped from a TensMul subtree, return
    the original expr untouched — avoids a TensMul(*new_args).doit()
    rebuild on every recursion when the expression has no 0-index Tensor
    factors anywhere in the subtree (the common case once stripping has
    been done at a higher level).
    """
    if isinstance(expr, Tensor):
        if len(expr.get_indices()) == 0:
            return S.One, [expr]
        return expr, []
    if isinstance(expr, TensMul):
        new_args = []
        scalars = []
        any_change = False
        for a in expr.args:
            sub_stripped, sub_scalars = _strip_zero_index_recursive(a)
            if sub_scalars or sub_stripped is not a:
                any_change = True
            scalars.extend(sub_scalars)
            if sub_stripped is not S.One and sub_stripped != 1:
                new_args.append(sub_stripped)
        if not any_change:
            return expr, []
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
