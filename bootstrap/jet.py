"""
Jet-space operations for the GR bootstrapping computation.

The "jet space" is the space of fields and their derivatives at a point.
Our jet variables are h_{μν}, h_{μν,ρ}, h_{μν,ρσ} (and similarly for 
matter fields). The key operations are:

1. jet_derivative: ∂/∂(field variable) — differentiates an expression 
   w.r.t. a jet variable, producing symmetrized Kronecker deltas.
   
2. total_derivative: ∂_τ — spacetime derivative acting via the chain 
   rule on jet variables (h_{μν} → h_{μν,τ}, etc.)

Both operations are the building blocks for the Euler-Lagrange derivative
and for the Helmholtz conditions.
"""

from functools import lru_cache
import os as _os
import time as _time

from sympy import S, Rational
from sympy.tensor.tensor import (
    TensAdd, TensMul, TensExpr, Tensor
)
from bootstrap.tensor_algebra import (
    metric, h, dh, ddh,
    canon, _JET_HIERARCHY,
)

# CanonAccumulator fold diagnostics (env-gated; see _fold/result). Reveals
# whether folding is re-canon'ing a non-shrinking total (the regression mode).
_REDEF_PROFILE = bool(_os.environ.get('GRB_REDEF_PROFILE'))
_CANON_PROFILE_MIN = int(_os.environ.get('GRB_REDEF_PROFILE_MIN', '200'))

# --- memory-pressure gate for folding/chunking ------------------------------
# `canon` cost is dominated by per-term Butler-Portugal (index/dummy
# permutations), NOT term count, so folding/chunking ADD canon calls = pure time
# overhead UNLESS we're actually near a RAM barrier (their only payoff is bounding
# the un-canon'd intermediate). So both are GATED on memory pressure: do the extra
# canons only when RSS exceeds a budget. Budget = GRB_MEM_BUDGET_GB if set
# (recommended: PBS script exports ~0.7x requested mem), else GRB_MEM_BUDGET_FRAC
# (default 0.7) x total system RAM. If RAM can't be measured -> no gate ->
# never fold/chunk (fast path, no memory safety). Linux /proc works without psutil.
try:
    import psutil as _psutil
except Exception:
    _psutil = None


def _total_ram_bytes():
    if _psutil is not None:
        try:
            return _psutil.virtual_memory().total
        except Exception:
            pass
    try:  # Linux fallback
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _rss_bytes():
    if _psutil is not None:
        try:
            return _psutil.Process().memory_info().rss
        except Exception:
            pass
    try:  # Linux fallback
        with open('/proc/self/statm') as f:
            return int(f.read().split()[1]) * _os.sysconf('SC_PAGE_SIZE')
    except Exception:
        return None


def _compute_mem_budget():
    gb = _os.environ.get('GRB_MEM_BUDGET_GB')
    if gb:
        try:
            return float(gb) * (1024 ** 3)
        except ValueError:
            pass
    total = _total_ram_bytes()
    if total is None:
        return None
    return total * float(_os.environ.get('GRB_MEM_BUDGET_FRAC', '0.7'))


_MEM_BUDGET = _compute_mem_budget()   # bytes, or None if unmeasurable


def _mem_pressure():
    """True iff process RSS is above the budget -> worth the extra canons of
    folding/chunking to bound the intermediate. False (fast path) if RAM can't
    be measured. Force with GRB_MEM_BUDGET_GB / GRB_MEM_BUDGET_FRAC for testing."""
    if _MEM_BUDGET is None:
        return False
    rss = _rss_bytes()
    return rss is not None and rss > _MEM_BUDGET


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sum_terms(terms):
    """One-shot sum of tensor terms, avoiding the O(N^2) repeated `result + t`.

    sympy's `TensAdd.__add__` calls `_tensAdd_collect_terms` each time, so
    a naive loop `result = result + t` re-normalizes the growing accumulator
    on every step. Building the TensAdd once from the full list runs the
    collect pass once. Skips S.Zero entries. Returns S.Zero if empty,
    the single term if length-1, otherwise a TensAdd.
    """
    nonzero = [t for t in terms if t is not S.Zero and t != 0]
    if not nonzero:
        return S.Zero
    if len(nonzero) == 1:
        return nonzero[0]
    return TensAdd(*nonzero)


class CanonAccumulator:
    """Incremental term accumulator that `canon`-folds periodically (opt #3).

    The hot builders (field-redef substitution, energy-momentum, the H2
    violation Z, the superpotential Psi) generate many terms that each carry
    distinct fresh dummies, so they do NOT combine in a plain `TensAdd` until
    `canon` normalizes the dummies. Building the whole list and `canon`-ing it
    once therefore holds every un-combined term in RAM at peak, and runs the
    single (superlinear-in-term-count) `canon` on the full pile.

    This accumulator instead folds the pending buffer into a `canon`-ed running
    total every `fold_every` additions. Because these sums combine/cancel
    heavily, the running total stays small, so peak live terms is bounded by
    (running total + fold_every) and each `canon` runs on a small input.

    Correctness: `+` is associative and `canon` is idempotent/canonicalizing,
    so the final `result()` equals `canon` of the one-shot sum of all added
    terms. (Validated per call-site via the A/B + regression checks.)

    Usage:
        acc = CanonAccumulator(fold_every=50)
        for t in produce_terms():
            acc.add(t)
        result = acc.result()        # canon-ed TensExpr (or S.Zero)
    """

    __slots__ = ('fold_every', '_buf', '_total', '_nfolds', '_nadded',
                 '_maxtotal', '_foldsecs', '_check_at')

    def __init__(self, fold_every=50):
        self.fold_every = fold_every
        self._buf = []          # pending, un-canon'd terms
        self._total = S.Zero    # canon'd running total
        self._check_at = fold_every  # buffer size at which to re-check pressure
        self._nfolds = 0        # diagnostics (GRB_REDEF_PROFILE)
        self._nadded = 0
        self._maxtotal = 0
        self._foldsecs = 0.0

    def add(self, term):
        if term is S.Zero or term == 0:
            return
        # Flatten a TensAdd into its terms so fold_every is a true term count
        # (tighter peak-RAM bound than counting each .add() as one).
        if isinstance(term, TensAdd):
            self._buf.extend(term.args)
            self._nadded += len(term.args)
        else:
            self._buf.append(term)
            self._nadded += 1
        # Fold (an extra canon) ONLY under memory pressure — otherwise accumulate
        # and canon once at result() (folding is per-term-BP overhead, see the
        # module gate). Re-check pressure every fold_every more terms.
        if len(self._buf) >= self._check_at:
            if _mem_pressure():
                self._fold()
                self._check_at = self.fold_every
            else:
                self._check_at = len(self._buf) + self.fold_every

    def _fold(self):
        if not self._buf:
            return
        if self._total is S.Zero:
            partial = _sum_terms(self._buf)
        else:
            partial = _sum_terms([self._total, *self._buf])
        if _REDEF_PROFILE:
            t0 = _time.time()
            self._total = canon(partial) if isinstance(partial, TensExpr) else partial
            self._foldsecs += _time.time() - t0
            self._nfolds += 1
            n = len(self._total.args) if isinstance(self._total, TensAdd) else 1
            if n > self._maxtotal:
                self._maxtotal = n
        else:
            self._total = canon(partial) if isinstance(partial, TensExpr) else partial
            self._nfolds += 1
        self._buf = []

    def result(self):
        self._fold()
        if _REDEF_PROFILE and self._nadded >= _CANON_PROFILE_MIN:
            final = len(self._total.args) if isinstance(self._total, TensAdd) else (
                0 if self._total is S.Zero else 1)
            print(f"      [acc] added={self._nadded} folds={self._nfolds} "
                  f"max_total={self._maxtotal} final={final} "
                  f"fold_canon={self._foldsecs:.1f}s", flush=True)
        return self._total


def apply_linear_chunked(F, expr, chunk_size=64, fold_every=128):
    """Apply a LINEAR builder `F` to a large TensAdd `expr` chunk-by-chunk.

    For any F with F(A+B) == F(A)+F(B) and F(0)==0 (the jet/total derivatives,
    the EL operator, the Hilbert/Belinfante EM -- all linear in the input
    Lagrangian), F(expr) = sum_j F(chunk_j) over a partition of expr's terms.
    Running the *whole* step (inflation AND its downstream reduction) per small
    chunk bounds the peak intermediate to ~one chunk's inflation instead of the
    full input's, while the cross-chunk combinations still happen in the final
    canon of the accumulated (already-reduced) per-chunk results.

    Handles two builder shapes:
      * bare:  F(x) -> tensor expr   (total_derivative, jet_derivative with the
        caller's FIXED free indices)
      * indexed:  F(x) -> (tensor, fresh_indices)   (euler_lagrange, the EM) --
        each chunk mints its own free indices, so we reindex every chunk's result
        onto the FIRST chunk's indices before accumulating, and return
        (result, those_indices) so the caller reindexes once, exactly as it would
        for the whole-input call.

    Equivalent to the whole-input call for any linear F -- but ONLY for linear F,
    so every call site must be A/B-validated (chunked == whole) before relying on
    it. Reuses CanonAccumulator for the accumulation half.

    GATED ON MEMORY PRESSURE: chunking calls F once per chunk (extra per-term-BP
    canons), which is pure time overhead unless near a RAM barrier. So when RSS is
    below budget we call F whole (no overhead); we only chunk under pressure, to
    bound the peak by trading canons against swap.
    """
    if (not isinstance(expr, TensAdd) or len(expr.args) <= chunk_size
            or not _mem_pressure()):
        return F(expr)
    args = expr.args
    acc = CanonAccumulator(fold_every=fold_every)
    target_idx = None
    indexed = False
    for i in range(0, len(args), chunk_size):
        chunk = _sum_terms(args[i:i + chunk_size])
        r = F(chunk)
        if isinstance(r, tuple):
            indexed = True
            rexpr, ridx = r
            if rexpr is S.Zero or rexpr == 0:
                continue
            if target_idx is None:
                target_idx = ridx
                acc.add(rexpr)
            else:
                acc.add(_reindex_free(rexpr, ridx, target_idx))
        else:
            acc.add(r)
    result = acc.result()
    return (result, target_idx) if indexed else result


def _reindex_free(expr, old_indices, new_indices):
    """Relabel free indices old->new (both signs) + canon. (Local copy to avoid
    a jet<->loop_helpers import cycle; mirrors loop_helpers._reindex_tensor.)"""
    if expr is S.Zero or expr == 0 or not hasattr(expr, 'substitute_indices'):
        return expr
    pairs = []
    for o, nw in zip(old_indices, new_indices):
        pairs.append((o, nw))
        pairs.append((-o, -nw))
    return canon(expr.substitute_indices(*pairs))


def _is_tensor_atom(expr):
    """Check if expr is a single tensor factor like h(-mu,-nu)."""
    return isinstance(expr, Tensor)

def _get_component(expr):
    """Get the TensorHead from a Tensor instance."""
    if isinstance(expr, Tensor):
        return expr.component
    return None

def _get_indices(expr):
    """Get the list of TensorIndex from a Tensor instance."""
    if isinstance(expr, Tensor):
        return list(expr.get_indices())
    return []

def _decompose_tensmul(expr):
    """Decompose a TensMul into (coefficient, [tensor_factors]).

    Returns:
        coeff: numerical/symbolic coefficient
        factors: list of Tensor objects (individual tensor factors)

    **Raises ValueError if the input is a TensMul wrapping a TensAdd.**
    This case (e.g. `(-1) * (A + B)` from `A − B` when sympy doesn't
    distribute the negation) used to silently drop the inner TensAdd's
    terms, producing wrong output downstream. The check now makes this
    failure loud so callers can fix their input — typically by calling
    `.expand()` (or `canon`) first. See MEMORY.md /
    project-decompose-tensmul-tensadd-pitfall for the historical bites.
    """
    if isinstance(expr, Tensor):
        return S.One, [expr]
    if not isinstance(expr, TensMul):
        # Might be a pure number
        return expr, []

    if any(isinstance(a, TensAdd) for a in expr.args):
        raise ValueError(
            "_decompose_tensmul received a TensMul wrapping a TensAdd "
            "(typical cause: undistributed (scalar)*(sum), e.g. from "
            "raw `A - B` where sympy keeps `(-1)*B` as TensMul(NegativeOne, "
            "TensAdd)). Call `.expand()` (or `canon`) on the input first to "
            "distribute. Silently dropping the inner TensAdd would lose terms."
        )

    coeff = expr.coeff
    factors = []
    for arg in expr.args:
        if isinstance(arg, Tensor):
            factors.append(arg)
        elif isinstance(arg, TensMul):
            c, fs = _decompose_tensmul(arg)
            coeff *= c
            factors.extend(fs)
    return coeff, factors

def _decompose_tensadd(expr):
    """Decompose a TensAdd into a list of its terms."""
    if isinstance(expr, TensAdd):
        return list(expr.args)
    return [expr]

def _rebuild_tensmul(coeff, factors):
    """Rebuild a TensMul from coefficient and tensor factors."""
    if not factors:
        return coeff
    result = coeff
    for f in factors:
        result = result * f
    return result

# ---------------------------------------------------------------------------
# Matching: does a tensor factor match a jet variable type?
# ---------------------------------------------------------------------------

def _get_h_heads():
    """Get the set of all h-type TensorHeads (h, dh, ddh)."""
    return {h, dh, ddh}

def _get_all_jet_heads():
    """Get the set of all registered jet TensorHeads."""
    return set(_JET_HIERARCHY.keys())

# ---------------------------------------------------------------------------
# Core: jet_derivative_of_factor
# ---------------------------------------------------------------------------

def _jet_derivative_of_factor(factor, wrt_head, wrt_indices):
    """Differentiate a single tensor factor w.r.t. a jet variable.
    
    Computes ∂(factor)/∂(wrt_head_{wrt_indices}).
    
    If factor has the same TensorHead as wrt_head, produces the appropriate
    symmetrized Kronecker delta product. Otherwise returns 0.
    
    The symmetrization accounts for the fact that h_{μν} = h_{νμ} (so
    ∂h_{μν}/∂h_{αβ} = ½(δ^α_μ δ^β_ν + δ^α_ν δ^β_μ)), and similarly
    for symmetric derivative index pairs.
    
    Args:
        factor: a Tensor instance (e.g., h(-mu, -nu))
        wrt_head: TensorHead we are differentiating by (e.g., h)
        wrt_indices: list of TensorIndex for the differentiation variable
            (e.g., [alpha, beta] for ∂/∂h_{αβ}). These should be 
            CONTRAVARIANT (positive) since the result of differentiating
            by a covariant variable gives a contravariant free index.
    
    Returns:
        Tensor expression (product of metrics/deltas), or S.Zero.
    """
    if _get_component(factor) != wrt_head:
        return S.Zero
    
    factor_indices = _get_indices(factor)
    
    if len(factor_indices) != len(wrt_indices):
        return S.Zero  #USER COMMENT: shouldn't this throw an error?
    
    # Get the symmetry information for this jet variable
    info = _JET_HIERARCHY.get(wrt_head, {})
    n_field = info.get('n_field_indices', 0)
    
    # Determine which groups of indices are symmetric
    # For h: indices [0,1] are symmetric (field indices)
    # For dh: indices [0,1] are symmetric (field), [2] is single
    # For ddh: indices [0,1] are symmetric (field), [2,3] are symmetric (deriv)
    # For scalar field: no field indices, derivative indices may be symmetric
    # For vector field: [0] is field, [1,2] may be symmetric (for 2nd deriv)
    
    # Build the list of symmetric groups of indices
    sym_groups = _get_symmetric_groups(wrt_head, n_field, len(wrt_indices))
    
    # Generate all permutations consistent with the symmetries.
    # For each symmetric group, we need to sum over all permutations
    # of the wrt_indices within that group, divided by the group size.
    # Compute the symmetrization factor and the set of index permutations
    perms, normalization = _symmetric_permutations(wrt_indices, sym_groups)
    
    # Build the result: sum over permutations of products of deltas
    perm_terms = []
    for perm_wrt in perms:
        # Each permutation gives a product of Kronecker deltas:
        # δ^{perm_wrt[0]}_{factor_idx[0]} * δ^{perm_wrt[1]}_{factor_idx[1]} * ...
        term = S.One
        for wi, fi in zip(perm_wrt, factor_indices):
            # wi should be contravariant, fi should be covariant
            # metric(wi, fi) where wi is up and fi is down gives δ^wi_fi
            term = term * metric(wi, fi)
        perm_terms.append(term)
    return normalization * _sum_terms(perm_terms)

def _get_symmetric_groups(head, n_field, n_total):
    """Get the symmetric index groups for a TensorHead.
    
    Returns a list of lists, each inner list being indices (positions)
    that are symmetric with each other.
    
    Uses n_field (number of field indices) and n_total to determine
    symmetry generically:
    - n_field == 2: field indices [0,1] are symmetric (rank-2 symmetric tensor)
    - derivative indices: symmetric if exactly 2 (second spacetime derivative)
    
    Examples:
        h   (n_field=2, n_total=2): [[0, 1]]
        dh  (n_field=2, n_total=3): [[0, 1]]
        ddh (n_field=2, n_total=4): [[0, 1], [2, 3]]
        dddh(n_field=2, n_total=5): [[0, 1], [2, 3, 4]]
        ginv(n_field=2, n_total=2): [[0, 1]]
        dg  (n_field=2, n_total=3): [[0, 1]]
        dphi (n_field=0, n_total=1): []
        ddphi(n_field=0, n_total=2): [[0, 1]]
        dA   (n_field=1, n_total=2): []
        ddA  (n_field=1, n_total=3): [[1, 2]]
    """
    groups = []
    
    # Field indices: symmetric if there are exactly 2 (rank-2 symmetric tensor)
    if n_field == 2 and n_total >= 2:
        groups.append([0, 1])
    
    # Derivative indices (positions n_field .. n_total-1):
    # symmetric if there are 2 or more (symmetric higher derivatives)
    n_deriv = n_total - n_field
    if n_deriv >= 2:
        groups.append(list(range(n_field, n_total)))
    
    return groups

@lru_cache(maxsize=None)
def _position_permutations(n_indices, sym_groups_key):
    """Position-only permutations for given symmetry groups; cacheable.

    sym_groups_key is the sym_groups list serialized as a tuple-of-tuples so it's
    hashable. Returns (list_of_position_tuples, normalization). Mapping to
    actual TensorIndex objects happens in `_symmetric_permutations` — that
    mapping is cheap, but the permutation enumeration here is the expensive
    part that was being recomputed identically on every jet_derivative call.
    """
    from itertools import permutations as iterperms

    sym_groups = [list(g) for g in sym_groups_key]
    if not sym_groups:
        return [tuple(range(n_indices))], S.One

    all_perms = {tuple(range(n_indices))}
    for group in sym_groups:
        new_perms = set()
        for existing_perm in all_perms:
            group_vals = [existing_perm[g] for g in group]
            for gp in iterperms(group_vals):
                new_perm = list(existing_perm)
                for g_pos, gp_val in zip(group, gp):
                    new_perm[g_pos] = gp_val
                new_perms.add(tuple(new_perm))
        all_perms = new_perms

    result = [tuple(perm) for perm in all_perms]
    normalization = Rational(1, len(result))
    return result, normalization


def _symmetric_permutations(indices, sym_groups):
    """Generate all distinct index permutations from symmetric groups.

    Args:
        indices: list of TensorIndex
        sym_groups: list of lists of positions that are symmetric

    Returns:
        (permutations, normalization_factor)
    """
    sym_groups_key = tuple(tuple(g) for g in sym_groups)
    pos_perms, normalization = _position_permutations(len(indices), sym_groups_key)
    idx_list = list(indices)
    result = [tuple(idx_list[p] for p in perm) for perm in pos_perms]
    return result, normalization


# ---------------------------------------------------------------------------
# Public API: jet_derivative
# ---------------------------------------------------------------------------

def jet_derivative(expr, wrt_head, wrt_indices):
    """Differentiate a tensor expression w.r.t. a jet variable.

    Computes ∂(expr)/∂(wrt_head_{wrt_indices}).

    For h-type variables, the result includes the ½-symmetrization
    factor for symmetric index pairs. Applies the Leibniz (product) rule
    on products of tensor factors.

    The result is automatically canonicalized to keep expressions compact.
    """
    if isinstance(expr, (int, float)):
        return S.Zero
    if expr == S.Zero:
        return S.Zero

    # Handle sums: distribute linearly
    if isinstance(expr, TensAdd):
        terms = [jet_derivative(t, wrt_head, wrt_indices) for t in expr.args]
        result = _sum_terms(terms)
        return canon(result) if isinstance(result, TensExpr) else result

    # Handle single tensor atom
    if isinstance(expr, Tensor):
        return canon(_jet_derivative_of_factor(expr, wrt_head, wrt_indices))

    # Handle product (TensMul): Leibniz rule
    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)

        if not factors:
            return S.Zero

        leibniz_terms = []
        for i, factor in enumerate(factors):
            dfactor = _jet_derivative_of_factor(factor, wrt_head, wrt_indices)
            if dfactor == S.Zero:
                continue
            other_factors = factors[:i] + factors[i+1:]
            term = coeff * dfactor
            for f in other_factors:
                term = term * f
            leibniz_terms.append(term)
        result = _sum_terms(leibniz_terms)
        return canon(result) if isinstance(result, TensExpr) else result

    # Handle expressions with scalar coefficient (like Rational * TensMul)
    if hasattr(expr, 'args'):
        try:
            coeff, factors = _decompose_tensmul(expr)
            if factors:
                inner = _rebuild_tensmul(S.One, factors)
                d_inner = jet_derivative(inner, wrt_head, wrt_indices)
                result = coeff * d_inner
                return canon(result) if isinstance(result, TensExpr) else result
        except (TypeError, AttributeError):
            pass

    return S.Zero


# ---------------------------------------------------------------------------
# Public API: total_derivative
# ---------------------------------------------------------------------------

def total_derivative(expr, deriv_index):
    """Take the total spacetime derivative ∂_τ of a tensor expression.

    This acts via the chain rule on each jet variable:
        ∂_τ h_{μν}     = h_{μν,τ}       (i.e., dh_{μν,τ})
        ∂_τ h_{μν,ρ}   = h_{μν,ρτ}      (i.e., ddh_{μν,ρτ})
        ∂_τ h_{μν,ρσ}  = h_{μν,ρστ}     — NOT SUPPORTED (3rd derivatives)
        ∂_τ φ          = φ_{,τ}          (dphi_τ)
        ∂_τ φ_{,μ}     = φ_{,μτ}        (ddphi_{μτ})
        ∂_τ η_{μν}     = 0              (background metric is constant)

    Applies the Leibniz rule on products.

    **Input is .expand()'d defensively.** The Leibniz loop relies on
    `_decompose_tensmul` to split each TensAdd arg into (coeff, factors).
    But `_decompose_tensmul` silently drops any TensAdd args of a TensMul
    (the long-standing footgun in MEMORY.md/project-decompose-tensmul-
    tensadd-pitfall). This bites total_derivative whenever the input
    contains a TensMul wrapping a TensAdd — most commonly from `A − B`
    where `A`, `B` are TensAdds: sympy represents this as
    `TensAdd(A, TensMul(-1, B))` and never distributes the −1 across B.
    `.expand()` here flattens those wraps so every leaf TensMul is a flat
    product of Tensors. Cheap if the input is already expanded; corrects
    the input if it isn't.

    Args:
        expr: tensor expression (should be in canonical form — see warning)
        deriv_index: TensorIndex for the derivative direction (should be
            covariant/negative, representing ∂_τ)

    Returns:
        Canonicalized tensor expression for ∂_τ(expr).

    Raises:
        ValueError: if the expression contains a jet variable at the
            highest level (e.g., ddh) that cannot be differentiated further.
    """
    if isinstance(expr, (int, float)):
        return S.Zero
    if expr == S.Zero:
        return S.Zero

    # Distribute any TensMul-wrapped TensAdd (e.g. (-1) * TensAdd from `A − B`)
    # so the Leibniz loop sees flat leaf TensMuls. See docstring above.
    if isinstance(expr, TensExpr):
        expr = expr.expand()

    if isinstance(expr, TensAdd):
        terms = [total_derivative(t, deriv_index) for t in expr.args]
        result = _sum_terms(terms)
        return canon(result) if isinstance(result, TensExpr) else result

    if isinstance(expr, Tensor):
        return _total_derivative_of_factor(expr, deriv_index)

    if isinstance(expr, TensMul):
        coeff, factors = _decompose_tensmul(expr)

        if not factors:
            return S.Zero

        leibniz_terms = []
        for i, factor in enumerate(factors):
            d_factor = _total_derivative_of_factor(factor, deriv_index)
            if d_factor == S.Zero:
                continue
            other_factors = factors[:i] + factors[i+1:]
            term = coeff * d_factor
            for f in other_factors:
                term = term * f
            leibniz_terms.append(term)
        result = _sum_terms(leibniz_terms)
        return canon(result) if isinstance(result, TensExpr) else result

    return S.Zero


def _total_derivative_of_factor(factor, deriv_index):
    """Take ∂_τ of a single tensor factor.
    
    Replaces the factor by its child in the jet hierarchy with the
    derivative index appended.
    """
    comp = _get_component(factor)
    if comp is None:
        return S.Zero
    
    # The metric is constant: ∂_τ η_{μν} = 0
    if comp == metric:
        return S.Zero
    
    info = _JET_HIERARCHY.get(comp)
    if info is None:
        # Unknown tensor head — treat as constant
        return S.Zero
    
    child = info.get('child')
    if child is None:
        raise ValueError(
            f"Cannot take total derivative of {comp}: "
            f"no higher jet variable defined (would need 3rd derivatives). "
            f"This likely means the expression has too many derivatives."
        )
    
    # Build the child tensor with the same indices plus the derivative index
    old_indices = _get_indices(factor)
    new_indices = old_indices + [deriv_index]

    return child(*new_indices)
