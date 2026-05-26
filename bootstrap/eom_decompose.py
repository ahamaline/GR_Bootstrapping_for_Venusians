"""Decompose tensor expressions into linear combinations of zeroth-order EOMs.

User's strategy (2026-05-26):

  Z = C_h^{cdef}_{αβ} · E_h^{(0)αβ}  +  Σ_i  C_{φ_i}^{cdef}_{α…} · E_{φ_i}^{(0) α…}
      + residual (should be 0)

with E_h^{(0)} = κ T_M (Hilbert or Belinfante) and E_{φ_i}^{(0)} = δL/δφ_i.

Notation (user 2026-05-26): the decomposition coefficients are
  - **Y** in step 3 of the bootstrap (decomposing the H2 violation Z),
  - **X** in the verification step (decomposing an E_h diff, e.g.
    Belinfante − Hilbert).
Same algorithm in both cases; this module's API uses generic "C" /
"coeff" names that the caller can rename as needed.

KEY OBSERVATION: matter EOMs E_{φ_i}^{(0)} contain only second derivatives; the
products of first derivatives all live in T_M (and so in C_h × T_M). So C_h can
be isolated by matching T_M's first-derivative-product signature in Z. Then a
trace term may need to be added if there are leftover contracted ∂φ·∂φ pieces.
What remains is purely a linear combination of matter EOMs; each matter EOM
has unique second-derivative terms that uniquely fix C_{φ_i}.

This module's MVP handles the trace-signature ddfield(α…, L, -L) for
rank-1 matter fields (downstairs A, upstairs V) with kinetic terms. Verified
on the Belinfante−Hilbert diffs:
  - EM: decomposes as C_A × EOM_A (Maxwell's equation × A), residual 0.
  - Full Proca: decomposes as C_V × EOM_V (Proca equation × V), residual 0.
Still to implement: C_h extraction (Y_h × T_M part), trace-signature
addition step, mass-only-EOM signature (no 2nd-derivative term),
pure-gravity h decomposition against the wave operator W.
"""

from sympy import S, Rational
from sympy.tensor.tensor import TensAdd, TensMul, Tensor, TensExpr
from bootstrap.tensor_algebra import (
    fresh_indices, canon, metric, _matter_fields,
    NATURAL_POSITIONS,
)
from bootstrap.jet import (
    _decompose_tensmul, _get_component, _get_indices,
)
from bootstrap.euler_lagrange import euler_lagrange, euler_lagrange_scalar


def compute_matter_eoms(L):
    """Compute zeroth-order EOM δL/δφ for each registered matter field.

    Returns: dict {name: (E_phi expression, idx tuple)} where idx is the
    tuple of free indices on E_phi (empty for scalar matter).
    """
    eoms = {}
    for name, info in _matter_fields.items():
        field = info['field']
        rank = info.get('rank', 0)
        if rank == 0:
            E = euler_lagrange_scalar(L, field)
            eoms[name] = (E, ())
        else:
            E, idx = euler_lagrange(L, field)
            eoms[name] = (E, idx)
    return eoms


def _is_dummy_pair(idx_a, idx_b):
    """Check if (idx_a, idx_b) is a contracted-dummy pair (one up, one down, same name)."""
    return idx_a.name == idx_b.name and (idx_a.is_up != idx_b.is_up)


def _find_trace_factor(factors, head, alpha_position):
    """Find a factor in `factors` whose component is `head` and whose indices form
    a `head(?, L, -L)` shape at the OTHER two positions (a dummy pair).

    For head with 3 indices (e.g., ddA, ddV: [field, deriv, deriv]), the deriv
    pair at positions (1, 2) must be a dummy pair if alpha_position == 0.
    For head with 2 indices (e.g., ddphi: [deriv, deriv]), alpha_position is
    irrelevant; the two indices form the trace pair.

    Returns (factor_index_in_factors, alpha_value_index) or None.
    """
    for i, f in enumerate(factors):
        if _get_component(f) != head:
            continue
        indices = _get_indices(f)
        n = len(indices)
        if n == 2:
            # Scalar ddphi case: the two indices must be a dummy pair.
            if _is_dummy_pair(indices[0], indices[1]):
                return i, None  # no alpha for scalar
        elif n == 3:
            # Vector ddA/ddV: alpha at alpha_position, other two are dummy pair.
            other = [j for j in range(3) if j != alpha_position]
            if _is_dummy_pair(indices[other[0]], indices[other[1]]):
                return i, indices[alpha_position]
    return None


def _extract_y_from_trace_signature(Z, head, alpha_position):
    """Extract Y by matching each term in Z to a `head(?, L, -L)` trace signature.

    For each matching term, strip the signature factor and contribute
    `stripped × metric(?, -α)` to Y (where α is a fresh up index introduced once).

    Args:
        Z: tensor expression.
        head: matter EOM second-derivative head (ddphi / ddA / ddV).
        alpha_position: which index slot is the "alpha" (contraction with Y).
            For scalar (2-index head), this is None.

    Returns: (Y, alpha_index) — Y has alpha (down sign for contraction with
    up-indexed EOM) as a free index, or (Y, None) for scalar (no alpha).
    """
    if Z == S.Zero:
        return S.Zero, None

    # For rank-r > 0 case, introduce a fresh α index (up) so Y has free α-up,
    # to later contract with EOM_φ which has free α-up (then we use metric to
    # convert to a clean δ when stripping).
    alpha = None
    if alpha_position is not None:
        alpha, = fresh_indices(1)

    terms = Z.args if isinstance(Z, TensAdd) else [Z]
    Y = S.Zero

    for term in terms:
        if isinstance(term, TensMul):
            coeff, factors = _decompose_tensmul(term)
        elif isinstance(term, Tensor):
            coeff, factors = S.One, [term]
        else:
            continue

        match = _find_trace_factor(factors, head, alpha_position)
        if match is None:
            continue
        factor_idx, star_idx = match

        # Strip the signature factor.
        other_factors = factors[:factor_idx] + factors[factor_idx + 1:]
        stripped = coeff
        for f in other_factors:
            stripped = stripped * f

        if alpha is None:
            # Scalar case: no α conversion needed; just accumulate the stripped coefficient.
            contribution = stripped
        else:
            # Vector case: convert star_idx to α via a Kronecker δ.
            # star_idx is up (the "field index" position of ddA/ddV is naturally down,
            # but in our convention we pass the up version to get clean Kroneckers);
            # metric(star_idx_up, -alpha_down) = δ^{star_idx}_α.
            contribution = stripped * metric(star_idx, -alpha)
        Y = Y + contribution

    if isinstance(Y, TensExpr):
        Y = canon(Y)
    return Y, alpha


def decompose_against_matter_eom(Z, L, field_name):
    """Try to express Z as Y × E_{field_name}^{(0)}.

    MVP: uses the trace-signature extraction for a single matter EOM whose
    "principal" second-derivative term has the `ddfield(α, L, -L)` shape
    (for rank-1 fields) or `ddφ(L, -L)` (for scalar).

    Returns (Y, alpha_index, residual). residual = Z − Y × E should be 0 if
    Z is purely a Y × E combination.
    """
    info = _matter_fields[field_name]
    rank = info.get('rank', 0)
    ddfield = info['ddfield']

    if rank == 0:
        # Scalar: signature is ddphi(L, -L) (trace), no α.
        Y, alpha = _extract_y_from_trace_signature(Z, ddfield, alpha_position=None)
        EOM_expr, EOM_idx = euler_lagrange_scalar(L, info['field']), ()
        contribution = Y * EOM_expr
        residual = canon(Z - contribution) if isinstance(Z - contribution, TensExpr) else Z - contribution
        return Y, None, residual

    # Rank-1 (downstairs A or upstairs V): signature ddfield(α, L, -L) at position 0.
    Y, alpha = _extract_y_from_trace_signature(Z, ddfield, alpha_position=0)
    if Y == S.Zero:
        return Y, alpha, Z

    EOM_expr, EOM_idx = euler_lagrange(L, info['field'])
    # EOM_idx is a tuple with one TensorIndex (the field index of EOM_φ).
    # We need to contract Y's α with EOM's index. Y has α up; EOM has α up (per our convention).
    # To contract: relabel EOM_idx[0] to α (matching).
    # Hmm but they're both up. Use metric to flip.
    # Actually Y × EOM with same-named indices contracts. So we relabel EOM_idx[0] → α via substitute.
    from bootstrap.bootstrap_loop import _reindex_tensor
    EOM_renamed = _reindex_tensor(EOM_expr, EOM_idx, (alpha,))
    contribution = Y * EOM_renamed
    if isinstance(contribution, TensExpr):
        contribution = canon(contribution)
    residual = Z - contribution
    if isinstance(residual, TensExpr):
        residual = canon(residual)
    return Y, alpha, residual
