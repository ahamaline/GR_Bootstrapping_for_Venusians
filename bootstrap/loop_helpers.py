"""General bootstrap-loop helper utilities — term counting / per-field
breakdown display, free-index reindexing, and the derivative-free predicate.

Extracted from bootstrap_loop.py so that it and bootstrap/traceless.py can
share them without an import cycle. Depends only on tensor_algebra (+ sympy).
"""
from sympy import S
from sympy.tensor.tensor import TensAdd, TensMul, TensExpr, Tensor
from bootstrap.tensor_algebra import (
    h, dh, ddh, canon, _matter_fields, get_tensors_in_expr,
)


def _count(expr):
    if expr == S.Zero:
        return 0
    if isinstance(expr, TensAdd):
        return len(expr.args)
    return 1


def _require_constant_coeff(coeff, signature_desc):
    """Return `coeff` if it is a field-INDEPENDENT constant (a pure function of
    the dimension d and coupling constants — no matter-field / curvature /
    h-derivative structure), else raise.

    This is the shared "the denominator must be a constant" guard. The recovery
    (traceless path) and the EOM decomposition (ordinary path) both recover a
    coefficient by stripping a signature from an operator and DIVIDING by that
    signature's coefficient in the operator. That divisor must be a constant:

      - traceless path: the ddh-signature coefficient of S^(1) / T2 (here a
        field-dependent value is "not the user's fault" — it would be a deeper
        feature of the model);
      - ordinary path: the kinetic / 2nd-derivative signature coefficient in
        T_M = E_h^(0) and in the matter EOMs E_phi^(0) (here a constant
        coefficient — e.g. canonical kinetic normalization — is a PRECONDITION
        on the user-supplied matter Lagrangian).

    Mechanically it is the same error either way: a field-dependent divisor
    cannot be handled without a polynomial division in the matter scalar, which
    is not implemented. Fail loudly with a clear message.

    Detection: "field-independent" means the coefficient contains NO matter-
    field or graviton heads (h / dh / ddh / phi / dphi / A / dA / V / dV / ...).
    A pure sympy scalar (e.g. d-2, +/-1) trivially qualifies; so does a coeff
    that carries only metric/delta index structure (e.g. a vector EOM's 2nd-
    derivative coefficient, which carries the field's vector index via a metric
    but no field dependence). A coeff containing any field head is the v != 0
    case and triggers the error.
    """
    if coeff == S.Zero or coeff == 0:
        return S.Zero
    if isinstance(coeff, TensExpr):
        field_heads = {h, dh, ddh}
        for info in _matter_fields.values():
            field_heads.update(
                [info['field'], info['dfield'], info['ddfield']])
        if set(get_tensors_in_expr(coeff)) & field_heads:
            raise RuntimeError(
                f"the coefficient of {signature_desc} includes field-dependent "
                f"terms ({_count(coeff)} terms): {coeff}. This signature requires "
                f"a constant (field-independent) coefficient; handling a field-"
                f"dependent one would need a more advanced extraction procedure "
                f"such as polynomial division.")
    return coeff


def _term_breakdown(expr):
    """Categorize the terms of `expr` by which matter fields they contain.

    Returns a dict {label: count} with labels:
      - "h-only" for terms with no matter-field factors,
      - "<name>" for terms containing exactly one matter field <name>,
      - "<n1>+<n2>+..." for terms containing multiple matter fields.

    Pure-gravity Hilbert paths show only "h-only". Matter Lagrangians show
    per-field splits, which makes it obvious whether (e.g.) a step 1 result
    is dominated by gravity, matter, or matter-gravity cross terms.
    """
    if expr == S.Zero:
        return {}
    matter_heads = {}
    for name, info in _matter_fields.items():
        for key in ('field', 'dfield', 'ddfield'):
            matter_heads[info[key]] = name
    terms = expr.args if isinstance(expr, TensAdd) else [expr]
    out = {}
    for term in terms:
        if isinstance(term, TensMul):
            factors = [a for a in term.args if hasattr(a, 'component')]
        elif hasattr(term, 'component'):
            factors = [term]
        else:
            factors = []
        names = set()
        for f in factors:
            head = f.component
            if head in matter_heads:
                names.add(matter_heads[head])
        if not names:
            label = 'h-only'
        elif len(names) == 1:
            label = next(iter(names))
        else:
            label = '+'.join(sorted(names))
        out[label] = out.get(label, 0) + 1
    return out

def _format_breakdown(expr):
    """Human-readable term count + breakdown by matter content.

    h-only = pure-gravity terms (h, dh, ddh and metric only).
    <name>  = terms with a single matter field <name> (may also have h).
    a+b     = terms with multiple matter fields (matter-matter cross terms).

    Examples:
      "10 terms (10 h-only)"          -> pure gravity
      "81 terms (51 h-only, 30 A)"    -> EM in gravity background
      "120 terms (40 h-only, 60 A, 20 A+phi)"  -> scalar QED with cross terms
    """
    total = _count(expr)
    if total == 0:
        return "0 terms"
    bd = _term_breakdown(expr)
    if len(bd) == 1 and 'h-only' in bd:
        return f"{total} terms"
    parts = []
    if 'h-only' in bd:
        parts.append(f"{bd['h-only']} h-only")
    for name in sorted(k for k in bd if k != 'h-only' and '+' not in k):
        parts.append(f"{bd[name]} {name}")
    for sig in sorted(k for k in bd if '+' in k):
        parts.append(f"{bd[sig]} {sig}")
    return f"{total} terms ({', '.join(parts)})"

def _derivative_heads():
    """The set of TensorHeads representing field derivatives — dh, ddh, and
    each registered matter field's dfield / ddfield. Used to recognize
    "derivative-free" expressions (Lagrangian-field-redef X coefficients)."""
    heads = {dh, ddh}
    for info in _matter_fields.values():
        heads.add(info['dfield'])
        heads.add(info['ddfield'])
    return heads

def _is_derivative_free(expr):
    """Return True iff expr contains no first- or second-derivative field
    heads (dh / ddh / dφ / ddφ / dA / ddA / dV / ddV / ...).

    A derivative-free X is the integrability-criterion signature of a
    Lagrangian field redefinition (paper §3): X must be a function of fields
    only, no derivatives. Constants, h, φ, A, V, η, deltas are all fine.
    """
    if expr == S.Zero or expr == 0:
        return True
    if not isinstance(expr, TensExpr):
        return True  # pure sympy scalar, no tensor heads
    heads_in_expr = set(get_tensors_in_expr(expr))
    return not (heads_in_expr & _derivative_heads())

def _reindex_tensor(expr, old_indices, new_indices):
    """Substitute old free indices with new ones (both signs)."""
    if expr == S.Zero or expr == 0:
        return expr
    if not hasattr(expr, 'substitute_indices'):
        return expr
    pairs = []
    for old, new in zip(old_indices, new_indices):
        pairs.append((old, new))
        pairs.append((-old, -new))
    return canon(expr.substitute_indices(*pairs))
