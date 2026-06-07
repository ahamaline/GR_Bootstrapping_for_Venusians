"""Covariantization of a DOWNSTAIRS vector A in matter_lagrangian_order.

Two facts must hold for the newly-added downstairs-A Christoffel correction
(covariant.py, step (b)):

  1. REGRESSION: for the EM Lagrangian L = -1/4 F_{μν} F^{μν} (F antisymmetric),
     the Christoffel term −Γ^τ_{ρσ} A_τ must cancel between the two terms of
     each F, so the order-≥1 expansion contains NO bare-A factor. (Bare A
     can only come from a surviving Christoffel correction; metric promotion
     (a) and √g never produce an undifferentiated A.)

  2. FEATURE: for a NON-antisymmetric combination of A's derivatives, e.g.
     L = (∂_μ A_ν)(∂^μ A^ν) (the symmetric square, not F²), the Christoffel
     correction does NOT cancel, so the order-1 expansion DOES contain
     bare-A terms. Before this change those terms were silently dropped.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S
from sympy.tensor.tensor import TensAdd, TensMul

from bootstrap.tensor_algebra import register_vector_field, fresh_indices, canon
from bootstrap.covariant import matter_lagrangian_order
from bootstrap.jet import _get_component


def _has_bare_field(expr, field_head):
    """True if any term of expr contains the undifferentiated field head."""
    if expr == S.Zero:
        return False
    terms = expr.args if isinstance(expr, TensAdd) else [expr]
    for term in terms:
        factors = (term.args if isinstance(term, TensMul) else [term])
        for f in factors:
            if _get_component(f) is field_head:
                return True
    return False


def main():
    A, dA, ddA = register_vector_field('A')  # downstairs A_mu

    # --- 1. EM: L = -1/4 F_{mn} F^{mn}, F_{mn} = dA(m,n) - dA(n,m) ---
    m, n = fresh_indices(2)
    L_em = -Rational(1, 4) * (dA(-m, -n) - dA(-n, -m)) * (dA(m, n) - dA(n, m))
    L_em = canon(L_em)

    em1 = matter_lagrangian_order(L_em, 1)
    em2 = matter_lagrangian_order(L_em, 2)
    assert not _has_bare_field(em1, A), (
        "REGRESSION: EM order-1 expansion has a bare-A term; the Christoffel "
        "correction failed to cancel in the antisymmetric F."
    )
    assert not _has_bare_field(em2, A), (
        "REGRESSION: EM order-2 expansion has a bare-A term; Christoffel "
        "correction failed to cancel."
    )
    print(f"[1] EM F^2: order-1 = {_n(em1)} terms, order-2 = {_n(em2)} terms; "
          f"no bare-A (Christoffel cancels in antisymmetric F). OK")

    # --- 2. Non-antisymmetric: L = (d_m A_n)(d^m A^n) ---
    a, b = fresh_indices(2)
    L_sym = dA(-a, -b) * dA(a, b)
    L_sym = canon(L_sym)

    sym1 = matter_lagrangian_order(L_sym, 1)
    assert _has_bare_field(sym1, A), (
        "FEATURE: non-antisymmetric (dA)(dA) order-1 expansion is MISSING the "
        "bare-A Christoffel term; the downstairs-A covariant derivative was "
        "not applied."
    )
    print(f"[2] (dA)(dA) symmetric square: order-1 = {_n(sym1)} terms, "
          f"bare-A Christoffel term present. OK")

    # --- 3. Upstairs-V guard: the refactor that unified V/A must keep the
    # upstairs-V Christoffel term. V does NOT cancel even in antisymmetric F,
    # so Proca's F^2 order-1 expansion must contain a bare-V term. ---
    from bootstrap.tensor_algebra import register_upstairs_vector_field
    V, dV, ddV = register_upstairs_vector_field('V')
    p, q = fresh_indices(2)
    L_proca = -Rational(1, 4) * (dV(-p, -q) - dV(-q, -p)) * (dV(p, q) - dV(q, p))
    L_proca = canon(L_proca)
    proca1 = matter_lagrangian_order(L_proca, 1)
    assert _has_bare_field(proca1, V), (
        "REFACTOR REGRESSION: upstairs-V Proca F^2 order-1 expansion is "
        "missing the bare-V Christoffel term (V must NOT cancel in F)."
    )
    print(f"[3] upstairs-V Proca F^2: order-1 = {_n(proca1)} terms, "
          f"bare-V Christoffel term present (no cancellation for V). OK")

    print("\n=== downstairs-A covariantization: regression + feature OK ===")


def _n(expr):
    if expr == S.Zero:
        return 0
    return len(expr.args) if isinstance(expr, TensAdd) else 1


if __name__ == '__main__':
    main()
