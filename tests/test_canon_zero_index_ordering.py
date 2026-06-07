"""Regression: canon must canonically ORDER products of 0-index Tensors
(matter fields), so commuting products in different orders combine/cancel.

Bug (2026-06-07, found via the traceless kitchen sink): canon's 0-index
workaround re-multiplied stripped 0-index factors in source order, so
phi1*phi2 and phi2*phi1 stayed distinct and `phi1*phi2 - phi2*phi1` did NOT
simplify to 0. That left a phantom 2-term residual in the trace-decomposition
of any matter sector with >= 2 multiplied scalar fields, defeating the
`residual == S.Zero` traceless check. Fixed by sorting on recombine.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import S
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, dh, metric,
)

p1, dp1, ddp1 = register_scalar_field('p1')
p2, dp2, ddp2 = register_scalar_field('p2')


def test_two_scalar_product_cancels():
    """phi1*phi2 - phi2*phi1 == 0 after canon."""
    e = canon(p1() * p2() - p2() * p1())
    assert e == S.Zero, f"expected 0, got {e}"


def test_two_scalar_product_same_canonical_form():
    """canon(phi1*phi2) and canon(phi2*phi1) are the SAME expression."""
    a = canon(p1() * p2())
    b = canon(p2() * p1())
    assert canon(a - b) == S.Zero, f"phi1*phi2 != phi2*phi1 under canon: {a} vs {b}"


def test_cancels_with_indexed_factor():
    """A derivative factor times the commuting product still cancels:
    dh(L,-L,m) (phi1 phi2 - phi2 phi1) == 0."""
    m, = fresh_indices(1)
    L0, = fresh_indices(1)
    X = dh(L0, -L0, m)
    e = canon(X * p1() * p2() - X * p2() * p1())
    assert e == S.Zero, f"expected 0, got {e}"


def test_three_scalar_permutations_cancel():
    """All orderings of phi1 phi2 phi1 agree (repeated head + distinct head)."""
    a = canon(p1() * p2() * p1())
    b = canon(p2() * p1() * p1())
    assert canon(a - b) == S.Zero, f"{a} vs {b}"


if __name__ == '__main__':
    test_two_scalar_product_cancels()
    test_two_scalar_product_same_canonical_form()
    test_cancels_with_indexed_factor()
    test_three_scalar_permutations_cancel()
    print("*** PASS: canon canonically orders 0-index tensor products. ***")
