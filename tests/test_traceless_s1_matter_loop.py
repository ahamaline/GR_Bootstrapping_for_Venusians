"""Fast check of the c_i != 0 matter half of the traceless recovery.

Massless upstairs-V at d=4 (Hilbert) is ON-shell traceless with c_V = -2 k V.
This exercises `_build_traceless_S1`'s `Sum c_i E_i^(1)` contraction (the piece
the off-shell EM case skips) WITHOUT a full order-2 run: only orders 0-1 are
needed, since S^(1) = eta E_h^(1) + c_V E_V^(1) depends only on E[1], L[1], c_V.

Asserts:
  - S^(1) is a SCALAR (no leaked free indices) -- catches a bad contraction;
  - the matter loop actually fired (S^(1) contains V);
  - the doubly-traced-ddh box signature is present;
  - its constant/matter split is (m=-2, v=0) for this theory. NOTE: v=0 here is
    EMPIRICAL for upstairs-V Maxwell, not a theorem -- ddh can in principle
    reach the matter part (via dh*dV terms in L^(1), or ddh in tr T_M[L^(1)]).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import register_upstairs_vector_field, fresh_indices, canon
from bootstrap.bootstrap_loop import BootstrapState, _extract_ddh_box_signature


def main():
    V, dV, ddV = register_upstairs_vector_field('V')
    a, b = fresh_indices(2)
    L_V = canon(-Rational(1, 2) * dV(a, b) * dV(-a, -b)
                + Rational(1, 2) * dV(a, b) * dV(-b, -a))

    state = BootstrapState(L_matter=L_V, em_procedure='hilbert',
                           n_max=2, verbose=False)
    state.run_order(0)
    assert state.traceless_T_M and state.traceless_c_i.get('V', S.Zero) != S.Zero, (
        "upstairs-V at d=4 should be on-shell traceless with c_V != 0")
    state.run_order(1)

    S1 = state._build_traceless_S1()
    free = list(S1.get_free_indices()) if hasattr(S1, 'get_free_indices') else []
    assert not free, f"S^(1) leaked free indices {free} -- matter contraction wrong"
    assert S1.has(V), "matter loop did not fire -- c_V E_V^(1) missing from S^(1)"

    box = _extract_ddh_box_signature(S1)
    assert box != S.Zero, "no doubly-traced-ddh signature in S^(1)"
    m, v = state._split_const_matter(box)
    assert m == -2, f"expected W-trace constant m = -2, got {m}"
    assert v == S.Zero, f"expected v = 0 for upstairs-V Maxwell, got {v}"

    print(f"S^(1): {len(S1.args) if isinstance(S1, TensAdd) else 1} terms, scalar, "
          f"contains V; box (m+v) = {box}  (m={m}, v={v})")
    print("\n*** c_i != 0 matter-loop (build_traceless_S1) check PASSES ***")


if __name__ == '__main__':
    main()
