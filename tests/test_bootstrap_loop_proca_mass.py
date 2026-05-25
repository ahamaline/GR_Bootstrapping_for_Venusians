"""BootstrapState driver with the Proca mass term only (no F.F).

  L_M = -1/2 m^2 V^mu V_mu

V is an upstairs-natural vector. This is the minimal test that exercises
the downstairs-metric infrastructure (Phase A): the lowering of V^mu inside
V^mu V_mu produces an explicit eta_{mu nu} factor, which becomes g_{mu nu}
under covariantization. With no derivatives of V there are no Christoffel
corrections to dV — those are exercised by test_bootstrap_loop_proca.

We run orders 0..2 and verify the closure cycle at each.
"""

import sys
import time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S, Symbol

from bootstrap.tensor_algebra import (
    register_upstairs_vector_field, fresh_indices, canon,
)
from bootstrap.bootstrap_loop import BootstrapState

V, dV, ddV = register_upstairs_vector_field('V')

m_sym = Symbol('m')

mu, = fresh_indices(1)
L_M = Rational(-1, 2) * m_sym**2 * V(mu) * V(-mu)
L_M = canon(L_M)
print(f"L_M (Proca mass only) = {L_M}")
print()

N_MAX = 2

state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True)

t_total = time.time()
for n in range(N_MAX + 1):
    t0 = time.time()
    state.run_order(n)
    print(f"  Order {n} total wall: {time.time()-t0:.1f}s")

print()
print("=" * 60)
print(f"  All orders 0..{N_MAX} passed verification.")
print(f"  Total wall: {time.time()-t_total:.1f}s")
print("=" * 60)
