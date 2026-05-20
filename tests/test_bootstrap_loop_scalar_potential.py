# Drives BootstrapState with a massive scalar matter field, orders 0..3.
# L_M = -1/2 dphi^mu dphi_mu - 1/2 m^2 phi^2
# This exercises:
#   - the matter L_ref expansion at orders 1..4 (with both dphi-quadratic
#     and phi^2 pieces)
#   - the n=1 integral formula (should give Psi^(1) = 0 on the default
#     path — Hilbert + no optional EOM)
#   - PsiForm (compute_superpotential_n2) at n=2 and n=3 with matter
#     contributions inside M
#   - EL self-consistency and L_ref verification at every step

import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S, Symbol
from sympy.tensor.tensor import TensAdd

from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon,
)
from bootstrap.energy_momentum import NATURAL_POSITIONS
from bootstrap.bootstrap_loop import BootstrapState

phi, dphi, ddphi = register_scalar_field('phi')
NATURAL_POSITIONS[dphi] = ['down']
NATURAL_POSITIONS[ddphi] = ['down', 'down']

m = Symbol('m')

mu_d, = fresh_indices(1)
L_M = canon(
    Rational(-1, 2) * dphi(mu_d) * dphi(-mu_d)
    + Rational(-1, 2) * m**2 * phi() * phi()
)
print(f"L_M = {L_M}")
print()

state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=3,
                       verbose=True)

t_total = time.time()
for n in range(4):
    t0 = time.time()
    state.run_order(n)
    print(f"  Order {n} total wall: {time.time()-t0:.1f}s")

print()
print("=" * 60)
print(f"  All orders 0..3 passed verification.")
print(f"  Total wall: {time.time()-t_total:.1f}s")
print("=" * 60)
