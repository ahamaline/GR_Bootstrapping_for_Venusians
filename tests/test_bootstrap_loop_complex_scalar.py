# BootstrapState with a complex scalar matter field, written as two real
# scalars phi1, phi2 with the same mass m. This is the standard
# representation of a complex scalar:
#   phi = (phi1 + i phi2) / sqrt(2)
#   L = -d_mu phi* d^mu phi - m^2 phi* phi
#     = -1/2 (dphi1)^2 - 1/2 (dphi2)^2 - 1/2 m^2 (phi1^2 + phi2^2)
# (cross terms vanish because phi* phi is real). U(1) symmetry is not
# enforced by anything in the bootstrap; we just write the Lagrangian as
# two independent fields and let the machinery handle them.
#
# Exercises:
#   - hilbert_energy_momentum on a multi-field Lagrangian
#   - matter_lagrangian_order with multiple registered scalars
#   - compute_superpotential_n1 summing over multiple matter fields
#     (still Psi^(1) = 0 in the default path, but with both fields
#     contributing potentially-cancelling pieces)
#   - PsiForm at n>=2 with both fields' contributions in M

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

phi1, dphi1, ddphi1 = register_scalar_field('phi1')
phi2, dphi2, ddphi2 = register_scalar_field('phi2')

NATURAL_POSITIONS[dphi1] = ['down']
NATURAL_POSITIONS[ddphi1] = ['down', 'down']
NATURAL_POSITIONS[dphi2] = ['down']
NATURAL_POSITIONS[ddphi2] = ['down', 'down']

m = Symbol('m')

mu, = fresh_indices(1)
L_M = canon(
    Rational(-1, 2) * dphi1(mu) * dphi1(-mu)
    + Rational(-1, 2) * dphi2(mu) * dphi2(-mu)
    + Rational(-1, 2) * m**2 * phi1() * phi1()
    + Rational(-1, 2) * m**2 * phi2() * phi2()
)
print(f"L_M (complex scalar via two real components) = {L_M}")
print()

# Start with n_max=2 — it's a smaller sanity check before we commit to
# the much more expensive n=3 run.
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
