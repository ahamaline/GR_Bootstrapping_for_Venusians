"""Example 02: free massless scalar coupled to gravity (Hilbert), n_max = 2.

Same bootstrap procedure as example 01, but now with matter: a real
scalar field `phi` with Lagrangian

    L_phi = -(1/2) (d phi)^2

(canonical kinetic, mostly-plus signature). The matter modifies what
the bootstrap produces at every order:

  - At n = 0, E^(0) is no longer 0; it equals kappa T_M[L_phi], the
    Hilbert energy-momentum tensor of the free scalar.
  - At higher orders, the bootstrap discovers the right h-phi couplings
    (the leading kappa * h_{mu nu} * T_M^{mu nu} interaction at n = 1
    and its descendants at n = 2).

This is the first case where you see term counts split by field
content. Each step's print line breaks down the terms as
`X h-only, Y phi` — useful for tracking whether gravity, matter, or
their interaction dominates a given step.

Expected wall time: about 3 minutes.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon,
)
from bootstrap.bootstrap_loop import BootstrapState


# Register the matter field. `register_scalar_field` creates the
# TensorHeads for phi, dphi, ddphi and records them in a global registry
# so the bootstrap machinery (energy-momentum, EL, etc.) knows about them.
phi, dphi, ddphi = register_scalar_field('phi')

# Free massless scalar: L_phi = -(1/2) (d_mu phi)(d^mu phi).
mu, = fresh_indices(1)
L_phi = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
print(f"Matter Lagrangian:  L_phi = {L_phi}")
print()

N_MAX = 2

state = BootstrapState(L_matter=L_phi, em_procedure='hilbert',
                       n_max=N_MAX, verbose=True)

t_total = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  Order {n} wall: {time.time() - t:.1f}s")

print()
print("=" * 60)
print(f"  Free scalar + gravity through order {N_MAX} closed and verified.")
print(f"  Total wall: {time.time() - t_total:.1f}s")
print("=" * 60)
