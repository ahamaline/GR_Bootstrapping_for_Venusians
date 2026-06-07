"""Hilbert scalar bootstrap with a user-supplied optional EOM term at n=0.

X_phi^{mu nu (0)} must be order 0 in h. Since order 0 means h-independent,
partial X / partial h vanishes trivially, so ANY derivative-free X is
Helmholtz-integrable. We pick

    X_phi^{mu nu (0)} = eta^{mu nu} * phi^2

i.e., the metric times a scalar function of phi. (We could use any other
scalar function of phi, or constants, or higher powers.) The corresponding
field redef recovered at the L_ref verification step is

    f_phi^(1) = (1/1) * h_{alpha beta} * X_phi^{alpha beta} = h * phi^2 (trace)
              = h^alpha_alpha * phi^2

so the verification cycle's recovered f at order 1 should match.

Runs through n_max=3 so we can see the optional X propagate via step 2
carryover at every higher order.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric,
)
from bootstrap.bootstrap_loop import BootstrapState


phi, dphi, ddphi = register_scalar_field('phi')
mu0, = fresh_indices(1)
L_M = canon(Rational(-1, 2) * dphi(mu0) * dphi(-mu0))
print(f"L_M (free scalar): {L_M}\n")

N_MAX = 3

state = BootstrapState(L_matter=L_M, em_procedure='hilbert',
                       n_max=N_MAX, verbose=True)
mE, nE = state.mu_E, state.nu_E

# Build X_phi^(0): order 0 in h, derivative-free, integrable by construction
# (h-independent => partial X / partial h = 0 => Helmholtz trivially satisfied).
X_phi_0 = canon(metric(mE, nE) * phi() * phi())
print(f"Optional EOM X_phi^(0) at n=0:  X_phi = {X_phi_0}")
state.add_optional_eom_term(n=0, field_name='phi', X_expr=X_phi_0)
print()

t0 = time.time()
for n in range(N_MAX + 1):
    tn = time.time()
    state.run_order(n)
    print(f"  Order {n} total wall: {time.time()-tn:.1f}s")

print()
print("=" * 60)
print(f"  Hilbert scalar + optional X_phi^(0) at n=0: orders 0..{N_MAX} closed.")
print(f"  Total wall: {time.time()-t0:.1f}s")
print("=" * 60)
