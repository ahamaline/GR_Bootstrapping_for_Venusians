"""Example 04: voluntary path — supply an optional EOM term at n = 0.

The paper's step 4 is the "voluntary path": at any order, the user is
free to add a term `X^(n) . E^(0)` to the field equation, provided that
X is derivative-free AND satisfies the Helmholtz integrability
condition (so that the added term actually comes from a Lagrangian
shift / field redefinition). The bootstrap accepts the X, validates it,
adds the term, and threads it through all subsequent steps.

This is how different physical conventions for the same theory are
represented in the bootstrap: every legal choice of X^(n) leads to a
SELF-CONSISTENT but DIFFERENT-LOOKING Lagrangian, all encoding the same
underlying physics.

In this example we take a free massless scalar (which has a manifest
shift symmetry phi -> phi + c at order 0) and pick a particular
optional term at n = 0:

    X_phi^{mu nu (0)} = eta^{mu nu} * phi^2

This is order 0 in h (h-independent), so trivially Helmholtz-integrable
(partial X / partial h = 0). The corresponding field redefinition is

    f_phi^(1) = h_{ab} X^{ab} = (tr h) phi^2,

i.e., phi -> phi + (tr h) phi^2. Substituting this redef into the free
scalar Lagrangian breaks the manifest shift symmetry and generates a
polynomial in phi at every higher order: f_phi^(2), f_phi^(3), ...

As we run the bootstrap, watch the f_phi^(k+1) that the verification
cycle RECOVERS at every order. Each one matches the polynomial
expansion of the inverse of our chosen X. (See the README at the
repo root for background on the recovery formula.)

Expected wall time: about 10-15 minutes (the optional X propagates via
step 2 carryover at every higher order, and the verification cycle
recovers and substitutes a new f at each order).
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric,
)
from bootstrap.bootstrap_loop import BootstrapState


phi, dphi, ddphi = register_scalar_field('phi')
mu, = fresh_indices(1)
L_phi = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
print(f"Matter Lagrangian:  L_phi = {L_phi}")
print()

N_MAX = 2

state = BootstrapState(L_matter=L_phi, em_procedure='hilbert',
                       n_max=N_MAX, verbose=True)

# Build the optional X. State.mu_E and state.nu_E are the canonical
# free indices the bootstrap uses everywhere for E^{mu nu}.
mE, nE = state.mu_E, state.nu_E
X_phi_0 = canon(metric(mE, nE) * phi() * phi())
print(f"Optional EOM coefficient (n = 0):  X_phi^(0) = {X_phi_0}")
print(f"  -> corresponds to the field redef  phi -> phi + (tr h) * phi^2.")
print()
state.add_optional_eom_term(n=0, field_name='phi', X_expr=X_phi_0)

t_total = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  Order {n} wall: {time.time() - t:.1f}s")

print()
print("=" * 60)
print(f"  Optional-EOM scalar through order {N_MAX} closed and verified.")
print(f"  Total wall: {time.time() - t_total:.1f}s")
print(f"")
print(f"  Note: the f_phi^(k+1) recovered at each verification step is")
print(f"  the additional redef beyond what was already in effect at order k.")
print(f"  The full composed redef is the product chain f^(1) -> f^(2) -> ...")
print("=" * 60)
