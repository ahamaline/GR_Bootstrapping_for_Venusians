"""Conformal scalar (xi=1/6, d=4) end-to-end closure with L_ref engaged.

Exercises the L_ref coupling expansion (covariant_coupling_order in
_init_L_ref) AND the E^(0) improvement together: the verification step checks
EL(L_ref^(n+1)) == E^(n). Order 0 must close by construction (improvement =
EL(coupling^(1))); orders 1-2 test whether the bootstrap-derived coupling
matches the covariant reference.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational
from bootstrap.tensor_algebra import register_scalar_field, fresh_indices, canon
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState

phi, dphi, ddphi = register_scalar_field('phi')
m, = fresh_indices(1)
L_M = canon(-Rational(1, 2) * dphi(m) * dphi(-m))
lam, a = fresh_indices(2)
coupling = -Rational(1, 2) * Rational(1, 6) * phi() * phi() * Riemann(lam, -a, -lam, a)

state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=2,
                       verbose=True, nonminimal_coupling=coupling)
state.run_order(0)
state.run_order(1)
state.run_order(2)
print("\n*** conformal scalar closed through n=2 ***")
