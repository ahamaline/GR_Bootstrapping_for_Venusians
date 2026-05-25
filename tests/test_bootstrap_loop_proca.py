"""BootstrapState driver with the full Proca Lagrangian (upstairs vector V).

  L_M = -1/4 F_{mn} F^{mn}  -  1/2 m^2 V^mu V_mu

  F_{mn} = ∂_m A_n − ∂_n A_m,  A_mu = V_mu = eta_{mu rho} V^rho.

V is registered as a naturally upstairs vector (V^mu). This exercises:

  - the downstairs-metric infrastructure (Phase A) via the eta_{nu rho}
    factors that lower V^rho inside F_{mn}, and via the eta_{mu nu} factor
    inside the mass term.
  - the upstairs-vector covariant-derivative path (Phase B): every dV
    factor inside F is expanded via ∇_rho V^sigma = dV(sigma, rho)
    + Gamma^sigma_{rho tau} V^tau in matter_lagrangian_order, and the same
    correction shows up inside hilbert_energy_momentum's Christoffel
    contribution.

The two pieces of F are connected by an antisymmetric exchange, so the
Christoffel corrections from the two dV's cancel inside the gauge-invariant
F_{mn} F^{mn} structure — but the algorithm has to *compute* them and let
the cancellation happen. The Christoffel corrections do NOT cancel in the
mass term (no dV's there), but the mass term contributes via the downstairs
metric only.

Runs orders 0..2 with the full closure verification cycle.
"""

import sys
import time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S, Symbol

from bootstrap.tensor_algebra import (
    register_upstairs_vector_field, fresh_indices, canon, metric,
)
from bootstrap.bootstrap_loop import BootstrapState


V, dV, ddV = register_upstairs_vector_field('V')

m_sym = Symbol('m')

# F_{mn} F^{mn} written in terms of V^mu and explicit eta factors:
#   F_{mn}  = eta_{n rho} dV(rho, -m) - eta_{m rho} dV(rho, -n)
#   F^{mn}  = eta^{m alpha} dV(n, -alpha) - eta^{n beta} dV(m, -beta)
# (dV convention: dV(field_up, deriv_down) = ∂_deriv V^field)
mu_F, nu_F = fresh_indices(2)
rho1, rho2 = fresh_indices(2)
alpha1, alpha2 = fresh_indices(2)

F_dn = (metric(-nu_F, -rho1) * dV(rho1, -mu_F)
        - metric(-mu_F, -rho2) * dV(rho2, -nu_F))
F_up = (metric(mu_F, alpha1) * dV(nu_F, -alpha1)
        - metric(nu_F, alpha2) * dV(mu_F, -alpha2))

L_M_FF = Rational(-1, 4) * F_dn * F_up

mu_m, = fresh_indices(1)
L_M_mass = Rational(-1, 2) * m_sym**2 * V(mu_m) * V(-mu_m)

L_M = canon(L_M_FF + L_M_mass)
print(f"L_M (Proca, full) = {L_M}")
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
