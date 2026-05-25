# BootstrapState with a charged complex scalar coupled to EM (scalar QED).
#
# Writing the complex scalar in real components phi = (phi1 + i phi2)/sqrt(2)
# and the EM potential A_mu, the scalar QED Lagrangian
#   L = -(D_mu phi)^* (D^mu phi) - m^2 |phi|^2 - 1/4 F.F
# with D_mu = ∂_mu - i e A_mu  expands to (no i factors remain, as expected):
#   L = -1/2 [∂phi1 . ∂phi1 + ∂phi2 . ∂phi2]      (kinetic for two real scalars)
#     - (m^2/2) (phi1^2 + phi2^2)                  (mass)
#     - 1/4 F_{mn} F^{mn}                          (EM)
#     - e A^mu (phi1 ∂_mu phi2 - phi2 ∂_mu phi1)   (minimal coupling current)
#     - (e^2/2) A^mu A_mu (phi1^2 + phi2^2)        (seagull)
#
# All three matter fields (phi1, phi2, A) are run through the bootstrap.
# Psi^(1) is computed via compute_superpotential_n1 (now actually running
# on vector matter, not skipped); we expect it to come out as 0 because
# we're on the default path (Hilbert + no optional EOM).

import sys
import time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S, Symbol
from sympy.tensor.tensor import TensAdd

from bootstrap.tensor_algebra import (
    register_scalar_field, register_vector_field, fresh_indices, canon,
)
from bootstrap.energy_momentum import NATURAL_POSITIONS
from bootstrap.bootstrap_loop import BootstrapState

# Two real scalars representing Re(phi), Im(phi).
phi1, dphi1, ddphi1 = register_scalar_field('phi1')
phi2, dphi2, ddphi2 = register_scalar_field('phi2')

# EM potential.
A, dA, ddA = register_vector_field('A')

NATURAL_POSITIONS[dphi1] = ['down']
NATURAL_POSITIONS[ddphi1] = ['down', 'down']
NATURAL_POSITIONS[dphi2] = ['down']
NATURAL_POSITIONS[ddphi2] = ['down', 'down']
NATURAL_POSITIONS[A] = ['down']
NATURAL_POSITIONS[dA] = ['down', 'down']
NATURAL_POSITIONS[ddA] = ['down', 'down', 'down']

m_sym = Symbol('m')
e_sym = Symbol('e')

# Kinetic terms.
mu_k1, = fresh_indices(1)
L_kin_1 = Rational(-1, 2) * dphi1(mu_k1) * dphi1(-mu_k1)
mu_k2, = fresh_indices(1)
L_kin_2 = Rational(-1, 2) * dphi2(mu_k2) * dphi2(-mu_k2)

# Mass term.
L_mass = Rational(-1, 2) * m_sym**2 * (phi1() * phi1() + phi2() * phi2())

# EM Lagrangian: -1/4 F.F, F_{mn} = ∂_m A_n - ∂_n A_m.
mu_F, nu_F = fresh_indices(2)
F_dn = dA(-nu_F, -mu_F) - dA(-mu_F, -nu_F)
F_up = dA(nu_F, mu_F) - dA(mu_F, nu_F)
L_EM = Rational(-1, 4) * F_dn * F_up

# Minimal coupling: -e A^mu (phi1 ∂_mu phi2 - phi2 ∂_mu phi1).
mu_c, = fresh_indices(1)
L_min_coupling = -e_sym * A(mu_c) * (phi1() * dphi2(-mu_c) - phi2() * dphi1(-mu_c))

# Seagull: -(e^2 / 2) A^mu A_mu (phi1^2 + phi2^2).
mu_s, = fresh_indices(1)
L_seagull = (Rational(-1, 2) * e_sym**2 * A(mu_s) * A(-mu_s)
             * (phi1() * phi1() + phi2() * phi2()))

L_M = L_kin_1 + L_kin_2 + L_mass + L_EM + L_min_coupling + L_seagull
L_M = canon(L_M)

print("L_M (charged complex scalar coupled to EM):")
print(f"  {L_M}")
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
