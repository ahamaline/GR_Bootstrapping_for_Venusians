# BootstrapState driver with the EM field as matter.
#   L_M = -1/4 F_{μν} F^{μν},   F_{μν} = ∂_μ A_ν − ∂_ν A_μ
#
# Per item 6 of DEVELOPMENT_STATUS.md, the antisymmetric structure of F
# means the Christoffel corrections in ∇_μ A_ν cancel, so covariantization
# of L_EM reduces to η→g in the metric contractions plus the √|g| factor —
# exactly what `matter_lagrangian_order` and `hilbert_energy_momentum`
# already do. No new mechanism needed beyond relaxing the two scalar-only
# guards that were already in place.
#
# This test runs orders 0..2 and verifies the closure cycle at each.
# (The n=1 Psi integral is exercised but skips the vector field per the
# default-path argument — see the comment in compute_superpotential_n1.)

import sys
import time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S
from sympy.tensor.tensor import TensAdd

from bootstrap.tensor_algebra import (
    register_vector_field, fresh_indices, canon,
)
from bootstrap.energy_momentum import NATURAL_POSITIONS
from bootstrap.bootstrap_loop import BootstrapState

A, dA, ddA = register_vector_field('A')

NATURAL_POSITIONS[A] = ['down']
NATURAL_POSITIONS[dA] = ['down', 'down']
NATURAL_POSITIONS[ddA] = ['down', 'down', 'down']

# L_EM = -1/4 F_{μν} F^{μν}
# dA convention: dA(field, deriv) = ∂_{deriv} A_{field}
# F_{μν} = ∂_μ A_ν − ∂_ν A_μ = dA(-ν, -μ) − dA(-μ, -ν)
# F^{μν} = dA(ν, μ) − dA(μ, ν)
mu, nu = fresh_indices(2)
F_dn = dA(-nu, -mu) - dA(-mu, -nu)
F_up = dA(nu, mu) - dA(mu, nu)
L_M = canon(Rational(-1, 4) * F_dn * F_up)
print(f"L_M (EM) = {L_M}")
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
