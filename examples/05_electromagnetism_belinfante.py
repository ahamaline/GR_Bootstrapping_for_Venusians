"""Example 05: electromagnetism on the Belinfante path, n_max = 1.

The most strongly Belinfante-dependent example in this folder.
Electromagnetism has a SPIN-1 field, so the symmetrized Belinfante
energy-momentum differs from Hilbert by genuine EOM-proportional pieces
ALREADY AT THE BARE MATTER LEVEL (the spin-tensor improvement term, §2
of the paper). This is on top of the h-spin contribution that also
makes Belinfante and Hilbert differ at every n >= 1 (see example 03).
Together the two effects make this the most active demonstration of
the EOM-correction + field-redefinition machinery:

  - Step 3's mandatory EOM correction fires at n = 1: the H2-violation
    tensor Z is nonzero, the orchestrator in
    `bootstrap/eom_decompose.py` decomposes Z as X_A . EOM_A (Maxwell's
    equation . A), and the bootstrap adds X_A . E_A^(0) to the field
    equation. A re-check confirms Z is now 0.

  - L_ref verification at n = 0 and n = 1: the bootstrap-derived L^(n+1)
    disagrees with the standard Einstein-Hilbert + matter expansion by
    EOM-proportional pieces. The integrability check recovers
    f_A^(1) = kappa A^L h_{rho L}  (n = 0)
    and at n = 1, both an h-redef AND an A-redef. Each one is applied to
    L_ref^(k) via the substitution machinery; the L_ref diff goes to 0
    after each application.

For the Hilbert version of this same theory, see
`tests/test_bootstrap_loop_em.py`.

Expected wall time: about 5 minutes.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational
from bootstrap.tensor_algebra import (
    register_vector_field, fresh_indices, canon,
)
from bootstrap.bootstrap_loop import BootstrapState


# Register the EM photon as a downstairs vector field A_mu.
A, dA, ddA = register_vector_field('A')

# Build L_EM = -(1/4) F^{mu nu} F_{mu nu}  with F = dA - dA antisymmetrized.
mu, nu = fresh_indices(2)
F_dn = dA(-nu, -mu) - dA(-mu, -nu)        # F_{mu nu}
F_up = dA(nu, mu) - dA(mu, nu)            # F^{mu nu}
L_EM = canon(Rational(-1, 4) * F_dn * F_up)
print(f"Matter Lagrangian:  L_EM = {L_EM}")
print()

N_MAX = 1

state = BootstrapState(L_matter=L_EM, em_procedure='belinfante',
                       n_max=N_MAX, verbose=True)

t_total = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  Order {n} wall: {time.time() - t:.1f}s")

print()
print("=" * 60)
print(f"  EM Belinfante through order {N_MAX} closed and verified.")
print(f"  Total wall: {time.time() - t_total:.1f}s")
print()
print(f"  Watch above for:")
print(f"    - step 3's 'EOM term added: X_A . E_A^(0)' line at n = 1,")
print(f"    - the 'Applying field redefinition' lines at the L_ref check,")
print(f"    - the 'Closure VERIFIED at this order' confirmation.")
print("=" * 60)
