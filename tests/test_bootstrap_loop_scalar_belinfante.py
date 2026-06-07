"""Bootstrap with em_procedure='belinfante' on a free scalar, through n_max=3.

For a free scalar (no spin), the symmetrized Belinfante tensor equals
Hilbert exactly. So the bootstrap should close identically to the Hilbert
scalar test at every order: step 3 sees Z = 0 and the L_ref verification
sees E_diff = 0 (no field redef needed).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational
from bootstrap.tensor_algebra import register_scalar_field, fresh_indices, canon
from bootstrap.bootstrap_loop import BootstrapState


phi, dphi, ddphi = register_scalar_field('phi')
mu, = fresh_indices(1)
L_M = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
print(f"L_M (free scalar): {L_M}\n")

print("=== Free scalar Belinfante, n_max=3, full cycle ===")
state = BootstrapState(L_matter=L_M, em_procedure='belinfante',
                       n_max=3, verbose=True)
t0 = time.time()
for n in range(4):
    tn = time.time()
    state.run_order(n)
    print(f"\n--- run_order({n}) done in {time.time()-tn:.1f}s ---\n")
print("*** Free scalar Belinfante closed through n=3 ***")
print(f"\nTotal wall: {time.time() - t0:.1f}s")
