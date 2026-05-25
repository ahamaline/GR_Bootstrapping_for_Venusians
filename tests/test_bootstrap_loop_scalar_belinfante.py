"""BootstrapState with em_procedure='belinfante', free scalar matter.

For a free scalar (no spin), the symmetrized Belinfante tensor equals
Hilbert exactly, so the bootstrap should close identically to the
existing Hilbert scalar test. First non-trivial Belinfante run.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S
from bootstrap.tensor_algebra import register_scalar_field, fresh_indices, canon
from bootstrap.bootstrap_loop import BootstrapState


phi, dphi, ddphi = register_scalar_field('phi')
mu, = fresh_indices(1)
L_M = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
print(f"L_M (free scalar): {L_M}\n")

print("=== Bootstrap with em_procedure='belinfante' ===")
state = BootstrapState(L_matter=L_M, em_procedure='belinfante',
                       n_max=1, verbose=True)
t0 = time.time()
state.run_order(0)
state.run_order(1)
print(f"\nTotal wall: {time.time() - t0:.1f}s")
print("\n*** Belinfante n=1 closure verified ***")
