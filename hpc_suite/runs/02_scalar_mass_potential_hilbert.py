"""HPC suite #2 — scalar with mass & potential, Hilbert, orders 0..7.

L_M = -1/2 dphi.dphi - 1/2 m^2 phi^2. Validates matter T_M + the n=1 integral
superpotential + PsiForm-with-matter + step-2 carryover at high order.

Usage:  python -u 02_scalar_mass_potential_hilbert.py [n_max]   (default 7)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import register_scalar_field, fresh_indices, canon
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 7


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


phi, dphi, ddphi = register_scalar_field('phi')
m = Symbol('m')
mu_d, = fresh_indices(1)
L_M = canon(Rational(-1, 2) * dphi(mu_d) * dphi(-mu_d)
            + Rational(-1, 2) * m**2 * phi() * phi())

print(f"=== massive scalar (kinetic + m^2 phi^2), Hilbert, n_max={N_MAX} ===",
      flush=True)
print(f"L_M = {L_M}", flush=True)
state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True)
t0 = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  >>> order {n} closed (no raise)  [{time.time() - t:.1f}s, "
          f"E^({n}): {_n(state.E.get(n, S.Zero))} terms]", flush=True)

print(f"\n*** PASS: massive scalar closes at every order 0-{N_MAX} "
      f"(total {time.time() - t0:.1f}s). ***", flush=True)
