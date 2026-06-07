"""HPC suite #3 — EM (-1/4 F.F), d=4, no injection, Hilbert, orders 0..7.

Downstairs gauge field A_mu. d=4 fixed (set_dimension). Validates the
downstairs-A vector matter path (covariantization, T_M, n=1 superpotential,
carryover) at high order.

Usage:  python -u 03_em_d4_hilbert.py [n_max]   (default 7)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# set_dimension(4) MUST run before any matter registration / other imports.
from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import register_vector_field, fresh_indices, canon
from bootstrap.energy_momentum import NATURAL_POSITIONS
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 7


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


A, dA, ddA = register_vector_field('A')
NATURAL_POSITIONS[A] = ['down']
NATURAL_POSITIONS[dA] = ['down', 'down']
NATURAL_POSITIONS[ddA] = ['down', 'down', 'down']

# F_{mu nu} = d_mu A_nu - d_nu A_mu;  dA(field, deriv) = d_{deriv} A_{field}.
mu, nu = fresh_indices(2)
F_dn = dA(-nu, -mu) - dA(-mu, -nu)
F_up = dA(nu, mu) - dA(mu, nu)
L_M = canon(Rational(-1, 4) * F_dn * F_up)

print(f"=== EM (-1/4 F.F), d=4, Hilbert, n_max={N_MAX} ===", flush=True)
print(f"L_M = {L_M}", flush=True)
state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True)
t0 = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  >>> order {n} closed (no raise)  [{time.time() - t:.1f}s, "
          f"E^({n}): {_n(state.E.get(n, S.Zero))} terms]", flush=True)

print(f"\n*** PASS: EM d=4 closes at every order 0-{N_MAX} "
      f"(total {time.time() - t0:.1f}s). ***", flush=True)
