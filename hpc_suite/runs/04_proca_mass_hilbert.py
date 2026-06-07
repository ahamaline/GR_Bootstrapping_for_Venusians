"""HPC suite #4 — full Proca (-1/4 F.F - 1/2 m^2 V.V), upstairs V, Hilbert,
orders 0..6.

Exercises BOTH the upstairs-vector covariant-derivative path (christoffel
corrections to grad V) AND the downstairs-metric expansion (the mass term's
g_{mu nu} V^mu V^nu) at high order.

Usage:  python -u 04_proca_mass_hilbert.py [n_max]   (default 6)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_upstairs_vector_field, fresh_indices, canon, metric,
)
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


V, dV, ddV = register_upstairs_vector_field('V')
m_sym = Symbol('m')

muF, nuF = fresh_indices(2)
r1, r2 = fresh_indices(2)
a1, a2 = fresh_indices(2)
F_dn = metric(-nuF, -r1) * dV(r1, -muF) - metric(-muF, -r2) * dV(r2, -nuF)
F_up = metric(muF, a1) * dV(nuF, -a1) - metric(nuF, a2) * dV(muF, -a2)
mu_m, = fresh_indices(1)
L_M = canon(Rational(-1, 4) * F_dn * F_up
            + Rational(-1, 2) * m_sym**2 * V(mu_m) * V(-mu_m))

print(f"=== full Proca (upstairs V, F.F + mass), Hilbert, n_max={N_MAX} ===",
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

print(f"\n*** PASS: full Proca closes at every order 0-{N_MAX} "
      f"(total {time.time() - t0:.1f}s). ***", flush=True)
