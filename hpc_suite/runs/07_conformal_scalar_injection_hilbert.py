"""HPC suite #7 — conformal scalar with dual injection, SYMBOLIC d, Hilbert,
orders 0..5.

The rigorous high-order test of BOTH traceless-T_M recoveries together:
  - verification-path recovery (missed order-(n-1) traceless redef), and
  - mandatory-step recovery (ddh in the H2 violation Z),
plus the rollback+augment of the order-(n-1) redef and the carryover of the
recovered X. Symbolic d (not d=4) so closure cannot be a numeric coincidence.

L_M = -1/2 dphi.dphi  with the conformal coupling -xi(d)/2 phi^2 R, xi=(d-2)/(4(d-1)).
A traceless-shape optional EOM is injected at order 1 (dual: h and phi).

Usage:  python -u 07_conformal_scalar_injection_hilbert.py [n_max]   (default 5)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric, h, dimension,
)
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
kappa = Symbol('kappa')
eps_inj = Symbol('eps_inj')
d = dimension()


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


phi, dphi, ddphi = register_scalar_field('phi')
m, = fresh_indices(1)
L_M = canon(-Rational(1, 2) * dphi(m) * dphi(-m))
lam, a = fresh_indices(2)
xi_d = (d - 2) / (4 * (d - 1))
coupling = -Rational(1, 2) * xi_d * phi() * phi() * Riemann(lam, -a, -lam, a)

print(f"=== conformal scalar + dual injection, symbolic d, Hilbert, "
      f"n_max={N_MAX} ===", flush=True)
state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True, nonminimal_coupling=coupling)

state.run_order(0)
assert state.traceless_T_M, "expected on-shell-traceless T_M for conformal scalar"
c_phi = state.traceless_c_i.get('phi', S.Zero)
print(f"  c_phi = {c_phi}", flush=True)

# Dual injection at order 1: a traceless-shape optional EOM (h and phi).
X_ab = eps_inj * kappa**2 * h(state.mu_E, state.nu_E)
cc, dd = fresh_indices(2)
state.add_optional_eom_term(1, 'h', canon(X_ab * metric(-cc, -dd)))
state.add_optional_eom_term(1, 'phi', canon(X_ab * c_phi))

t0 = time.time()
for n in range(1, N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  >>> order {n} closed (no raise)  [{time.time() - t:.1f}s, "
          f"E^({n}): {_n(state.E.get(n, S.Zero))} terms]", flush=True)

print(f"\n  recovered_traceless_X keys: {sorted(state.recovered_traceless_X)}",
      flush=True)
print(f"\n*** PASS: conformal scalar (symbolic d, dual injection) closes at "
      f"every order 0-{N_MAX} (total {time.time() - t0:.1f}s). ***", flush=True)
