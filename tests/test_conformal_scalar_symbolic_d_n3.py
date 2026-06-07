"""Conformal scalar (symbolic d), no injection, full closure orders 0..3.

The cleanest validation of the symbolic-d machinery: the conformally coupled
scalar with xi(d) = (d-2)/(4(d-1)) is on-shell traceless for ALL d. With NO
optional-EOM injection, the bootstrap should close against L_ref at every
order purely on the standard path -- no field redefinitions, no traceless
recovery. This exercises:
  - traceless-T_M detection with symbolic d (canon now cancels the d-rational
    trace to 0),
  - the d-rational coefficients flowing through T_H, the n=1 integral
    superpotential, PsiForm at n=2,3, and the L_ref verification, all kept
    clean by the canon-level coefficient simplification.

`run_order(n)` raises if EL self-consistency or L_ref verification fails, so
"orders 0..3 complete without raising" IS the closure proof.

Heavy: order-3 step-5 superpotential dominates (~20 min CPU). verbose=True
prints per-order progress.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, dimension,
)
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


# Symbolic d -- the SAME Symbol the metric traces produce, so xi(d) cancels.
d = dimension()

phi, dphi, ddphi = register_scalar_field('phi')
m, = fresh_indices(1)
L_M = canon(-Rational(1, 2) * dphi(m) * dphi(-m))
lam, a = fresh_indices(2)
xi_d = (d - 2) / (4 * (d - 1))
coupling = -Rational(1, 2) * xi_d * phi() * phi() * Riemann(lam, -a, -lam, a)

state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=3,
                       verbose=True, nonminimal_coupling=coupling)

print("##### run_order(0) #####", flush=True)
state.run_order(0)
assert state.traceless_T_M, "expected on-shell-traceless T_M for conformal scalar"
print(f"  c_phi = {state.traceless_c_i.get('phi', S.Zero)}", flush=True)

for n in (1, 2, 3):
    print(f"\n##### run_order({n}) #####", flush=True)
    state.run_order(n)
    print(f"  >>> order {n} closed against L_ref (no raise).", flush=True)

assert 3 in state.E and state.E[3] != S.Zero, "E^(3) should be populated"
print(f"\nE^(3): {_n(state.E[3])} terms", flush=True)
print("\n*** PASS: conformal scalar (symbolic d) closes against L_ref "
      "at every order 0-3, no injection, no field redef. ***", flush=True)
