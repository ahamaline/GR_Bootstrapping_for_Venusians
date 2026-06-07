"""Nonminimal vector-curvature coupling: L = -1/4 F.F + xi V^a V^b R_ab,
upstairs vector, Hilbert, NO optional EOM terms, orders 0..3.

First exercise of the nonminimal-coupling machinery (covariant_coupling_order
+ the order-0 improvement EL_h of the coupling) on a VECTOR-Ricci-TENSOR
coupling -- so far it had only been run on the scalar phi^2 R (Ricci scalar).
This stresses (a) the abstract-Riemann expansion with the Ricci tensor index
pattern Riemann(lam,-a,-lam,-b) contracted into V^a V^b, and (b) the upstairs-
vector covariant machinery together with the curvature coupling, end to end.

Standard closure test: run_order raises on any EL-self-consistency or L_ref
verification failure, so "orders 0..3 complete without raising" IS the proof.

d=4 for speed (heavy: vector matter at order 3). verbose prints per-order
progress.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_upstairs_vector_field, fresh_indices, canon, metric,
)
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState

xi = Symbol('xi')


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


V, dV, ddV = register_upstairs_vector_field('V')

# Base kinetic: L = -1/4 F_{mn} F^{mn}, F from the upstairs vector (as in the
# Proca test): F_{mn} = eta_{n r} dV(r,-m) - eta_{m r} dV(r,-n).
muF, nuF = fresh_indices(2)
r1, r2 = fresh_indices(2)
a1, a2 = fresh_indices(2)
F_dn = metric(-nuF, -r1) * dV(r1, -muF) - metric(-muF, -r2) * dV(r2, -nuF)
F_up = metric(muF, a1) * dV(nuF, -a1) - metric(nuF, a2) * dV(muF, -a2)
L_M = canon(Rational(-1, 4) * F_dn * F_up)

# Nonminimal coupling: xi V^a V^b R_ab, with the Ricci tensor
# R_ab = Riemann(lam, -a, -lam, -b) (lam self-contracted; -a,-b free, then
# contracted with V^a V^b). Mirrors the conformal scalar's xi phi^2 R, which
# used the Ricci SCALAR Riemann(lam,-a,-lam,a).
lam, a, b = fresh_indices(3)
coupling = xi * V(a) * V(b) * Riemann(lam, -a, -lam, -b)
print(f"L_M (vector kinetic) = {L_M}", flush=True)
print(f"coupling (xi V^a V^b R_ab) = {coupling}", flush=True)

state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=3,
                       verbose=True, nonminimal_coupling=coupling)

for n in range(4):
    print(f"\n##### run_order({n}) #####", flush=True)
    state.run_order(n)
    print(f"  >>> order {n} closed against L_ref (no raise).", flush=True)

assert 3 in state.E and state.E[3] != S.Zero, "E^(3) should be populated"
print(f"\nE^(3): {_n(state.E[3])} terms", flush=True)
print("\n*** PASS: nonminimal vector-Ricci coupling closes against L_ref "
      "at every order 0-3. ***", flush=True)
