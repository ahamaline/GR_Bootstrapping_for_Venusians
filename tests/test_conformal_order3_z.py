"""ORDER-3 Z HUNT on the tractable conformal scalar (xi=1/6, d=4).

Conformal scalar is on-shell traceless with c_phi = -kappa*phi. Dual-inject a
traceless-shape term at order 1 with X^{ab} = eps kappa^2 h^{ab}:
    X_h^{ab}_{cd} = X^{ab} eta_{cd}     (h-EOM)
    X_phi^{ab}    = X^{ab} c_phi         (scalar-EOM; rank 0, no extra index)
so X_h.E_h^(0) + X_phi.E_phi^(0) = X^{ab}(tr E^(0) + c_phi E_phi^(0)) = 0
(invisible at order 1). At order 3 the carryover is X . S^(2), and
d S^(2)/dh retains ddh (S^(2) has ddh*h terms) -> H2(X . S^(2)) should carry
ddh -> the mandatory-step trigger we've never seen. The scalar's light term
counts (order-2 step1 ~100 vs the vector's ~460) make order 3 tractable.

Run 0-2, then order-3 step1 + step2 + Z, report ddh in Z split by source
(eps-tagged = injection, untagged = natural). n_max=None (skip L_ref).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric, h,
    get_tensors_in_expr, ddh,
)
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState
from bootstrap.helmholtz import compute_h2_violation

kappa = Symbol('kappa')
eps_inj = Symbol('eps_inj')


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


def _terms(e):
    return [] if e == S.Zero else (list(e.args) if isinstance(e, TensAdd) else [e])


phi, dphi, ddphi = register_scalar_field('phi')
m, = fresh_indices(1)
L_M = canon(-Rational(1, 2) * dphi(m) * dphi(-m))
lam, a = fresh_indices(2)
coupling = -Rational(1, 2) * Rational(1, 6) * phi() * phi() * Riemann(lam, -a, -lam, a)

state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=None,
                       verbose=True, nonminimal_coupling=coupling)

print("##### run_order(0) #####")
state.run_order(0)
c_phi = state.traceless_c_i.get('phi', S.Zero)
print(f"\nc_phi = {c_phi}")

X_ab = eps_inj * kappa**2 * h(state.mu_E, state.nu_E)
cc, dd = fresh_indices(2)
state.add_optional_eom_term(1, 'h', canon(X_ab * metric(-cc, -dd)))
state.add_optional_eom_term(1, 'phi', canon(X_ab * c_phi))

for n in (1, 2):
    print(f"\n##### run_order({n}) #####")
    state.run_order(n)
print(f"\n>>> invisibility: eps-tagged terms in E^(1) = "
      f"{len([t for t in _terms(state.E[1]) if t.has(eps_inj)])} (expect 0)")

print("\n##### order 3: step1 + step2 + Z (stop before steps 4-6) #####")
E = state._step1_energy_momentum(3)
print(f"order-3 step1 E_1: {_n(E)} terms")
E = state._step2_eom_carryover(E, 3)
print(f"order-3 step2 E_2: {_n(E)} terms")
Z, _ = compute_h2_violation(E, (state.mu_E, state.nu_E))
ddh_terms = [t for t in _terms(Z) if ddh in set(get_tensors_in_expr(t))]
ddh_eps = [t for t in ddh_terms if t.has(eps_inj)]
print(f"\n>>> order-3 Z: {_n(Z)} terms;  ddh-bearing: {len(ddh_terms)}")
print(f">>>   injection-sourced (eps): {len(ddh_eps)};  natural: {len(ddh_terms)-len(ddh_eps)}")
if ddh_terms:
    print(">>> *** ddh IN Z at order 3 -- mandatory-step trigger FOUND. ***")
    for t in ddh_terms[:8]:
        print(f"      {t}")
else:
    print(">>> no ddh in Z at order 3.")
