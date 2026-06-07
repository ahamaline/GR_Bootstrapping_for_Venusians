"""Mandatory-EOM-step traceless path at order 3 (conformal scalar, injected).

Builds on test_conformal_order3_z: the dual-injected conformal scalar produces
a ddh-bearing H2 violation Z at order 3 (4 ddh terms, all injection-sourced).
This test exercises the NEW step-3 traceless absorber
(_recover_traceless_mandatory_eom): it recovers a new order-2 mandatory
traceless-shape X via the ddh box signature, adds X·S^(1) to E so its H2
cancels Z's ddh, then decomposes the (now ddh-free) residual normally and
verifies post-correction H2 = 0.

Runs orders 0-2, then the FULL step 3 at order 3 (step1 + step2 + step3),
stopping before the expensive steps 4-6 (superpotential/close). n_max=None.

Expected:
  - step 3 returns without raising (NotImplementedError would mean the
    residual didn't decompose; RuntimeError would mean ddh survived or the
    post-correction H2 != 0),
  - the recovered X^(ab,2) is recorded in state.recovered_traceless_X[3],
  - compute_h2_violation(E_3) == 0.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric, h,
)
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState

kappa = Symbol('kappa')
eps_inj = Symbol('eps_inj')


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


phi, dphi, ddphi = register_scalar_field('phi')
m, = fresh_indices(1)
L_M = canon(-Rational(1, 2) * dphi(m) * dphi(-m))
lam, a = fresh_indices(2)
coupling = -Rational(1, 2) * Rational(1, 6) * phi() * phi() * Riemann(lam, -a, -lam, a)

state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=None,
                       verbose=True, nonminimal_coupling=coupling)

print("##### run_order(0) #####", flush=True)
state.run_order(0)
assert state.traceless_T_M, "expected on-shell-traceless T_M for conformal scalar"
c_phi = state.traceless_c_i.get('phi', S.Zero)
print(f"\nc_phi = {c_phi}", flush=True)

X_ab = eps_inj * kappa**2 * h(state.mu_E, state.nu_E)
cc, dd = fresh_indices(2)
state.add_optional_eom_term(1, 'h', canon(X_ab * metric(-cc, -dd)))
state.add_optional_eom_term(1, 'phi', canon(X_ab * c_phi))

for n in (1, 2):
    print(f"\n##### run_order({n}) #####", flush=True)
    state.run_order(n)

print("\n##### order 3: step1 + step2 + step3 (mandatory EOM) #####", flush=True)
E = state._step1_energy_momentum(3)
print(f"order-3 step1 E_1: {_n(E)} terms", flush=True)
E = state._step2_eom_carryover(E, 3)
print(f"order-3 step2 E_2: {_n(E)} terms", flush=True)

# The mandatory step (now with the traceless absorber).
# Recovery method asserts Z is ddh-free post-correction.
E3 = state._step3_mandatory_eom(E, 3)
print(f"order-3 step3 E_3: {_n(E3)} terms", flush=True)

# Validations: recovered X recorded and stored for carryover.
assert 3 in state.recovered_traceless_X, \
    "step 3 should have recorded a recovered traceless X at n=3"
X = state.recovered_traceless_X[3]
print(f"recovered X^(ab,2) = {X}", flush=True)

# X^(ab,2) should be stored for carryover.
assert 2 in state.eom_terms_h, "X_h^(2) should be stored for carryover"
assert state.eom_terms_matter.get(2, {}).get('phi', S.Zero) != S.Zero, \
    "X_phi^(2) should be stored for carryover"

print("\n*** PASS: mandatory-step traceless absorber closes H2 at order 3. ***",
      flush=True)
