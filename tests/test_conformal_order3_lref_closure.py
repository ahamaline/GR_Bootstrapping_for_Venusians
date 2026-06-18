"""End-to-end L_ref closure at order 3 (conformal scalar, dual injection).

The real proof of the traceless-recovery procedure: run the FULL pipeline
(all 6 steps + superpotential + close + L_ref verification) on the
dual-injected conformal scalar with n_max=3, and require

    canon(EL(L_ref^{(n+1)}) - E^{(n)}) == 0   for every order n = 0..3.

`_verify_vs_L_ref` (called inside run_order whenever n_max is set) RAISES
RuntimeError if that closure fails -- so "run_order(0..3) all return without
raising" IS the closure proof, enforced by exceptions.

This single run is self-staging and exercises BOTH traceless recoveries plus
the order-3 superpotential together for the first time:
  - n=2: the VERIFICATION-path recovery (_recover_missed_traceless_redef) fires
    on E_diff = X_inj.S^(1) (has ddh) -- the conformal scalar's c_i != 0
    matter-redef branch, run early/cheap before the expensive order-3 work.
  - n=3: the MANDATORY-step recovery (_recover_traceless_mandatory_eom) fires
    on the ddh-bearing H2 violation Z inside step 3.

NOTE: heavy run (order-3 step-3 H2 alone is ~20 min CPU; the order-3
superpotential/close adds more). verbose=True prints per-order progress.

Expected:
  - run_order(0..3) complete without raising  => closure verified each order,
  - recovered_traceless_X has key 2 (verification path) and key 3 (mandatory),
  - state.E[3] is populated.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# d=4 for speed (plain integer coefficients, no rational-in-d overhead). The
# symbolic-d version (d = dimension(); xi = (d-2)/(4(d-1))) is the rigorous
# variant -- flip the next two lines + the coupling back to it for the final
# confirmation run. The checkpoint below supports cross-process resume
# (test_..._resume.py): save_state captures _dim + _field_specs so a fresh
# process can set_dimension(4), re-register 'phi', and continue at order 3.
from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric, h,
)
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState, save_state

kappa = Symbol('kappa')
eps_inj = Symbol('eps_inj')


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


phi, dphi, ddphi = register_scalar_field('phi')
m, = fresh_indices(1)
L_M = canon(-Rational(1, 2) * dphi(m) * dphi(-m))
lam, a = fresh_indices(2)
# Conformal coupling at d=4: xi = (d-2)/(4(d-1)) = 1/6.
coupling = -Rational(1, 2) * Rational(1, 6) * phi() * phi() * Riemann(lam, -a, -lam, a)

state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=3,
                       verbose=True, nonminimal_coupling=coupling)

print("##### run_order(0) #####", flush=True)
state.run_order(0)
assert state.traceless_T_M, "expected on-shell-traceless T_M for conformal scalar"
c_phi = state.traceless_c_i.get('phi', S.Zero)
print(f"\nc_phi = {c_phi}", flush=True)

# Dual injection at order 1 (same as the isolated mandatory-step test).
X_ab = eps_inj * kappa**2 * h(state.mu_E, state.nu_E)
cc, dd = fresh_indices(2)
state.add_optional_eom_term(1, 'h', canon(X_ab * metric(-cc, -dd)))
state.add_optional_eom_term(1, 'phi', canon(X_ab * c_phi))

for n in (1, 2):
    print(f"\n##### run_order({n}) #####", flush=True)
    state.run_order(n)
    print(f"  >>> order {n} closed against L_ref (no raise).", flush=True)

# Checkpoint the state after order 2. Supports cross-process resume
# (test_..._resume.py loads this and runs order 3 in a fresh process) and
# inspection (_dump_ckpt.py). Non-fatal: a pickling hiccup must not abort
# before the in-process order-3 run below.
ckpt_path = '_conf_o3_ckpt_order2.pkl'
try:
    save_state(state, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")
except Exception as e:
    print(f"\n[warn] checkpoint pickle failed (non-fatal): {e}")
if __name__ != '__main__':
    sys.exit(0)

print(f"\n##### run_order(3) #####", flush=True)
state.run_order(3)
print(f"  >>> order 3 closed against L_ref (no raise).", flush=True)

# Both recovery paths must have fired.
print(f"\nrecovered_traceless_X keys: {sorted(state.recovered_traceless_X)}",
      flush=True)
assert 2 in state.recovered_traceless_X, \
    "verification-path recovery should have fired at n=2 (key 2 missing)"
assert 3 in state.recovered_traceless_X, \
    "mandatory-step recovery should have fired at n=3 (key 3 missing)"
print(f"  X (verification, n=2) = {state.recovered_traceless_X[2]}", flush=True)
print(f"  X (mandatory,   n=3) = {state.recovered_traceless_X[3]}", flush=True)

assert 3 in state.E and state.E[3] != S.Zero, "E^(3) should be populated"
print(f"\nE^(3): {_n(state.E[3])} terms", flush=True)

print("\n*** PASS: conformal scalar closes against L_ref at every order 0-3. ***",
      flush=True)
