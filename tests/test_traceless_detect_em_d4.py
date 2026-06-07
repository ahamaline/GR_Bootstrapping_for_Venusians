"""Smoke test for traceless-T_M detection on EM with set_dimension(4).

Maxwell electromagnetism is famously traceless in d=4: T_M^a_a = (1 - d/4) F.F
vanishes at d=4. We exercise this by:
  1. Calling set_dimension(4) before any work.
  2. Building L_EM and running BootstrapState.run_order(0).
  3. Verifying that _check_T_M_traceless fires and sets state.traceless_T_M = True
     with state.traceless_c_i empty (off-shell traceless, no matter EOMs needed).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Step 1: import set_dimension ALONE and call it FIRST. Importing anything
# else from the bootstrap package before this would bind the importer's
# local h/dh/ddh names to the symbolic-d heads, and set_dimension's rebuild
# would then leave those local names pointing at stale objects.
from bootstrap.tensor_algebra import set_dimension
set_dimension(4)
print("set_dimension(4) called.")

# Step 2: only NOW import the rest of the bootstrap package; these grab
# the just-rebuilt heads.
from sympy import Rational
from bootstrap.tensor_algebra import register_vector_field, fresh_indices, canon
from bootstrap.bootstrap_loop import BootstrapState

# Standard EM Lagrangian L = -(1/4) F^{mu nu} F_{mu nu}.
A, dA, ddA = register_vector_field('A')
mu, nu = fresh_indices(2)
F_dn = dA(-nu, -mu) - dA(-mu, -nu)
F_up = dA(nu, mu) - dA(mu, nu)
L_EM = canon(Rational(-1, 4) * F_dn * F_up)
print(f"L_EM = {L_EM}")
print()

# Run order 0 only -- detection happens at end of run_order(0).
state = BootstrapState(L_matter=L_EM, em_procedure='hilbert',
                       n_max=None, verbose=True)
state.run_order(0)
print()

print(f"state.traceless_T_M = {state.traceless_T_M}")
print(f"state.traceless_c_i = {state.traceless_c_i}")

assert state.traceless_T_M is True, (
    f"Expected traceless_T_M = True for EM at d=4; got {state.traceless_T_M}"
)
assert state.traceless_c_i == {}, (
    f"Expected empty c_i (off-shell traceless); got {state.traceless_c_i}"
)
print("\n*** EM-d=4 off-shell-traceless detection PASSES ***")
