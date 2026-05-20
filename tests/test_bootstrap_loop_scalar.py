# Drives the bootstrap loop with a single massless scalar matter field.
# Smoke-test of matter ingestion: run_order(0) should give the expected
# matter T_M source, and the EL self-consistency check at n=0 should pass.
#
# This is Phase A of matter support — verification against L_ref is skipped
# (L_ref expansion for matter is Phase B; n_max=None disables that path).

import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S
from sympy.tensor.tensor import TensAdd

from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, h,
)
from bootstrap.energy_momentum import NATURAL_POSITIONS
from bootstrap.bootstrap_loop import BootstrapState, kappa

# Register a single scalar field and tell the Hilbert procedure that its
# derivative indices are naturally down.
phi, dphi, ddphi = register_scalar_field('phi')
NATURAL_POSITIONS[dphi] = ['down']
NATURAL_POSITIONS[ddphi] = ['down', 'down']

# L_M = -1/2 dphi^mu dphi_mu  (massless free scalar)
mu, = fresh_indices(1)
L_M = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
print(f"L_M = {L_M}")

# Pure-gravity-Hilbert + scalar matter. n_max=1 forces L_ref to be built
# through order 2 — that exercises both the order-1 (κ h T_M) and the
# order-2 (mixed EH × matter √|g|·g^{-1} expansion) pieces of matter
# L_ref construction. The bootstrap itself is only run at n=0; n=1 with
# matter needs the integral formula (Phase C).
print()
state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=1,
                       verbose=True)
# Sanity: L_ref^(2) for our massless free scalar should be nonzero —
# the EH part contributes (it's L_EH^{(2)}) and the matter part
# contributes (κ² × something × T_M, from the order-2 of √|g| g^{-1}).
print()
n_Lref2 = len(state.L_ref[2].args) if hasattr(state.L_ref[2], 'args') else 1
print(f"L_ref^(2) sanity: {n_Lref2} terms — nontrivial as expected.")
assert state.L_ref[2] != 0, "L_ref^(2) should be nonzero with matter present"

# Run n=0: should compute E^(0) = kappa * T_M and L^(1) = kappa * h * T_M,
# and the verification cycle should pass against L_ref^(1) = kappa * h * T_M
# (the order-1 part of sqrt(|g|) * L_tilde_M for our scalar L_M).
state.run_order(0)

# Independent sanity check: E^(0) free indices are state's canonical (mu_E, nu_E),
# and the value matches the textbook T_M^{mn} for a massless scalar.
m, n = state.mu_E, state.nu_E
rho, = fresh_indices(1)
from bootstrap.tensor_algebra import metric
T_M_expected = (dphi(m) * dphi(n)
                + Rational(-1, 2) * metric(m, n) * dphi(rho) * dphi(-rho))
E_expected = canon(kappa * T_M_expected)

E_actual = canon(state.E[0])
diff = canon(E_actual - E_expected)
n_diff = len(diff.args) if isinstance(diff, TensAdd) else (0 if diff == S.Zero else 1)
print()
print(f"E^(0) expected (textbook kappa*T_M): {E_expected}")
print(f"E^(0) actual:                       {E_actual}")
print(f"Difference: {n_diff} terms")
assert diff == S.Zero, "E^(0) does not match the textbook kappa*T_M for a massless scalar"
print()
print("*** n=0 matter ingestion verified: E^(0) == kappa*T_M (textbook). ***")
print("    EL self-consistency (EL(L^(1)) == E^(0)) also verified by run_order.")

print()
print("=" * 60)
print("  Now: bootstrap n=1 with the scalar matter")
print("=" * 60)
state.run_order(1)
print()
print("*** n=1 matter closure verified end-to-end. ***")
