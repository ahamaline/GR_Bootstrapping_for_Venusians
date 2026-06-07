"""Order-0 gate for the conformal-scalar pivot: a massless scalar with the
conformal nonminimal coupling -1/2 xi phi^2 R (xi = 1/6 in d=4) must come out
ON-shell traceless (c_phi != 0). Validates the coupling wiring (improvement
added to E^(0)) + traceless detection on the scalar branch.

With symbolic xi, tr E^(0) = (6k xi - k)(dphi)^2 + 6k xi (box phi) phi; the
non-EOM (dphi)^2 term vanishes at xi = 1/6 = (d-2)/(4(d-1)), leaving
tr E^(0) = k phi E_phi -> on-shell traceless. Cheap: order 0, n_max=None.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational, S
from bootstrap.tensor_algebra import register_scalar_field, fresh_indices, canon, metric
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState


def main():
    xi = Rational(1, 6)  # conformal value in d=4
    phi, dphi, ddphi = register_scalar_field('phi')
    m, = fresh_indices(1)
    L_M = canon(-Rational(1, 2) * dphi(m) * dphi(-m))

    lam, a = fresh_indices(2)
    R_scalar = Riemann(lam, -a, -lam, a)              # Ricci scalar, intuitive
    coupling = -Rational(1, 2) * xi * phi() * phi() * R_scalar

    state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=None,
                           verbose=False, nonminimal_coupling=coupling)
    state.run_order(0)

    tr = canon(state.E[0] * metric(-state.mu_E, -state.nu_E))
    print(f"tr E^(0) at xi=1/6 (expect purely box-phi*phi): {tr}")
    print(f"traceless_T_M = {state.traceless_T_M};  c_i = {state.traceless_c_i}")

    assert state.traceless_T_M is True, (
        "conformal scalar at xi=1/6, d=4 should be on-shell traceless")
    c_phi = state.traceless_c_i.get('phi', S.Zero)
    assert c_phi != S.Zero and c_phi.has(phi), (
        f"expected nonzero c_phi proportional to phi, got {c_phi}")
    print(f"\n*** conformal-scalar order-0 gate PASSES (c_phi = {c_phi}) ***")


if __name__ == '__main__':
    main()
