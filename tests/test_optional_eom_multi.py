"""Hilbert scalar bootstrap with optional EOM terms:
  - Case A: a single X_h^(1) at n=1 (exercises the h-redef branch of step 4).
  - Case B: X_phi^(0) at n=0 plus X_h^(1) at n=1, run together (multiple
    optional terms at different orders, mixing matter and h).

Both cases run through n_max=2 so step 2's carryover fires at n=1 and n=2
on the user-supplied X's.

Construction: every X is X = partial f / partial h for an arbitrary f, so
Helmholtz integrability holds by Schwarz's theorem on the resulting X.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric, h,
    _matter_fields,
)
from bootstrap.jet import jet_derivative
from bootstrap.bootstrap_loop import BootstrapState


def _fresh_scalar_state(n_max):
    """Reset registry and build a BootstrapState with a free scalar."""
    _matter_fields.clear()
    register_scalar_field('phi')
    info = _matter_fields['phi']
    dphi = info['dfield']
    mu, = fresh_indices(1)
    L_M = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
    return BootstrapState(L_matter=L_M, em_procedure='hilbert',
                          n_max=n_max, verbose=True), info


def build_X_h_n1(mE, nE):
    """X_h^(1) = partial f^(2) / partial h, with f^(2)_{kappa, lambda} =
    h(-kappa, alpha) h(-lambda, -alpha). Order 1 in h, derivative-free,
    Helmholtz-integrable by Schwarz."""
    kappa, lam = fresh_indices(2)
    alpha, = fresh_indices(1)
    f = canon(h(-kappa, alpha) * h(-lam, -alpha))
    X = canon(jet_derivative(f, h, [mE, nE]))
    return X


def build_X_phi_n0(mE, nE, phi_info):
    """X_phi^(0) = eta^{mu nu} * phi^2 (order 0 in h, trivially integrable
    since h-independent)."""
    phi = phi_info['field']
    return canon(metric(mE, nE) * phi() * phi())


def case_A_h_only():
    """Single optional X_h^(1) at n=1, scalar Hilbert, n_max=2."""
    print("\n" + "=" * 70)
    print("  CASE A: optional X_h^(1) at n=1 (Hilbert scalar, n_max=2)")
    print("=" * 70 + "\n")
    state, phi_info = _fresh_scalar_state(n_max=2)
    mE, nE = state.mu_E, state.nu_E
    X_h_1 = build_X_h_n1(mE, nE)
    print(f"Optional EOM X_h^(1) at n=1:")
    print(f"  X_h^(1) = {X_h_1}")
    state.add_optional_eom_term(n=1, field_name='h', X_expr=X_h_1)
    print()
    t0 = time.time()
    for n in range(3):
        tn = time.time()
        state.run_order(n)
        print(f"  Order {n} total wall: {time.time()-tn:.1f}s")
    print(f"\n  CASE A total wall: {time.time()-t0:.1f}s")


def case_B_multiple():
    """Multiple optional terms: X_phi^(0) at n=0 AND X_h^(1) at n=1."""
    print("\n" + "=" * 70)
    print("  CASE B: X_phi^(0) at n=0 + X_h^(1) at n=1 (Hilbert scalar, n_max=2)")
    print("=" * 70 + "\n")
    state, phi_info = _fresh_scalar_state(n_max=2)
    mE, nE = state.mu_E, state.nu_E

    X_phi_0 = build_X_phi_n0(mE, nE, phi_info)
    X_h_1 = build_X_h_n1(mE, nE)
    print(f"Optional EOM X_phi^(0) at n=0:")
    print(f"  X_phi^(0) = {X_phi_0}")
    print(f"Optional EOM X_h^(1) at n=1:")
    print(f"  X_h^(1) = {X_h_1}")
    state.add_optional_eom_term(n=0, field_name='phi', X_expr=X_phi_0)
    state.add_optional_eom_term(n=1, field_name='h', X_expr=X_h_1)
    print()
    t0 = time.time()
    for n in range(3):
        tn = time.time()
        state.run_order(n)
        print(f"  Order {n} total wall: {time.time()-tn:.1f}s")
    print(f"\n  CASE B total wall: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    t_overall = time.time()
    case_A_h_only()
    case_B_multiple()
    print("\n" + "=" * 70)
    print(f"  Both cases passed. Overall wall: {time.time()-t_overall:.1f}s")
    print("=" * 70)
