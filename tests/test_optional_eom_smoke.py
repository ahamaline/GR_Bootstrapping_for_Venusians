"""Smoke test for Step 4 (optional EOM / field redefinition).

Four cases exercise the validation + apply paths in `_step4_optional_eom`:

  0. Constructive accept: X = partial f / partial h for an arbitrary f
     (order 2 in h). X is then automatically Helmholtz-integrable because
     the antisymmetric-in-(mu nu <-> kappa lambda) part of partial X /
     partial h is the commutator of partial / partial h with itself = 0.
     Step 4 accepts; bootstrap closes through n=1.
  1. X has the wrong order in h (constant X applied at n=1). Rejected by
     the order-in-h check that keeps the (1/(n+1)) E h closure consistent.
  2. X contains a derivative factor (dh). Rejected by `_is_derivative_free`.
  3. X is derivative-free with the right order but is NOT Helmholtz-
     integrable. Rejected by `_check_X_integrable`.

Each case constructs a FRESH BootstrapState (and re-registers fields) to
avoid state bleeding between cases.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric, h, dh,
    _matter_fields,
)
from bootstrap.jet import jet_derivative
from bootstrap.bootstrap_loop import BootstrapState


def _fresh_scalar_state():
    """Reset registry and build a fresh BootstrapState with a free scalar.
    Avoids state bleed when multiple cases run back-to-back."""
    _matter_fields.clear()
    register_scalar_field('phi')
    info = _matter_fields['phi']
    dphi = info['dfield']
    mu, = fresh_indices(1)
    L_M = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
    return BootstrapState(L_matter=L_M, em_procedure='hilbert',
                          n_max=1, verbose=False)


def _expect_value_error(label, fn, must_contain):
    """Run fn(); expect a ValueError whose message contains all the strings
    in must_contain (a list). Print pass/fail."""
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        if all(s in msg for s in must_contain):
            print(f"  *** {label} OK -- raised as expected: "
                  f"{msg[:80]}{'...' if len(msg) > 80 else ''}")
            return
        print(f"  *** {label} FAIL: raised ValueError but message missing "
              f"{[s for s in must_contain if s not in msg]}\n"
              f"      full message: {msg}")
        return
    print(f"  *** {label} FAIL: did NOT raise (expected ValueError)")


def case_0_constructive_accept():
    """Pick arbitrary f^(2)_{kappa, lambda} (order 2 in h), compute
    X^{mu, nu}_{kappa, lambda} = partial f / partial h_{mu, nu} via
    `jet_derivative`. X is order 1 in h, derivative-free, and Helmholtz-
    integrable by construction (Schwarz: partials commute). Step 4
    accepts; bootstrap should close at n=1."""
    print("\n=== Case 0: constructive (X = d f / d h for an arbitrary f) ===")
    state = _fresh_scalar_state()
    state.verbose = True
    mE, nE = state.mu_E, state.nu_E
    kappa, lam = fresh_indices(2)
    alpha, = fresh_indices(1)
    # Pick f^(2)_{kappa, lambda} = h(-kappa, alpha) h(-lambda, -alpha) -- a
    # quadratic-in-h matter-free tensor with two free (kappa, lambda) indices.
    f = canon(h(-kappa, alpha) * h(-lam, -alpha))
    # X^{mu nu}_{kappa lambda} = partial f / partial h_{mu nu}.
    X = canon(jet_derivative(f, h, [mE, nE]))
    print(f"  f^(2)_(kappa lambda)  = {f}")
    print(f"  X^(1)_(mu nu, kappa lambda)  = partial f / partial h  = {X}")
    state.add_optional_eom_term(n=1, field_name='h', X_expr=X)
    state.run_order(0)
    state.run_order(1)
    print("  *** case 0 OK (constructive X_h^(1) accepted, bootstrap closed) ***")


def case_1_wrong_order():
    """X is constant (order 0) but applied at n=1. The order-in-h check
    in step 4 must reject it."""
    print("\n=== Case 1: X has wrong order in h ===")
    state = _fresh_scalar_state()
    mE, nE = state.mu_E, state.nu_E
    kappa, lam = fresh_indices(2)
    # Constant rank-4 = (1/2)(eta^mu_kappa eta^nu_lambda + sym (mu<->nu)).
    X_const = canon(Rational(1, 2) * (
        metric(mE, -kappa) * metric(nE, -lam)
        + metric(mE, -lam) * metric(nE, -kappa)
    ))
    state.add_optional_eom_term(n=1, field_name='h', X_expr=X_const)
    state.run_order(0)
    _expect_value_error(
        "case 1",
        lambda: state.run_order(1),
        must_contain=["order", "in h"],
    )


def case_2_has_derivative():
    """X contains a dh factor. _is_derivative_free must reject it.
    The structural index-shape check happens later, so a derivative-laden
    X gets caught for its derivative content first."""
    print("\n=== Case 2: X contains a derivative ===")
    state = _fresh_scalar_state()
    mE, nE = state.mu_E, state.nu_E
    sigma, = fresh_indices(1)
    # X_bad has a dh in it: dh(mE, nE, sigma) -- 3 free indices, order 1
    # in h. The validation rejects on `_is_derivative_free` before the
    # structural (4-free-indices) check, which is fine for our purposes.
    X_bad = dh(mE, nE, sigma)
    state.add_optional_eom_term(n=1, field_name='h', X_expr=X_bad)
    state.run_order(0)
    _expect_value_error(
        "case 2",
        lambda: state.run_order(1),
        must_contain=["derivative"],
    )


def case_3_not_integrable():
    """X is derivative-free and order 1, but NOT (mu nu <-> alpha beta)-
    symmetric under d/dh. _check_X_integrable must reject it."""
    print("\n=== Case 3: X is order 1 + derivative-free but not Helmholtz-integrable ===")
    state = _fresh_scalar_state()
    mE, nE = state.mu_E, state.nu_E
    kappa, lam = fresh_indices(2)
    # X = h^{mu alpha} delta^alpha_kappa delta^nu_lambda -- order 1 in h,
    # derivative-free, but NOT (mu nu <-> kappa lambda)-symmetric (nu and
    # lambda are tied to different positions than mu and kappa).
    alpha, = fresh_indices(1)
    X_nonint = canon(h(mE, alpha) * metric(-alpha, -kappa) * metric(nE, -lam))
    state.add_optional_eom_term(n=1, field_name='h', X_expr=X_nonint)
    state.run_order(0)
    _expect_value_error(
        "case 3",
        lambda: state.run_order(1),
        must_contain=["Helmholtz", "integrability"],
    )


if __name__ == '__main__':
    case_0_constructive_accept()
    case_1_wrong_order()
    case_2_has_derivative()
    case_3_not_integrable()
    print("\nDone.")
