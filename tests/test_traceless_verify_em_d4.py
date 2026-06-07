"""Regression test: verification-step traceless recovery on EM at d=4 (Hilbert).

EM at d=4 is OFF-shell traceless (c_i = {}). We inject a traceless-SHAPE
optional h-EOM term at order 1,

    X_h^{ab}_{cd} = eps_inj * kappa^2 * h^{ab} * eta_{cd}   (order 1 in h),

which is INVISIBLE at order 1 (eta_{cd} E^(0)cd = tr E^(0) = 0 at d=4) but
resurfaces at order 2 as ddh content in E_diff = E^(2) - EL(L_ref^(3)). The
verification-step recovery must:
  - detect the ddh, extract X^{ab} via the doubly-traced-ddh signature,
  - recover EXACTLY the injected X^{ab} = eps_inj kappa^2 h^{ab},
  - apply the missed redef and drive E_diff to 0 (order 2 closes).

Without the recovery this raised RuntimeError (non-integrable residual).

NOTE: runs the bootstrap to order 2 at d=4 -- a few minutes wall.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# set_dimension(4) MUST be the first bootstrap import (rebuilds the heads).
from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_vector_field, fresh_indices, canon, metric, h,
)
from bootstrap.bootstrap_loop import BootstrapState

kappa = Symbol('kappa')
eps_inj = Symbol('eps_inj')


def _terms_with(expr, sym):
    if expr == S.Zero:
        return []
    terms = expr.args if isinstance(expr, TensAdd) else [expr]
    return [t for t in terms if t.has(sym)]


def main():
    A, dA, ddA = register_vector_field('A')
    mu, nu = fresh_indices(2)
    F_dn = dA(-nu, -mu) - dA(-mu, -nu)
    F_up = dA(nu, mu) - dA(mu, nu)
    L_EM = canon(Rational(-1, 4) * F_dn * F_up)

    state = BootstrapState(L_matter=L_EM, em_procedure='hilbert',
                           n_max=2, verbose=False)

    # Injected traceless-shape term at order 1.
    cc, dd = fresh_indices(2)
    X_inject = canon(eps_inj * kappa**2 * h(state.mu_E, state.nu_E)
                     * metric(-cc, -dd))
    state.add_optional_eom_term(1, 'h', X_inject)

    state.run_order(0)
    assert state.traceless_T_M is True, "EM d=4 should detect traceless_T_M"
    assert state.traceless_c_i == {}, (
        f"EM d=4 is off-shell traceless; expected c_i = {{}}, "
        f"got {state.traceless_c_i}")

    state.run_order(1)
    assert len(_terms_with(state.E[1], eps_inj)) == 0, (
        "injected traceless term must be INVISIBLE at order 1 "
        "(no eps_inj terms in E^(1))")

    # Order 2: the recovery must fire and close. (Raised before the fix.)
    state.run_order(2)

    # Known-answer check: recovered X == injected eps_inj kappa^2 h^{ab}.
    assert 2 in state.recovered_traceless_X, (
        "verification-step recovery did not fire at order 2 "
        "(no recovered X recorded)")
    X_rec = state.recovered_traceless_X[2]
    X_expected = canon(eps_inj * kappa**2 * h(state.mu_E, state.nu_E))
    assert canon(X_rec - X_expected) == S.Zero, (
        f"recovered X mismatch:\n  got      {X_rec}\n  expected {X_expected}")

    print(f"recovered X^(ab) = {X_rec}  (== injected) OK")
    print("\n*** EM-d=4 verification-step traceless recovery PASSES ***")


if __name__ == '__main__':
    main()
