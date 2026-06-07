"""Sanity test `decompose_against_eoms` on the Belinfante - Hilbert diffs
for representative matter cases. The diffs are EOM-proportional by
construction (see test_belinfante_smoke.py), and the orchestrator must
decompose them cleanly.

Expectations:
  - Free scalar: T_SymBel = T_Hilbert -> diff = 0 -> trivial.
  - EM: diff = X_A . EOM_A (Maxwell . A). Y_h = 0 (no kinetic T_M match
    in the diff), X_A nonzero, residual 0.
  - Proca-mass-only: known precondition violation - EOM_V = m^2 V has no
    2nd-derivative monomial for the trace-signature matcher to bite on,
    so the orchestrator returns a nonzero residual. Intentionally NOT
    fixed (see DEVELOPMENT_STATUS open work).
  - Full Proca: kinetic + mass; diff = X_V . EOM_V, residual 0 (the
    kinetic term provides the 2nd-derivative monomial).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S, Symbol
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, register_vector_field, register_upstairs_vector_field,
    fresh_indices, canon, metric,
)
from bootstrap.energy_momentum import (
    hilbert_energy_momentum, symmetrized_belinfante,
)
from bootstrap.bootstrap_loop import _reindex_tensor
from bootstrap.eom_decompose import decompose_against_eoms


def n(x):
    return 0 if x == S.Zero else (len(x.args) if isinstance(x, TensAdd) else 1)


def run_case(label, L):
    print(f"\n=== {label} ===")
    T_H, h_idx = hilbert_energy_momentum(L)
    T_B, b_idx = symmetrized_belinfante(L)
    T_B_re = _reindex_tensor(T_B, b_idx, h_idx)
    diff = canon(T_B_re - T_H)
    print(f"  diff = T_SymBel - T_H: {n(diff)} terms")
    if diff == S.Zero:
        print("  (trivial: nothing to decompose)")
        return
    result = decompose_against_eoms(diff, L, em_procedure='hilbert', verbose=True)
    print(f"  result:")
    print(f"    Y_h: {n(result['Y_h'])} terms, alphas={result['alphas_h']}")
    for name, (C, alphas) in result['X_phi'].items():
        print(f"    X_{name}: {n(C)} terms, alphas={alphas}")
    print(f"    residual: {n(result['residual'])} terms")
    if result['residual'] == S.Zero:
        print(f"  *** SUCCESS ({label}) ***")
    else:
        print(f"  *** FAIL: nonzero residual ({label}) ***")
        if n(result['residual']) < 12:
            print(f"      = {result['residual']}")


# (1) Free scalar — diff = 0 (no spin)
phi, dphi, ddphi = register_scalar_field('phi')
mu, = fresh_indices(1)
L_scalar = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
run_case("Free scalar", L_scalar)

# (2) EM (downstairs A)
A, dA, ddA = register_vector_field('A')
mu3, nu3 = fresh_indices(2)
F_A_dn = dA(-nu3, -mu3) - dA(-mu3, -nu3)
F_A_up = dA(nu3, mu3) - dA(mu3, nu3)
L_EM = canon(Rational(-1, 4) * F_A_dn * F_A_up)
run_case("EM (downstairs A)", L_EM)

# (3) Proca mass term (upstairs V)
V, dV, ddV = register_upstairs_vector_field('V')
m_V = Symbol('m_V')
mu4, = fresh_indices(1)
L_V_mass = canon(Rational(-1, 2) * m_V**2 * V(mu4) * V(-mu4))
run_case("Proca mass (upstairs V)", L_V_mass)

# (4) Full Proca (F^2 + mass, upstairs V)
mu5, nu5 = fresh_indices(2)
rho1, rho2 = fresh_indices(2)
alpha1, alpha2 = fresh_indices(2)
F_V_dn = (metric(-nu5, -rho1) * dV(rho1, -mu5)
          - metric(-mu5, -rho2) * dV(rho2, -nu5))
F_V_up = (metric(mu5, alpha1) * dV(nu5, -alpha1)
          - metric(nu5, alpha2) * dV(mu5, -alpha2))
mu5b, = fresh_indices(1)
L_proca = canon(Rational(-1, 4) * F_V_dn * F_V_up
                + Rational(-1, 2) * m_V**2 * V(mu5b) * V(-mu5b))
run_case("Full Proca (F^2 + mass, upstairs V)", L_proca)
