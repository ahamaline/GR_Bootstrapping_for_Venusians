"""Verify that T_SymBel - T_Hilbert for each matter type decomposes as Y_field x EOM_field.

For the diffs from the Belinfante smoke test, the decomposition should recover
a Y for each matter field such that Y x EOM = diff (residual = 0).
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
    symmetrized_belinfante, hilbert_energy_momentum,
)
from bootstrap.bootstrap_loop import _reindex_tensor
from bootstrap.eom_decompose import (
    decompose_against_matter_eom, compute_matter_eoms,
)


def n(x):
    return 0 if x == S.Zero else (len(x.args) if isinstance(x, TensAdd) else 1)


def try_decompose(L, field_name, label):
    print(f"\n=== {label} ===")
    T_sym, sym_idx = symmetrized_belinfante(L)
    T_H, h_idx = hilbert_energy_momentum(L)
    T_sym_re = _reindex_tensor(T_sym, sym_idx, h_idx)
    diff = canon(T_sym_re - T_H)
    print(f"  diff: {n(diff)} terms")

    if diff == S.Zero:
        print("  (diff is 0; trivially EOM-proportional)")
        return

    eoms = compute_matter_eoms(L)
    if field_name in eoms:
        E, E_idx = eoms[field_name]
        print(f"  EOM_{field_name}: {n(E)} terms (free: {E_idx})")
        print(f"    = {E}")

    Y, alpha, residual = decompose_against_matter_eom(diff, L, field_name)
    print(f"  Y_{field_name}: {n(Y)} terms, alpha={alpha}")
    if n(Y) < 8:
        print(f"    = {Y}")
    print(f"  residual: {n(residual)} terms")
    if n(residual) > 0 and n(residual) < 8:
        print(f"    = {residual}")
    if residual == S.Zero:
        print(f"  *** SUCCESS: diff = Y_{field_name} x EOM_{field_name} ***")
    else:
        print(f"  *** FAIL: residual nonzero ***")


# --- Free scalar ---
phi, dphi, ddphi = register_scalar_field('phi')
mu, = fresh_indices(1)
L_scalar = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
try_decompose(L_scalar, 'phi', "Free scalar")

# --- EM ---
A, dA, ddA = register_vector_field('A')
mu_F, nu_F = fresh_indices(2)
F_A_dn = dA(-nu_F, -mu_F) - dA(-mu_F, -nu_F)
F_A_up = dA(nu_F, mu_F) - dA(mu_F, nu_F)
L_EM = canon(Rational(-1, 4) * F_A_dn * F_A_up)
try_decompose(L_EM, 'A', "EM (downstairs A)")

# --- Proca mass only (upstairs V) ---
V, dV, ddV = register_upstairs_vector_field('V')
m_V = Symbol('m_V')
mu_v, = fresh_indices(1)
L_V_mass = canon(Rational(-1, 2) * m_V**2 * V(mu_v) * V(-mu_v))
try_decompose(L_V_mass, 'V', "Proca mass only")

# --- Full Proca ---
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
try_decompose(L_proca, 'V', "Full Proca")
