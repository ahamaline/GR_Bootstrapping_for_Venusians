"""Smoke test the canonical_noether_tensor and symmetrized_belinfante on
each matter case we already test for Hilbert.

For matter that's scalar (no spin), T_SymBel = T_Hilbert exactly.
For matter with spin (vector, h), T_SymBel = T_Hilbert + EOM-proportional
terms — the difference should be expressible as the field's EOM times the
field itself (no leftover dA-dA or dh-dh structure).
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
    canonical_noether_tensor, symmetrized_belinfante, hilbert_energy_momentum,
)
from bootstrap.covariant import einstein_hilbert_lagrangian_order
from bootstrap.bootstrap_loop import _reindex_tensor


def n(x):
    return 0 if x == S.Zero else (len(x.args) if isinstance(x, TensAdd) else 1)


def show(name, x):
    print(f"  {name} ({n(x)} terms): {x}")


def compare(L, label, expect_zero_diff=True):
    print(f"\n=== {label} ===")
    print(f"  L = {L}")
    T_sym, sym_idx = symmetrized_belinfante(L)
    T_H, h_idx = hilbert_energy_momentum(L)
    print(f"  T_SymBel: {n(T_sym)} terms")
    print(f"  T_Hilbert: {n(T_H)} terms")

    T_sym_re = _reindex_tensor(T_sym, sym_idx, h_idx)
    diff = canon(T_sym_re - T_H)
    print(f"  T_SymBel - T_H: {n(diff)} terms"
          + ("" if not expect_zero_diff else " (expect 0)"))
    if diff != S.Zero and n(diff) <= 6:
        show("    diff", diff)


# (1) Free scalar — T_Bel = T_H exactly
phi, dphi, ddphi = register_scalar_field('phi')
mu, = fresh_indices(1)
L_scalar = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
compare(L_scalar, "Free scalar", expect_zero_diff=True)

# (2) Massive scalar — same
m_phi = Symbol('m_phi')
mu2, = fresh_indices(1)
L_scalar_m = canon(Rational(-1, 2) * dphi(mu2) * dphi(-mu2)
                   + Rational(-1, 2) * m_phi**2 * phi() * phi())
compare(L_scalar_m, "Massive scalar", expect_zero_diff=True)

# (3) EM (downstairs A) — should give EOM-proportional diff
A, dA, ddA = register_vector_field('A')
mu3, nu3 = fresh_indices(2)
F_A_dn = dA(-nu3, -mu3) - dA(-mu3, -nu3)
F_A_up = dA(nu3, mu3) - dA(mu3, nu3)
L_EM = canon(Rational(-1, 4) * F_A_dn * F_A_up)
compare(L_EM, "EM (downstairs A)", expect_zero_diff=False)

# (4) Proca mass term (upstairs V) — should give EOM-proportional diff
V, dV, ddV = register_upstairs_vector_field('V')
m_V = Symbol('m_V')
mu4, = fresh_indices(1)
L_V_mass = canon(Rational(-1, 2) * m_V**2 * V(mu4) * V(-mu4))
compare(L_V_mass, "Proca mass (upstairs V)", expect_zero_diff=False)

# (5) Full Proca (F^2 + mass, upstairs V)
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
compare(L_proca, "Full Proca (F^2 + mass, upstairs V)", expect_zero_diff=False)

# (6) Pure gravity L_h^{(2)} — should give EOM-proportional diff (EOM_h = W)
L_h2 = canon(einstein_hilbert_lagrangian_order(2))
compare(L_h2, "Pure gravity L_h^{(2)}", expect_zero_diff=False)
