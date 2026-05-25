"""Test: T_H for the Proca mass term using an upstairs vector V^mu.

L_mass = -1/2 m^2 V^mu V_mu = -1/2 m^2 eta_{mu nu} V^mu V^nu

The expected Hilbert stress-energy is:
    T_H^{mn} = -m^2 V^m V^n - (1/2) m^2 eta^{mn} V^a V_a

Derivation:
    T_H^{mn} = 2 delta(sqrt|g| L~)/delta g_{mn}
    delta(sqrt|g|)/delta g_{mn} = (1/2) sqrt|g| g^{mn}      → A: eta^{mn} L
    delta g_{ab}/delta g_{mn}    = (1/2)(d^m_a d^n_b + ...)  → D: -m^2 V^m V^n
    (no g^{ab} factors in L_mass; no derivatives → no Christoffel)

This test exercises the new g_down jet head and term_D path in
hilbert_energy_momentum: at g=eta the explicit eta_{mu nu} factor (from
lowering V^mu inside V^mu V_mu) becomes a g_down(...) jet variable that
must be varied directly.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd

from bootstrap.tensor_algebra import (
    register_upstairs_vector_field, fresh_indices, canon, metric,
)
from bootstrap.energy_momentum import hilbert_energy_momentum


V, dV, ddV = register_upstairs_vector_field('V')
m_sym = Symbol('m')

# L_mass = -1/2 m^2 V^sigma V_sigma  — implicit eta contraction lowers one V.
sigma, = fresh_indices(1)
L_mass = Rational(-1, 2) * m_sym**2 * V(sigma) * V(-sigma)
L_mass = canon(L_mass)
print(f"L_mass = {L_mass}")

t0 = time.time()
T, T_idx = hilbert_energy_momentum(L_mass)
dt = time.time() - t0
n_T = len(T.args) if isinstance(T, TensAdd) else (0 if T == S.Zero else 1)
print(f"T_H has {n_T} terms ({dt:.2f}s)")
print(f"T_H = {T}")
print(f"Free indices: {T_idx}")

mu_T, nu_T = T_idx
alpha, = fresh_indices(1)
T_expected = (
    -m_sym**2 * V(mu_T) * V(nu_T)
    + Rational(-1, 2) * m_sym**2 * metric(mu_T, nu_T) * V(alpha) * V(-alpha)
)
T_expected = canon(T_expected)
print(f"\nExpected T_H = {T_expected}")

diff = canon(T - T_expected)
n_diff = len(diff.args) if isinstance(diff, TensAdd) else (0 if diff == S.Zero else 1)
print(f"\nDifference: {n_diff} terms")
if diff == S.Zero:
    print("*** EXACT MATCH! Phase A (downstairs metric) verified. ***")
else:
    print(f"MISMATCH:\n  diff = {diff}")
    raise SystemExit(1)
