"""Test: Hilbert energy-momentum tensor of a free scalar field.

The scalar field Lagrangian is L_M = -1/2 dphi^mu dphi_mu.
The known Hilbert energy-momentum tensor is:
    T_H^{mn} = dphi^m dphi^n + eta^{mn} L_M
             = dphi^m dphi^n - 1/2 eta^{mn} dphi^rho dphi_rho

We test that our Hilbert procedure reproduces this exactly.
"""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from bootstrap.energy_momentum import (
    uncontract_metrics, replace_metric_with_ginv,
    hilbert_energy_momentum, ginv, NATURAL_POSITIONS
)
from bootstrap.jet import jet_derivative
from sympy import Rational
from sympy.tensor.tensor import TensAdd

# Register the scalar field
phi, dphi, ddphi = register_scalar_field('phi')

# Natural index positions for the scalar field derivatives
# dphi_mu has one naturally-down index
NATURAL_POSITIONS[dphi] = ['down']
NATURAL_POSITIONS[ddphi] = ['down', 'down']

# Build L_M = -1/2 dphi^mu dphi_mu
mu, = fresh_indices(1)
L_M = Rational(-1, 2) * dphi(mu) * dphi(-mu)
L_M = canon(L_M)
print(f"L_M = {L_M}")

# Compute T_H[L_M]
print("\n=== Computing T_H^{{mn}}[L_M] ===")
t0 = time.time()
T, T_idx = hilbert_energy_momentum(L_M)
dt = time.time() - t0
n_T = len(T.args) if isinstance(T, TensAdd) else (0 if T == S.Zero else 1)
print(f"T_H has {n_T} terms ({dt:.1f}s)")
print(f"T_H = {T}")
print(f"Free indices: {T_idx}")

# Build the expected answer:
# T_H^{mn} = dphi^m dphi^n - 1/2 eta^{mn} dphi^rho dphi_rho
m, n = T_idx
rho, = fresh_indices(1)
T_expected = dphi(m) * dphi(n) + Rational(-1, 2) * metric(m, n) * dphi(rho) * dphi(-rho)
T_expected = canon(T_expected)
print(f"\nExpected T_H = {T_expected}")

# Compare
diff = canon(T - T_expected)
print(f"\nDifference = {diff}")
if diff == S.Zero:
    print("*** EXACT MATCH! Hilbert procedure gives correct scalar field T^mn ***")
else:
    print(f"MISMATCH: difference has {len(diff.args) if isinstance(diff, TensAdd) else 1} terms")

print("\n=== Done ===")
