"""Test the Hilbert energy-momentum procedure on L_h^{(2)}."""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from bootstrap.covariant import einstein_hilbert_lagrangian_order
from bootstrap.energy_momentum import (
    uncontract_metrics, replace_metric_with_ginv, hilbert_energy_momentum, ginv
)
from bootstrap.jet import jet_derivative
from sympy import Rational
from sympy.tensor.tensor import TensAdd

# Build L_h^{(2)}
print("=== Building L_h^(2) ===")
Lh2 = einstein_hilbert_lagrangian_order(2)
n = len(Lh2.args) if isinstance(Lh2, TensAdd) else 1
print(f"L_h^(2) has {n} terms")
print(f"L_h^(2) = {Lh2}")

# Step 1: Uncontract metrics
print("\n=== Step 1: Uncontract metrics ===")
t0 = time.time()
L_unc = uncontract_metrics(Lh2)
print(f"Uncontracted: {L_unc}")
print(f"({time.time()-t0:.1f}s)")

# Step 2: Replace metric → ginv
print("\n=== Step 2: Replace metric -> ginv ===")
L_cov = replace_metric_with_ginv(L_unc)
print(f"Covariantized: {L_cov}")

# Test: jet derivative by ginv should be nonzero
print("\n=== Test: jet_derivative by ginv ===")
a, b = fresh_indices(2)
pi = jet_derivative(L_cov, ginv, [a, b])
print(f"pi^ab = dL/dg^ab has type: {type(pi)}")
print(f"pi^ab = {pi}")

# Full Hilbert energy-momentum
print("\n=== Computing T_H^{mn}[L_h^(2)] ===")
t0 = time.time()
try:
    T, T_idx = hilbert_energy_momentum(Lh2)
    n_T = len(T.args) if isinstance(T, TensAdd) else (0 if T == S.Zero else 1)
    print(f"T_H has {n_T} terms ({time.time()-t0:.1f}s)")
    print(f"T_H = {T}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Done ===")
