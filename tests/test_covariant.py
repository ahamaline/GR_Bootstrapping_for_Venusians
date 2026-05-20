"""Test: compute L_EH at order 3 (the cubic graviton vertex)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from bootstrap.covariant import (
    einstein_hilbert_lagrangian_order, ricci_scalar_order,
    sqrt_det_g_order, christoffel_order, inverse_metric_order
)
from sympy import Symbol
from sympy.tensor.tensor import TensAdd

kappa = Symbol('kappa')

print("=== Testing covariant expansion components ===")

# Test inverse metric at order 2
m1, m2 = fresh_indices(2)
ginv2 = inverse_metric_order(2, m1, m2)
print(f"g^{{mn}} order 2 = {ginv2}")

# Test sqrt(|g|) at order 1
sg1 = sqrt_det_g_order(1)
print(f"sqrt(|g|) order 1 = {sg1}")

sg2 = sqrt_det_g_order(2)
print(f"sqrt(|g|) order 2 = {sg2}")

# Test Christoffel at order 1
l1, m3, m4 = fresh_indices(3)
chris1 = christoffel_order(1, l1, -m3, -m4)
print(f"Christoffel order 1 = {chris1}")

# Test Ricci scalar at order 2
print("\nComputing R at order 2...")
R2 = ricci_scalar_order(2)
n_terms_R2 = len(R2.args) if isinstance(R2, TensAdd) else (0 if R2 == S.Zero else 1)
print(f"R^(2) has {n_terms_R2} terms")
print(f"R^(2) = {R2}")

# Test L_EH at order 2
print("\nComputing L_EH at order 2...")
L2 = einstein_hilbert_lagrangian_order(2)
n_terms_L2 = len(L2.args) if isinstance(L2, TensAdd) else (0 if L2 == S.Zero else 1)
print(f"L_EH^(2) has {n_terms_L2} terms")

# Test L_EH at order 3 (the cubic vertex!)
print("\nComputing L_EH at order 3 (this may take a moment)...")
try:
    L3 = einstein_hilbert_lagrangian_order(3)
    n_terms_L3 = len(L3.args) if isinstance(L3, TensAdd) else (0 if L3 == S.Zero else 1)
    print(f"L_EH^(3) has {n_terms_L3} terms")
    print(f"L_EH^(3) = {L3}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Done ===")
