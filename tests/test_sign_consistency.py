"""Consistency check: does the expanded L_EH^(2) give the same W as the hard-coded L_h^(2)?

This checks at the EL derivative level, which is independent of boundary terms (IBP).
If they disagree, the sign in the expansion is wrong relative to the hard-coded form.
"""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from bootstrap.euler_lagrange import euler_lagrange
from bootstrap.covariant import ricci_scalar_order, kappa, einstein_hilbert_lagrangian_order
from sympy import Rational
from sympy.tensor.tensor import TensAdd

# W from the hard-coded L_h^{(2)}
print("=== EL of hard-coded L_h^(2) ===")
Lh2_hc = einstein_hilbert_lagrangian_order(2)
W_hc, W_idx = euler_lagrange(Lh2_hc, h)
nW = len(W_hc.args) if isinstance(W_hc, TensAdd) else 1
print(f"W (hard-coded): {nW} terms")
print(f"W = {W_hc}")

# EL of expanded L_EH^(2) with POSITIVE sign (+1/2*kappa^-2)
print("\n=== EL of expanded L_EH^(2), POSITIVE sign ===")
R2 = ricci_scalar_order(2)
L2_pos = canon(Rational(1, 2) * kappa**(-2) * R2)
W_pos, W_pos_idx = euler_lagrange(L2_pos, h)
nW_pos = len(W_pos.args) if isinstance(W_pos, TensAdd) else 1
print(f"W (positive): {nW_pos} terms")

# Align indices
mu_hc, nu_hc = W_idx
mu_pos, nu_pos = W_pos_idx
W_pos_aligned = W_pos.substitute_indices(
    (mu_pos, mu_hc), (-mu_pos, -mu_hc),
    (nu_pos, nu_hc), (-nu_pos, -nu_hc)
)
W_pos_aligned = canon(W_pos_aligned)

diff_pos = canon(W_hc - W_pos_aligned)
print(f"W_hc - W_pos = {diff_pos}")

# EL of expanded L_EH^(2) with NEGATIVE sign (-1/2*kappa^-2) 
print("\n=== EL of expanded L_EH^(2), NEGATIVE sign ===")
L2_neg = canon(Rational(-1, 2) * kappa**(-2) * R2)
W_neg, W_neg_idx = euler_lagrange(L2_neg, h)
nW_neg = len(W_neg.args) if isinstance(W_neg, TensAdd) else 1
print(f"W (negative): {nW_neg} terms")

mu_neg, nu_neg = W_neg_idx
W_neg_aligned = W_neg.substitute_indices(
    (mu_neg, mu_hc), (-mu_neg, -mu_hc),
    (nu_neg, nu_hc), (-nu_neg, -nu_hc)
)
W_neg_aligned = canon(W_neg_aligned)

diff_neg = canon(W_hc - W_neg_aligned)
print(f"W_hc - W_neg = {diff_neg}")

print("\n=== Done ===")
