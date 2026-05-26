"""Verify that T_SymBel - T_Hilbert for EM decomposes as Y_A × EOM_A.

Tests the simplest case of the EOM-decomposition machinery: the EM diff is
a pure Y_A × E_A^(0) combination (no Y_h × T_M part since the diff has no
products of first derivatives).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_vector_field, fresh_indices, canon, metric,
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


A, dA, ddA = register_vector_field('A')

# Build L_EM
mu_F, nu_F = fresh_indices(2)
F_A_dn = dA(-nu_F, -mu_F) - dA(-mu_F, -nu_F)
F_A_up = dA(nu_F, mu_F) - dA(mu_F, nu_F)
L_EM = canon(Rational(-1, 4) * F_A_dn * F_A_up)

# Compute T_SymBel and T_Hilbert
T_sym, sym_idx = symmetrized_belinfante(L_EM)
T_H, h_idx = hilbert_energy_momentum(L_EM)
T_sym_re = _reindex_tensor(T_sym, sym_idx, h_idx)
diff = canon(T_sym_re - T_H)

print(f"L_EM: {n(L_EM)} terms")
print(f"T_SymBel: {n(T_sym)} terms")
print(f"T_Hilbert: {n(T_H)} terms")
print(f"diff = T_SymBel - T_Hilbert: {n(diff)} terms")
print(f"  diff = {diff}\n")

# Print the matter EOM for A
matter_eoms = compute_matter_eoms(L_EM)
EOM_A, EOM_A_idx = matter_eoms['A']
print(f"EOM_A: {n(EOM_A)} terms (free index {EOM_A_idx})")
print(f"  EOM_A = {EOM_A}\n")

# Decompose diff against EOM_A
Y_A, alpha, residual = decompose_against_matter_eom(diff, L_EM, 'A')
print(f"Y_A: {n(Y_A)} terms, free alpha index: {alpha}")
print(f"  Y_A = {Y_A}\n")
print(f"residual = diff - Y_A × EOM_A: {n(residual)} terms")
if residual != S.Zero and n(residual) < 20:
    print(f"  residual = {residual}")

print()
if residual == S.Zero:
    print("*** SUCCESS: diff = Y_A × EOM_A exactly ***")
else:
    print(f"*** FAIL: residual has {n(residual)} terms ***")
