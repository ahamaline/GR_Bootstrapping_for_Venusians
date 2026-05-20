"""Debug: Examine each piece of T_H[L_h^(2)] separately."""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from bootstrap.energy_momentum import (
    uncontract_metrics, replace_metric_with_ginv, ginv,
    _replace_ginv_with_metric, _christoffel_contribution,
    NATURAL_POSITIONS
)
from bootstrap.euler_lagrange import euler_lagrange
from bootstrap.covariant import einstein_hilbert_lagrangian_order, kappa
from bootstrap.jet import jet_derivative, total_derivative
from sympy import Rational
from sympy.tensor.tensor import TensAdd

Lh2 = einstein_hilbert_lagrangian_order(2)
mu, nu = fresh_indices(2)

# --- Term A: metric(mu,nu) * L ---
term_A = canon(metric(mu, nu) * Lh2)
nA = len(term_A.args) if isinstance(term_A, TensAdd) else 1
print(f"Term A (sqrt|g| variation): {nA} terms")

# --- Term B: ginv variation ---
L_unc = uncontract_metrics(Lh2)
L_cov = replace_metric_with_ginv(L_unc)
alpha, beta = fresh_indices(2)
pi_ab = jet_derivative(L_cov, ginv, [alpha, beta])
if pi_ab != S.Zero:
    pi_ab_eta = _replace_ginv_with_metric(pi_ab)
    variation_B = -Rational(1, 2) * (
        pi_ab_eta * metric(-alpha, mu) * metric(-beta, nu)
        + pi_ab_eta * metric(-alpha, nu) * metric(-beta, mu)
    )
    term_B = 2 * canon(variation_B)
else:
    term_B = S.Zero
nB = len(term_B.args) if isinstance(term_B, TensAdd) else (0 if term_B == S.Zero else 1)
print(f"Term B (ginv variation): {nB} terms")

# --- Term C: Christoffel contribution ---
t0 = time.time()
term_C = _christoffel_contribution(Lh2, mu, nu)
nC = 0 if term_C == S.Zero else (len(term_C.args) if isinstance(term_C, TensAdd) else 1)
print(f"Term C (Christoffel): {nC} terms ({time.time()-t0:.1f}s)")

# --- Total T_H ---
T_H = canon(term_A + term_B + term_C)
nT = len(T_H.args) if isinstance(T_H, TensAdd) else 1
print(f"\nTotal T_H: {nT} terms")

# --- Print each contribution ---
print(f"\nA = {term_A}")
print(f"\nB = {term_B}")
print(f"\nC = {term_C}")

# --- E2 from EL(L_EH^(3)) ---
L3 = einstein_hilbert_lagrangian_order(3)
E2, E2_idx = euler_lagrange(L3, h)
mu_e, nu_e = E2_idx
E2 = E2.substitute_indices((mu_e, mu), (-mu_e, -mu), (nu_e, nu), (-nu_e, -nu))
E2 = canon(E2)
nE = len(E2.args) if isinstance(E2, TensAdd) else 1
print(f"\nE2 = EL(L_EH^(3)): {nE} terms")

# --- Delta ---
Delta = canon(E2 / kappa - T_H)
nD = len(Delta.args) if isinstance(Delta, TensAdd) else (0 if Delta == S.Zero else 1)
print(f"Delta = E2/kappa - T_H: {nD} terms")

# --- Divergence of T_H alone ---
print(f"\n--- Checking div(T_H) ---")
div_TH = total_derivative(T_H, -nu)
div_TH = canon(div_TH)
nDivT = 0 if div_TH == S.Zero else (len(div_TH.args) if isinstance(div_TH, TensAdd) else 1)
print(f"div(T_H): {nDivT} terms")

# --- Divergence of each piece ---
print(f"\n--- Checking div(A) ---")
div_A = total_derivative(term_A, -nu)
div_A = canon(div_A)
nDivA = 0 if div_A == S.Zero else (len(div_A.args) if isinstance(div_A, TensAdd) else 1)
print(f"div(A): {nDivA} terms")

print(f"\n--- Checking div(B) ---")
div_B = total_derivative(term_B, -nu)
div_B = canon(div_B)
nDivB = 0 if div_B == S.Zero else (len(div_B.args) if isinstance(div_B, TensAdd) else 1)
print(f"div(B): {nDivB} terms")

print(f"\n--- Checking div(C) ---")
div_C = total_derivative(term_C, -nu)
div_C = canon(div_C) if isinstance(div_C, TensExpr) else div_C
nDivC = 0 if div_C == S.Zero else (len(div_C.args) if isinstance(div_C, TensAdd) else 1)
print(f"div(C): {nDivC} terms")

print("\n=== Done ===")
