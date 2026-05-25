"""Test: Hilbert energy-momentum tensor of the EM field A_mu.

L_EM = -1/4 F_{μν} F^{μν}, where F_{μν} = ∂_μ A_ν − ∂_ν A_μ.

The textbook Maxwell stress-energy tensor is:
    T_EM^{mn} = F^{m ρ} F^n_{  ρ} - 1/4 η^{mn} F_{ρσ} F^{ρσ}

We test that our Hilbert procedure reproduces this exactly.

Note on covariantization: F_{μν} = ∇_μ A_ν − ∇_ν A_μ
                                = ∂_μ A_ν − ∂_ν A_μ
because Γ^λ_{μν} is symmetric in (μ,ν) and cancels in the antisymmetric
combination. So the covariantized L̃_EM differs from L_EM only by the
η → g substitution in the metric contractions and the √|g| factor —
no Christoffel corrections to dA are needed. The existing
`hilbert_energy_momentum` already does exactly that (the
_christoffel_contribution path only fires on dh-containing Lagrangians).
"""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import (
    register_vector_field, fresh_indices, canon, metric, S,
)
from bootstrap.energy_momentum import hilbert_energy_momentum, NATURAL_POSITIONS
from sympy import Rational
from sympy.tensor.tensor import TensAdd

# Register a vector field A. The TensorHeads:
#   A(mu)        -- A_mu
#   dA(mu, nu)   -- A_{mu, nu} = ∂_nu A_mu
#   ddA(mu, nu, rho)  -- A_{mu, nu rho} (symmetric in last two)
A, dA, ddA = register_vector_field('A')

# Natural positions: A_mu and its derivatives are written with all indices down.
NATURAL_POSITIONS[A] = ['down']
NATURAL_POSITIONS[dA] = ['down', 'down']
NATURAL_POSITIONS[ddA] = ['down', 'down', 'down']

# Build L_M = -1/4 F_{μν} F^{μν}.
# F_{mu, nu} = dA(-nu, -mu) - dA(-mu, -nu)  (using dA(field, deriv) convention,
# so dA(-nu, -mu) = ∂_mu A_nu = A_{ν,μ}).
# F^{μν} requires raising both indices via η.
mu, nu = fresh_indices(2)
F_dn = dA(-nu, -mu) - dA(-mu, -nu)          # F_{mu, nu}
F_up = dA(nu, mu) - dA(mu, nu)              # F^{mu, nu} via raising
L_M = Rational(-1, 4) * F_dn * F_up
L_M = canon(L_M)
print(f"L_M = {L_M}")
print()

# Compute T_H[L_M]
print("=== Computing T_H^{{mn}}[L_EM] ===")
t0 = time.time()
T, T_idx = hilbert_energy_momentum(L_M)
dt = time.time() - t0
n_T = len(T.args) if isinstance(T, TensAdd) else (0 if T == S.Zero else 1)
print(f"T_H has {n_T} terms ({dt:.1f}s)")
print(f"T_H = {T}")
print(f"Free indices: {T_idx}")

# Expected Maxwell stress-energy tensor.
# T^{mn} = eta_{ρσ} F^{m ρ} F^{n σ} − 1/4 eta^{mn} F_{ρσ} F^{ρσ}
# Convention: dA(α, β) = ∂_β A_α, so F^{αβ} = dA(β, α) − dA(α, β).
m, n = T_idx
rho, sig = fresh_indices(2)
F_m_rho = dA(rho, m) - dA(m, rho)           # F^{m, rho}
F_n_sig = dA(sig, n) - dA(n, sig)           # F^{n, sigma}
F_rs_dn = dA(-sig, -rho) - dA(-rho, -sig)   # F_{rho, sigma}
F_rs_up = dA(sig, rho) - dA(rho, sig)       # F^{rho, sigma}
T_expected = (metric(-rho, -sig) * F_m_rho * F_n_sig
              - Rational(1, 4) * metric(m, n) * F_rs_dn * F_rs_up)
T_expected = canon(T_expected)
print(f"\nExpected T_H = {T_expected}")

# Compare
diff = canon(T - T_expected)
n_diff = len(diff.args) if isinstance(diff, TensAdd) else (0 if diff == S.Zero else 1)
print(f"\nDifference: {n_diff} terms")
if diff == S.Zero:
    print("*** EXACT MATCH! Hilbert procedure gives the textbook Maxwell T_EM. ***")
else:
    print(f"MISMATCH:\n  diff = {diff}")
