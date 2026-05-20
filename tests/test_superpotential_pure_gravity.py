#USER ADDED TEST
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from bootstrap.helmholtz import compute_superpotential_n2, superpotential_divergence, verify_psi_symmetries
from bootstrap.energy_momentum import hilbert_energy_momentum
from bootstrap.euler_lagrange import euler_lagrange, remove_second_derivatives
from bootstrap.covariant import einstein_hilbert_lagrangian_order, kappa
from bootstrap.jet import total_derivative
from sympy import Rational
from sympy.tensor.tensor import TensAdd

# Use two fixed free indices for E^{mn}
mu_E, nu_E = fresh_indices(2)

# ---- Step 1: Compute T_H[L_h^{(2)}] ----
print("=== Step 1: Compute T_H[L_h^(2)] ===")
t0 = time.time()
Lh2 = einstein_hilbert_lagrangian_order(2)
Lh2 = remove_second_derivatives(Lh2)     #USER COMMENT: I added this because otherwise we get higher order derivatives
T_H, T_idx = hilbert_energy_momentum(Lh2)
n_T = len(T_H.args) if isinstance(T_H, TensAdd) else (0 if T_H == S.Zero else 1)
print(f"T_H has {n_T} terms ({time.time()-t0:.1f}s)")

# Align T_H's free indices to (mu_E, nu_E)
mu_T, nu_T = T_idx
from sympy.tensor.tensor import TensorIndex
T_H = T_H.substitute_indices((mu_T, mu_E), (-mu_T, -mu_E), (nu_T, nu_E), (-nu_T, -nu_E))
T_H = canon(T_H)
print(f"T_H (reindexed) has {len(T_H.args) if isinstance(T_H, TensAdd) else 1} terms")

# ---- Step 2: Compute Psi ----
print("=== Step 1: Compute Psi ===")
t0 = time.time()
Psi, Psi_indices = compute_superpotential_n2(T_H, 2, (mu_E, nu_E))
print(f"Psi has {len(Psi.args) if isinstance(Psi, TensAdd) else 1} terms ({time.time()-t0:.1f}s)")
print(f"Psi = {Psi}")

#---- Step 3: Verify s-symmetries of Psi ----
print("=== Step 3: Verify s-symmetries of Psi ===")
t0 = time.time()
verify_psi_symmetries(Psi, Psi_indices)
print(f"Psi s-symmetries verified ({time.time()-t0:.1f}s)")

# ---- Step 4: Compute Delta ----
print("=== Step 4: Compute Delta ===")
t0 = time.time()
Delta = superpotential_divergence(Psi, Psi_indices)
print(f"Delta has {len(Delta.args) if isinstance(Delta, TensAdd) else 1} terms ({time.time()-t0:.1f}s)")
print(f"Delta = {Delta}")

## ---- Step 5: Compute EL(L_EH^{(3)}) ----
print("\n=== Step 5: Compute E^mn(2) = EL(L_EH^(3)) ===")
t0 = time.time()
L3 = einstein_hilbert_lagrangian_order(3)
n_L3 = len(L3.args) if isinstance(L3, TensAdd) else 1
print(f"L_EH^(3) has {n_L3} terms ({time.time()-t0:.1f}s)")

t0 = time.time()
E2, E2_idx = euler_lagrange(L3, h)
n_E2 = len(E2.args) if isinstance(E2, TensAdd) else 1
print(f"E^mn(2) has {n_E2} terms ({time.time()-t0:.1f}s)")

# Align E2's free indices to (mu_E, nu_E)
mu_e, nu_e = E2_idx
E2 = E2.substitute_indices((mu_e, mu_E), (-mu_e, -mu_E), (nu_e, nu_E), (-nu_e, -nu_E))
E2 = canon(E2)
print(f"E2 (reindexed) has {len(E2.args) if isinstance(E2, TensAdd) else 1} terms")

# ---- Step 6: Compute E2 - kappa*T_H ----
print("\n=== Step 6: Difference E2 - kappa*T_H ===")
diff = canon(E2 - kappa*T_H)
n_diff = len(diff.args) if isinstance(diff, TensAdd) else (0 if diff == S.Zero else 1)
print(f"Difference has {n_diff} terms")

# ---- Step 7: The Bootstrap closure check ----
# Bootstrap claim at n=2 (pure gravity, Hilbert):
#   E^{mn(2)} = kappa T_H[L_h^(2)] + Delta,   Delta = Psi_{,rs}
# and E^{mn(2)} = EL(L_EH^{(3)}). So:
#   E2 - kappa*T_H  ==  kappa * Delta
# (Psi was computed from M = T_H rather than kappa*T_H, so by linearity
# in M the formula's Delta carries no kappa and we put it in here.)
print("\n=== Step 7: Bootstrap closure check ===")
t0 = time.time()
# Align Delta's free indices (the first two of Psi_indices) to (mu_E, nu_E).
mu_psi, nu_psi, _, _ = Psi_indices
Delta_aligned = Delta.substitute_indices(
    (mu_psi, mu_E), (-mu_psi, -mu_E), (nu_psi, nu_E), (-nu_psi, -nu_E)
)
Delta_aligned = canon(Delta_aligned)
discrepancy = canon(diff - kappa * Delta_aligned)
n_disc = (len(discrepancy.args) if isinstance(discrepancy, TensAdd)
          else (0 if discrepancy == S.Zero else 1))
print(f"Discrepancy (E2 - kappa*T_H - kappa*Delta) has {n_disc} terms ({time.time()-t0:.1f}s)")
if discrepancy == S.Zero:
    print("*** BOOTSTRAP CLOSURE VERIFIED at n=2 (pure gravity). ***")
else:
    print("Discrepancy is nonzero; bootstrap closure FAILED.")
    if isinstance(discrepancy, TensAdd):
        for i, t in enumerate(discrepancy.args[:8]):
            print(f"  term {i}: {t}")
        if n_disc > 8:
            print(f"  ... and {n_disc - 8} more")
    else:
        print(f"  discrepancy = {discrepancy}")

assert discrepancy == S.Zero, "Bootstrap closure failed at n=2 (pure gravity)"

