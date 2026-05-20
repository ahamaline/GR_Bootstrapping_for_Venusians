"""Test: Compare T_H[L_h^{(2)}] against the order-2 part of sqrt(|g|)G^{mn}.

The paper says (Section 4, step 1):
    E^{mn(n)} = kappa * T_H^{mn}[L^{(n)}] + kappa * Delta^{mn}
    
where Delta is a superpotential term. Also E^{mn(n)} = EL(L^{(n+1)}).

So at n=2:
    EL(L_EH^{(3)}) = kappa * T_H[L_h^{(2)}] + kappa * Delta^{mn}
    
=> T_H[L_h^{(2)}] = (1/kappa) * EL(L_EH^{(3)}) - Delta^{mn}

=> Delta^{mn} = (1/kappa) * EL(L_EH^{(3)}) - T_H[L_h^{(2)}]

We verify that Delta^{mn}_{,n} = 0 (identically conserved).
"""

import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
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

# ---- Step 2: Compute EL(L_EH^{(3)}) ----
print("\n=== Step 2: Compute E^mn(2) = EL(L_EH^(3)) ===")
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

# ---- Step 3: Compute Delta = (1/kappa)*E2 - T_H ----
print("\n=== Step 3: Difference Delta = (1/kappa)*E2 - T_H ===")
diff = canon(E2 / kappa - T_H)
n_diff = len(diff.args) if isinstance(diff, TensAdd) else (0 if diff == S.Zero else 1)
print(f"Delta has {n_diff} terms")

if diff == S.Zero:
    print("  T_H equals (1/kappa)*E2 exactly (no superpotential needed!)")
else:
    # Check if Delta is identically conserved: Delta^{mn}_{,n} = 0
    print(f"\n=== Check: is Delta identically conserved? ===")
    t0 = time.time()
    div_diff = total_derivative(diff, -nu_E)
    div_diff = canon(div_diff) if isinstance(div_diff, TensExpr) else div_diff
    dt = time.time() - t0
    print(f"Divergence computation: {dt:.1f}s")
    if div_diff == S.Zero:
        print("*** DIVERGENCE IS ZERO! ***")
        print("  Delta is identically conserved -> it's a superpotential term.")
        print("  This confirms: T_H = (1/kappa)*EL(L_EH^(3)) - Psi_{,rs}")
    else:
        n_div = len(div_diff.args) if isinstance(div_diff, TensAdd) else 1
        print(f"  Divergence has {n_div} terms (should be zero!)")
        # Show a few terms
        if isinstance(div_diff, TensAdd):
            for i, t in enumerate(div_diff.args[:5]):
                print(f"  term {i}: {t}")
            if n_div > 5:
                print(f"  ... and {n_div - 5} more")
        else:
            print(f"  div = {div_diff}")

print("\n=== Done ===")
