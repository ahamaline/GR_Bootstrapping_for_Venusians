# Same bootstrap closure check as the n=2/n=3 superpotential tests,
# bumped to n=4. Uses L_EH^{(4)} (after IBP) as L^{(4)}.
#
# We expect:
#   EL(L_EH^{(5)}) - kappa T_H[L_EH^{(4)}]  ==  kappa * Psi^{(4)}_{,rs}
# where Psi^{(4)} = PsiForm applied to T_H[L_EH^{(4)}] at n=4.

import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from bootstrap.helmholtz import compute_superpotential_n2, superpotential_divergence, verify_psi_symmetries
from bootstrap.energy_momentum import hilbert_energy_momentum
from bootstrap.euler_lagrange import euler_lagrange, remove_second_derivatives
from bootstrap.covariant import einstein_hilbert_lagrangian_order, kappa
from sympy import Rational
from sympy.tensor.tensor import TensAdd

mu_E, nu_E = fresh_indices(2)

# ---- Step 1: L^{(4)} = L_EH^{(4)} (after IBP) ----
print("=== Step 1: Compute L_EH^(4) ===")
t0 = time.time()
L4 = einstein_hilbert_lagrangian_order(4)
L4 = remove_second_derivatives(L4)
n_L4 = len(L4.args) if isinstance(L4, TensAdd) else 1
print(f"L_EH^(4) has {n_L4} terms ({time.time()-t0:.1f}s)")

# ---- Step 2: T_H[L^{(4)}] ----
print("\n=== Step 2: Compute T_H[L^(4)] ===")
t0 = time.time()
T_H, T_idx = hilbert_energy_momentum(L4)
n_T = len(T_H.args) if isinstance(T_H, TensAdd) else (0 if T_H == S.Zero else 1)
print(f"T_H has {n_T} terms ({time.time()-t0:.1f}s)")

mu_T, nu_T = T_idx
T_H = T_H.substitute_indices((mu_T, mu_E), (-mu_T, -mu_E), (nu_T, nu_E), (-nu_T, -nu_E))
T_H = canon(T_H)
print(f"T_H (reindexed) has {len(T_H.args) if isinstance(T_H, TensAdd) else 1} terms")

# ---- Step 3: Psi^{(4)} ----
print("\n=== Step 3: Compute Psi^(4) ===")
t0 = time.time()
Psi, Psi_indices = compute_superpotential_n2(T_H, 4, (mu_E, nu_E))
n_Psi = len(Psi.args) if isinstance(Psi, TensAdd) else (0 if Psi == S.Zero else 1)
print(f"Psi has {n_Psi} terms ({time.time()-t0:.1f}s)")

# ---- Step 4: verify Psi symmetries ----
print("\n=== Step 4: Verify Psi symmetries ===")
t0 = time.time()
sym = verify_psi_symmetries(Psi, Psi_indices)
print(f"  sym_mn={sym.get('sym_mn')}  sym_rs={sym.get('sym_rs')}  cyclic={sym.get('cyclic')}  ({time.time()-t0:.1f}s)")

# ---- Step 5: Delta ----
print("\n=== Step 5: Delta = Psi_{,rs} ===")
t0 = time.time()
Delta = superpotential_divergence(Psi, Psi_indices)
n_D = len(Delta.args) if isinstance(Delta, TensAdd) else (0 if Delta == S.Zero else 1)
print(f"Delta has {n_D} terms ({time.time()-t0:.1f}s)")

# ---- Step 6: L_EH^{(5)} and EL ----
print("\n=== Step 6: Compute E^mn(4) = EL(L_EH^(5)) ===")
t0 = time.time()
L5 = einstein_hilbert_lagrangian_order(5)
n_L5 = len(L5.args) if isinstance(L5, TensAdd) else 1
print(f"L_EH^(5) has {n_L5} terms ({time.time()-t0:.1f}s)")

t0 = time.time()
E4, E4_idx = euler_lagrange(L5, h)
n_E4 = len(E4.args) if isinstance(E4, TensAdd) else 1
print(f"E^mn(4) has {n_E4} terms ({time.time()-t0:.1f}s)")

mu_e, nu_e = E4_idx
E4 = E4.substitute_indices((mu_e, mu_E), (-mu_e, -mu_E), (nu_e, nu_E), (-nu_e, -nu_E))
E4 = canon(E4)
print(f"E4 (reindexed) has {len(E4.args) if isinstance(E4, TensAdd) else 1} terms")

# ---- Step 7: closure ----
print("\n=== Step 7: Bootstrap closure check ===")
t0 = time.time()
diff = canon(E4 - kappa * T_H)
n_diff = len(diff.args) if isinstance(diff, TensAdd) else (0 if diff == S.Zero else 1)
print(f"E4 - kappa*T_H has {n_diff} terms")

mu_psi, nu_psi, _, _ = Psi_indices
Delta_aligned = Delta.substitute_indices(
    (mu_psi, mu_E), (-mu_psi, -mu_E), (nu_psi, nu_E), (-nu_psi, -nu_E)
)
Delta_aligned = canon(Delta_aligned)

discrepancy = canon(diff - kappa * Delta_aligned)
n_disc = (len(discrepancy.args) if isinstance(discrepancy, TensAdd)
          else (0 if discrepancy == S.Zero else 1))
print(f"Discrepancy has {n_disc} terms ({time.time()-t0:.1f}s)")
if discrepancy == S.Zero:
    print("*** BOOTSTRAP CLOSURE VERIFIED at n=4 (pure gravity). ***")
else:
    print("Closure FAILED.")
    if isinstance(discrepancy, TensAdd):
        for i, t in enumerate(discrepancy.args[:8]):
            print(f"  term {i}: {t}")
        if n_disc > 8:
            print(f"  ... and {n_disc - 8} more")

assert discrepancy == S.Zero, "Bootstrap closure failed at n=4 (pure gravity)"
