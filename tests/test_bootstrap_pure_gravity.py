"""
Pure gravity bootstrap self-consistency test.

For pure gravity with the Hilbert procedure, the bootstrap should reproduce
the Einstein-Hilbert Lagrangian order by order. The test is:

1. Compute L_EH^{(n+1)} from the covariant expansion
2. Take its EL derivative to get E^{mn(n)} 
3. "Close the loop": L_bootstrap^{(n+1)} = 1/(n+1) E^{mn(n)} h_{mn}
4. Integration by parts to remove second derivatives
5. Compare L_bootstrap^{(n+1)} vs L_EH^{(n+1)}

If they match, the bootstrap reproduces the EFE at this order.

We test at order n=1 (L^{(2)}), which is the simplest nontrivial case.
Higher orders test the superpotential formula.
"""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from bootstrap.jet import jet_derivative, total_derivative
from bootstrap.euler_lagrange import euler_lagrange, remove_second_derivatives
from bootstrap.covariant import einstein_hilbert_lagrangian_order
from sympy import Rational
from sympy.tensor.tensor import TensAdd

# =======================================================================
# Test at order n=1: L^{(2)} -> E^{mn(1)} = W -> L_bootstrap^{(2)}
# =======================================================================

print("=" * 60)
print("  PURE GRAVITY BOOTSTRAP: ORDER n=1")
print("=" * 60)

print("\n--- Step 1: Compute L_EH^(2) from covariant expansion ---")
t0 = time.time()
L2_ref = einstein_hilbert_lagrangian_order(2)
n_terms = len(L2_ref.args) if isinstance(L2_ref, TensAdd) else 1
print(f"  L_EH^(2) has {n_terms} terms ({time.time()-t0:.1f}s)")

print("\n--- Step 2: Take EL derivative -> W^mn ---")
t0 = time.time()
W, W_idx = euler_lagrange(L2_ref, h)
n_terms_W = len(W.args) if isinstance(W, TensAdd) else 1
print(f"  W^mn has {n_terms_W} terms ({time.time()-t0:.1f}s)")

print("\n--- Step 3: Close the loop -> L^(2)_bootstrap = 1/2 W h ---")
mu, nu = W_idx
L2_bootstrap_raw = Rational(1, 2) * W * h(-mu, -nu)
L2_bootstrap_raw = canon(L2_bootstrap_raw)
n_raw = len(L2_bootstrap_raw.args) if isinstance(L2_bootstrap_raw, TensAdd) else 1
print(f"  Raw (before IBP): {n_raw} terms")

print("\n--- Step 4: Integration by parts to remove ddh ---")
t0 = time.time()
L2_bootstrap = remove_second_derivatives(L2_bootstrap_raw)
L2_bootstrap = canon(L2_bootstrap)
n_ibp = len(L2_bootstrap.args) if isinstance(L2_bootstrap, TensAdd) else 1
print(f"  After IBP: {n_ibp} terms ({time.time()-t0:.1f}s)")

print("\n--- Step 5: Compare L_bootstrap^(2) vs L_EH^(2) ---")
diff = canon(L2_bootstrap - L2_ref)

if diff == S.Zero:
    print("  *** EXACT MATCH! The bootstrap reproduces L_h^(2). ***")
else:
    n_diff = len(diff.args) if isinstance(diff, TensAdd) else 1
    print(f"  DIFFERENCE has {n_diff} terms:")
    print(f"  {diff}")
    print("  (This difference should be a total divergence / boundary term)")

# =======================================================================
# Test at order n=2: L^{(3)} -> E^{mn(2)} -> L_bootstrap^{(3)}
# This is where the superpotential first matters!
# =======================================================================

print("\n" + "=" * 60)
print("  PURE GRAVITY BOOTSTRAP: ORDER n=2")
print("=" * 60)

print("\n--- Step 1: Compute L_EH^(3) from covariant expansion ---")
t0 = time.time()
L3_ref = einstein_hilbert_lagrangian_order(3)
n_terms = len(L3_ref.args) if isinstance(L3_ref, TensAdd) else 1
print(f"  L_EH^(3) has {n_terms} terms ({time.time()-t0:.1f}s)")

print("\n--- Step 2: Take EL derivative of L_EH^(3) ---")
t0 = time.time()
E2, E2_idx = euler_lagrange(L3_ref, h)
n_terms_E = len(E2.args) if isinstance(E2, TensAdd) else 1
print(f"  E^mn(2) has {n_terms_E} terms ({time.time()-t0:.1f}s)")

print("\n--- Step 3: Close the loop -> L^(3)_bootstrap = 1/3 E^(2) h ---")
mu2, nu2 = E2_idx
L3_bootstrap_raw = Rational(1, 3) * E2 * h(-mu2, -nu2)
L3_bootstrap_raw = canon(L3_bootstrap_raw)
n_raw = len(L3_bootstrap_raw.args) if isinstance(L3_bootstrap_raw, TensAdd) else 1
print(f"  Raw (before IBP): {n_raw} terms")

print("\n--- Step 4: Integration by parts ---")
t0 = time.time()
L3_bootstrap = remove_second_derivatives(L3_bootstrap_raw)
L3_bootstrap = canon(L3_bootstrap)
n_ibp = len(L3_bootstrap.args) if isinstance(L3_bootstrap, TensAdd) else 1
print(f"  After IBP: {n_ibp} terms ({time.time()-t0:.1f}s)")

print("\n--- Step 5: Compare L_bootstrap^(3) vs L_EH^(3) ---")
diff3 = canon(L3_bootstrap - L3_ref)

if diff3 == S.Zero:
    print("  *** EXACT MATCH! Bootstrap reproduces L_EH^(3). ***")
else:
    n_diff = len(diff3.args) if isinstance(diff3, TensAdd) else 1
    print(f"  DIFFERENCE has {n_diff} terms")
    print(f"  (Nonzero difference is expected if superpotential is needed)")

print("\n=== All tests completed ===")
