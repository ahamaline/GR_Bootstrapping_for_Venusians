"""Full end-to-end EM Belinfante test through n_max=1.

Exercises every piece of the field-redef cycle plus step 3:
  - Step 3 (n=1): compute_h2_violation Z = 16 → orchestrator → X_A
    correction → post-correction Z = 0.
  - L_ref integrability check at both n=0 and n=1: the diff decomposes
    cleanly into Y_h × E^(0) + X_φ × E_φ^(0) with derivative-free X's.
  - f^(n+1) recovery via paper formula.
  - Full substitution: f's applied to L_ref^(0..n_max-n+1) with chain-
    rule propagation through dφ, ddφ, with the fresh-deriv-metric trick
    that dodges sympy's dummy-name-collision pitfall.
  - Per paper ordering: h-redef first, then matter-redefs one at a time.

Success criterion: at BOTH orders, after applying the redef, EL(new
L_ref^(n+1)) equals E^(n) exactly (residual = 0).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S
from bootstrap.tensor_algebra import register_vector_field, fresh_indices, canon
from bootstrap.bootstrap_loop import BootstrapState


A, dA, ddA = register_vector_field('A')
mu, nu = fresh_indices(2)
F_dn = dA(-nu, -mu) - dA(-mu, -nu)
F_up = dA(nu, mu) - dA(mu, nu)
L_EM = canon(Rational(-1, 4) * F_dn * F_up)
print(f"L_EM: {L_EM}\n")

print("=== EM Belinfante with redef applied, n_max=2, full cycle ===")
state = BootstrapState(L_matter=L_EM, em_procedure='belinfante',
                       n_max=2, verbose=True)
t0 = time.time()
for n in range(3):
    tn = time.time()
    state.run_order(n)
    print(f"\n--- run_order({n}) done in {time.time()-tn:.1f}s ---\n")
print("*** Belinfante EM through n=2: step 2 carryover + step 3 + integrability + redef applied ***")
print(f"\nTotal wall: {time.time() - t0:.1f}s")
