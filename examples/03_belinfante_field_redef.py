"""Example 03: scalar bootstrap with symmetrized-Belinfante T̂, n_max = 2.

This example demonstrates two things at once:

  (a) The alternative energy-momentum prescription. Instead of computing
      T̂[L] by varying sqrt(|g|) L_cov with respect to g_{mu nu} (the
      Hilbert way), we use the symmetrized Belinfante tensor: the
      canonical Noether T_can plus a spin-tensor "improvement," then
      explicitly (mu nu)-symmetrized.

  (b) The field-redefinition cycle. Symmetrized Belinfante differs from
      Hilbert by EOM-proportional pieces — the bootstrap built from
      Belinfante is a valid description of the SAME theory, just in a
      different field convention. When we compare the bootstrap's
      L^(n+1) against the standard Einstein-Hilbert L_ref^(n+1), we get
      a nonzero diff. The code then:
        - runs an EOM-decomposition check on the diff,
        - if it decomposes cleanly into derivative-free X coefficients,
          recovers f^(n+1) = (1/(n+1)) h_{ab} X^{ab}^(n),
        - substitutes h -> h + f_h and/or phi -> phi + f_phi into
          L_ref^(k) (with chain rule propagation through derivatives),
        - re-verifies that EL(updated L_ref^(n+1)) now equals E^(n).
      Subtle point: even though the scalar has no spin at the BARE T_M
      level (so Hilbert and Belinfante agree on L^(0) = L_phi), step 1
      at every n >= 1 applies T̂ to L^(n) which by then contains
      h x phi structures — and h itself has spin. Belinfante's spin
      improvement on h then makes T̂[L^(n)] differ from the Hilbert
      version, and the L_ref check sees nonzero diffs that need
      h-redefs to absorb. Watch the n=1 verification step for the
      recovered f_h^(2), then n=2 for f_h^(3), etc.

Expected wall time: about 10-15 minutes (the field-redef machinery
runs at every order >= 1, including the substitution of the recovered
f_h into L_ref^(k) for k up to n_max+1).
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon,
)
from bootstrap.bootstrap_loop import BootstrapState


phi, dphi, ddphi = register_scalar_field('phi')
mu, = fresh_indices(1)
L_phi = canon(Rational(-1, 2) * dphi(mu) * dphi(-mu))
print(f"Matter Lagrangian:  L_phi = {L_phi}")
print()

N_MAX = 2

state = BootstrapState(L_matter=L_phi, em_procedure='belinfante',
                       n_max=N_MAX, verbose=True)

t_total = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  Order {n} wall: {time.time() - t:.1f}s")

print()
print("=" * 60)
print(f"  Scalar Belinfante through order {N_MAX} closed.")
print(f"  Total wall: {time.time() - t_total:.1f}s")
print("=" * 60)
