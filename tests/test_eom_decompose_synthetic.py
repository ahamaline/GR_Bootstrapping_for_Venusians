"""Synthetic test for the Y_h kinetic extraction path of decompose_against_eoms.

Construct Z := Y_test^{cdef}_{ab} x T_M^{ab} for the free-scalar Lagrangian
(T_M = d^muphi d^nuphi - ½ eta^{munu} (dphi)²) with an arbitrary derivative-free Y_test.
Then run the orchestrator and verify it returns the same Y_h (modulo
relabeling of a, b), residual = 0.

This exercises:
  - the kinetic-signature picker (`_pick_kinetic_signature`),
  - the two-factor Y_h extractor (`_extract_coeff_two_factor`),
  - the trace-term extractor (`_extract_trace_coeff_two_factor`) — needed
    because T_M's -½ eta^{ab} (dphi)² piece contributes contracted dphi.dphi terms
    that the bare kinetic extraction won't account for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric, h,
)
from bootstrap.energy_momentum import hilbert_energy_momentum
from bootstrap.bootstrap_loop import _reindex_tensor
from bootstrap.eom_decompose import decompose_against_eoms


def n(x):
    return 0 if x == S.Zero else (len(x.args) if isinstance(x, TensAdd) else 1)


# Register the free scalar
phi, dphi, ddphi = register_scalar_field('phi')
mu0, = fresh_indices(1)
L_scalar = canon(Rational(-1, 2) * dphi(mu0) * dphi(-mu0))

# Compute T_M for the free scalar
T_M, (mu_T, nu_T) = hilbert_energy_momentum(L_scalar)
print(f"T_M (free scalar): {n(T_M)} terms — indices ({mu_T}, {nu_T})")
print(f"  = {T_M}")

# Build a synthetic Y_test^{cdef}_{ab} with NO derivatives.
# Pick a simple shape with explicit free indices c, d, e, f and EOM indices a, b.
c, d, e, f = fresh_indices(4)
alpha, beta = fresh_indices(2)

# Y_test^{cdef}_{ab} = h(c, d) x eta(e, -a) x eta(f, -b)  [simple "delta-style"]
# (Note: eta with mixed signs is a Kronecker delta after canon.)
Y_test = canon(h(c, d) * metric(e, -alpha) * metric(f, -beta))
print(f"\nY_test (synthetic): {n(Y_test)} terms — free indices (c, d, e, f, a, b)")

# Build Z := Y_test x T_M^{ab} (with T_M's indices renamed to a, b)
T_M_re = _reindex_tensor(T_M, (mu_T, nu_T), (alpha, beta))
Z = canon(Y_test * T_M_re)
print(f"Z = Y_test x T_M: {n(Z)} terms")
print(f"  free indices should be (c, d, e, f) — a, b are contracted out")

# Now run the orchestrator on Z
print("\n--- decompose_against_eoms ---")
result = decompose_against_eoms(Z, L_scalar, em_procedure='hilbert', verbose=True)
Y_h = result['Y_h']
alphas_h = result['alphas_h']
residual = result['residual']
print(f"\nY_h recovered: {n(Y_h)} terms, alphas={alphas_h}")
print(f"  = {Y_h}")
print(f"residual: {n(residual)} terms")

# Verify Y_h x T_M reproduces Z (Y_h is unique only up to the (a,b) symmetry
# of T_M, so we check equivalence by reconstruction, not literal equality).
T_M_recover = _reindex_tensor(T_M, (mu_T, nu_T), alphas_h)
Z_recovered = canon(Y_h * T_M_recover)
diff_Z = canon(Z_recovered - Z)
print(f"\nZ_recovered = Y_h x T_M: {n(Z_recovered)} terms")
print(f"Z_recovered - Z: {n(diff_Z)} terms")

# Also report the literal Y_h - Y_test diff for diagnostic purposes
Y_h_re = _reindex_tensor(Y_h, alphas_h, (alpha, beta))
diff_Y = canon(Y_h_re - Y_test)
print(f"Y_h - Y_test (literal, after relabel): {n(diff_Y)} terms"
      + " (expected nonzero if Y_test isn't symmetric in a<->b)")

if residual == S.Zero and diff_Z == S.Zero:
    print("\n*** SUCCESS: Y_h x T_M = Z, residual = 0 ***")
elif residual == S.Zero:
    print(f"\n*** FAIL: residual=0 but Y_h x T_M differs from Z by {n(diff_Z)} terms")
    print(f"  Diff: {diff_Z}")
else:
    print(f"\n*** FAIL: residual is {n(residual)} terms")
    if n(residual) < 12:
        print(f"  = {residual}")
