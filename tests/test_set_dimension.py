"""Unit test for set_dimension: verify that the Lorentz rebuild produces
heads whose fully-contracted metric trace yields the requested dim, and
that a (Lorentz.dim - 4)-coefficient term then vanishes automatically.

Three cases:
  1. Default symbolic mode: metric trace -> Symbol('d').
  2. set_dimension(4): metric trace -> 4, and a coefficient built from the
     CURRENT Lorentz.dim (Lorentz.dim - 4 = 0) collapses to zero.
  3. set_dimension(None) restores symbolic mode.

Important notes about authoring code that uses set_dimension:

- set_dimension REBUILDS the module-level h/metric, so any local binding
  via `from bootstrap.tensor_algebra import h, metric` goes stale the
  moment set_dimension fires. This test references them via the module
  object (`ta.h`, `ta.metric`) so each lookup is fresh.

- A stray `Symbol('d')` typed in user code is NOT the same object as the
  current `Lorentz.dim` after set_dimension(4) — there is no global hook
  to substitute `Symbol('d') -> 4` everywhere. Code that wants a
  dim-dependent coefficient should read `ta.Lorentz.dim`. Contraction
  traces (e.g. metric(L, -L)) already yield Lorentz.dim directly, so
  this is mainly an issue if user code is manually mixing in a 'd'
  it created itself.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Symbol, S
from sympy.tensor.tensor import TensAdd
import bootstrap.tensor_algebra as ta
from bootstrap.tensor_algebra import set_dimension, canon, fresh_indices


def n_terms(expr):
    return 0 if expr == S.Zero else (
        len(expr.args) if isinstance(expr, TensAdd) else 1)


# Case 1: default symbolic mode.
d_symbolic = Symbol('d')
L1, = fresh_indices(1)
canon_trace_default = canon(ta.metric(L1, -L1))
print(f"Default canon(metric(L, -L))  = {canon_trace_default}  (expect: d)")
assert canon_trace_default == d_symbolic, f"Expected d, got {canon_trace_default}"
assert ta.Lorentz.dim == d_symbolic, f"Expected Lorentz.dim = d, got {ta.Lorentz.dim}"

# Case 2: set_dimension(4).
set_dimension(4)
assert ta.Lorentz.dim == 4, f"Expected Lorentz.dim = 4, got {ta.Lorentz.dim}"

L1, = fresh_indices(1)
canon_trace_d4 = canon(ta.metric(L1, -L1))
print(f"After set_dimension(4):  canon(metric(L, -L))  = {canon_trace_d4}  (expect: 4)")
assert canon_trace_d4 == 4, f"Expected 4, got {canon_trace_d4}"

# A coefficient built from the CURRENT Lorentz.dim is concretely 4 - 4 = 0,
# so canon((Lorentz.dim - 4) * h) drops to zero immediately.
mu, nu = fresh_indices(2)
canon_test = canon((ta.Lorentz.dim - 4) * ta.h(mu, nu))
print(f"After set_dimension(4):  canon((Lorentz.dim - 4)*h)  = {canon_test}  (expect: 0)")
assert canon_test == S.Zero, f"Expected 0, got {canon_test}"

# The metric-trace path: 4 * (metric(L,-L) - 4) * h is identically zero.
mu, nu = fresh_indices(2)
L2, = fresh_indices(1)
canon_via_trace = canon((ta.metric(L2, -L2) - 4) * ta.h(mu, nu))
print(f"After set_dimension(4):  canon((metric(L,-L) - 4)*h) = {canon_via_trace}  (expect: 0)")
assert canon_via_trace == S.Zero, f"Expected 0, got {canon_via_trace}"

# Case 3: restore symbolic mode.
set_dimension(None)
assert ta.Lorentz.dim == d_symbolic, f"Expected Lorentz.dim = d, got {ta.Lorentz.dim}"
L1, = fresh_indices(1)
canon_trace_restored = canon(ta.metric(L1, -L1))
print(f"After set_dimension(None):  canon(metric(L, -L))  = {canon_trace_restored}  (expect: d)")
assert canon_trace_restored == d_symbolic, (
    f"Expected d after restore, got {canon_trace_restored}")

print()
print("*** All set_dimension tests pass ***")
