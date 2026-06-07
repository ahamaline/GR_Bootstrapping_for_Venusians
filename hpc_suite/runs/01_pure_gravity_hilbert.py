"""HPC suite #1 — pure gravity, Hilbert, orders 0..8.

The grand-challenge run: the bare bootstrap (no matter) pushed to order 8.
Validates the superpotential PsiForm + closure uniformly at high order.

run_order raises on any EL-self-consistency or L_ref verification failure, so
"every order completes without raising" IS the closure proof.

Usage:  python -u 01_pure_gravity_hilbert.py [n_max]   (default 8)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sympy import S
from sympy.tensor.tensor import TensAdd
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


print(f"=== pure gravity, Hilbert, n_max={N_MAX} ===", flush=True)
state = BootstrapState(L_matter=None, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True)
t0 = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  >>> order {n} closed (no raise)  [{time.time() - t:.1f}s, "
          f"E^({n}): {_n(state.E.get(n, S.Zero))} terms]", flush=True)

print(f"\n*** PASS: pure gravity closes at every order 0-{N_MAX} "
      f"(total {time.time() - t0:.1f}s). ***", flush=True)
