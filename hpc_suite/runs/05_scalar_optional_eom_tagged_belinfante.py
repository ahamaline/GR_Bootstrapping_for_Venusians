"""HPC suite #5 — free scalar, Belinfante, with TAGGED optional EOM terms
(both h and phi) added at every order, orders 0..6.

Stresses the optional-EOM (voluntary field-redefinition) machinery under the
Belinfante procedure at high order: at each order n we register a distinct,
tagged, integrable optional X_h^(n) and X_phi^(n). Each tag is a unique symbol
(th_n / tp_n) so its propagation (step-2 carryover into higher orders, and the
field redefs it induces) is visible in the log.

  X_h^(n)   = d/dh [ th_n * h(-k,a)h(-l,-a) * (tr h)^(n-1) ]  (rank-2 potential
              f_{kl} of order n+1; X is rank-4 (mu nu, k l), integrable)
  X_phi^(n) = tp_n * (tr h)^n * eta^{mn} * phi^2             (Schwarz-symmetric in h)

run_order raises on any closure failure, so completing all orders IS the proof.
NOTE: this is also a genuine test of whether free-scalar Belinfante closes at
all (the graviton sector's Belinfante != Hilbert), driven through the
field-redef machinery.

Usage:  python -u 05_scalar_optional_eom_tagged_belinfante.py [n_max]  (default 6)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, fresh_indices, canon, metric, h,
)
from bootstrap.jet import jet_derivative
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


def tr_h_power(p):
    """(tr h)^p = product of p factors h(a_i, -a_i) with distinct dummies."""
    out = S.One
    for _ in range(p):
        a, = fresh_indices(1)
        out = out * h(a, -a)
    return out


phi, dphi, ddphi = register_scalar_field('phi')
mu_d, = fresh_indices(1)
L_M = canon(Rational(-1, 2) * dphi(mu_d) * dphi(-mu_d))

print(f"=== free scalar + tagged optional EOM (h & phi) every order, "
      f"Belinfante, n_max={N_MAX} ===", flush=True)
state = BootstrapState(L_matter=L_M, em_procedure='belinfante', n_max=N_MAX,
                       verbose=True)
mE, nE = state.mu_E, state.nu_E

# Register a distinct tagged, integrable optional X_h and X_phi at every order.
for k in range(1, N_MAX + 1):
    th = Symbol(f'th_{k}')   # tag for the h-redef at order k
    tp = Symbol(f'tp_{k}')   # tag for the phi-redef at order k
    # rank-2 potential f_{kap,lam} of order k+1 -> X_h is rank-4 (mE,nE,kap,lam).
    kap, lam = fresh_indices(2)
    al, = fresh_indices(1)
    f_pot = canon(th * h(-kap, al) * h(-lam, -al) * tr_h_power(k - 1))
    X_h = canon(jet_derivative(f_pot, h, [mE, nE]))      # order k, integrable
    X_phi = canon(tp * tr_h_power(k) * metric(mE, nE) * phi() * phi())
    state.add_optional_eom_term(k, 'h', X_h)
    state.add_optional_eom_term(k, 'phi', X_phi)
    print(f"  tagged optional at order {k}: X_h^({k}) [{_n(X_h)} terms, tag th_{k}], "
          f"X_phi^({k}) [{_n(X_phi)} terms, tag tp_{k}]", flush=True)

t0 = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  >>> order {n} closed (no raise)  [{time.time() - t:.1f}s, "
          f"E^({n}): {_n(state.E.get(n, S.Zero))} terms]", flush=True)

print(f"\n*** PASS: tagged-optional-EOM scalar Belinfante closes at every order "
      f"0-{N_MAX} (total {time.time() - t0:.1f}s). ***", flush=True)
