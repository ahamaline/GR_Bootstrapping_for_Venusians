"""HPC suite #9 — TRACELESS kitchen sink, d=4, Hilbert, orders 0..5.

A fully conformal (on-shell-traceless) charged-matter sector, to stress the
traceless-T_M DETECTION on rich complex matter:
  - massless EM (-1/4 F_A^2),
  - charged conformal complex scalar (phi1,phi2): kinetic + minimal EM coupling
    + seagull + the conformal coupling xi (phi1^2+phi2^2) R, xi=1/6 at d=4,
  - charged MASSLESS complex vector (V1,V2): charged-Proca F_V^2 (no mass),
  - phibar phi Vbar V interaction: g (phi1^2+phi2^2)(V1.V1 + V2.V2)  [marginal].

Everything is massless / marginal, so each piece is classically conformal and
T_M is on-shell traceless. NOTE (design to confirm): masslessness is required
for tracelessness; xi must be exactly the conformal value.

Usage:  python -u 09_traceless_kitchen_sink_hilbert.py [n_max]   (default 5)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, register_vector_field, register_upstairs_vector_field,
    fresh_indices, canon, metric,
)
from bootstrap.energy_momentum import NATURAL_POSITIONS
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


A, dA, ddA = register_vector_field('A')
phi1, dphi1, ddphi1 = register_scalar_field('phi1')
phi2, dphi2, ddphi2 = register_scalar_field('phi2')
V1, dV1, ddV1 = register_upstairs_vector_field('V1')
V2, dV2, ddV2 = register_upstairs_vector_field('V2')
NATURAL_POSITIONS[A] = ['down']
NATURAL_POSITIONS[dA] = ['down', 'down']
NATURAL_POSITIONS[ddA] = ['down', 'down', 'down']

e = Symbol('e')
g_coup = Symbol('g')
xi = Rational(1, 6)   # conformal value at d=4

# L_EM (massless)
mu_F, nu_F = fresh_indices(2)
F_A_dn = dA(-nu_F, -mu_F) - dA(-mu_F, -nu_F)
F_A_up = dA(nu_F, mu_F) - dA(mu_F, nu_F)
L_EM = Rational(-1, 4) * F_A_dn * F_A_up

# charged scalar: kinetic + minimal coupling + seagull (NO mass)
mu_k1, = fresh_indices(1)
mu_k2, = fresh_indices(1)
L_phi_kin = (Rational(-1, 2) * dphi1(mu_k1) * dphi1(-mu_k1)
             + Rational(-1, 2) * dphi2(mu_k2) * dphi2(-mu_k2))
mu_c, = fresh_indices(1)
L_phi_min = -e * A(mu_c) * (phi1() * dphi2(-mu_c) - phi2() * dphi1(-mu_c))
mu_s, = fresh_indices(1)
L_phi_sea = (Rational(-1, 2) * e**2 * A(mu_s) * A(-mu_s)
             * (phi1() * phi1() + phi2() * phi2()))


def charged_proca_F_squared(V_re_dfield, V_im_field, sign_e):
    mu, nu = fresh_indices(2)
    rho1, rho2, rho3, rho4 = fresh_indices(4)
    F_dn = (metric(-nu, -rho1) * V_re_dfield(rho1, -mu)
            - metric(-mu, -rho2) * V_re_dfield(rho2, -nu)
            + sign_e * e * A(-mu) * metric(-nu, -rho3) * V_im_field(rho3)
            - sign_e * e * A(-nu) * metric(-mu, -rho4) * V_im_field(rho4))
    alpha1, alpha2, beta1, beta2 = fresh_indices(4)
    F_up = (metric(mu, alpha1) * V_re_dfield(nu, -alpha1)
            - metric(nu, alpha2) * V_re_dfield(mu, -alpha2)
            + sign_e * e * metric(mu, beta1) * A(-beta1) * V_im_field(nu)
            - sign_e * e * metric(nu, beta2) * A(-beta2) * V_im_field(mu))
    return F_dn * F_up


# charged MASSLESS vector
L_V1_kin = Rational(-1, 4) * charged_proca_F_squared(dV1, V2, sign_e=+1)
L_V2_kin = Rational(-1, 4) * charged_proca_F_squared(dV2, V1, sign_e=-1)

# phibar phi Vbar V (marginal quartic):  g (phi1^2+phi2^2)(V1.V1 + V2.V2)
mu_q1, = fresh_indices(1)
mu_q2, = fresh_indices(1)
L_phiV = g_coup * (phi1() * phi1() + phi2() * phi2()) * (
    V1(mu_q1) * V1(-mu_q1) + V2(mu_q2) * V2(-mu_q2))

L_M = canon(L_EM + L_phi_kin + L_phi_min + L_phi_sea
            + L_V1_kin + L_V2_kin + L_phiV)

# Conformal coupling for the complex scalar:  xi (phi1^2+phi2^2) R.
# canon to DISTRIBUTE the (phi1^2+phi2^2)*Riemann product (else the coupling is
# a TensMul wrapping a TensAdd, which the expansion's _decompose_tensmul rejects).
lam, aR = fresh_indices(2)
coupling = canon(-Rational(1, 2) * xi * (phi1() * phi1() + phi2() * phi2())
                 * Riemann(lam, -aR, -lam, aR))

print(f"=== TRACELESS kitchen sink, d=4, Hilbert, n_max={N_MAX} ===", flush=True)
print(f"L_M built: {_n(L_M)} terms", flush=True)
state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True, nonminimal_coupling=coupling)
t0 = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    if n == 0:
        print(f"  traceless_T_M = {state.traceless_T_M}; "
              f"c_i keys = {list(state.traceless_c_i)}", flush=True)
    print(f"  >>> order {n} closed (no raise)  [{time.time() - t:.1f}s, "
          f"E^({n}): {_n(state.E.get(n, S.Zero))} terms]", flush=True)

print(f"\n*** PASS: traceless kitchen sink closes at every order 0-{N_MAX} "
      f"(total {time.time() - t0:.1f}s). ***", flush=True)
