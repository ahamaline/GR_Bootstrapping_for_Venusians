"""HPC suite #8 — kitchen sink (EM + charged complex scalar + charged complex
vector + phi-bar D V + H.C.), Hilbert, orders 0..5.

The everything-bagel charged-matter Lagrangian (the n=4 milestone, pushed to 5).
Stresses the full covariantization + cross-couplings + n=1 superpotential +
carryover under the heaviest matter content.

Usage:  python -u 08_kitchen_sink_hilbert.py [n_max]   (default 5)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, register_vector_field, register_upstairs_vector_field,
    fresh_indices, canon, metric,
)
from bootstrap.energy_momentum import NATURAL_POSITIONS
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


A, dA, ddA = register_vector_field('A')            # downstairs EM photon
phi1, dphi1, ddphi1 = register_scalar_field('phi1')
phi2, dphi2, ddphi2 = register_scalar_field('phi2')
V1, dV1, ddV1 = register_upstairs_vector_field('V1')
V2, dV2, ddV2 = register_upstairs_vector_field('V2')
NATURAL_POSITIONS[A] = ['down']
NATURAL_POSITIONS[dA] = ['down', 'down']
NATURAL_POSITIONS[ddA] = ['down', 'down', 'down']

m_phi = Symbol('m_phi')
m_V = Symbol('m_V')
e = Symbol('e')
g_coup = Symbol('g')

# L_EM
mu_F, nu_F = fresh_indices(2)
F_A_dn = dA(-nu_F, -mu_F) - dA(-mu_F, -nu_F)
F_A_up = dA(nu_F, mu_F) - dA(mu_F, nu_F)
L_EM = Rational(-1, 4) * F_A_dn * F_A_up

# L_phi: kinetic + mass + minimal coupling + seagull
mu_k1, = fresh_indices(1)
L_phi_kin = Rational(-1, 2) * dphi1(mu_k1) * dphi1(-mu_k1)
mu_k2, = fresh_indices(1)
L_phi_kin = L_phi_kin + Rational(-1, 2) * dphi2(mu_k2) * dphi2(-mu_k2)
L_phi_mass = Rational(-1, 2) * m_phi**2 * (phi1() * phi1() + phi2() * phi2())
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


L_V1_kin = Rational(-1, 4) * charged_proca_F_squared(dV1, V2, sign_e=+1)
L_V2_kin = Rational(-1, 4) * charged_proca_F_squared(dV2, V1, sign_e=-1)
mu_V1, = fresh_indices(1)
L_V_mass = Rational(-1, 2) * m_V**2 * V1(mu_V1) * V1(-mu_V1)
mu_V2, = fresh_indices(1)
L_V_mass = L_V_mass + Rational(-1, 2) * m_V**2 * V2(mu_V2) * V2(-mu_V2)

mu_pV1, = fresh_indices(1)
mu_pV2, = fresh_indices(1)
L_phiDV_partial = g_coup * (phi1() * dV1(mu_pV1, -mu_pV1)
                           + phi2() * dV2(mu_pV2, -mu_pV2))
mu_pA, = fresh_indices(1)
L_phiDV_A = g_coup * e * A(-mu_pA) * (phi1() * V2(mu_pA) - phi2() * V1(mu_pA))

L_M = canon(L_EM + L_phi_kin + L_phi_mass + L_phi_min + L_phi_sea
            + L_V1_kin + L_V2_kin + L_V_mass + L_phiDV_partial + L_phiDV_A)

print(f"=== kitchen sink, Hilbert, n_max={N_MAX} ===", flush=True)
print(f"L_M built: {_n(L_M)} terms after canon", flush=True)
state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True)
t0 = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  >>> order {n} closed (no raise)  [{time.time() - t:.1f}s, "
          f"E^({n}): {_n(state.E.get(n, S.Zero))} terms]", flush=True)

print(f"\n*** PASS: kitchen sink closes at every order 0-{N_MAX} "
      f"(total {time.time() - t0:.1f}s). ***", flush=True)
