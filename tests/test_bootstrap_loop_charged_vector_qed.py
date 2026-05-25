"""Kitchen-sink BootstrapState: EM + charged complex scalar + charged complex
vector (upstairs V) + a phi-bar D V + H.C. interaction coupling them.

Real-component encoding of the complex fields:
  phi   = (phi_1 + i phi_2) / sqrt(2)
  V^mu  = (V_1^mu + i V_2^mu) / sqrt(2)
  D_mu  = partial_mu - i e A_mu

Full L_M (real-component form):

  L_EM      = -1/4 F^A_{mn} F^A^{mn}                          (EM kinetic)

  L_phi_kin = -1/2 sum_i (d phi_i)^2                          (charged scalar kinetic)
  L_phi_mas = -(m_phi^2 / 2) sum_i phi_i^2                    (scalar mass)
  L_phi_min = -e A^mu (phi_1 d_mu phi_2 - phi_2 d_mu phi_1)   (scalar minimal coupling)
  L_phi_sea = -(e^2 / 2) A^2 sum_i phi_i^2                    (scalar seagull)

  L_V_kin   = -1/4 sum_i F^{V_i}^{cov} . F^{V_i}^{cov}        (charged Proca kinetic,
              where F^{V_i}^{cov}_munu uses (∂ + signed e A x partner) in each slot)
  L_V_mass  = -(m_V^2 / 2) sum_i V_i . V_i                    (vector mass)

  L_phiDV   = +g (phi_1 d_mu V_1^mu + phi_2 d_mu V_2^mu)
              + g e A^mu (phi_1 V_2_mu - phi_2 V_1_mu)        (phi-bar D V + H.C.)

The phi-bar D V interaction has gauge structure: phi-bar transforms with charge -1
and V transforms with charge +1, so the whole thing is U(1)-invariant.

This is the first test that combines:
  - downstairs gauge vector (A)
  - rank-0 complex matter (phi_1, phi_2)
  - upstairs rank-1 complex matter (V_1, V_2)
  - a non-gauge-invariant-per-piece interaction (phi_i d_mu V_i^mu) that is
    only gauge invariant via the +H.C. combination
  - a phi V A cubic coupling (from the phi-bar D V's A piece)

We aim for n=0 verification first (cheapest, confirms the infrastructure threads
through). Bumping N_MAX to 1 or 2 is straightforward but expensive.
"""

import sys
import time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational, S, Symbol
from sympy.tensor.tensor import TensAdd

from bootstrap.tensor_algebra import (
    register_scalar_field, register_vector_field, register_upstairs_vector_field,
    fresh_indices, canon, metric,
)
from bootstrap.bootstrap_loop import BootstrapState


# --- Field registrations ----------------------------------------------------

A, dA, ddA = register_vector_field('A')           # downstairs EM photon
phi1, dphi1, ddphi1 = register_scalar_field('phi1')
phi2, dphi2, ddphi2 = register_scalar_field('phi2')
V1, dV1, ddV1 = register_upstairs_vector_field('V1')
V2, dV2, ddV2 = register_upstairs_vector_field('V2')

# --- Coupling and mass symbols ---------------------------------------------

m_phi = Symbol('m_phi')
m_V = Symbol('m_V')
e = Symbol('e')
g_coup = Symbol('g')   # coupling for phi-bar D V + H.C.


# --- L_EM ------------------------------------------------------------------

mu_F, nu_F = fresh_indices(2)
F_A_dn = dA(-nu_F, -mu_F) - dA(-mu_F, -nu_F)        # F^A_{mu nu}
F_A_up = dA(nu_F, mu_F) - dA(mu_F, nu_F)            # F^A^{mu nu}
L_EM = Rational(-1, 4) * F_A_dn * F_A_up


# --- L_phi: kinetic + mass + EM couplings (real-component scalar QED) -----

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


# --- L_V: charged Proca (covariant in EM via the +/- e A x partner pattern)

def charged_proca_F_squared(V_re_dfield, V_im_field, sign_e):
    """Returns F^{V}_munu * F^{V}^{munu} for a charged Proca field whose F is
        F_{munu} = partial_mu V_re_nu - partial_nu V_re_mu
                  + sign_e * e (A_mu V_im_nu - A_nu V_im_mu)
    and similarly raised for F^{munu}. V_re_dfield is the d-tensor head for the
    real (or imag) component being squared; V_im_field is the field head of the
    partner (imag or real), unsuffixed. sign_e is +1 for the real-component
    contribution and -1 for the imag-component contribution, matching the real
    and imaginary parts of the complex F^{V} = D V - D V.

    Naturals: V's natural position is UP (upstairs), A's is DOWN.
    """
    mu, nu = fresh_indices(2)
    rho1, rho2, rho3, rho4 = fresh_indices(4)
    F_dn = (
        metric(-nu, -rho1) * V_re_dfield(rho1, -mu)
        - metric(-mu, -rho2) * V_re_dfield(rho2, -nu)
        + sign_e * e * A(-mu) * metric(-nu, -rho3) * V_im_field(rho3)
        - sign_e * e * A(-nu) * metric(-mu, -rho4) * V_im_field(rho4)
    )
    alpha1, alpha2, beta1, beta2 = fresh_indices(4)
    F_up = (
        metric(mu, alpha1) * V_re_dfield(nu, -alpha1)
        - metric(nu, alpha2) * V_re_dfield(mu, -alpha2)
        + sign_e * e * metric(mu, beta1) * A(-beta1) * V_im_field(nu)
        - sign_e * e * metric(nu, beta2) * A(-beta2) * V_im_field(mu)
    )
    return F_dn * F_up

L_V1_kin = Rational(-1, 4) * charged_proca_F_squared(dV1, V2, sign_e=+1)  # Re
L_V2_kin = Rational(-1, 4) * charged_proca_F_squared(dV2, V1, sign_e=-1)  # Im

mu_V1, = fresh_indices(1)
L_V_mass = Rational(-1, 2) * m_V**2 * V1(mu_V1) * V1(-mu_V1)
mu_V2, = fresh_indices(1)
L_V_mass = L_V_mass + Rational(-1, 2) * m_V**2 * V2(mu_V2) * V2(-mu_V2)


# --- L_phiDV: phi-bar D V + H.C. in real components ------------------------

# Partial-derivative piece:  g (phi_1 d_mu V_1^mu  +  phi_2 d_mu V_2^mu)
mu_pV1, = fresh_indices(1)
mu_pV2, = fresh_indices(1)
L_phiDV_partial = g_coup * (
    phi1() * dV1(mu_pV1, -mu_pV1) + phi2() * dV2(mu_pV2, -mu_pV2)
)

# A-coupling piece:  g e A^mu (phi_1 V_2_mu  -  phi_2 V_1_mu)
mu_pA, = fresh_indices(1)
L_phiDV_A = g_coup * e * A(-mu_pA) * (phi1() * V2(mu_pA) - phi2() * V1(mu_pA))


# --- Assemble L_M ----------------------------------------------------------

L_M = canon(
    L_EM
    + L_phi_kin + L_phi_mass + L_phi_min + L_phi_sea
    + L_V1_kin + L_V2_kin + L_V_mass
    + L_phiDV_partial + L_phiDV_A
)

n_terms = len(L_M.args) if isinstance(L_M, TensAdd) else 1
print(f"L_M built: {n_terms} terms after canon")
print()

N_MAX = 4   # bump to 2 once we know n=1 is tractable

state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True)

t_total = time.time()
for n in range(N_MAX + 1):
    t0 = time.time()
    state.run_order(n)
    print(f"  Order {n} total wall: {time.time()-t0:.1f}s")

print()
print("=" * 60)
print(f"  All orders 0..{N_MAX} passed verification.")
print(f"  Total wall: {time.time()-t_total:.1f}s")
print("=" * 60)
