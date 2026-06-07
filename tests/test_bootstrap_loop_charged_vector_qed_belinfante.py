"""Kitchen-sink BootstrapState with em_procedure='belinfante', n_max=3.

Same Lagrangian as `test_bootstrap_loop_charged_vector_qed.py` (EM + charged
complex scalar + charged complex vector + phi-bar D V + H.C.), but routed
through the symmetrized Belinfante energy-momentum tensor and exercising
the full field-redef cycle (integrability check, recovery, substitution
into L_ref) at every order. Aims to confirm the new machinery works for a
non-trivial multi-matter Lagrangian beyond EM.

Conservative wall-time estimate: kitchen-sink Hilbert at n=4 was ~4.8 h
(and n=3 a few hours less). Belinfante adds the per-order field-redef
work (substitute h -> h + f_h and each matter field, re-expand L_ref^(k)
for k up to n_max+1). Plan accordingly.
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
g_coup = Symbol('g')


# --- L_EM ------------------------------------------------------------------

mu_F, nu_F = fresh_indices(2)
F_A_dn = dA(-nu_F, -mu_F) - dA(-mu_F, -nu_F)
F_A_up = dA(nu_F, mu_F) - dA(mu_F, nu_F)
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

L_V1_kin = Rational(-1, 4) * charged_proca_F_squared(dV1, V2, sign_e=+1)
L_V2_kin = Rational(-1, 4) * charged_proca_F_squared(dV2, V1, sign_e=-1)

mu_V1, = fresh_indices(1)
L_V_mass = Rational(-1, 2) * m_V**2 * V1(mu_V1) * V1(-mu_V1)
mu_V2, = fresh_indices(1)
L_V_mass = L_V_mass + Rational(-1, 2) * m_V**2 * V2(mu_V2) * V2(-mu_V2)


# --- L_phiDV: phi-bar D V + H.C. -------------------------------------------

mu_pV1, = fresh_indices(1)
mu_pV2, = fresh_indices(1)
L_phiDV_partial = g_coup * (
    phi1() * dV1(mu_pV1, -mu_pV1) + phi2() * dV2(mu_pV2, -mu_pV2)
)

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
print(f"L_M (kitchen sink): {n_terms} terms after canon")
print()

N_MAX = 3

state = BootstrapState(L_matter=L_M, em_procedure='belinfante',
                       n_max=N_MAX, verbose=True)

t_total = time.time()
for n in range(N_MAX + 1):
    t0 = time.time()
    state.run_order(n)
    print(f"  Order {n} total wall: {time.time()-t0:.1f}s")

print()
print("=" * 60)
print(f"  Kitchen-sink Belinfante: orders 0..{N_MAX} all closed.")
print(f"  Total wall: {time.time()-t_total:.1f}s")
print("=" * 60)
