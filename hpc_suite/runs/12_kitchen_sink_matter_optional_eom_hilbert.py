"""HPC suite #12 — MAXIMALLY MESSY: kitchen sink (run 8) + tagged optional EOM
terms whose h-redefinition potential f_h DEPENDS ON THE MATTER FIELDS, Hilbert.

This exercises something never tested: optional E_h terms (voluntary field
redefinitions of the graviton) that are functions of the MATTER fields, not just h.
As always, integrable X tensors are obtained as the h-jet-derivative of an arbitrary
potential f:  X_h = d/dh_{mu_E nu_E} f_h ,  with f_h rank-2 (its two free indices are
the EOM-direction pair that contracts E_h). Here f_h is a deliberately messy,
NON-gauge-invariant polynomial in BARE A, V1, V2, phi1, phi2 and h, and in several
terms ONE OR BOTH of f_h's free indices are carried by a matter field (A, V1, V2)
rather than by h/eta. (f_h is derivative-free -> X_h is too, as the optional-EOM
machinery requires; bare A/V means manifestly not gauge invariant.)

f_h is symmetrized in its two free indices (h is symmetric). A distinct tag t{i}_{n}
per term per order makes the propagation (step-2 carryover, induced redefs) visible.

run_order raises on any closure failure, so completing all orders IS the result. If
it raises, that pins where the matter-dependent optional-E_h path breaks.

Usage:  python -u 12_kitchen_sink_matter_optional_eom_hilbert.py [n_max]  (default 3)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sympy import Rational, Symbol, S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import (
    register_scalar_field, register_vector_field, register_upstairs_vector_field,
    fresh_indices, canon, metric, h,
)
from bootstrap.jet import jet_derivative
from bootstrap.energy_momentum import NATURAL_POSITIONS
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def _n(e):
    return 0 if e == S.Zero else (len(e.args) if isinstance(e, TensAdd) else 1)


def tr_h_power(p):
    """(tr h)^p = product of p factors h(a_i, -a_i) with distinct dummies."""
    out = S.One
    for _ in range(p):
        a, = fresh_indices(1)
        out = out * h(a, -a)
    return out


# ---- kitchen-sink matter (identical to run 8) ---------------------------------
A, dA, ddA = register_vector_field('A')
phi1, dphi1, ddphi1 = register_scalar_field('phi1')
phi2, dphi2, ddphi2 = register_scalar_field('phi2')
V1, dV1, ddV1 = register_upstairs_vector_field('V1')
V2, dV2, ddV2 = register_upstairs_vector_field('V2')
NATURAL_POSITIONS[A] = ['down']
NATURAL_POSITIONS[dA] = ['down', 'down']
NATURAL_POSITIONS[ddA] = ['down', 'down', 'down']

m_phi = Symbol('m_phi'); m_V = Symbol('m_V'); e = Symbol('e'); g_coup = Symbol('g')

mu_F, nu_F = fresh_indices(2)
F_A_dn = dA(-nu_F, -mu_F) - dA(-mu_F, -nu_F)
F_A_up = dA(nu_F, mu_F) - dA(mu_F, nu_F)
L_EM = Rational(-1, 4) * F_A_dn * F_A_up
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
mu_pV1, = fresh_indices(1); mu_pV2, = fresh_indices(1)
L_phiDV_partial = g_coup * (phi1() * dV1(mu_pV1, -mu_pV1)
                           + phi2() * dV2(mu_pV2, -mu_pV2))
mu_pA, = fresh_indices(1)
L_phiDV_A = g_coup * e * A(-mu_pA) * (phi1() * V2(mu_pA) - phi2() * V1(mu_pA))

L_M = canon(L_EM + L_phi_kin + L_phi_mass + L_phi_min + L_phi_sea
            + L_V1_kin + L_V2_kin + L_V_mass + L_phiDV_partial + L_phiDV_A)


# ---- messy matter-dependent h-redef potential ---------------------------------
def Vd(field, idx):
    """Lower an upstairs vector to a single DOWN free index `idx`."""
    a, = fresh_indices(1)
    return metric(-idx, -a) * field(a)


def matter_X_h(n, mE, nE):
    """X_h^(n) = d/dh f_h, with f_h a symmetric rank-2 (free k,l) order-(n+1)
    derivative-free polynomial in BARE matter fields and h, free indices on matter."""
    k, l = fresh_indices(2)
    t1, t2, t3, t4 = (Symbol(f't{i}_{n}') for i in range(1, 5))
    # both free indices on A
    term1 = t1 * A(-k) * A(-l) * tr_h_power(n + 1)
    # free indices on V1, V2 (symmetrized k<->l)
    term2 = t2 * Rational(1, 2) * (Vd(V1, k) * Vd(V2, l) + Vd(V1, l) * Vd(V2, k)) \
        * tr_h_power(n + 1)
    # one free on A, one on h (symmetrized); depends on A
    c1, = fresh_indices(1); c2, = fresh_indices(1)
    term3 = t3 * Rational(1, 2) * (A(-k) * h(-l, c1) * A(-c1)
                                   + A(-l) * h(-k, c2) * A(-c2)) * tr_h_power(n)
    # both free on h, with a matter (phi) scalar spectator
    term4 = t4 * h(-k, -l) * (phi1() * phi1() + phi2() * phi2()) * tr_h_power(n)
    f_h = canon(term1 + term2 + term3 + term4)
    return canon(jet_derivative(f_h, h, [mE, nE]))


print(f"=== kitchen sink + MATTER-DEPENDENT optional E_h (tagged, every order), "
      f"Hilbert, n_max={N_MAX} ===", flush=True)
print(f"L_M built: {_n(L_M)} terms after canon", flush=True)
state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True)
mE, nE = state.mu_E, state.nu_E
for k in range(1, N_MAX + 1):
    X_h = matter_X_h(k, mE, nE)
    state.add_optional_eom_term(k, 'h', X_h)
    print(f"  tagged matter-dependent optional X_h^({k}): {_n(X_h)} terms "
          f"(tags t1_{k}..t4_{k})", flush=True)

t0 = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  >>> order {n} closed (no raise)  [{time.time() - t:.1f}s, "
          f"E^({n}): {_n(state.E.get(n, S.Zero))} terms]", flush=True)

print(f"\n*** PASS: kitchen sink + matter-dependent optional E_h closes at every "
      f"order 0-{N_MAX} (total {time.time() - t0:.1f}s). ***", flush=True)
