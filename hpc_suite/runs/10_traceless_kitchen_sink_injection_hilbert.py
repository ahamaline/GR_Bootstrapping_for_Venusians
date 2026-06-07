"""HPC suite #10 — TRACELESS kitchen sink WITH dual injection, d=4, Hilbert,
orders 0..4.

Same conformal charged-matter sector as #9 (so T_M is on-shell traceless), but
with a traceless-shape optional EOM injected at order 1 for the graviton AND
every traceless matter field. This drives BOTH traceless recoveries on the
richest matter content: the verification-path recovery (missed order-(n-1)
redef) and the mandatory-step recovery, plus rollback+augment and carryover.

Usage:  python -u 10_traceless_kitchen_sink_injection_hilbert.py [n_max]  (default 4)
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
    fresh_indices, canon, metric, h,
)
from bootstrap.energy_momentum import NATURAL_POSITIONS
from bootstrap.covariant import Riemann
from bootstrap.bootstrap_loop import BootstrapState

N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 4
kappa = Symbol('kappa')
eps_inj = Symbol('eps_inj')


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
xi = Rational(1, 6)

mu_F, nu_F = fresh_indices(2)
L_EM = Rational(-1, 4) * (dA(-nu_F, -mu_F) - dA(-mu_F, -nu_F)) * (dA(nu_F, mu_F) - dA(mu_F, nu_F))
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


L_V1_kin = Rational(-1, 4) * charged_proca_F_squared(dV1, V2, sign_e=+1)
L_V2_kin = Rational(-1, 4) * charged_proca_F_squared(dV2, V1, sign_e=-1)
mu_q1, = fresh_indices(1)
mu_q2, = fresh_indices(1)
L_phiV = g_coup * (phi1() * phi1() + phi2() * phi2()) * (
    V1(mu_q1) * V1(-mu_q1) + V2(mu_q2) * V2(-mu_q2))

L_M = canon(L_EM + L_phi_kin + L_phi_min + L_phi_sea + L_V1_kin + L_V2_kin + L_phiV)
lam, aR = fresh_indices(2)
coupling = canon(-Rational(1, 2) * xi * (phi1() * phi1() + phi2() * phi2())
                 * Riemann(lam, -aR, -lam, aR))

print(f"=== TRACELESS kitchen sink + dual injection, d=4, Hilbert, "
      f"n_max={N_MAX} ===", flush=True)
state = BootstrapState(L_matter=L_M, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True, nonminimal_coupling=coupling)

state.run_order(0)
assert state.traceless_T_M, "expected on-shell-traceless T_M for the conformal sector"
print(f"  c_i fields = {list(state.traceless_c_i)}", flush=True)

# Dual injection at order 1: traceless-shape optional EOM for h AND every
# traceless matter field (X^{ab} = eps kappa^2 h^{ab}, times eta_cd for h and
# times c_i for each field).
X_ab = eps_inj * kappa**2 * h(state.mu_E, state.nu_E)
cc, dd = fresh_indices(2)
state.add_optional_eom_term(1, 'h', canon(X_ab * metric(-cc, -dd)))
for name, c_i in state.traceless_c_i.items():
    if c_i != S.Zero:
        state.add_optional_eom_term(1, name, canon(X_ab * c_i))

t0 = time.time()
for n in range(1, N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  >>> order {n} closed (no raise)  [{time.time() - t:.1f}s, "
          f"E^({n}): {_n(state.E.get(n, S.Zero))} terms]", flush=True)

print(f"\n  recovered_traceless_X keys: {sorted(state.recovered_traceless_X)}",
      flush=True)
print(f"\n*** PASS: traceless kitchen sink + injection closes at every order "
      f"0-{N_MAX} (total {time.time() - t0:.1f}s). ***", flush=True)
