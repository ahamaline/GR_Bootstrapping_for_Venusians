"""Compose the coupled h- and V-redefs from run 13 (Proca, Belinfante) and identify
the UPSTAIRS vector V^mu redefinition. Same recursion as compose_A_scalar.py, but V
has an UPPER index, so the prediction is the INVERSE vielbein:

    V_orig^mu = (e^{-kappa H})^mu_nu V_final^nu = (g^{-1/2})^mu_nu V_final^nu,

i.e. per h-eigenvalue  psi_V(y) = e^{-y}  (coefficients 1, -1, +1/2, -1/6, ...).

Run 13 has only reached order 2 so far -> f_V^(1..3); enough to test through order 3.
"""
from sympy import symbols, Rational, expand, Poly, S, factorial

y = symbols('y')
N = 3
# h-redef coeffs (same as always); c_1 = 0
c = {2: S(1), 3: Rational(2, 3)}
# V-redef coeffs b_n from run 13:  f_V^(n) = b_n kappa^n (h^n V)
b = {1: S(-1), 2: Rational(3, 2), 3: S(2)}


def trunc(e):
    p = Poly(expand(e), y)
    return sum(co * y**k for (k,), co in p.terms() if k <= N)


lam = y
psi = S.One
for n in range(N, 0, -1):
    lam_old = lam
    psi = trunc(psi * (1 + b.get(n, 0) * lam_old**n))
    lam = trunc(lam_old + c.get(n, 0) * lam_old**n)

print("V scaling  psi_V(y) = V_orig / V_final  (per h-eigenvalue, kappa=1):")
pc = {k: co for (k,), co in Poly(psi, y).terms()}
for k in range(N + 1):
    print(f"   y^{k}: {pc.get(k, S.Zero)}")

print("\ncompare to candidates (Taylor coefficients through y^%d):" % N)
cands = {
    "e^{-y} = 1/sqrt(g) (inverse vielbein)": [Rational((-1)**k, factorial(k)) for k in range(N + 1)],
    "e^{+y} = sqrt(g)":                      [Rational(1, factorial(k)) for k in range(N + 1)],
    "e^{-2y} = 1/g":                         [Rational((-2)**k, factorial(k)) for k in range(N + 1)],
}
for name, coeffs in cands.items():
    ok = all(pc.get(k, S.Zero) == coeffs[k] for k in range(N + 1))
    print(f"   {name:38s}: {'MATCH' if ok else 'no'}")
