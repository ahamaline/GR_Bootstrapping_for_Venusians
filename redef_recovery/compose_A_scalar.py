"""Compose the COUPLED h- and A-redefs from run 6 (EM, Belinfante) and identify the
A (photon) field redefinition. Per eigenvalue of the matrix h (kappa=1):

  h-redef order n:  lambda -> lambda + c_n lambda^n      (f_h^(n) = c_n k^(n-1) h^n)
  A-redef order n:  A      -> A (1 + a_n lambda^n)        (f_A^(n) = a_n k^n (h^n A))

applied for n = N..1 (innermost = highest order first), tracking lambda (the current
h eigenvalue) and psi (the cumulative A scaling = A_orig / A_final). The h-eigenvalue
in f_A^(n) is the level-n h, i.e. the current lambda BEFORE that order's update.

Candidates for psi(y) (y = final-H eigenvalue):  the metric eigenvalue is g = e^{2y},
so e^y = sqrt(g) = vielbein eigenvalue.
"""
from sympy import symbols, Rational, expand, Poly, S, factorial

y = symbols('y')
N = 7
# h-redef coeffs c_n (same as run 11); c_1 = 0 (no f_h^(1))
c = {2: S(1), 3: Rational(2, 3), 4: S(-1), 5: Rational(32, 15), 6: Rational(-8, 3),
     7: Rational(404, 63)}
# A-redef coeffs a_n from run 6; a_3 = 0 (no f_A^(3))
a = {1: S(1), 2: Rational(-1, 2), 4: Rational(7, 8), 5: S(-1), 6: Rational(65, 24),
     7: Rational(-35, 6)}


def trunc(e):
    p = Poly(expand(e), y)
    return sum(co * y**k for (k,), co in p.terms() if k <= N)


lam = y          # current h eigenvalue, starts at final H eigenvalue
psi = S.One      # cumulative A scaling A_orig/A_final
for n in range(N, 0, -1):
    lam_old = lam
    psi = trunc(psi * (1 + a.get(n, 0) * lam_old**n))   # f_A^(n) uses level-n h
    lam = trunc(lam_old + c.get(n, 0) * lam_old**n)      # f_h^(n)

print("A scaling  psi(y) = A_orig / A_final  (per h-eigenvalue, kappa=1):")
for (k,), co in sorted(Poly(psi, y).terms()):
    print(f"   y^{k}: {co}")

print("\ncompare psi(y) to candidates (Taylor coefficients):")
cands = {
    "e^y = sqrt(g) (vielbein)": [Rational(1, factorial(k)) for k in range(N + 1)],
    "e^{-y} = 1/sqrt(g)":       [Rational((-1)**k, factorial(k)) for k in range(N + 1)],
    "e^{2y} = g":              [Rational(2**k, factorial(k)) for k in range(N + 1)],
}
pc = {k: co for (k,), co in Poly(psi, y).terms()}
for name, coeffs in cands.items():
    ok = all(pc.get(k, S.Zero) == coeffs[k] for k in range(N + 1))
    print(f"   {name:28s}: {'MATCH' if ok else 'no'}")
