"""Fast scalar (per-eigenvalue) version of the redef composition.

Because every recovered redef f_h^(n) = c_n kappa^(n-1) (h^n) is a single, trace-free
matrix power of h, all the h-endomorphisms commute and the composition has no traces,
so it reduces EXACTLY to a scalar power series in one eigenvalue. (The tensor-level
proof in recover_exponential_metric.py confirms this reduction is exact.)

The c_n below are the coefficients of f_h^(2)..f_h^(9) in f_h_redefs.txt; the rigorous
script verifies f_h^(n) == c_n kappa^(n-1) (h^n) against the scraped tensor objects.

Result (per eigenvalue of the (1,1) endomorphism g^mu_nu = g_mu_rho eta^rho_nu):
    g^mu_nu = exp(2 kappa H^mu_nu)   <=>   H = (1/2kappa) log(g^mu_nu).
"""
from sympy import symbols, Rational, expand, Poly, S, factorial

y = symbols('y')               # eigenvalue of the REDEFINED graviton H (kappa=1)
c = {2: S(1), 3: Rational(2, 3), 4: S(-1), 5: Rational(32, 15), 6: Rational(-8, 3),
     7: Rational(404, 63), 8: Rational(-34, 3), 9: Rational(14336, 405),
     10: Rational(-6418, 135)}
DEG = 10


def trunc(e):
    p = Poly(expand(e), y)
    return sum(co * y**k for (k,), co in p.terms() if k <= DEG)


# compose h = phi2(phi3(...phi9(H))),  phi_n(z) = z + c_n z^n   (kappa set to 1)
xi = y
for n in range(DEG, 1, -1):
    xi = trunc(xi + c[n] * xi**n)

print("composed h_orig eigenvalue  xi(y)  (= h in terms of redefined H, kappa=1):")
for (k,), co in sorted(Poly(xi, y).terms()):
    print(f"   y^{k}: {co}")

# The metric is g = eta + 2*kappa*h 
# Written as a function of the REDEFINED graviton H (eigenvalue y), per eigenvalue
# (kappa=1):  g(y) = 1 + 2*xi(y).
g_of_y = expand(1 + 2 * xi)
# Rescale to the exponent variable u = 2*y (= 2*kappa * H-eigenvalue) -- purely so the
# series reads as exp(u) = sum_k u^k/k!, i.e. clean 1/k! coefficients. 
u = symbols('u')
g_of_u = expand(g_of_y.subs(y, u / 2))
print("\nthe metric g, as a function of the redefined H (in u = 2*kappa*H):")
ok = True
for (k,), co in sorted(Poly(g_of_u, u).terms()):
    want = Rational(1, factorial(k))
    ok = ok and (co == want)
    print(f"   u^{k}: {co}     [1/{k}! = {want}]  {'ok' if co == want else 'MISMATCH'}")

print(f"\n=> g eigenvalue = sum_k u^k/k! = exp(u) = exp(2 kappa H eigenvalue), so "
      f"g^mu_nu = exp(2 kappa H).  {'CONFIRMED' if ok else 'MISMATCH'}")
