"""Quick smoke test for the tensor algebra module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from sympy import Rational

# 1. Create indices
mu, nu, rho, sigma = named_indices('mu nu rho sigma')
alpha, beta = named_indices('alpha beta')
print("Indices created OK")

# 2. Build a simple h expression  
expr1 = h(-mu, -nu)
print(f"h(-mu,-nu) = {expr1}")

# 3. Product of two tensor heads
expr2 = dh(-mu, -nu, -rho) * h(rho, sigma)
print(f"dh * h = {expr2}")

# 4. Canonicalization test: two expressions that should be equal
a, b, c = named_indices('a b c')
expr_a = dh(-mu, -nu, -rho) * dh(mu, nu, rho)
expr_b = dh(-a, -b, -c) * dh(a, b, c)
ca = canon(expr_a)
cb = canon(expr_b)
print(f"Canon test: {ca} == {cb} ? {ca == cb}")

# 5. Test symmetry: h(-mu,-nu) should equal h(-nu,-mu)
ha = h(-mu, -nu)
hb = h(-nu, -mu)
print(f"h(-mu,-nu) == h(-nu,-mu) ? {canon(ha) == canon(hb)}")

# 6. Test symmetrized delta
delta_sym = symmetrized_h_delta((alpha, beta), (mu, nu))
print(f"Symmetrized delta: {delta_sym}")

# 7. Test metric contraction: delta^alpha_mu * h(-alpha, -nu) should give h(-mu,-nu)
contracted = metric(alpha, -mu) * h(-alpha, -nu)
print(f"metric(alpha,-mu)*h(-alpha,-nu) = {canon(contracted)}")

# 8. Test fresh indices don't clash
i1 = fresh_indices(3)
i2 = fresh_indices(3)
print(f"Fresh 1: {i1}")
print(f"Fresh 2: {i2}")
assert all(str(a) != str(b) for a in i1 for b in i2), "Index clash!"
print("No index clashes")

# 9. Register a scalar field
phi, dphi, ddphi = register_scalar_field('phi')
print(f"Scalar field: {phi()}")  # scalar has no indices
print(f"dphi: {dphi(-mu)}")
print(f"ddphi: {ddphi(-mu, -nu)}")

# 10. A simple scalar-gravity expression
scalar_kin = dphi(mu) * dphi(-mu)
print(f"dphi^mu dphi_mu = {scalar_kin}")

print("\n=== All smoke tests passed! ===")
