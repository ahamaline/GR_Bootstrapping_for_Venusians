"""
Test: Compute W^{μν} from L_h^{(2)} and verify the linearized Bianchi identity.

This tests the full chain: tensor algebra -> jet derivatives -> EL derivative.

L_h^{(2)} = -½ h^{μν,ρ} h_{μν,ρ} + h^{μν,ρ} h_{μρ,ν} 
           + ½ h^μ_μ^{,ρ} h^ν_{ν,ρ} - h^μ_μ^{,ρ} h_{ρ}^{ν}_{,ν}

W^{μν} = δL_h^{(2)}/δh_{μν} should be the linearized Einstein tensor.

W^{μν} = h^{μν,ρ}_{,ρ} + h^ρ_{ρ}^{,μν} - h^{μρ,ν}_{,ρ} 
        - h^{νρ,μ}_{,ρ} - η^{μν} h^ρ_{ρ}^{,σ}_{,σ} + η^{μν} h^{ρσ}_{,ρσ}
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.tensor_algebra import *
from bootstrap.jet import jet_derivative, total_derivative
from bootstrap.euler_lagrange import euler_lagrange

from sympy import Rational

# Create named indices for building L_h^{(2)}
mu, nu, rho, sigma, tau = named_indices('mu nu rho sigma tau')
alpha, beta, gamma, delta = named_indices('alpha beta gamma delta')
a, b, c = named_indices('a b c')

print("=== Building L_h^{(2)} ===")

# L_h^{(2)} = -½ dh^{μν,ρ} dh_{μν,ρ} + dh^{μν,ρ} dh_{μρ,ν}
#            + ½ (dh^μ_{μ}^{,ρ})(dh^ν_{ν,ρ}) - (dh^μ_{μ}^{,ρ})(dh_{ρ}^{ν}_{,ν})

# Term 1: -½ h^{μν,ρ} h_{μν,ρ} = -½ dh(mu,nu,rho) * dh(-mu,-nu,-rho)
term1 = Rational(-1, 2) * dh(mu, nu, rho) * dh(-mu, -nu, -rho)
print(f"Term 1: {canon(term1)}")

# Term 2: h^{μν,ρ} h_{μρ,ν} = dh(mu,nu,rho) * dh(-mu,-rho,-nu)
term2 = dh(mu, nu, rho) * dh(-mu, -rho, -nu)
print(f"Term 2: {canon(term2)}")

# Term 3: ½ h^μ_μ^{,ρ} h^ν_{ν,ρ} = ½ dh(mu,-mu,rho) * dh(nu,-nu,-rho)
term3 = Rational(1, 2) * dh(mu, -mu, rho) * dh(nu, -nu, -rho)
print(f"Term 3: {canon(term3)}")

# Term 4: -h^μ_μ^{,ρ} h_ρ^ν_{,ν} = -dh(mu,-mu,rho) * dh(-rho,nu,-nu)
term4 = -dh(mu, -mu, rho) * dh(-rho, nu, -nu)
print(f"Term 4: {canon(term4)}")

Lh2 = term1 + term2 + term3 + term4
Lh2 = canon(Lh2)
print(f"\nL_h^(2) = {Lh2}")

print("\n=== Computing W^ab = dL_h^(2)/dh_ab ===")
W, W_indices = euler_lagrange(Lh2, h)
print(f"Free indices: {W_indices}")
print(f"W = {W}")

# Also check: is it a sum? how many terms?
if isinstance(W, TensAdd):
    print(f"Number of terms: {len(W.args)}")
    for i, t in enumerate(W.args):
        print(f"  term {i}: {t}")
else:
    print(f"Single term: {W}")
