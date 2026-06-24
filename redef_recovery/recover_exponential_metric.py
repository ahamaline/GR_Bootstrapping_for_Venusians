"""Recover the field redefinition implied by the Belinfante bootstrap, and show it
is the EXPONENTIAL PARAMETRIZATION of the metric.

Background
----------
Run 11 (massive scalar, Belinfante energy-momentum) closes the bootstrap by applying,
at each order n, a graviton field redefinition  h -> h + f_h^(n+1)(h)  that reconciles
the Belinfante stress tensor with the covariant (Hilbert) reference. The recovered
redefs (scraped to f_h_redefs.txt) are each a single, trace-free MATRIX POWER of h:

    f_h^(n)_{mu nu} = c_n * kappa^(n-1) * (h^n)_{mu nu},
    c = {1, 2/3, -1, 32/15, -8/3, 404/63, -34/3, 14336/405}   (n = 2..9)

Composing them recursively (original h in terms of the fully-redefined H) and using
g = eta + 2 kappa h, this script verifies -- entirely in tensor algebra, no scalar
reduction -- that, as a (1,1) ENDOMORPHISM (index raised with eta),

    g^{mu}{}_{nu} = ( exp(2 kappa H) )^{mu}{}_{nu} ,   i.e.   H^{mu}{}_{nu} = (1/2kappa) log( g^{mu}{}_{nu} ),

where  g^{mu}{}_{nu} = g_{mu rho} eta^{rho nu}  is the metric viewed as a linear map
on the tangent space (positive-definite in Euclidean signature, so the matrix log is
well-defined -- NOTE: it is the log of the mixed-index g^{mu}{}_{nu}, not of the
bilinear form g_{mu nu}). The down-down metric is then
    g_{mu nu} = eta_{mu rho} ( exp(2 kappa H) )^{rho}{}_{nu}.
The "matrix powers" (h^n) below are powers of this endomorphism h^{mu}{}_{nu}, chained
through eta (which is exactly what matmul does).

So the Belinfante stress tensor is the metric stress tensor of the EXPONENTIAL
parametrization of the metric; the bootstrapped Einstein equation is markedly more
compact in this frame than in the linear split g = eta + 2 kappa h.

Steps
-----
1. scrape f_h^(n) (f_h_redefs.txt) -> SymPy tensor objects, free indices (-MU,-NU);
2. verify each scraped object == c_n * kappa^(n-1) * matpow(h,n) (canon(diff)==0);
3. compose phi2 . phi3 . ... . phi9 with real tensor matrix-multiplication, trunc @9;
4. check the composed h == expansion of g=exp(2 kappa H), i.e.
   h = sum_{k>=1} (2 kappa)^(k-1)/k! (H^k), term by term.

Run:  python -u recover_exponential_metric.py [f_h_redefs.txt]
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sympy import Rational, factorial, S
from sympy.tensor.tensor import TensorIndex, TensAdd
from bootstrap.tensor_algebra import h, metric, Lorentz, canon, fresh_indices, order_in_h
from bootstrap.covariant import kappa

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "f_h_redefs.txt")
NMAX = 10
_IDX = re.compile(r'_i\d+|L_\d+')

MU, NU = fresh_indices(2)   # global rank-2 free indices (down-down: -MU,-NU)


def parse_tensor(s):
    """eval a printed tensor string into a TensExpr; int/int kept Rational."""
    ns = {'h': h, 'kappa': kappa, 'Rational': Rational}
    for nm in sorted(set(_IDX.findall(s))):
        ns[nm] = TensorIndex(nm, Lorentz)
    return eval(re.sub(r'(\d+)/(\d+)', r'Rational(\1,\2)', s), ns)


def to_MUNU(expr):
    free = list(expr.get_free_indices())
    pairs = []
    for old, new in zip(free, (-MU, -NU)):
        pairs += [(old, new), (-old, -new)]
    return canon(expr.substitute_indices(*pairs))


def matmul(A, B):
    """(A.B)_{-MU,-NU} = A_{-MU}{}^{c} B_{c,-NU} for A,B with free (-MU,-NU)."""
    if A == 0 or B == 0:
        return S.Zero
    c, d = fresh_indices(2)
    A2 = A.substitute_indices((-NU, -c), (NU, c))
    B2 = B.substitute_indices((-MU, -d), (MU, d))
    return canon((A2 * metric(c, d) * B2).expand())


def matpow(A, n):
    R = A
    for _ in range(n - 1):
        R = matmul(R, A)
    return R


def order_part(expr, k):
    """the order-k (k h-factors) part of expr."""
    terms = expr.args if isinstance(expr, TensAdd) else [expr]
    keep = [t for t in terms if t != 0 and order_in_h(t) == k]
    return TensAdd(*keep) if len(keep) > 1 else (keep[0] if keep else S.Zero)


def truncate(expr, dmax=NMAX):
    terms = expr.args if isinstance(expr, TensAdd) else [expr]
    keep = [t for t in terms if t != 0 and order_in_h(t) <= dmax]
    return TensAdd(*keep) if len(keep) > 1 else (keep[0] if keep else S.Zero)


def n_terms(e):
    return len(e.args) if isinstance(e, TensAdd) else (0 if e == 0 else 1)


# ---- 1. scrape ----------------------------------------------------------------
scraped = {}
with open(DATA, encoding='utf-8') as fh:
    for line in fh:
        m = re.search(r'f_h\^\((\d+)\) = (.+)', line)
        if m and int(m.group(1)) not in scraped:
            scraped[int(m.group(1))] = to_MUNU(parse_tensor(m.group(2).strip()))
print(f"1. scraped {len(scraped)} redefs from {os.path.basename(DATA)}: orders {sorted(scraped)}\n")

H = canon(h(-MU, -NU))   # the fully-redefined graviton

# ---- 2. verify scraped f_h^(n) == c_n kappa^(n-1) matpow(h,n) ------------------
coef = {}
print("2. verify each scraped redef is c_n * kappa^(n-1) * (h^n):")
for n in sorted(scraped):
    pown = matpow(H, n)
    c_n = scraped[n].coeff / (pown.coeff * kappa**(n - 1))
    coef[n] = c_n
    ok = (canon(scraped[n] - c_n * kappa**(n - 1) * pown) == 0)
    print(f"     f_h^({n}): c_{n} = {str(c_n):>12}   == c_n k^(n-1) h^n : {ok}")
    assert ok

# ---- 3. compose phi2 . phi3 . ... . phi9 (tensor matmul, truncate @ NMAX) ------
print("\n3. compose  h = phi2(phi3(...phi9(H)))  at the tensor level:")
result = H
for n in range(NMAX, 1, -1):
    result = truncate(canon(result + coef[n] * kappa**(n - 1) * matpow(result, n)))
h_orig = result
print(f"     composed h_orig: {n_terms(h_orig)} terms (orders 1..{NMAX})")

# ---- 4. compare to g = exp(2 kappa H):  h = sum (2k)^(k-1)/k! matpow(H,k) ------
print("\n4. compare to the expansion of g = exp(2*kappa*H):")
exp_series = canon(sum(Rational(2**(k - 1), 1) / factorial(k) * kappa**(k - 1) * matpow(H, k)
                       for k in range(1, NMAX + 1)))
allok = True
for k in range(1, NMAX + 1):
    ok = (canon(order_part(h_orig, k) - order_part(exp_series, k)) == 0)
    allok = allok and ok
    print(f"     order {k}: composed == exp-series : {ok}")

print(f"\n=== RESULT: g^mu_nu = exp(2*kappa*H)^mu_nu  "
      f"(H = (1/2kappa) log(g^mu_nu), g^mu_nu = g_mu_rho eta^rho_nu),  "
      f"orders 1..{NMAX}: {'CONFIRMED' if allok else 'MISMATCH'} ===")
sys.exit(0 if allok else 1)
