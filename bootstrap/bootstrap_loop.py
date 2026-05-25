"""
The bootstrap loop: the 6-step procedure from Section 4 of the paper.

Given an initial matter Lagrangian L_M (or None for pure gravity) and a
choice of energy-momentum procedure (Hilbert or Belinfante), this module
iteratively constructs the gravitational field equation order by order in
h_{μν}. Each call to `run_order(n)` advances the state by one order.

Steps per the paper:
    1. E_1 = κ T̂[L^{(n)}] + δ_{1,n} W^{μν}
    2. E_2 = E_1 + Σ_{m<n} X^{(m)} · E^{(n−m)}                    (carryover)
    3. E_3 = E_2 + X^{(n)} · E^{(0)}                                (mandatory EOM, from H2)
    4. E_4 = E_3 + X'^{(n)} · E^{(0)}                               (optional, field-redef)
    5. E   = E_4 + Δ^{(n)},     Δ^{(n)} = Ψ_{,ρσ}                 (superpotential, from H3)
    6. L^{(n+1)} = (1/(n+1)) E^{(n)} h + b.t.                       (close the loop)

If `n_max` is supplied at construction, a reference Lagrangian
L_ref^{(k)} (the standard covariant Einstein-Hilbert expansion + matter)
is pre-computed for all k=0..n_max+1, and after each `run_order(n)` the
bootstrap-derived L^{(n+1)} is checked against L_ref^{(n+1)} — see
`_verify_vs_L_ref` for the full flow.
"""

from sympy import S, Rational, Symbol
from sympy.tensor.tensor import TensAdd, TensMul, TensExpr

from bootstrap.tensor_algebra import (
    h, fresh_indices, canon,
)
from bootstrap.euler_lagrange import euler_lagrange, remove_second_derivatives
from bootstrap.helmholtz import (
    compute_superpotential_n2, compute_superpotential_n1,
    superpotential_divergence, verify_psi_symmetries,
)
from bootstrap.energy_momentum import hilbert_energy_momentum
from bootstrap.covariant import einstein_hilbert_lagrangian_order, matter_lagrangian_order


kappa = Symbol('kappa')


def _count(expr):
    if expr == S.Zero:
        return 0
    if isinstance(expr, TensAdd):
        return len(expr.args)
    return 1


def _reindex_tensor(expr, old_indices, new_indices):
    """Substitute old free indices with new ones (both signs)."""
    if expr == S.Zero or expr == 0:
        return expr
    if not hasattr(expr, 'substitute_indices'):
        return expr
    pairs = []
    for old, new in zip(old_indices, new_indices):
        pairs.append((old, new))
        pairs.append((-old, -new))
    return canon(expr.substitute_indices(*pairs))


class BootstrapState:
    """Holds the state of the bootstrap computation and drives it forward.

    Public attributes:
        L: dict {n: L^{(n)}}                     -- the bootstrap Lagrangians
        E: dict {n: E^{μν(n)}}                   -- the field equation terms
        mu_E, nu_E: the canonical (reserved) free indices for E^{μν}
        n_max: highest order we plan to reach (drives L_ref pre-computation)
        L_ref: dict {n: L_ref^{(n)}}              -- the reference Lagrangians
        eom_terms_h: dict {n: X_h^{μν}_{ρσ}^{(n)}}     -- accumulated mandatory + optional gravity EOM coefficients
        eom_terms_matter: dict {n: {field_name: X_φ^{μν(n)}}}
    """

    def __init__(self, L_matter=None, em_procedure='hilbert', n_max=None,
                 verbose=True):
        if em_procedure not in ('hilbert', 'belinfante'):
            raise ValueError(f"em_procedure must be 'hilbert' or 'belinfante', got {em_procedure!r}")
        if em_procedure == 'belinfante':
            raise NotImplementedError("Belinfante procedure not yet implemented")

        self.em_procedure = em_procedure
        self.verbose = verbose
        self.n_max = n_max

        # Reserve the canonical free indices for E^{μν(n)} — used at every
        # order, so all field equations are in a comparable form without
        # having to substitute_indices every time we want to combine them.
        self.mu_E, self.nu_E = fresh_indices(2)

        # Pre-compute the wave operator W^{μν} once (paper eq. for W).
        # Equivalent to EL(L_EH^{(2)}); reindexed to our canonical (μ, ν).
        Lh2 = einstein_hilbert_lagrangian_order(2)
        W, (w_mu, w_nu) = euler_lagrange(Lh2, h)
        self.W = _reindex_tensor(W, (w_mu, w_nu), (self.mu_E, self.nu_E))

        # Lagrangians and field equations from the bootstrap, by order.
        self.L = {0: L_matter if L_matter is not None else S.Zero}
        self.E = {}

        # EOM coefficients accumulated across orders (steps 3 and 4).
        self.eom_terms_h = {}
        self.eom_terms_matter = {}

        self.max_order_run = -1

        # Reference Lagrangian L_ref^{(k)}, pre-computed if n_max given.
        self.L_ref = {}
        if n_max is not None:
            self._init_L_ref(n_max)

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------

    def run_order(self, n):
        """Execute the bootstrap at order n. Updates self.E[n] and self.L[n+1]."""
        if n > 0 and (n - 1) not in self.L:
            raise ValueError(f"Must run order {n-1} before order {n}")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  BOOTSTRAP ORDER n = {n}")
            print(f"{'='*60}")

        # Step 1
        E = self._step1_energy_momentum(n)
        if self.verbose:
            print(f"  Step 1 (E_1): {_count(E)} terms")

        # Step 2 — EOM carryover from earlier orders
        E = self._step2_eom_carryover(E, n)
        if self.verbose:
            print(f"  Step 2 (E_2): {_count(E)} terms")

        # Step 3 — mandatory EOM correction (H2)
        E = self._step3_mandatory_eom(E, n)
        if self.verbose:
            print(f"  Step 3 (E_3): {_count(E)} terms")

        # Step 4 — optional EOM (field redefinition, voluntary). Not exposed
        # via run_order yet; would take user-provided X' coefficients.
        # E = self._step4_optional_eom(E, n, ...)
        if self.verbose:
            print(f"  Step 4 (E_4): {_count(E)} terms (no optional terms applied)")

        # Step 5 — superpotential correction (H3)
        E = self._step5_superpotential(E, n)
        if self.verbose:
            print(f"  Step 5 (E^({n})): {_count(E)} terms")

        self.E[n] = E

        # Step 6 — close the loop
        L_next = self._step6_close_loop(E, n)
        self.L[n + 1] = L_next
        if self.verbose:
            print(f"  Step 6 (L^({n+1})): {_count(L_next)} terms")

        # Self-consistency: EL(L^{(n+1)}) should reproduce E^{(n)} on the nose.
        self._verify_el(n)

        # External consistency: bootstrap result vs reference EH expansion.
        if self.n_max is not None:
            self._verify_vs_L_ref(n)

        self.max_order_run = n
        return E

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _step1_energy_momentum(self, n):
        """E_1^{μν(n)} = κ T̂[L^{(n)}] + δ_{1,n} W^{μν}."""
        L_n = self.L.get(n, S.Zero)

        if L_n != S.Zero:
            if self.em_procedure == 'hilbert':
                T_mn, T_idx = hilbert_energy_momentum(L_n)
            else:
                raise NotImplementedError(self.em_procedure)
            T_mn = _reindex_tensor(T_mn, T_idx, (self.mu_E, self.nu_E))
            E = kappa * T_mn
        else:
            E = S.Zero

        if n == 1:
            # Add the wave operator
            E = E + self.W if E != S.Zero else self.W

        if isinstance(E, TensExpr):
            E = canon(E)
        return E

    def _step2_eom_carryover(self, E, n):
        """Add order-n parts of EOM terms chosen at earlier orders.

        For each m < n with stored X_h^{(m)} (or X_φ^{(m)}), the term
        X^{(m)} · E^{(n−m)} contributes here. No EOM terms have been
        added in the pure-gravity Hilbert path so this is a no-op there.
        """
        if not self.eom_terms_h and not self.eom_terms_matter:
            return E
        # No EOM terms have been added yet in any test path — when matter
        # arrives, fill this in.
        raise NotImplementedError(
            "Step 2 (EOM carryover) is not yet wired up; reachable only when "
            "self.eom_terms_h or self.eom_terms_matter is non-empty, which "
            "requires implementing Step 3 first."
        )

    def _step3_mandatory_eom(self, E, n):
        """Add the mandatory EOM correction required by H2 (Helmholtz #2).

        Compute Z^{μνρσ(n-1)} (paper eq. for Z), decompose Z = Y · E^{(0)},
        set X = -1/(2(n+1)) Y · h, append X · E^{(0)} to E. For pure gravity
        with the Hilbert procedure (Butcher's claim) Z is zero, so no
        correction is needed at any order. Until we have a Z-decomposition
        algorithm, this step assumes Z=0 and skips the work.
        """
        if n == 0:
            return E
        # Pure-gravity-Hilbert: Z=0 (Butcher). We don't currently verify
        # this directly — the n=2/n=3/n=4 closure tests imply it. With
        # matter, this stub becomes load-bearing: see open-work item 4 in
        # DEVELOPMENT_STATUS.md ("explicit H2-violation check").
        return E

    def _step4_optional_eom(self, E, n, X_prime_h=None, X_prime_matter=None):
        """Add optional integrable EOM terms (field redefinitions). Not used on
        the default path; not exposed via run_order yet."""
        if X_prime_h is None and X_prime_matter is None:
            return E
        raise NotImplementedError("Optional EOM terms not yet wired up")

    def _step5_superpotential(self, E, n):
        """Add Δ^{(n)} = Ψ^{(n)}_{,ρσ}.

        Ψ comes from the paper's PsiForm formula at n≥2, or the integral
        formula (paper eq. 23) at n=1 with matter. For n=1 pure gravity,
        E_4 = W already satisfies H3 with Ψ=0.
        """
        if n == 0 or E == S.Zero:
            # n=0: any superpotential here is a physical modification, skip.
            return E

        if n == 1:
            if self.L[0] == S.Zero:
                return E  # pure gravity at n=1: no superpotential
            Psi, psi_idx = compute_superpotential_n1(
                E, (self.mu_E, self.nu_E)
            )
        else:
            Psi, psi_idx = compute_superpotential_n2(
                E, n, (self.mu_E, self.nu_E)
            )

        if Psi == S.Zero or psi_idx is None:
            return E

        # Sanity-check Ψ's three symmetries; the paper guarantees they hold
        # when the bootstrap has a valid continuation.
        sym = verify_psi_symmetries(Psi, psi_idx)
        ok = all(sym.get(k, True) for k in ('sym_mn', 'sym_rs', 'cyclic'))
        if not ok:
            raise RuntimeError(
                f"Psi^({n}) failed symmetry checks: {sym}. Either the "
                "input E is not a valid bootstrap field equation or the "
                "implementation is broken."
            )
        if self.verbose:
            print(f"    Psi^({n}) symmetries: OK")

        Delta = superpotential_divergence(Psi, psi_idx)
        Delta = _reindex_tensor(Delta, (psi_idx[0], psi_idx[1]),
                                (self.mu_E, self.nu_E))

        return canon(E + Delta) if Delta != S.Zero else E

    def _step6_close_loop(self, E, n):
        """L^{(n+1)} = (1/(n+1)) · E^{μν(n)} · h_{μν} + boundary terms.

        Boundary terms are absorbed by `remove_second_derivatives` so the
        Lagrangian comes out in standard first-order form.
        """
        if E == S.Zero:
            return S.Zero
        L_raw = Rational(1, n + 1) * E * h(-self.mu_E, -self.nu_E)
        if isinstance(L_raw, TensExpr):
            # .expand() to distribute Rational * TensAdd * Tensor — without it,
            # sympy can leave L_raw as a TensMul wrapping a TensAdd, and
            # remove_second_derivatives (which uses _decompose_tensmul) would
            # silently skip the wrapped TensAdd's terms. Same dropped-TensAdd
            # footgun as in compute_superpotential_n1 (see
            # project-decompose-tensmul-tensadd-pitfall memory).
            L_raw = L_raw.expand()
            L_raw = canon(L_raw)
        return remove_second_derivatives(L_raw)

    # ------------------------------------------------------------------
    # Verifications
    # ------------------------------------------------------------------

    def _verify_el(self, n):
        """Self-consistency: EL(L^{(n+1)}) must reproduce E^{(n)}.

        This is automatic in theory (paper §3 identity), but it catches
        bugs in `remove_second_derivatives` (IBP), `euler_lagrange`, and
        `_step6_close_loop` working in concert.
        """
        L_next = self.L[n + 1]
        E_target = self.E[n]
        if L_next == S.Zero:
            if E_target != S.Zero:
                raise RuntimeError(
                    f"L^({n+1}) is zero but E^({n}) is not — Step 6 dropped content"
                )
            if self.verbose:
                print(f"  Verify EL(L^({n+1})) == E^({n}): both zero, OK")
            return True

        E_check, E_check_idx = euler_lagrange(L_next, h)
        E_check = _reindex_tensor(E_check, E_check_idx, (self.mu_E, self.nu_E))

        diff = canon(E_check - E_target)
        if diff != S.Zero:
            n_diff = _count(diff)
            raise RuntimeError(
                f"EL(L^({n+1})) does not equal E^({n}): residual has {n_diff} terms"
            )
        if self.verbose:
            print(f"  Verify EL(L^({n+1})) == E^({n}): OK")
        return True

    # ------------------------------------------------------------------
    # Reference Lagrangian (verification cycle)
    # ------------------------------------------------------------------

    def _init_L_ref(self, n_max):
        """Pre-compute L_ref^{(k)} for k=0..n_max+1.

        L_ref is the "true" raw Lagrangian expansion — NOT IBP'd.
        For pure gravity: L_ref^{(k)} = L_EH^{(k)}. With matter, the
        matter Lagrangian's covariantized expansion (√|g| L̃_M)^{(k)} is
        added to each order; L_ref^{(0)} = L_M and L_ref^{(1)} is now
        non-zero (= κ h_{μν} T_M^{μν}) when matter is present.

        The bootstrap's own L^{(n+1)} comes out of `_step6_close_loop`
        with IBP applied (per the paper's recipe), so it's always going
        to differ from L_ref by boundary terms; the verification cycle
        uses the EL-equivalence check (case b) which sees through that.
        """
        L_M = self.L[0]
        has_matter = (L_M != S.Zero)

        if self.verbose:
            kind = "with matter" if has_matter else "pure gravity"
            print(f"  Pre-computing L_ref^{{(0..{n_max+1})}} ({kind})...")

        self.L_ref[0] = L_M
        if has_matter:
            self.L_ref[1] = matter_lagrangian_order(L_M, 1)
            if self.verbose:
                print(f"    L_ref^(1): {_count(self.L_ref[1])} terms (matter)")
        else:
            self.L_ref[1] = S.Zero

        for k in range(2, n_max + 2):
            L_EH = einstein_hilbert_lagrangian_order(k)
            if has_matter:
                L_mat = matter_lagrangian_order(L_M, k)
                L_k = L_EH + L_mat if L_mat != S.Zero else L_EH
                if isinstance(L_k, TensExpr):
                    L_k = canon(L_k)
            else:
                L_k = L_EH
            self.L_ref[k] = L_k
            if self.verbose:
                print(f"    L_ref^({k}): {_count(L_k)} terms")

    def _verify_vs_L_ref(self, n):
        """Verification cycle (USER COMMENT in the paper's roadmap).

        After running order n, check that L^{(n+1)} is equivalent (modulo
        boundary terms) to L_ref^{(n+1)} by comparing their EL derivatives:

          canon(EL(L_ref^{(n+1)}) − E^{(n)}) == 0

        We skip the "literal equality" check that would compare L's
        directly: L_ref is the raw EH expansion (with second derivatives)
        and our L^{(n+1)} comes out of `_step6_close_loop` after IBP, so
        literal equality essentially never holds. The EL comparison is
        the real test anyway.

        If the EL difference is nonzero, the paper's prescription is to
        decompose it into integrable EOM terms (yielding field
        redefinitions to apply to L_ref). That decomposition is stubbed
        — when we reach a setting where this fires we want a loud failure
        so we know we need to fill it in.
        """
        target_n = n + 1
        if target_n not in self.L_ref:
            if self.verbose:
                print(f"  Verify vs L_ref: no L_ref^({target_n}) (n_max too low?), skipped")
            return None

        L_r = self.L_ref[target_n]
        if L_r == S.Zero:
            E_r = S.Zero
        else:
            E_r, E_r_idx = euler_lagrange(L_r, h)
            E_r = _reindex_tensor(E_r, E_r_idx, (self.mu_E, self.nu_E))
        E_diff = self.E[n] - E_r
        if isinstance(E_diff, TensExpr):
            E_diff = canon(E_diff)
        if E_diff == S.Zero:
            if self.verbose:
                print(f"  Verify vs L_ref^({target_n}): EL-equivalent (OK)")
            return True

        # Nonzero EL difference. The paper's flow says: decompose E_diff
        # into integrable EOM coefficients, extract h- and matter-field
        # redefinitions, apply them to L_ref. Skeletoned in `_apply_field_redef`.
        n_E_diff = _count(E_diff)
        raise NotImplementedError(
            f"EL(L_ref^({target_n})) does not match E^({n}): residual has "
            f"{n_E_diff} terms. The verification cycle expects this to be "
            f"handled by a field-redefinition decomposition (see "
            f"_apply_field_redef), which is not yet implemented. Reachable "
            f"when using the Belinfante procedure or when adding optional "
            f"EOM terms."
        )

    def _apply_field_redef(self, E_diff, n):
        """SKELETON: decompose E_diff into integrable EOM coefficients, derive
        the corresponding field redefinitions, and update L_ref accordingly.

        Detailed flow (user's USER COMMENT in the older bootstrap_loop.py
        comments, now lives in DEVELOPMENT_STATUS.md open-work item 3):
          1-2. (already done in _verify_vs_L_ref above)
          3.   Decompose E_diff as Y_h · E^{(0)} + Σ Y_φ_i · E_φ_i^{(0)}.
               Same algorithm as H2 step (open-work item 7) — not done yet.
          4.   From the Y's, recover the field redefinitions (paper formula):
                   f^{(n+1)}_{ρσ}    = (1/(n+1)) h_{αβ} X_h^{αβ}_{ρσ}^{(n)}
                   f^{(n+1)}_{φ_i}   = (1/(n+1)) h_{αβ} X_{φ_i}^{αβ(n)}
          5.   Apply h → h + f^{(n+1)} to L_ref^{(k)} for k = 1..n_max+2-n.
               Re-expand in h, truncate at order n_max+1. Use fresh dummies
               on every occurrence (NOT for a combinatorial explosion
               reason — for the index-clash pitfall when re-using φ in φ^n).
          6.   Apply φ_i → φ_i + f_φ_i^{(n+1)} similarly, one matter field
               at a time.
          7.   Sanity check: updated L_ref^{(n+1)} should now equal L^{(n+1)}.
        """
        raise NotImplementedError("Field-redefinition decomposition is a skeleton")
