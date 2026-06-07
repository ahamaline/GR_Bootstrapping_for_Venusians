"""Trace-free T_M loophole: detection of on-shell-traceless matter and the
two field-redefinition recoveries (verification-step and mandatory-EOM-step),
plus the ddh-signature extractors they use.

The recoveries are methods of `TracelessRecoveryMixin`, which `BootstrapState`
inherits; they operate on the usual state attributes (self.E, self.L,
self.L_ref, self.traceless_c_i, self.mu_E/nu_E, ...). See DEVELOPMENT_STATUS
item 0 for the theory (T1 = X^{mn} S^(1), the T2 extra term, m = d-2, etc.).
"""
from sympy import S, Rational
from sympy.tensor.tensor import TensAdd, TensMul, TensExpr, Tensor
from bootstrap.tensor_algebra import (
    h, ddh, metric, fresh_indices, canon, _matter_fields, get_tensors_in_expr,
)
from bootstrap.euler_lagrange import euler_lagrange, euler_lagrange_scalar
from bootstrap.jet import (
    jet_derivative, _decompose_tensmul, _get_component, _get_indices, _sum_terms,
)
from bootstrap.helmholtz import compute_h2_violation
from bootstrap.eom_decompose import _extract_coeff_from_trace_signature
from bootstrap.loop_helpers import (
    _count, _format_breakdown, _reindex_tensor, _is_derivative_free,
    _require_constant_coeff,
)


def _extract_ddh_box_signature(expr):
    """Coefficient of the doubly-traced ddh scalar signature ddh(L,-L,M,-M)
    in `expr` (BOTH the field-pair, positions 0,1, AND the derivative-pair,
    positions 2,3, contracted dummy pairs).

    For each term carrying such a ddh factor, strip it and accumulate
    (coeff x other_factors). For a scalar `expr` this returns a scalar; for a
    rank-2 `expr` (free mu,nu) it carries those free indices. Terms without
    the signature contribute nothing; returns S.Zero if none match.

    This is the signature used by the traceless-T_M verification recovery:
    its coefficient in S^(1) is the (m+v) denominator (m = W-trace constant,
    v = matter scalar), and its prefactor in E_diff is X^{ab}.(m+v).
    """
    if expr == S.Zero or expr == 0:
        return S.Zero

    def _dummy_pair(a, b):
        return a.name == b.name and (a.is_up != b.is_up)

    terms = expr.args if isinstance(expr, TensAdd) else [expr]
    out = []
    for term in terms:
        if isinstance(term, TensMul):
            coeff, factors = _decompose_tensmul(term)
        elif isinstance(term, Tensor):
            coeff, factors = S.One, [term]
        else:
            continue
        hit = None
        for i, f in enumerate(factors):
            if _get_component(f) is not ddh:
                continue
            idx = _get_indices(f)
            if (len(idx) == 4 and _dummy_pair(idx[0], idx[1])
                    and _dummy_pair(idx[2], idx[3])):
                hit = i
                break
        if hit is None:
            continue
        rest = coeff
        for j, f in enumerate(factors):
            if j != hit:
                rest = rest * f
        out.append(rest)
    if not out:
        return S.Zero
    C = _sum_terms(out)
    return canon(C) if isinstance(C, TensExpr) else C


def _extract_ddh_deriv_signature(expr, mu, nu):
    """Coefficient of the ddh whose DERIVATIVE pair (positions 2,3) equals the
    free index pair (mu, nu) (matched by name, either order/variance). Strip
    that ddh and accumulate the rest. For the FIELD pair (positions 0,1):

      - if it is a self-traced dummy pair, resupply two fresh field indices via
        a metric (eta^{ab}) — the "X-trace reinterpretation": a term
        c·ddh(L,-L, mu,nu) is read as c·eta^{ab}·ddh(-a,-b, mu,nu), so its
        contribution to the coefficient is c·eta^{ab};
      - otherwise leave the rest as-is (the field indices were either contracted
        with the unknown X — which then survives with those free indices — or
        are extra free indices that vanish with the stripped ddh).

    This is the signature used by the traceless-T_M VERIFICATION recovery
    (missed field redef). Per the paper, the redef's effect on E_h^{mn(n)} is
    X^{ab} S_{ab}^{mn} where S = T1 + T2; only T2 reaches this signature (T1 =
    X^{mn}S^(1) keeps the ddh entirely inside the scalar S^(1), so its ddh never
    carries the free indices mu,nu on a derivative slot). In T2 the leading piece
    is a number m times ddh(-a,-b,-mu,-nu), so this extractor returns m (a
    scalar) from the operator and m·X^{ab} from E_diff; divide to recover X.
    """
    if expr == S.Zero or expr == 0:
        return S.Zero

    def _dummy_pair(a, b):
        return a.name == b.name and (a.is_up != b.is_up)

    mset = {mu.name, nu.name}
    # ONE common field-index pair (A,B), UP, shared by every contribution so
    # the per-term results can be summed (different terms otherwise free
    # differently-named dummies -> TensAdd "all tensors must have the same
    # indices"). For the operator T2 the field slots are free and vanish on
    # stripping (-> scalar m); for E_diff they are contracted with X and
    # survive (-> m X^{AB}); the trace branch resupplies them via eta^{AB}.
    A, B = fresh_indices(2)
    terms = expr.args if isinstance(expr, TensAdd) else [expr]
    out = []
    for term in terms:
        if isinstance(term, TensMul):
            coeff, factors = _decompose_tensmul(term)
        elif isinstance(term, Tensor):
            coeff, factors = S.One, [term]
        else:
            continue
        hit = None
        for i, f in enumerate(factors):
            if _get_component(f) is not ddh:
                continue
            idx = _get_indices(f)
            if len(idx) == 4 and {idx[2].name, idx[3].name} == mset:
                hit = i
                break
        if hit is None:
            continue
        fidx = _get_indices(factors[hit])
        rest = coeff
        for j, f in enumerate(factors):
            if j != hit:
                rest = rest * f
        if _dummy_pair(fidx[0], fidx[1]):
            # X-trace reinterpretation: the field pair was self-traced on the
            # ddh, so resupply the two field indices via eta^{AB}.
            rest = rest * metric(A, B)
        else:
            # Field pair was either contracted with X (E_diff -> rest has the
            # two partner indices free) or free on the operator (T2 -> they
            # vanished with the ddh, rest is scalar). Rename any freed pair to
            # the common (A,B); leave a scalar rest as-is.
            gf = (sorted(rest.get_free_indices(), key=lambda x: x.name)
                  if isinstance(rest, TensExpr) else [])
            if len(gf) == 2:
                rest = rest * metric(-gf[0], A) * metric(-gf[1], B)
            elif len(gf) != 0:
                raise RuntimeError(
                    f"_extract_ddh_deriv_signature: matched ddh term left "
                    f"{len(gf)} free indices {gf} after stripping (expected 0 "
                    f"for the operator or 2 for E_diff). Unmodeled structure.")
        rest = canon(rest) if isinstance(rest, TensExpr) else rest
        out.append(rest)
    if not out:
        return S.Zero
    C = _sum_terms(out)
    return canon(C) if isinstance(C, TensExpr) else C


class TracelessRecoveryMixin:
    """Traceless-T_M detection + field-redef recoveries, mixed into BootstrapState."""

    def _ensure_traceless_operators(self):
        """Compute, ONCE, the order-1 traceless operators reused at every order,
        and cache them on self. Idempotent (subsequent calls are no-ops).

        Caches:
          self.traceless_S1       — S^(1) = eta_cd E_h^(1)cd + Sum_i c_i E_i^(1)
                                    (needed at every order to form F = X.S^(1)
                                    in the mandatory step).
          self.traceless_m_box    — numerical coefficient of the doubly-traced
                                    ddh box signature in S^(1) (mandatory step).
          self.traceless_m_deriv  — numerical coefficient of the deriv-pair=free-
                                    indices ddh signature in T2 (verification).

        Both m's must be field-INDEPENDENT constants (v=0); `_require_constant_m`
        raises if a v-contamination term ever appears. The operators depend only
        on E[0], E[1] and the c_i (all fixed after order 1), so this runs once at
        the end of the n=1 loop (and lazily as a safety net inside the
        recoveries). No-op when T_M is not on-shell traceless.
        """
        if getattr(self, '_traceless_ops_ready', False):
            return
        self.traceless_S1 = S.Zero
        self.traceless_m_box = S.Zero
        self.traceless_m_deriv = S.Zero
        if not getattr(self, 'traceless_T_M', False):
            self._traceless_ops_ready = True
            return
        S1 = self._build_traceless_S1()
        if S1 != S.Zero:
            self.traceless_S1 = S1
            self.traceless_m_box = _require_constant_coeff(
                _extract_ddh_box_signature(S1), "ddh(L,-L,M,-M) in S^(1)")
        T2 = self._build_traceless_redef_operator()
        if T2 != S.Zero:
            self.traceless_m_deriv = _require_constant_coeff(
                _extract_ddh_deriv_signature(T2, self.mu_E, self.nu_E),
                "ddh(a,b,mu,nu) [deriv pair = free indices] in T2")
        self._traceless_ops_ready = True
        if self.verbose:
            print(f"    [traceless] cached operators: m_box = "
                  f"{self.traceless_m_box}, m_deriv = {self.traceless_m_deriv}")

    def _check_T_M_traceless(self):
        """Detect whether the zeroth-order h-EOM trace is on-shell-traceless,
        i.e., E^(0)^a_a + Σ_i c_i E_i^(0) = 0 for derivative-free c_i.

        Sets:
          self.traceless_T_M    = True/False
          self.traceless_c_i    = {field_name: c_i (tensor expr)} if True

        Off-shell-traceless (E^(0)^a_a = 0 identically) is the special case
        with all c_i = 0; we report this as traceless with an empty c_i dict.

        Decomposition strategy: try to express T = eta_{ab} E^(0)^{ab} as
        Σ_i C_i E_i^(0) via the same trace-signature matcher the orchestrator
        uses for matter EOMs. The c_i are then -C_i (T = Σ C_i E_i ⟹
        T = -Σ (-C_i) E_i, so c_i = -C_i). Each C_i must be derivative-free.
        """
        E0 = self.E.get(0, S.Zero)
        if E0 == S.Zero:
            return False
        T_trace = E0 * metric(-self.mu_E, -self.nu_E)
        if isinstance(T_trace, TensExpr):
            T_trace = canon(T_trace)
        if T_trace == S.Zero:
            # Off-shell traceless: η_{ab} E^(0)^{ab} = 0 identically.
            self.traceless_T_M = True
            self.traceless_c_i = {}
            if self.verbose:
                print(f"  Traceless-T_M check: OFF-shell traceless "
                      f"(eta_{{ab}} E^(0)^{{ab}} = 0 identically).")
            return True
        # Try on-shell: decompose T_trace against each matter EOM.
        residual = T_trace
        c_dict = {}
        for name, info in _matter_fields.items():
            rank = info.get('rank', 0)
            ddhead = info['ddfield']
            field = info['field']
            if rank == 0:
                field_positions = ()
                trace_positions = (0, 1)
            elif rank == 1:
                field_positions = (0,)
                trace_positions = (1, 2)
            else:
                continue
            C, alphas = _extract_coeff_from_trace_signature(
                residual, ddhead, field_positions, trace_positions
            )
            if C == S.Zero:
                continue
            if not _is_derivative_free(C):
                # A derivative-laden coefficient is IMPOSSIBLE for a genuine
                # 2-derivative matter Lagrangian: c_i multiplies the 2nd-deriv
                # signature dd{field}, so a derivative in c_i means the trace
                # eta_{ab} E^(0)^{ab} carries a term with > 2 derivatives. That
                # can only come from a bug in T_M / the trace-signature matcher
                # (or a Lagrangian that broke the 2-derivative limit). Fail loud
                # rather than silently mis-reporting it as "not traceless".
                raise RuntimeError(
                    f"_check_T_M_traceless: trace-decomposition coefficient for "
                    f"field {name!r} is NOT derivative-free: c_{name} = {C}. "
                    f"This implies eta_{{ab}} E^(0)^{{ab}} has a term with more "
                    f"than two derivatives, which is impossible for a "
                    f"two-derivative matter Lagrangian -- a bug in T_M or the "
                    f"trace-signature extraction.")
            if rank == 0:
                EOM_phi = euler_lagrange_scalar(self.L[0], field)
                contribution = C * EOM_phi
            else:
                EOM_phi, EOM_idx = euler_lagrange(self.L[0], field)
                EOM_renamed = _reindex_tensor(EOM_phi, EOM_idx, alphas)
                contribution = C * EOM_renamed
            if isinstance(contribution, TensExpr):
                contribution = canon(contribution)
            residual = residual - contribution
            if isinstance(residual, TensExpr):
                residual = canon(residual)
            # In T = Σ C_i E_i convention; we want T + Σ c_i E_i = 0 ⟹ c_i = -C_i.
            c_dict[name] = canon(-C) if isinstance(C, TensExpr) else -C
        if residual == S.Zero:
            self.traceless_T_M = True
            self.traceless_c_i = c_dict
            if self.verbose:
                names = list(c_dict.keys())
                print(f"  Traceless-T_M check: ON-shell traceless via matter "
                      f"EOMs {names}.")
                for name, c in c_dict.items():
                    print(f"    c_{name} = {c}")
            return True
        if self.verbose:
            print(f"  Traceless-T_M check: NOT traceless "
                  f"(residual {_count(residual)} terms after matter EOM "
                  f"decomposition).")
            print(f"    [traceless] c_i found so far: "
                  f"{ {k: str(v) for k, v in c_dict.items()} }")
            _rterms = (residual.args if isinstance(residual, TensAdd)
                       else [residual])
            for _t in _rterms[:12]:
                print(f"    [traceless] residual term: {_t}")
        return False

    def _build_traceless_S1(self):
        """Build the order-1 EOM-combination scalar
            S^(1) = eta_{cd} E_h^{(1)cd} + Sum_i c_i E_i^{(1)},
        whose order-0 analogue vanishes by the on-shell traceless condition.

        E_h^(1) is the order-1 field equation self.E[1]; E_i^(1) is the order-1
        matter EOM EL(self.L[1], field_i). The c_i contraction mirrors
        `_check_T_M_traceless`: c_i carries the EOM-side free index, and we
        rename E_i^(1)'s index to its negative so the two contract. Returns
        S.Zero if order 1 hasn't been run.
        """
        if 1 not in self.E:
            return S.Zero
        S1 = self.E[1] * metric(-self.mu_E, -self.nu_E)
        if isinstance(S1, TensExpr):
            S1 = canon(S1)
        L1 = self.L.get(1, S.Zero)
        for name, c_i in self.traceless_c_i.items():
            if c_i == S.Zero or L1 == S.Zero:
                continue
            info = _matter_fields[name]
            rank = info.get('rank', 0)
            field = info['field']
            if rank == 0:
                E_i1 = euler_lagrange_scalar(L1, field)
                if E_i1 == S.Zero:
                    continue
                contrib = c_i * E_i1
            else:
                E_i1, E_i1_idx = euler_lagrange(L1, field)
                if E_i1 == S.Zero:
                    continue
                c_extra = (list(c_i.get_free_indices())
                           if isinstance(c_i, TensExpr) else [])
                E_i1_ren = _reindex_tensor(E_i1, E_i1_idx,
                                           tuple(-idx for idx in c_extra))
                contrib = c_i * E_i1_ren
            if isinstance(contrib, TensExpr):
                contrib = canon(contrib)
            S1 = S1 + contrib
            if isinstance(S1, TensExpr):
                S1 = canon(S1)
        return S1

    def _build_traceless_redef_operator(self):
        """Build the T2 part of the missed-redef effect on E_h^{mn(n)} (the
        ddh-bearing 'extra term' the paper flags beyond X^{mn}S^(1)):

            T2_{ab}^{mn} = [ (dE_h^{mn(1)}/dh_{jk,cd}) eta_{jk}
                             + Sum_i (dE_h^{mn(0)}/dphi_{i,cd}) c_i ] h_{ab,cd}

        free indices a,b (the X-contraction pair) and mn (= mu_E, nu_E). It
        contains ddh = h_{ab,cd}. Derived by substituting the field redef into
        E_h: f_h_{jk} = f eta_{jk} into E_h^(1) and f_{phi_i} = f c_i into
        E_h^(0), keeping only the ddh part of the second derivative f_{,cd}
        (= X^{ab} h_{ab,cd}).

        Returns S.Zero if order 0/1 data is missing. Used by the verification
        recovery: the coefficient of ddh(-a,-b,-mu,-nu) in T2 is the number m
        (with v=0 in the cases seen so far), and the same signature in E_diff is
        m·X^{ab}, so X = (that)/m.
        """
        if 0 not in self.E or 1 not in self.E:
            return S.Zero
        J, K, C, D = fresh_indices(4)
        a, b = fresh_indices(2)
        # First piece: (dE_h^(1)/dddh(J,K,C,D)) . eta_{JK} . ddh(a,b,-C,-D).
        T2 = S.Zero
        dE1 = jet_derivative(self.E[1], ddh, (J, K, C, D))
        if dE1 != S.Zero:
            dE1_tr = canon(dE1 * metric(-J, -K))
            if dE1_tr != S.Zero:
                T2 = canon(dE1_tr * ddh(a, b, -C, -D))
        # Second piece: Sum_i (dE_h^(0)/dddphi_i(...,C,D)) . c_i . ddh(a,b,-C,-D).
        # E_h^(0) carries ddphi_i for any matter field whose T_M has second
        # derivatives (the conformal scalar's improvement; and, crucially, the
        # upstairs-vector T_M genuinely contains ddV via the metric/Christoffel
        # variation of grad V). The c_i contraction follows the field rank:
        # scalar c_i is a number, rank-1 c_i carries one vector index that
        # contracts the ddV field slot.
        for name, c_i in self.traceless_c_i.items():
            if c_i == S.Zero:
                continue
            info = _matter_fields[name]
            rank = info.get('rank', 0)
            ddfield = info['ddfield']
            cc, dd = fresh_indices(2)
            if rank == 0:
                dE0 = jet_derivative(self.E[0], ddfield, (cc, dd))
                if dE0 == S.Zero:
                    continue
                contrib = canon(dE0 * c_i * ddh(a, b, -cc, -dd))
            elif rank == 1:
                sg, = fresh_indices(1)
                dE0 = jet_derivative(self.E[0], ddfield, (sg, cc, dd))
                if dE0 == S.Zero:
                    continue
                # Contract the vector index sg with c_i's free index.
                c_free = (list(c_i.get_free_indices())
                          if isinstance(c_i, TensExpr) else [])
                if len(c_free) != 1:
                    raise RuntimeError(
                        f"_build_traceless_redef_operator: rank-1 c_{name} has "
                        f"{len(c_free)} free indices (expected 1).")
                c_ren = _reindex_tensor(c_i, (c_free[0],), (-sg,))
                contrib = canon(dE0 * c_ren * ddh(a, b, -cc, -dd))
            else:
                raise NotImplementedError(
                    f"_build_traceless_redef_operator: matter field {name!r} has "
                    f"rank {rank}; only scalar (0) and vector (1) c_i are "
                    f"implemented for the T2 second piece.")
            T2 = canon(T2 + contrib) if T2 != S.Zero else contrib
        return T2

    def _recover_missed_traceless_redef(self, n, E_diff):
        """Recover the order-(n-1) traceless-shape redef that was invisible at
        its own order and now shows up as ddh in E_diff, apply it to L_ref, and
        return the recomputed E_diff. Returns None if nothing is recoverable
        (caller then falls through to the normal decomposition path).

        Mechanism (paper-corrected): the redef's effect on E_h^{mn(n)} is
        X^{ab} S_{ab}^{mn}, where S = T1 + T2 and
            T1 = X^{mn} S^(1)        (S^(1) = eta_cd E_h^{cd(1)} + Sum_i c_i E_i^{(1)})
            T2 = [ (dE_h^{mn(1)}/dh_{jk,cd}) eta_jk
                   + Sum_i (dE_h^{mn(0)}/dphi_{i,cd}) c_i ] h_{ab,cd}   (= _build_traceless_redef_operator).
        T1 (the old template) is NOT enough — T2 carries additional ddh. We
        recover X via a signature T1 CANNOT reach: ddh whose DERIVATIVE pair
        equals the free indices (mu_E, nu_E). In T1 the ddh sits inside the
        scalar S^(1) so its derivative slots are never the free indices; only
        T2 hits this signature, with leading coefficient a number m. Thus
            m = self.traceless_m_deriv  (cached deriv-signature coeff of T2; a
                                        constant, v=0 enforced at precompute)
            P = _extract_ddh_deriv_signature(E_diff, mu_E, nu_E)  (= m X^{ab})
            X^{ab} = P / m.
        The missed order-n redefs are then
            f_h^{(n)}_{cd} = (1/n) h_{ab} X^{ab} eta_{cd},
            f_i^{(n)}      = (1/n) h_{ab} X^{ab} c_i.
        Apply (as the order-(n-1) verification's redef) and recompute E_diff.

        Validated end-to-end via a synthetic round-trip (X_test -> E_diff =
        X_test.T2 -> recover): both a general X (h^{ab}) and a pure-trace X
        (eta^{ab}, exercising the eta reinterpretation in the extractor) come
        back exactly. A v_{ab}^{cd} h_{cd,mn} contamination term (never seen in
        practice) would make m field-dependent; `_require_constant_m` then raises
        at precompute ("work harder"). Rollback of a *non-empty* order-(n-1)
        redef set is handled below (snapshot + augment).
        """
        if n < 2:
            return None  # X is order n-1 >= 1; the loophole surfaces at n >= 2.
        self._ensure_traceless_operators()
        m = self.traceless_m_deriv  # cached constant; v=0 checked at precompute
        if m == S.Zero:
            return None
        P = _extract_ddh_deriv_signature(E_diff, self.mu_E, self.nu_E)  # = m X^{ab}
        if P == S.Zero:
            return None
        X = canon(P / m)
        if X == S.Zero:
            return None
        # P came from the ddh FIELD pair, so X's two free indices are fresh
        # dummies (and may be either variance). Remap them to the reserved
        # (mu_E, nu_E) UP, so the downstream hX = (1/n) X h(-mu,-nu) contracts
        # exactly as in the box-method code. (X is symmetric, so which free
        # index maps to mu_E vs nu_E is immaterial.)
        if isinstance(X, TensExpr):
            xf = list(X.get_free_indices())
            if len(xf) != 2:
                raise RuntimeError(
                    f"missed-redef recovery at n={n}: recovered X has "
                    f"{len(xf)} free indices (expected 2). Extraction is off.")
            X = canon(X * metric(-xf[0], self.mu_E) * metric(-xf[1], self.nu_E))
        self.recovered_traceless_X[n] = X

        # (1/n) h_{ab} X^{ab}: contract X's (mu_E, nu_E) with h_{mu_E nu_E}.
        hX = Rational(1, n) * X * h(-self.mu_E, -self.nu_E)
        if isinstance(hX, TensExpr):
            hX = canon(hX)

        redefs = {'h': None, 'phi': {}}
        cc, dd = fresh_indices(2)
        f_h = hX * metric(-cc, -dd)
        if isinstance(f_h, TensExpr):
            f_h = canon(f_h)
        f_h_free = (tuple(f_h.get_free_indices())
                    if isinstance(f_h, TensExpr) else ())
        redefs['h'] = (f_h, f_h_free)
        for name, c_i in self.traceless_c_i.items():
            if c_i == S.Zero:
                continue
            f_i = hX * c_i
            if isinstance(f_i, TensExpr):
                f_i = canon(f_i)
            f_i_free = (tuple(f_i.get_free_indices())
                        if isinstance(f_i, TensExpr) else ())
            redefs['phi'][name] = (f_i, f_i_free)

        if self.verbose:
            print(f"    [traceless] recovered missed X^(ab) via ddh signature; "
                  f"applying f^({n}) redef, recomputing E_diff")
            print(f"      X^(ab) = {X}")

        # Augment order-(n-1)'s normal redef (stored as recovered_redefs[n]
        # when _verify_vs_L_ref(n-1) ran) with this missed contribution. If
        # that order was NOT clean, applying `redefs` on top would double-count
        # the chain-rule cross terms; instead roll L_ref back to the snapshot
        # taken just before that normal redef, sum the two, and re-apply once.
        prev = self.recovered_redefs.get(n)
        prev_nonempty = bool(prev) and (
            prev.get('h') is not None or prev.get('phi'))
        if prev_nonempty and (n - 1) in self.L_ref_history:
            if self.verbose:
                print(f"    [traceless] order-{n-1} had a normal redef; rolling "
                      f"L_ref back and augmenting it with the missed redef")
            self.L_ref = dict(self.L_ref_history[n - 1])
            redefs = self._add_redefs(prev, redefs)
        self._apply_field_redefs_to_L_ref(redefs, n - 1)
        # Record the (possibly augmented) order-(n-1) redef for a later rollback.
        self.recovered_redefs[n] = redefs

        # Applying the missed redef rewrites L_ref^(n) (= order-(n-1)'s
        # verification target). Confirm it did NOT disturb that lower order:
        # EL(L_ref^(n)) must still equal E^(n-1). Since the missed redef is
        # traceless-shape (invisible at order n-1), its EL contribution there
        # must vanish; a nonzero diff means the recovery corrupted order n-1.
        if (n - 1) in self.E:
            L_prev = self.L_ref.get(n, S.Zero)
            if L_prev == S.Zero:
                E_prev = S.Zero
            else:
                E_prev, E_prev_idx = euler_lagrange(L_prev, h)
                E_prev = _reindex_tensor(E_prev, E_prev_idx,
                                         (self.mu_E, self.nu_E))
            prev_diff = self.E[n - 1] - E_prev
            if isinstance(prev_diff, TensExpr):
                prev_diff = canon(prev_diff)
            if prev_diff != S.Zero:
                raise RuntimeError(
                    f"Traceless recovery at n={n} disturbed order-{n-1} "
                    f"verification: EL(L_ref^({n})) - E^({n-1}) = "
                    f"{_count(prev_diff)} terms (must be 0 -- the missed redef "
                    f"is traceless-shape and should be invisible at order "
                    f"{n-1}).")
            if self.verbose:
                print(f"    [traceless] order-{n-1} verification still holds "
                      f"after the missed redef (EL(L_ref^({n})) == E^({n-1}))")

        L_r = self.L_ref.get(n + 1, S.Zero)
        if L_r == S.Zero:
            E_r = S.Zero
        else:
            E_r, E_r_idx = euler_lagrange(L_r, h)
            E_r = _reindex_tensor(E_r, E_r_idx, (self.mu_E, self.nu_E))
        new_E_diff = self.E[n] - E_r
        if isinstance(new_E_diff, TensExpr):
            new_E_diff = canon(new_E_diff)
        return new_E_diff

    def _recover_traceless_mandatory_eom(self, n, E, Z, h_indices):
        """Mandatory-EOM-step traceless path (DEVELOPMENT_STATUS item 0).

        When T_M is on-shell traceless, the H2 violation Z at order n can
        carry ddh that is NOT decomposable as Y · E^(0) (E^(0) = κ T_M has no
        ddh). It is the order-n surfacing of a NEW mandatory traceless-shape
        EOM coefficient X^{ab(n-1)} (order n-1) whose visible contribution at
        this order is

            F^{ab} = X^{ab(n-1)} · S^(1),
            S^(1)  = η_{cd} E_h^{(1)cd} + Σ_i c_i E_i^{(1)}   (= _build_traceless_S1),

        added to E_h^(n). Its H2 violation cancels Z's ddh.

        Recovery (spec lines 181-200): the ddh in H2(X·S^(1)) arises ONLY from
        2(∂X/∂h)_anti · S^(1) — the E^(1) pieces are order 1 in h, so their
        ∂/∂h and ∂/∂dh are matter-only (no ddh). So the doubly-traced ddh box
        signature isolates X:

            P  = box-signature coeff of Z            (= Y · m)
            m  = box-signature coeff of S^(1)         (self.traceless_m_box, a
                                                      constant; v=0 is enforced
                                                      by _require_constant_m)
            Y  = P / m
            X^{ab(n-1)} = -1/(2n) · Y^{ab}_{AB} · h^{AB}

        The -1/(2n) sign is chosen so H2(X·S^(1)) cancels Z's ddh; the
        denominator is n (not n+1) because this X is order n-1, not n.

        Then F = X·S^(1) is added to E, Z gains H2(F) (must become ddh-free),
        and X is folded — with η_{cd} / c_i factors — into eom_terms_h[n-1] /
        eom_terms_matter[n-1] so step 2 of higher orders carries it forward.

        Returns (E_new, Z_new) with Z_new ddh-free, or None if there is
        nothing to recover (caller then proceeds with the normal decomposition).
        """
        if n < 2:
            return None  # X is order n-1 >= 1; ddh surfaces at n >= 2.
        self._ensure_traceless_operators()
        S1 = self.traceless_S1
        m = self.traceless_m_box  # cached constant; v=0 checked at precompute
        if S1 == S.Zero or m == S.Zero:
            return None
        P = _extract_ddh_box_signature(Z)  # = Y . m
        if P == S.Zero:
            return None
        Y = canon(P / m) if isinstance(P, TensExpr) else P / m
        if Y == S.Zero:
            return None
        A_idx, B_idx = h_indices
        # X^{ab(n-1)} = -1/(2n) Y^{ab}_{AB} h^{AB} (contract the box-derivative
        # pair returned by compute_h2_violation).
        X = Rational(-1, 2 * n) * Y * h(-A_idx, -B_idx)
        if isinstance(X, TensExpr):
            X = canon(X)
        if X == S.Zero:
            return None
        self.recovered_traceless_X[n] = X
        if self.verbose:
            print(f"    [traceless step-3] recovered new mandatory "
                  f"X^(ab,{n-1}) via ddh signature: {_format_breakdown(X)}")
            print(f"      X = {X}")

        # Visible contribution this order: F = X · S^(1).
        F = X * S1
        if isinstance(F, TensExpr):
            F = canon(F)

        # Full H2 of the product (compute_h2_violation is the non-truncated H2
        # of any expression — no separate utility needed). Adding F to E adds
        # H2(F) to Z; the -1/(2n) sign makes that cancel Z's ddh.
        # compute_h2_violation generates its OWN fresh box-derivative pair, so
        # reindex it onto Z's (h_indices) before the TensAdd — otherwise the
        # two differ in dummy NAME and sympy rejects the sum.
        Z_prod, prod_idx = compute_h2_violation(F, (self.mu_E, self.nu_E))
        if Z_prod != S.Zero:
            Z_prod = _reindex_tensor(Z_prod, prod_idx, h_indices)
            Z_new = canon(Z + Z_prod)
        else:
            Z_new = Z

        if ddh in set(get_tensors_in_expr(Z_new)):
            n_ddh = len([t for t in (Z_new.args if isinstance(Z_new, TensAdd)
                                     else [Z_new])
                         if ddh in set(get_tensors_in_expr(t))])
            raise RuntimeError(
                f"Traceless step-3 recovery at n={n}: ddh still present in Z "
                f"after adding X·S^(1) ({n_ddh} ddh terms). The recovered X "
                f"(sign/factor) or the ddh-box extraction is off.")

        E_new = canon(E + F) if F != S.Zero else E

        # Store X for step-2 carryover at higher orders: X_h^{(n-1)} += X η_{cd},
        # X_i^{(n-1)} += X c_i (the traceless shape — no special label needed).
        cc, dd = fresh_indices(2)
        Xh = X * metric(-cc, -dd)
        if isinstance(Xh, TensExpr):
            Xh = canon(Xh)
        self.eom_terms_h[n - 1] = self._merge_eom_coeff(
            self.eom_terms_h.get(n - 1, S.Zero), Xh)
        for name, c_i in self.traceless_c_i.items():
            if c_i == S.Zero:
                continue
            Xi = X * c_i
            if isinstance(Xi, TensExpr):
                Xi = canon(Xi)
            phi_dict = self.eom_terms_matter.setdefault(n - 1, {})
            phi_dict[name] = self._merge_eom_coeff(
                phi_dict.get(name, S.Zero), Xi)

        if self.verbose:
            print(f"    [traceless step-3] added X·S^(1) to E "
                  f"({_format_breakdown(F)}); Z now ddh-free "
                  f"({_format_breakdown(Z_new)}); stored X^(ab,{n-1}) for carryover")
        return E_new, Z_new
