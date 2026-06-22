"""Validate parallel_apply_linear == whole, on a FORK node, for the builders the
original bench (bench_parallel_builder.py, total_derivative) did NOT cover:

  [H2]    compute_h2_violation  -- INDEX-RETURNING, with heavy cross-chunk
          cancellation. This is the exact builder that produced the Z != 0 fork
          bug (combine-merge), now fixed by canon-at-merge + substitute-only
          reindex. THE regression test that gap let slip through.
  [SUBST] _substitute_field      -- the newly-parallelized field-redef path (bare).

Fast (targeted builders, not a full loop). On Zeus it exercises real fork; on
Windows parallel_apply_linear falls back to F(whole) so it trivially passes
(checks imports/logic only). The fork path is what matters -- run on Zeus.

Run:  GRB_N_WORKERS unused here (K is passed directly); just be on a fork node:
      python -u tests/bench_parallel_correctness.py [EH_ORDER] [K]
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sympy.tensor.tensor import TensAdd
from bootstrap import jet
from bootstrap.tensor_algebra import canon, h, dh, ddh, fresh_indices
from bootstrap.jet import parallel_apply_linear, _reindex_free
from bootstrap.covariant import einstein_hilbert_lagrangian_order
from bootstrap.energy_momentum import hilbert_energy_momentum
from bootstrap.helmholtz import compute_h2_violation
from bootstrap.bootstrap_loop import _substitute_field, _build_deriv_cache


def n(e):
    return len(e.args) if isinstance(e, TensAdd) else (0 if e == 0 else 1)


def eq_indexed(aw, iaw, bw, ibw):
    """Compare two index-returning results up to free-index relabeling."""
    if iaw is not None and ibw is not None:
        bw = _reindex_free(bw, ibw, iaw)
    return canon(aw + (-1) * bw) == 0


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    K = int(sys.argv[2]) if len(sys.argv) > 2 else max(2, min(os.cpu_count() or 4, 8))
    print(f"EH_ORDER={N}  K={K}  fork={jet._FORK_OK}", flush=True)

    # [H2] index-returning builder with cross-chunk cancellation (the bug case).
    L = einstein_hilbert_lagrangian_order(N)
    T, tidx = hilbert_energy_momentum(L)
    E = canon(T)
    nE = n(E)
    chunk = max(1, math.ceil(nE / K))
    F_h2 = lambda x: compute_h2_violation(x, tidx)
    t = time.time(); Zw, zw = F_h2(E); tw = time.time() - t
    t = time.time(); Zp, zp = parallel_apply_linear(F_h2, E, K, chunk_size=chunk); tp = time.time() - t
    ok_h2 = eq_indexed(Zw, zw, Zp, zp)
    print(f"[H2]    E={nE} terms  whole {tw:6.1f}s -> {n(Zw):4d}  "
          f"parallel {tp:6.1f}s -> {n(Zp):4d}  == whole: {ok_h2}", flush=True)

    # [SUBST] field-redef substitution (bare, the newly-parallelized path).
    a, b, c = fresh_indices(3)
    f_expr = canon(h(a, c) * h(b, -c)); f_idx = (a, b)
    hinfo = {'field': h, 'dfield': dh, 'ddfield': ddh, 'rank': 2, 'name': 'h'}
    Lr = canon(einstein_hilbert_lagrangian_order(2) + einstein_hilbert_lagrangian_order(3))
    cache = _build_deriv_cache(hinfo, f_expr, f_idx)
    subst = lambda ch: _substitute_field(ch, hinfo, f_expr, f_idx, target_order=5, deriv_cache=cache)
    nL = n(Lr); chunk2 = max(1, math.ceil(nL / K))
    t = time.time(); Sw = subst(Lr); tsw = time.time() - t
    t = time.time(); Sp = parallel_apply_linear(subst, Lr, K, chunk_size=chunk2); tsp = time.time() - t
    ok_sub = (canon(Sw + (-1) * Sp) == 0)
    print(f"[SUBST] L={nL} terms  whole {tsw:6.1f}s -> {n(Sw):4d}  "
          f"parallel {tsp:6.1f}s -> {n(Sp):4d}  == whole: {ok_sub}", flush=True)

    if ok_h2 and ok_sub:
        print("\n*** parallel == whole for compute_h2_violation AND field-redef subst ***", flush=True)
    else:
        print("\n*** CORRECTNESS FAIL ***", flush=True)
        sys.exit(1)
