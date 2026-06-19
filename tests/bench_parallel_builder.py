"""Builder-chunk parallelism: validate + bench the PRODUCTION parallel_apply_linear.

Parallelize a whole linear builder F over input chunks: each forked worker runs
F(chunk) (incl. its first-canons), parent merges with combine_canonical. F linear
=> sum_j F(chunk_j) == F(expr). This attacks the per-term work INSIDE the builders
(first-canons), where high-order time lives after combine.

  [whole]    F(L)                                   -- baseline.
  [serial]   manual chunk + combine == whole        -- validates the chunking
             logic everywhere (no processes; runs on Windows).
  [parallel] jet.parallel_apply_linear(F, L, K)      -- on Linux/Zeus forks K
             workers (COW + index ranges, no chunk pickling); on Windows falls
             back to F(L). Validates == whole and measures the [fork/COW] speedup.

Run: python -u tests/bench_parallel_builder.py [N] [K]
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sympy import S
from sympy.tensor.tensor import TensAdd, TensExpr
from bootstrap import jet
from bootstrap.tensor_algebra import fresh_indices, combine_canonical, set_index_counter, get_index_counter
from bootstrap.jet import total_derivative, parallel_apply_linear, _sum_terms
from bootstrap.covariant import einstein_hilbert_lagrangian_order

_STRIDE = jet._PARALLEL_STRIDE


def is_zero(e):
    return e is S.Zero or e == 0


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    K = int(sys.argv[2]) if len(sys.argv) > 2 else min(os.cpu_count() or 4, 8)

    L = einstein_hilbert_lagrangian_order(N)
    nL = len(L.args) if isinstance(L, TensAdd) else 1
    tau, = fresh_indices(1)
    F = lambda x: total_derivative(x, -tau)
    chunk = max(1, math.ceil(nL / K))
    print(f"N={N} K={K} fork={jet._FORK_OK} L_EH^({N})={nL} terms chunk={chunk} "
          f"cpu={os.cpu_count()}", flush=True)

    t = time.time(); whole = F(L); T_whole = time.time() - t
    nout = len(whole.args) if isinstance(whole, TensAdd) else 1
    print(f"[whole]    F(L)                       {T_whole:7.2f}s -> {nout} terms", flush=True)

    # Serial chunk + combine (validates the chunking logic, no processes).
    args = list(L.args)
    base = get_index_counter() + 1
    t = time.time()
    parts = []
    for w, lo in enumerate(range(0, nL, chunk)):
        set_index_counter(base + w * _STRIDE)
        sub = args[lo:lo + chunk]
        parts.append(F(_sum_terms(sub)))
    merged_s = combine_canonical(_sum_terms([p for p in parts if not is_zero(p)]))
    T_serial = time.time() - t
    ok_s = (merged_s == whole)
    print(f"[serial]   chunk+combine              {T_serial:7.2f}s   == whole: {ok_s}", flush=True)

    # Production parallel_apply_linear (fork on Zeus, fallback on Windows).
    t = time.time(); par = parallel_apply_linear(F, L, K, chunk_size=chunk); T_par = time.time() - t
    ok_p = (par == whole)
    tag = "fork/COW" if jet._FORK_OK else "fallback"
    print(f"[parallel] parallel_apply_linear ({tag}) {T_par:7.2f}s   == whole: {ok_p}", flush=True)
    if jet._FORK_OK:
        print(f"\n=== [fork/COW] speedup vs whole: {T_whole / T_par:.2f}x "
              f"(production parallel_apply_linear, K={K}) ===", flush=True)

    if not (ok_s and ok_p):
        print("\n*** CORRECTNESS FAIL ***", flush=True)
        sys.exit(1)
    print("\n*** parallel_apply_linear == F(whole) confirmed ***", flush=True)
