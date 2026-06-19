"""De-risking spike for BUILDER-chunk parallelism (the next parallelization step).

bench_parallel_canon proved fork/COW parallelizes a single big canon. But after
the combine_canonical rollout, the bulk of high-order time is the per-term work
INSIDE the linear builders (first-canons + the symbolic-d cancel), not the
recombination canons. So the real lever is parallelizing a whole linear builder
F over its input chunks: each worker runs F(chunk) -- doing that chunk's
first-canons -- and the parent merges with combine_canonical. F linear =>
sum_i F(chunk_i) == F(sum_i chunk_i).

This benches it on total_derivative (recursive, canon-heavy, linear). It:
  * validates parallel == whole STRUCTURALLY (serial-chunk path, runs anywhere),
  * on Linux, measures the [fork/COW] speedup (COW + index ranges, no chunk
    pickling) -- the production-representative number Windows can't produce.

Workers get DISJOINT fresh-index ranges via set_index_counter, so any fresh
indices F allocates internally can't collide across workers; canon normalizes
dummies per term, so the per-chunk results combine cleanly.

Run: python -u tests/bench_parallel_builder.py [N] [K]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sympy import S
from sympy.tensor.tensor import TensAdd, TensExpr
from bootstrap.tensor_algebra import (
    fresh_indices, canon, combine_canonical, get_index_counter, set_index_counter,
)
from bootstrap.jet import total_derivative, _sum_terms
from bootstrap.covariant import einstein_hilbert_lagrangian_order

# Worker-disjoint index stride: each worker offsets its fresh-index counter by
# widx*STRIDE so F's internal allocations never overlap across workers.
_STRIDE = 10 ** 7
_G = {}  # populated in parent BEFORE fork; inherited copy-on-write by children


def _apply_chunk(spec):
    """Worker: F(sum of L.args[lo:hi]) with a disjoint fresh-index range."""
    lo, hi, base, widx = spec
    set_index_counter(base + widx * _STRIDE)
    sub = _G['args'][lo:hi]
    chunk = TensAdd(*sub) if len(sub) > 1 else sub[0]
    return _G['F'](chunk)


def is_zero(e):
    return e is S.Zero or e == 0


def _ranges(n, k):
    size = (n + k - 1) // k
    return [(i, min(i + size, n)) for i in range(0, n, size)]


if __name__ == '__main__':
    import multiprocessing as mp
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    K = int(sys.argv[2]) if len(sys.argv) > 2 else min(os.cpu_count() or 4, 8)

    L = einstein_hilbert_lagrangian_order(N)
    nL = len(L.args) if isinstance(L, TensAdd) else 1
    tau, = fresh_indices(1)
    F = lambda x: total_derivative(x, -tau)   # bare linear builder (shares -tau)
    print(f"N={N}, K={K}, L_EH^({N})={nL} terms, cpu_count={os.cpu_count()}",
          flush=True)

    # Baseline: F over the whole input.
    t = time.time(); whole = F(L); T_whole = time.time() - t
    nout = len(whole.args) if isinstance(whole, TensAdd) else 1
    print(f"[whole]   F(L)                    {T_whole:7.2f}s -> {nout} terms", flush=True)

    args = list(L.args)
    rngs = _ranges(len(args), K)

    # Serial-chunk + combine: validates correctness everywhere (no processes).
    base = get_index_counter() + 1
    t = time.time()
    parts = []
    for i, (lo, hi) in enumerate(rngs):
        set_index_counter(base + i * _STRIDE)
        sub = args[lo:hi]
        parts.append(F(TensAdd(*sub) if len(sub) > 1 else sub[0]))
    merged_s = combine_canonical(_sum_terms([p for p in parts if not is_zero(p)]))
    T_serial = time.time() - t
    ok_s = (merged_s == whole)
    print(f"[serial]  chunks+combine          {T_serial:7.2f}s   == whole: {ok_s}", flush=True)

    # Fork/COW path (Linux): index ranges only, no chunk pickling.
    if 'fork' in mp.get_all_start_methods():
        ctx = mp.get_context('fork')
        _G['args'] = args
        _G['F'] = F
        base = get_index_counter() + 1
        specs = [(lo, hi, base, i) for i, (lo, hi) in enumerate(rngs)]
        t = time.time()
        with ctx.Pool(processes=K) as pool:
            parts_f = pool.map(_apply_chunk, specs)
        merged_f = combine_canonical(_sum_terms([p for p in parts_f if not is_zero(p)]))
        T_fork = time.time() - t
        ok_f = (merged_f == whole)
        print(f"[fork/COW] pool F(chunks)+combine {T_fork:7.2f}s   == whole: {ok_f}", flush=True)
        print(f"\n=== [fork/COW] speedup vs whole: {T_whole / T_fork:.2f}x "
              f"(production-representative) ===", flush=True)
    else:
        ok_f = True
        print("[fork/COW] skipped (no fork -- Windows). Run on Zeus/Linux.", flush=True)

    if not (ok_s and ok_f):
        print("\n*** CORRECTNESS FAIL: builder-chunk != whole ***", flush=True)
        sys.exit(1)
    print("\n*** builder-chunk (F over chunks + combine) == F(whole) confirmed ***",
          flush=True)
