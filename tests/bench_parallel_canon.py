"""Benchmark + correctness check for process-parallel canon.

Tracked (not scratch) because hpc_suite/submit_parallel_spike.sh runs it on
Zeus to get the production-representative [fork/COW] speedup that the Windows
dev box cannot measure.

The crux: our canon -> canon_bp on the WHOLE TensAdd, with no already-canonical
fast path. So naive chunk-parallel canon would re-pay Butler-Portugal serially
on merge and win nothing (the trap that sank serial chunking). This spike tests
the only thing that can rescue parallelism:

  combine_canonical(partials): collect ALREADY-canonical partial results via
  TensAdd(*).doit() (+ the symbolic-d coeff pass), WITHOUT calling canon_bp
  again. Feasible because canon_bp assigns deterministic per-term canonical
  dummy names, so identical terms from different workers already match.

Measures three numbers on a heavy, high-collapse TensAdd:
  T_whole         = canon(whole)                       (baseline)
  T_serial_chunk  = sum(canon(chunk)) + combine        (proves merge != re-BP
                                                         if ~= T_whole, not 2x)
  T_parallel      = ProcessPool canon(chunks) + combine (the actual win)
and asserts merged == whole in all cases (structural: canon(merged - whole)==0).

Default symbolic d, only gravity heads (h/dh/ddh) -> workers need no field
registration; the picklable task is the module-level `canon` itself.
Run from repo root: python -u tests/bench_parallel_canon.py [R] [K]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import S
from sympy.tensor.tensor import TensAdd
from bootstrap.tensor_algebra import h, dh, ddh, fresh_indices, canon
from bootstrap.tensor_algebra import _simplify_d_coeffs


def combine_canonical(partials):
    """Merge already-canonical partial results WITHOUT re-running canon_bp.

    Each partial must be the output of canon() (so every term is in canonical
    BP form with deterministic dummy names). We flatten, collect like terms via
    TensAdd.doit() (collection only, no BP), then run the gated symbolic-d coeff
    simplification once. This is the candidate production merge primitive.
    """
    terms = []
    for p in partials:
        if p == S.Zero or p == 0:
            continue
        if isinstance(p, TensAdd):
            terms.extend(p.args)
        else:
            terms.append(p)
    if not terms:
        return S.Zero
    result = terms[0] if len(terms) == 1 else TensAdd(*terms).doit()
    return _simplify_d_coeffs(result)


# --- heavy, high-rank templates (distinct canonical structures) -------------
def _A():
    a, b, c, d = fresh_indices(4)
    return h(a, b) * h(-a, -b) * h(c, d) * h(-c, -d)

def _B():
    a, b, c, d = fresh_indices(4)
    return h(a, b) * h(-b, c) * h(-c, d) * h(-d, -a)

def _C():
    a, b, c, d, e, f = fresh_indices(6)
    return dh(a, b, c) * dh(-a, -b, -c) * h(d, e) * h(-d, -e) * h(f, -f)

def _D():
    a, b, c, d, e = fresh_indices(5)
    return ddh(a, b, c, -c) * h(-a, d) * h(-b, -d) * h(e, -e)

def _E():
    a, b, c, d, e, f = fresh_indices(6)
    return dh(a, b, c) * dh(-d, -b, -c) * h(-a, d) * h(e, f) * h(-e, -f)

def _F():
    a, b, c, d = fresh_indices(4)
    return ddh(a, -a, b, c) * h(-b, -c) * h(d, -d)

TEMPLATES = [_A, _B, _C, _D, _E, _F]


def build_expr(R):
    """6*R pre-canon terms: each template instantiated R times with fresh
    dummies (so canon must BP each, then collapse the R copies per template)."""
    terms = []
    for _ in range(R):
        for t in TEMPLATES:
            terms.append(t())
    return TensAdd(*terms)


def chunks(args, k):
    n = len(args)
    size = (n + k - 1) // k
    return [TensAdd(*args[i:i + size]) for i in range(0, n, size)]


# --- fork-optimized path (Linux/Zeus): workers inherit the term list via COW
# and receive only an (lo, hi) index range, so the chunk is NEVER pickled to
# the worker. Only the (small) canon'd result is pickled back. This models the
# production-optimal dispatch that Windows spawn cannot do.
_G = {}  # populated in parent BEFORE fork; inherited copy-on-write by children

def _canon_range(rng):
    lo, hi = rng
    sub = _G['args'][lo:hi]
    return canon(TensAdd(*sub) if len(sub) > 1 else sub[0])

def _ranges(n, k):
    size = (n + k - 1) // k
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def is_zero(expr):
    return expr == S.Zero or expr == 0


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    K = int(sys.argv[2]) if len(sys.argv) > 2 else min(os.cpu_count() or 4, 8)
    print(f"R={R} (-> {6*R} pre-canon terms), K={K} workers, "
          f"cpu_count={os.cpu_count()}", flush=True)

    expr = build_expr(R)
    print(f"built {len(expr.args)} terms", flush=True)

    # Baseline: canon the whole thing.
    t = time.time()
    whole = canon(expr)
    T_whole = time.time() - t
    nout = len(whole.args) if isinstance(whole, TensAdd) else 1
    print(f"\n[whole]        canon(expr)                 {T_whole:7.2f}s -> {nout} terms", flush=True)

    parts = chunks(expr.args, K)
    print(f"split into {len(parts)} chunks of ~{len(parts[0].args)} terms", flush=True)

    # Serial-chunk + BP-free merge: isolates whether combine re-pays BP.
    t = time.time()
    partials = [canon(c) for c in parts]
    t_canon_serial = time.time() - t
    t = time.time()
    merged_s = combine_canonical(partials)
    t_merge_s = time.time() - t
    T_serial_chunk = t_canon_serial + t_merge_s
    diff_s = canon(TensAdd(merged_s, -whole)) if not is_zero(merged_s) else whole
    ok_s = is_zero(diff_s)
    print(f"[serial chunk] canon(chunks)+combine        {T_serial_chunk:7.2f}s "
          f"(canon {t_canon_serial:.2f} + merge {t_merge_s:.2f})  "
          f"merged==whole: {ok_s}", flush=True)

    # Parallel-chunk + BP-free merge: the actual measurement.
    t = time.time()
    with ProcessPoolExecutor(max_workers=K) as ex:
        partials_p = list(ex.map(canon, parts))
    t_canon_par = time.time() - t
    t = time.time()
    merged_p = combine_canonical(partials_p)
    t_merge_p = time.time() - t
    T_parallel = t_canon_par + t_merge_p
    diff_p = canon(TensAdd(merged_p, -whole)) if not is_zero(merged_p) else whole
    ok_p = is_zero(diff_p)
    print(f"[parallel]     pool canon(chunks)+combine    {T_parallel:7.2f}s "
          f"(canon {t_canon_par:.2f} + merge {t_merge_p:.2f})  "
          f"merged==whole: {ok_p}", flush=True)

    print(f"\n[map/pickle]   speedup vs whole: {T_whole/T_parallel:.2f}x  "
          f"(merge re-pays BP? serial-chunk/whole = {T_serial_chunk/T_whole:.2f}, "
          f"~1.0 = no)", flush=True)

    # Fork-optimized path (Linux only): no chunk pickling (COW + index ranges).
    # This is the production-representative measurement; Windows spawn skips it.
    import multiprocessing as mp
    if 'fork' in mp.get_all_start_methods():
        ctx = mp.get_context('fork')
        _G['args'] = list(expr.args)              # set BEFORE fork -> inherited COW
        rngs = _ranges(len(_G['args']), K)
        t = time.time()
        with ctx.Pool(processes=K) as pool:
            partials_f = pool.map(_canon_range, rngs)
        t_canon_fork = time.time() - t
        t = time.time()
        merged_f = combine_canonical(partials_f)
        t_merge_f = time.time() - t
        T_fork = t_canon_fork + t_merge_f
        diff_f = canon(TensAdd(merged_f, -whole)) if not is_zero(merged_f) else whole
        ok_f = is_zero(diff_f)
        print(f"[fork/COW]     pool canon(ranges)+combine   {T_fork:7.2f}s "
              f"(canon {t_canon_fork:.2f} + merge {t_merge_f:.2f})  "
              f"merged==whole: {ok_f}", flush=True)
        print(f"[fork/COW]     speedup vs whole: {T_whole/T_fork:.2f}x  "
              f"(THIS is the production-representative number)", flush=True)
    else:
        ok_f = True
        print("[fork/COW]     skipped (no fork start method -- Windows). "
              "Run on Zeus/Linux for the production-representative number.", flush=True)

    if not (ok_s and ok_p and ok_f):
        print("*** CORRECTNESS FAIL: BP-free merge != full canon ***", flush=True)
        sys.exit(1)
    print("\n*** combine_canonical == canon confirmed (all paths) ***", flush=True)
