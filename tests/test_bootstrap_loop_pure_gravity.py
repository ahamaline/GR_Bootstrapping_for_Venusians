# Drives the standardized bootstrap loop (BootstrapState) at orders 0..n_max
# in the pure-gravity / Hilbert case, asserting:
#   (a) EL(L^{(n+1)}) == E^{(n)}                          (self-consistency)
#   (b) L^{(n+1)} matches L_ref^{(n+1)} (or EL-equivalent) (verification cycle)
#
# This is the standardized entry point; the older
# test_superpotential_pure_gravity{,_n3,_n4}.py scripts remain as direct
# regression coverage of the underlying primitives.

import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.bootstrap_loop import BootstrapState

N_MAX = 3  # bumped via command-line override below

if len(sys.argv) > 1:
    try:
        N_MAX = int(sys.argv[1])
    except ValueError:
        sys.exit(f"usage: {sys.argv[0]} [n_max]")

print(f"Running bootstrap loop (pure gravity, Hilbert) up to n={N_MAX}")
print(f"  (BootstrapState will pre-compute L_ref^(0..{N_MAX+1}))")

t_init = time.time()
state = BootstrapState(L_matter=None, em_procedure='hilbert', n_max=N_MAX,
                       verbose=True)
print(f"  Initialization: {time.time()-t_init:.1f}s\n")

for n in range(N_MAX + 1):
    t0 = time.time()
    state.run_order(n)
    print(f"  Order {n} total: {time.time()-t0:.1f}s")

print()
print("=" * 60)
print(f"  All orders 0..{N_MAX} passed verification.")
print("=" * 60)
