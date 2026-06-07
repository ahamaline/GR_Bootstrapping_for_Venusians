"""Example 01: pure-gravity bootstrap through order 2.

The simplest non-trivial case: no matter, Hilbert energy-momentum
prescription. We start from nothing and iteratively build the
gravitational field equation order by order in `h_{mu nu}`. By order 2
we will have recovered the first two non-trivial orders of the Einstein
tensor.

The script is verbose: at each order it prints the 6 steps of the
procedure with term counts. Re-read the README in this folder for what
each step does.

Expected wall time: about 2 minutes (the n=2 superpotential is the
dominant cost).
"""

import sys, os, time
# Allow `python examples/01_...py` from anywhere by adding the parent dir
# to sys.path so we can import the `bootstrap` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap.bootstrap_loop import BootstrapState


# We don't register any matter fields. `BootstrapState` with `L_matter=None`
# means pure gravity: at order 0, E_h^(0) = 0 (there's no matter to source
# the wave operator); at order 1, the wave operator W^{mu nu} = EL(L_h^(2))
# appears in step 1; the bootstrap then proceeds to discover the
# higher-order pieces of Einstein's equation.

N_MAX = 2

print(f"Running pure-gravity bootstrap (Hilbert) through order n = {N_MAX}.")
print("L^(0) = 0 (no matter); the gravitational field equation will be")
print("built order by order from the spin-2 wave operator W^{mu nu}.")
print()

state = BootstrapState(L_matter=None, em_procedure='hilbert',
                       n_max=N_MAX, verbose=True)

t_total = time.time()
for n in range(N_MAX + 1):
    t = time.time()
    state.run_order(n)
    print(f"  Order {n} wall: {time.time() - t:.1f}s")

print()
print("=" * 60)
print(f"  All orders 0..{N_MAX} closed and verified against the")
print(f"  Einstein-Hilbert expansion. Total wall: {time.time() - t_total:.1f}s")
print("=" * 60)
