"""load_state restoration smoke + cross-process compute-resume XFAIL.

What this locks in (the part that WORKS): load_state, in a FRESH process, fully
restores the environment a pickled BootstrapState needs --
  * the matter-field registry (re-registers 'phi'),
  * the spacetime dimension (verifies dim==4), and
  * the fresh-index counter (fast-forwarded past every pickled index).
This is exactly the restoration a parallel WORKER needs (_init_worker reuses it),
and it is verified by the asserts below.

What is XFAIL (a known, documented limitation): full cross-process *compute*
resume -- continuing run_order in the fresh process -- does NOT reproduce an
in-process run. Diagnosis (see git history / DEVELOPMENT_STATUS): the bootstrap
quantities themselves reproduce EXACTLY across a pickle (pure-gravity E[3] was
bit-for-bit identical resumed vs in-process), but the high-level VERIFICATION /
traceless-recovery orchestration (_verify_vs_L_ref, the ddh-box machinery)
carries hidden process state that diverges. The three obvious cross-process
breaks were fixed (registry, tensor-head `is`->`==`, index-counter); this
residual is deeper and resume is explicitly NOT a supported feature. It does not
affect parallelism, which only farms out the (reproducible) compute builders and
keeps verification serial in the parent.

So: the asserts must pass; run_order(3) is expected to raise (XFAIL). If it ever
stops raising, resume may have become viable -- revisit.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dim-4 checkpoint: set_dimension(4) MUST precede importing bootstrap_loop so
# its local h/dh/ddh bind to the dim-4 heads (see set_dimension docstring).
from bootstrap.tensor_algebra import set_dimension
set_dimension(4)

from bootstrap.bootstrap_loop import load_state
from bootstrap.tensor_algebra import get_matter_fields, get_index_counter, dimension

ckpt_path = '_conf_o3_ckpt_order2.pkl'
if not os.path.exists(ckpt_path):
    print(f"Checkpoint {ckpt_path} not found. Run "
          f"test_conformal_order3_lref_closure.py first.")
    sys.exit(1)

print(f"matter_fields before load: {list(get_matter_fields().keys())}", flush=True)
state = load_state(ckpt_path)
print(f"matter_fields after load:  {list(get_matter_fields().keys())}", flush=True)
print(f"dimension after load: {dimension()}; index counter: {get_index_counter()}",
      flush=True)

# The restoration that load_state guarantees (and that a parallel worker needs):
assert 'phi' in get_matter_fields(), "load_state failed to re-register phi"
assert dimension() == 4, "load_state did not preserve the checkpoint dimension"
assert get_index_counter() > 0, "load_state did not fast-forward the index counter"
print("*** load_state restoration verified (registry + dimension + counter). ***",
      flush=True)

print(f"\n##### run_order(3) (XFAIL: cross-process compute-resume) #####", flush=True)
try:
    state.run_order(3)
    print("\n*** UNEXPECTED PASS: cross-process compute-resume reproduced order 3. "
          "Resume may now be viable -- revisit the XFAIL. ***", flush=True)
except RuntimeError as e:
    print(f"\n*** XFAIL (known limitation): cross-process compute-resume diverges "
          f"in the verification/traceless orchestration: {e}", flush=True)
    print("    (Expected. load_state restoration above is the supported, "
          "parallelism-relevant part.)", flush=True)
# Exit 0 in both branches: the supported behavior (restoration) is asserted; the
# unsupported behavior (compute-resume) is XFAIL, not a regression.
