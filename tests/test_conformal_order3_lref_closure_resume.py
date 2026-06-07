"""Order-3-only resume test: load post-order-2 checkpoint and run order 3 with diagnostics.

Designed for fast iteration on order-3 closure debugging. Loads the
post-order-2 BootstrapState from _conf_o3_ckpt_order2.pkl (created by
test_conformal_order3_lref_closure.py) and runs order 3, printing all
E_diff and decomposition details if the closure check fails.

Expected:
  - run_order(3) completes without raising  => order-3 closure verified,
  - if RuntimeError is raised, full residual and context are printed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

ckpt_path = '_conf_o3_ckpt_order2.pkl'
if not os.path.exists(ckpt_path):
    print(f"Checkpoint {ckpt_path} not found. Run test_conformal_order3_lref_closure.py first.")
    sys.exit(1)

print(f"Loading checkpoint from {ckpt_path}...", flush=True)
with open(ckpt_path, 'rb') as f:
    state = pickle.load(f)

print(f"Checkpoint loaded. Max order completed: {state.max_order_run}", flush=True)

print(f"\n##### run_order(3) #####", flush=True)
try:
    state.run_order(3)
    print(f"  >>> order 3 closed against L_ref (no raise).", flush=True)
    print("\n*** PASS: conformal scalar closes against L_ref at order 3. ***",
          flush=True)
except RuntimeError as e:
    print(f"\n*** FAIL: {e}", flush=True)
    sys.exit(1)
