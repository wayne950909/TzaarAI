"""Verify the C++ bridge is functional."""
import sys
import os

# Add the build output directory to path
sys.path.insert(0, os.path.join("cpp", "build", "Release"))

import tzaar_cpp as tz

print("=== Basic constants ===")
print(f"N_ACTIONS = {tz.N_ACTIONS}")
print(f"PASS_ACTION_IDX = {tz.PASS_ACTION_IDX}")
print(f"CAPTURE_OFFSET = {tz.CAPTURE_OFFSET}")
print(f"REINFORCE_OFFSET = {tz.REINFORCE_OFFSET}")
print(f"N_EDGE_ACTIONS  = {tz.N_EDGE_ACTIONS}")

print("\n=== Edge table ===")
edges = tz.action_edge_table()
print(f"Edge table length: {len(edges)}")
print(f"First 3 edges: {edges[:3]}")
print(f"Last 3 edges: {edges[-3:]}")

print("\n=== Encode / Decode ===")
idx = tz.encode_action_idx("capture", 0)
print(f"encode capture edge=0 -> {idx}")
decoded = tz.decode_action_idx(idx)
print(f"Decoded: {decoded}")

idx_pass = tz.encode_action_idx("pass", None)
print(f"encode pass -> {idx_pass}")
decoded_pass = tz.decode_action_idx(idx_pass)
print(f"Decoded pass: {decoded_pass}")

print("\n=== PhaseGameState basic ===")
state = tz.PhaseGameState()
print(f"is_done: {state.is_done()}")
print(f"current_player: {state.current_player()}")
print(f"turn_number: {state.turn_number()}")
print(f"winner: {state.winner()}")
print(f"phase: {state.phase()}")

mask = state.legal_mask()
print(f"legal_mask length: {len(mask)}")
print(f"Number of legal actions (step1): {sum(mask)}")

snap = state.leaf_snapshot(node_id=0)
print(f"snapshot node_id: {snap.node_id}")
print(f"snapshot current_player: {snap.current_player}")
print(f"snapshot winner: {snap.winner}")
print(f"snapshot is_done: {snap.is_done}")
print(f"snapshot phase: {snap.phase}")
print(f"snapshot white_counts: {snap.white_counts}")
print(f"snapshot black_counts: {snap.black_counts}")
print(f"snapshot board_state_flat shape: {len(snap.board_state_flat)}")
print(f"snapshot global_features shape: {len(snap.global_features)}")
print(f"snapshot legal_mask length: {len(snap.legal_mask)}")

print("\n=== Applying an action (first step capture) ===")
# Find the first legal capture action
legal_indices = [i for i, v in enumerate(mask) if v and i < tz.CAPTURE_OFFSET + tz.N_EDGE_ACTIONS]
print(f"First 5 legal capture indices: {legal_indices[:5]}")
if legal_indices:
    action = legal_indices[0]
    print(f"Applying action {action}...")
    print(f"Before apply: phase={state.phase()}")
    state.apply_action(action)
    print(f"After apply: phase={state.phase()}, is_done={state.is_done()}")

print("\n=== Clone ===")
cloned = state.clone()
print(f"Cloned state phase: {cloned.phase()}")

print("\n=== piece_counts ===")
counts = state.piece_counts()
print(f"piece_counts: {counts}")

print("\n=== board_cells ===")
cells = state.board_cells()
print(f"Number of non-empty cells: {len(cells)}")
if cells:
    print(f"First cell: {cells[0]}")

print("\n=== SearchConfig ===")
cfg = tz.SearchConfig()
cfg.simulations = 10
cfg.leaf_batch_size = 4
cfg.puct_c = 1.5
cfg.add_root_dirichlet_noise = False
print(f"simulations: {cfg.simulations}")
print(f"leaf_batch_size: {cfg.leaf_batch_size}")
print(f"puct_c: {cfg.puct_c}")

print("\n=== SearchSession basic ===")
state2 = tz.PhaseGameState()
session = tz.SearchSession(state2, cfg)
print(f"has_pending_leaves: {session.has_pending_leaves()}")

# Collect pending leaves
packed = session.collect_pending_leaves_packed(max_batch=4)
print(f"Packed keys: {list(packed.keys())}")
for k, v in packed.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print("\n=== Test single leaf eval ===")
if packed["node_ids"].shape[0] > 0:
    nid = int(packed["node_ids"][0])
    priors = [1.0 / tz.N_ACTIONS] * tz.N_ACTIONS
    value = 0.0
    session.submit_leaf_eval(nid, priors, value)
    print(f"Submitted eval for node {nid}")

print("\n=== Test batch leaf eval ===")
packed2 = session.collect_pending_leaves_packed(max_batch=4)
if packed2["node_ids"].shape[0] > 0:
    bsz = packed2["node_ids"].shape[0]
    import numpy as np
    ids = np.array(packed2["node_ids"], dtype=np.int32)
    priors = np.full((bsz, tz.N_ACTIONS), 1.0 / tz.N_ACTIONS, dtype=np.float32)
    vals = np.zeros(bsz, dtype=np.float32)
    session.submit_leaf_eval_batch(ids, priors, vals)
    print(f"Submitted batch eval for {bsz} nodes")

print("\n=== Finish search ===")
result = session.finish()
print(f"Result type: {type(result)}")
print(f"root_node_id: {result.root_node_id}")
print(f"root_player: {result.root_player}")
print(f"winner: {result.winner}")
print(f"is_done: {result.is_done}")
print(f"is_complete: {result.is_complete}")
print(f"needs_root_eval: {result.needs_root_eval}")
print(f"simulations_requested: {result.simulations_requested}")
print(f"simulations_processed: {result.simulations_processed}")
print(f"pending_leaf_count: {result.pending_leaf_count}")
print(f"root_value: {result.root_value}")

print("\n=== ALL TESTS PASSED ===")
