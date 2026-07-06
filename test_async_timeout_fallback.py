"""Force async batch failure and verify training falls back to sync path."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TZAAR_STATE_BACKEND"] = "cpp"

from config import (
    ASYNC_MCTS_CFG,
    GATE_CFG,
    MCTS_CFG,
    REPLAY_CFG,
    SELFPLAY_CFG,
    TRAINING_CFG,
)
import mcts.mcts_api as mcts_api
from training.loop import run


def main() -> None:
    # Minimal run to validate fallback behavior quickly.
    TRAINING_CFG.total_updates = 1
    TRAINING_CFG.games_per_update = 2
    TRAINING_CFG.log_every = 1
    TRAINING_CFG.checkpoint_every_updates = 999

    SELFPLAY_CFG.temp_high = 1.0
    SELFPLAY_CFG.temp_low = 0.5
    SELFPLAY_CFG.temp_switch_decision = 3

    MCTS_CFG.simulations = 64
    MCTS_CFG.leaf_batch_size = 4
    MCTS_CFG.puct_c = 1.5
    MCTS_CFG.use_root_dirichlet_noise = False

    REPLAY_CFG.enabled = False
    GATE_CFG.eval_every_updates = 999

    # Keep async enabled so the training loop enters the async path first.
    ASYNC_MCTS_CFG.enabled = True
    ASYNC_MCTS_CFG.parallel_games = 2
    ASYNC_MCTS_CFG.infer_max_batch = 32
    ASYNC_MCTS_CFG.infer_max_wait_ms = 1.0
    ASYNC_MCTS_CFG.request_queue_size = 4
    ASYNC_MCTS_CFG.response_timeout_s = 0.1

    original_async_batch = mcts_api.run_mcts_cpp_batch_async

    def _forced_async_failure(*args, **kwargs):
        raise TimeoutError("forced async failure for fallback regression test")

    print("=" * 60)
    print("Async Timeout Fallback Test")
    print("=" * 60)
    print("Running training with forced async batch failure...")

    mcts_api.run_mcts_cpp_batch_async = _forced_async_failure
    try:
        run("async_timeout_fallback_test")
    finally:
        mcts_api.run_mcts_cpp_batch_async = original_async_batch

    print("\nPASS: training completed with async timeout fallback")


if __name__ == "__main__":
    main()
