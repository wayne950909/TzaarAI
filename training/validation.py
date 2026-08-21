"""
training/validation.py — 訓練常數驗證

在訓練開始前驗證所有超參數的合法性。
"""

from __future__ import annotations

from config import (
    TRAINING_CFG,
    SELFPLAY_CFG,
    MCTS_CFG,
    OPTIMIZER_CFG,
    GATE_CFG,
    REPLAY_CFG,
    NETWORK_CFG,
    ASYNC_MCTS_CFG,
    ELO_CFG,
)


def validate_constants() -> None:
    """驗證所有訓練常數的合法性。"""
    if TRAINING_CFG.total_updates <= 0:
        raise ValueError("TOTAL_UPDATES must be >= 1")
    if TRAINING_CFG.games_per_update <= 0:
        raise ValueError("GAMES_PER_UPDATE must be >= 1")
    if not (0.0 <= TRAINING_CFG.selfplay_model_game_ratio <= 1.0):
        raise ValueError("SELFPLAY_MODEL_GAME_RATIO must be in [0, 1]")
    if TRAINING_CFG.train_epochs_per_update <= 0:
        raise ValueError("TRAIN_EPOCHS_PER_UPDATE must be >= 1")
    if TRAINING_CFG.batch_size <= 0:
        raise ValueError("BATCH_SIZE must be >= 1")
    if MCTS_CFG.simulations <= 0:
        raise ValueError("SIMULATIONS_PER_DECISION must be >= 1")
    if TRAINING_CFG.checkpoint_every_updates <= 0:
        raise ValueError("CHECKPOINT_EVERY_UPDATES must be >= 1")
    if GATE_CFG.eval_every_updates <= 0:
        raise ValueError("GATE_EVERY_UPDATES must be >= 1")
    if TRAINING_CFG.log_every <= 0:
        raise ValueError("LOG_EVERY must be >= 1")
    if TRAINING_CFG.optimization_passes_per_update <= 0:
        raise ValueError("OPTIMIZATION_PASSES_PER_UPDATE must be >= 1")
    if GATE_CFG.eval_games <= 0:
        raise ValueError("GATE_EVAL_GAMES must be >= 1")
    if not (0.0 < GATE_CFG.winrate_threshold < 1.0):
        raise ValueError("GATE_WINRATE_THRESHOLD must be in (0, 1)")
    if MCTS_CFG.heuristic_softmax_temperature <= 0:
        raise ValueError("HEURISTIC_SOFTMAX_TEMPERATURE must be > 0")
    if not (0.0 <= MCTS_CFG.heuristic_prior_weight <= 1.0):
        raise ValueError("HEURISTIC_PRIOR_WEIGHT must be in [0, 1]")
    if ASYNC_MCTS_CFG.enabled:
        if ASYNC_MCTS_CFG.parallel_games <= 0:
            raise ValueError("ASYNC_PARALLEL_GAMES must be >= 1")
        if ASYNC_MCTS_CFG.num_threads <= 0:
            raise ValueError("ASYNC_NUM_THREADS must be >= 1")
        if ASYNC_MCTS_CFG.infer_max_batch <= 0:
            raise ValueError("ASYNC_INFER_MAX_BATCH must be >= 1")
        if ASYNC_MCTS_CFG.request_queue_size <= 0:
            raise ValueError("ASYNC_REQUEST_QUEUE_SIZE must be >= 1")
        if ASYNC_MCTS_CFG.infer_max_wait_ms < 0:
            raise ValueError("ASYNC_INFER_MAX_WAIT_MS must be >= 0")
        if ASYNC_MCTS_CFG.response_timeout_s <= 0:
            raise ValueError("ASYNC_RESPONSE_TIMEOUT_S must be > 0")
    if REPLAY_CFG.enabled:
        if REPLAY_CFG.max_samples <= 0:
            raise ValueError(
                "REPLAY_BUFFER_MAX_SAMPLES must be >= 1 "
                "when replay buffer is enabled"
            )
        if REPLAY_CFG.train_samples_per_update <= 0:
            raise ValueError(
                "REPLAY_TRAIN_SAMPLES_PER_UPDATE must be >= 1 "
                "when replay buffer is enabled"
            )
        if REPLAY_CFG.min_train_samples <= 0:
            raise ValueError(
                "REPLAY_MIN_TRAIN_SAMPLES must be >= 1 "
                "when replay buffer is enabled"
            )
        if REPLAY_CFG.min_train_samples > REPLAY_CFG.max_samples:
            raise ValueError(
                "REPLAY_MIN_TRAIN_SAMPLES must be <= "
                "REPLAY_BUFFER_MAX_SAMPLES"
            )
    if ELO_CFG.enabled:
        if ELO_CFG.pool_size <= 0:
            raise ValueError("ELO_POOL_SIZE must be >= 1")
        if ELO_CFG.k_factor <= 0:
            raise ValueError("ELO_K_FACTOR must be > 0")
        if ELO_CFG.round_robin_pairs_games <= 0:
            raise ValueError("ELO_ROUND_ROBIN_PAIRS_GAMES must be >= 1")
        if ELO_CFG.round_robin_simulations <= 0:
            raise ValueError("ELO_ROUND_ROBIN_SIMULATIONS must be >= 1")
        if not (0.0 <= ELO_CFG.elo_selfplay_ratio <= 1.0):
            raise ValueError("ELO_SELFPLAY_RATIO must be in [0, 1]")
