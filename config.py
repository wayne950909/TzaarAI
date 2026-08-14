"""
config.py — 所有超參數集中管理

使用方式
--------
    from config import MCTS_CFG, TRAINING_CFG, GATE_CFG, NETWORK_CFG, ...

    # 可在執行前修改
    MCTS_CFG.simulations = 800
    TRAINING_CFG.games_per_update = 200

這個檔案在目前架構中同時扮演三個角色：
1. 訓練主流程的超參數來源
2. Python / C++ 搜尋共用的關鍵設定來源
3. 執行期後端狀態（_ACTIVE_STATE_BACKEND / _ACTIVE_CPP_MODULE）保存位置

因此它不只是靜態設定檔，也是一個小型 runtime registry。
"""

from __future__ import annotations
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════
# 基本設定
# ══════════════════════════════════════════════════════════════════════

TITLE = "mcts_cnn"
GUARD_TITLE = f"{TITLE}_guard"
LOG_DIR = "logs"
SEED = 42
DEVICE = "auto"          # auto / cpu / cuda / cuda:N
INFERENCE_DEVICE = "cuda:0"  # MCTS 推論裝置（大 batch 用 GPU）
CHECKPOINT_DIR = "checkpoints"
MAX_HEIGHT_NORM = 8.0
HEAD_ACTION = "action"


# ══════════════════════════════════════════════════════════════════════
# 狀態後端設定
# ══════════════════════════════════════════════════════════════════════

STATE_BACKEND = "cpp"          # python / cpp
CPP_BACKEND_REQUIRED = True


# ══════════════════════════════════════════════════════════════════════
# 神經網路架構
# ══════════════════════════════════════════════════════════════════════

@dataclass
class NetworkConfig:
    """PolicyNetCNNMin17 的超參數"""
    global_feature_dim: int = 12
    dropout: float = 0
    channels: int = 64
    num_res_blocks: int = 4
    fc_hidden: int = 512
    fc_hidden_2: int = 256


NETWORK_CFG = NetworkConfig()

# ══════════════════════════════════════════════════════════════════════
# MCTS 搜尋
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MCTSConfig:
    """MCTS 搜尋引擎的超參數"""
    simulations: int = 256  
    puct_c: float = 1.5
    leaf_batch_size: int = 32

    # Root Dirichlet noise（訓練時啟用）
    use_root_dirichlet_noise: bool = True
    root_dirichlet_eps: float = 0.25
    root_dirichlet_alpha: float = 0.05

    # 啟發式先驗（設為 0 則停用）
    heuristic_prior_weight: float = 0.0
    heuristic_softmax_temperature: float = 1.0

    # 啟發式評分權重
    material_base_tzaar: float = 100.0
    material_base_tzarra: float = 40.0
    material_base_tott: float = 15.0
    mobility_capture_weight: float = 10.0
    scarcity_bonus_two_or_less: float = 200.0
    scarcity_bonus_one_or_less: float = 500.0


MCTS_CFG = MCTSConfig()


# ══════════════════════════════════════════════════════════════════════
# 非同步 MCTS Pipeline
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AsyncMCTSConfig:
    """多 CPU worker + 共享 GPU worker 的非同步 pipeline"""
    enabled: bool = True
    # 同時進行的 active game pool 大小（例如 200 局平行推進）
    parallel_games: int = 400
    # C++ SearchManager 的常駐 worker thread 數量（固定不變）
    num_threads: int = 10
    # 單次 GPU forward 最多聚合多少個 leaf states。
    infer_max_batch: int = 12800
    # GPU worker 最多等待多久，再把目前已收集到的 request 一起送進模型。
    infer_max_wait_ms: float = 2.0
    # Python async worker request queue 的容量上限。
    request_queue_size: int = 30
    # CPU worker 等待 GPU 回傳結果的超時秒數；
    # 超時時目前訓練會在 selfplay engine 退回 sync-single。
    response_timeout_s: float = 30.0

    # 累積多少 leaf 才送 GPU 做一次 batch forward
    # 設為 0 則用 infer_max_batch 當 threshold
    min_batch_for_swap: int = 1

    # ── Worker-Local Double Buffer 參數 ──────────────────────
    # 每個 worker 每側邊緩衝區的最大容量 = 樹數量 × 此值
    # （adjust.md：buffer 最大容量 = 樹數量 * 32，填不滿）
    buffer_capacity_per_tree: int = 16
    # 單一側邊緩衝區尚未滿載前，「到達一定資料量」即觸發 is_ready 的 leaf 數。
    # 此值不是緩衝區最大容量，而是 worker 依 adjust.md 判定 ready 的資料量下限。
    ready_flush_leaves: int = 64

    # ── C++ SearchManager 內部 Debug Log ──────────────────
    # 在 config.py 設定，經 cpp_manager._build_config 傳入 C++ SearchManager。
    # 關閉時 C++ 端只做一次 relaxed atomic 檢查便跳過，完全不打擾效能。
    # 開啟時會把「worker buffer ready / 主執行緒取資料 / result_handler 處理」
    # 三類事件寫入 debug_log_path 指定的文字檔。
    debug_log_enabled: bool = False
    debug_log_path: str = "logs/cpp_debug.log"


ASYNC_MCTS_CFG = AsyncMCTSConfig()


# ══════════════════════════════════════════════════════════════════════
# 自我對弈（溫度排程）
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SelfPlayConfig:
    """自我對弈的動作採樣溫度排程"""
    temp_high: float = 1.0     # 前期探索溫度
    temp_low: float = 0.1      # 後期利用溫度
    temp_switch_decision: int = 8  # 第幾步之後切換到低溫


SELFPLAY_CFG = SelfPlayConfig()


# ══════════════════════════════════════════════════════════════════════
# 訓練主迴圈
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """訓練主迴圈的超參數"""
    total_updates: int = 200
    games_per_update: int = 400
    selfplay_model_game_ratio: float = 1.0  # 純自我對弈比例 (0~1)
    train_epochs_per_update: int = 1
    optimization_passes_per_update: int = 5
    batch_size: int = 128
    checkpoint_every_updates: int = 20
    log_every: int = 10
    selfplay_progress_log_interval: int = 1  # <=0 停用進度log

    # Resume 行為
    require_resume: bool = False
    resume_any_title: bool = True


TRAINING_CFG = TrainingConfig()


# ══════════════════════════════════════════════════════════════════════
# 優化器 / 損失函數
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OptimizerConfig:
    """Adam 優化器與損失加權"""
    learning_rate_start: float = 0.0007
    learning_rate_end: float = 0.0007   # = start 表示固定學習率
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    entropy_weight: float = 0.0


OPTIMIZER_CFG = OptimizerConfig()


# ══════════════════════════════════════════════════════════════════════
# Gatekeeper
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GatekeeperConfig:
    """Gatekeeper 評估的超參數"""
    eval_games: int = 100
    winrate_threshold: float = 0.55
    temperature: float = 0.1           # 評估時的採樣溫度
    simulations_per_decision: int = 400  # 評估時的 MCTS 模擬數
    eval_every_updates: int = 1        # 每 N 次更新執行一次
    keep_optimizer_on_reject: bool = False
    keep_replay_on_reject: bool = True


GATE_CFG = GatekeeperConfig()


# ══════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ReplayConfig:
    """Replay Buffer 設定"""
    enabled: bool = True
    max_samples: int = 50000
    train_samples_per_update: int = 8192
    min_train_samples: int = 256
    snapshot_version: int = 1
    snapshot_tag: str = "latest"


REPLAY_CFG = ReplayConfig()


# ══════════════════════════════════════════════════════════════════════
# 運行時狀態（由 training.loop 或 update_cnn_mcts_no_heuristic.py 設定）
# ══════════════════════════════════════════════════════════════════════

from typing import Any, Optional

# 活躍的狀態後端名稱（"python" 或 "cpp"）
_ACTIVE_STATE_BACKEND: str = "python"
# 已載入的 C++ 模組（若無則為 None）
_ACTIVE_CPP_MODULE: Optional[Any] = None


# ══════════════════════════════════════════════════════════════════════
# 驗證函式（確保設定值合理）
# ══════════════════════════════════════════════════════════════════════

def validate_configs() -> None:
    """在訓練開始前檢查所有設定值是否合理。"""
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
    if REPLAY_CFG.enabled:
        if REPLAY_CFG.max_samples <= 0:
            raise ValueError("REPLAY_BUFFER_MAX_SAMPLES must be >= 1")
        if REPLAY_CFG.train_samples_per_update <= 0:
            raise ValueError("REPLAY_TRAIN_SAMPLES_PER_UPDATE must be >= 1")
        if REPLAY_CFG.min_train_samples <= 0:
            raise ValueError("REPLAY_MIN_TRAIN_SAMPLES must be >= 1")
        if REPLAY_CFG.min_train_samples > REPLAY_CFG.max_samples:
            raise ValueError("REPLAY_MIN_TRAIN_SAMPLES must be <= REPLAY_BUFFER_MAX_SAMPLES")
