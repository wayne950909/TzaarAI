"""
test_training_config.py — 小參數訓練測試配置

用法：
    python test_training_config.py

這個腳本會用極小參數執行一次訓練循環，驗證新 C++ 架構是否正確運作。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 設定小參數測試覆蓋
os.environ["TZAAR_STATE_BACKEND"] = "cpp"

# 匯入 config 並覆蓋為測試參數
from config import (
    TRAINING_CFG,
    SELFPLAY_CFG,
    MCTS_CFG,
    NETWORK_CFG,
    REPLAY_CFG,
    GATE_CFG,
    ELO_CFG,
    SIMS_CFG,
)

# ─── 測試用極小參數 ─────────────────────────────────────
print("=" * 60)
print("王柏崴你好")
print("=" * 60)

# 訓練配置
TRAINING_CFG.total_updates = 50        # 只做 2 個 update
TRAINING_CFG.games_per_update = 400      # 每輪 5 場遊戲
TRAINING_CFG.log_every = 1
TRAINING_CFG.checkpoint_every_updates = 10  # 測試時不存

# Self-play
SELFPLAY_CFG.temp_high = 1.0
SELFPLAY_CFG.temp_low = 0.1
SELFPLAY_CFG.temp_switch_decision = 6

# MCTS
MCTS_CFG.simulations = 768     # 極少模擬
MCTS_CFG.leaf_batch_size = 16
MCTS_CFG.puct_c = 1.5
MCTS_CFG.use_root_dirichlet_noise = True

# 隨機模擬次數排程器（self-play 資料產生用）
SIMS_CFG.enabled = True
SIMS_CFG.low_start = (64, 92)      # 測試：固定小範圍，不上升
SIMS_CFG.low_end = (384, 512)
SIMS_CFG.high_start = (512, 640)     # 測試：固定高範圍
SIMS_CFG.high_end = (832, 960)
SIMS_CFG.high_probability = 0.25

# Gate — 測試時跳過（設為極大值）
GATE_CFG.eval_every_updates = 1

# ─── ELO 歷史對手池（測試參數） ────────────────────────
ELO_CFG.enabled = True
ELO_CFG.pool_size = 6
ELO_CFG.round_robin_pairs_games = 20     # 每對內戰局數（極少）
ELO_CFG.round_robin_simulations = 768   # 內戰模擬數（極少）
ELO_CFG.round_robin_temperature = 0.1
ELO_CFG.elo_selfplay_ratio = 1.0

print("\n訓練參數：")
print(f"  total_updates = {TRAINING_CFG.total_updates}")
print(f"  games_per_update = {TRAINING_CFG.games_per_update}")
print(f"  mcts_simulations = {MCTS_CFG.simulations}")
print(f"  cpp_backend = {os.environ.get('TZAAR_STATE_BACKEND', 'cpp')}")
print(f"  elo_opponent_pool = {ELO_CFG.enabled} | pool_size={ELO_CFG.pool_size}")
print(f"  random_sims_scheduler = {SIMS_CFG.enabled} "
      f"| low={SIMS_CFG.low_start} high={SIMS_CFG.high_start} "
      f"| high_prob={SIMS_CFG.high_probability}")

print("\n正在啟動訓練...\n")

# 匯入主訓練循環
from training.loop import run

try:
    run("cpp_restructure_test")
    print("\n✅ 測試完成！新 C++ 架構訓練正常運作。")
except Exception as e:
    print(f"\n❌ 測試失敗：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
