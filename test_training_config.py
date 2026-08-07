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
)

# ─── 測試用極小參數 ─────────────────────────────────────
print("=" * 60)
print("C++ 重構測試 — 小參數訓練驗證")
print("=" * 60)

# 訓練配置
TRAINING_CFG.total_updates = 1          # 只做 2 個 update
TRAINING_CFG.games_per_update = 200      # 每輪 5 場遊戲
TRAINING_CFG.log_every = 1
TRAINING_CFG.checkpoint_every_updates = 10  # 測試時不存

# Self-play
SELFPLAY_CFG.temp_high = 1.0
SELFPLAY_CFG.temp_low = 0.5
SELFPLAY_CFG.temp_switch_decision = 5

# MCTS
MCTS_CFG.simulations = 128         # 極少模擬
MCTS_CFG.leaf_batch_size = 16
MCTS_CFG.puct_c = 1.5
MCTS_CFG.use_root_dirichlet_noise = False

# Gate — 測試時跳過（設為極大值）
GATE_CFG.eval_every_updates = 999

print("\n訓練參數：")
print(f"  total_updates = {TRAINING_CFG.total_updates}")
print(f"  games_per_update = {TRAINING_CFG.games_per_update}")
print(f"  mcts_simulations = {MCTS_CFG.simulations}")
print(f"  cpp_backend = {os.environ.get('TZAAR_STATE_BACKEND', 'cpp')}")

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
