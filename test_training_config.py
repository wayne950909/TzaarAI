"""
test_training_config.py — 小參數訓練測試配置

用法：
    python test_training_config.py [start_level]

    start_level（可選）：從階級表的哪一階開始訓練（0-based）。
    None / 省略 = 預設從第 0 階開始。

這個腳本會用極小參數執行一次訓練循環，並可指定要從「模擬次數 + 學習率 + 混合權重」的哪一階開始。

階級表（GATE_CFG.level_schedule）：
    level 0  -> sims=128,   lr=0.0008, vq=0.2
    level 1  -> sims=256,   lr=0.0006, vq=0.3
    level 2  -> sims=384,   lr=0.0004, vq=0.4
    level 3  -> sims=512,   lr=0.0003, vq=0.5
    level 4  -> sims=768,   lr=0.0002, vq=0.6
    level 5  -> sims=1280,  lr=0.0002, vq=0.7
    level 6  -> sims=1600,  lr=0.0001, vq=0.8

範例：
    python test_training_config.py          # 從 level 0（128 sims）開始
    python test_training_config.py 3        # 從 level 3（512 sims）開始
    python test_training_config.py 6        # 從 level 6（1600 sims）封頂階開始
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── 設定起始階層（可從命令列參數覆寫，也可直接改這一行）─────
# 0 = 第一階，最大 = 最後一階（封頂）。例如 3 = 從 sims=512, lr=0.0003 開始。
START_LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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
)

# ─── 測試用極小參數 ─────────────────────────────────────
print("=" * 60)
print("王柏崴你好")
print("=" * 60)

# 訓練配置
TRAINING_CFG.total_updates = 75        # 只做 2 個 update
TRAINING_CFG.games_per_update = 400      # 每輪 5 場遊戲
TRAINING_CFG.log_every = 1
TRAINING_CFG.checkpoint_every_updates = 10  # 測試時不存

# Self-play
SELFPLAY_CFG.temp_high = 1.0
SELFPLAY_CFG.temp_low = 0.1
SELFPLAY_CFG.temp_switch_decision = 9 #25, 256/18
# 混合式 value target：value_target = (1-λ)*終局 + λ*MCTS根Q
# 注意：value_q_weight 已改由「階級表（level_schedule）」隨階層控制，
# 因此這裡不再手動覆寫（保留給階級表驅動）。

# ── 最新模型 vs 隨機近期歷史模型 ─────────────────────
# 每輪除了自我對弈（games_per_update=400 局）之外，額外讓最新模型
# 對上一個隨機近期歷史 guard 模型，再產生 vs_past_games=100 局的資料，
# 總共每輪 self-play 產生 400 + 100 = 500 場。
SELFPLAY_CFG.vs_past_enabled = True
SELFPLAY_CFG.vs_past_games = 200
SELFPLAY_CFG.vs_past_recent_n = 10

# MCTS
MCTS_CFG.simulations = 768     # 極少模擬
MCTS_CFG.leaf_batch_size = 16
MCTS_CFG.puct_c = 1.25
MCTS_CFG.use_root_dirichlet_noise = True

# Gate — 測試時跳過（設為極大值）
GATE_CFG.eval_every_updates = 1

# ─── ELO 歷史對手池（測試參數） ────────────────────────
ELO_CFG.enabled = True
ELO_CFG.pool_size = 6
ELO_CFG.round_robin_pairs_games = 50     # 每對內戰局數（極少）
ELO_CFG.round_robin_simulations = 768   # 內戰模擬數（極少）
ELO_CFG.round_robin_temperature = 0.1
ELO_CFG.elo_selfplay_ratio = 1.0

# ─── 印出階級表與本次起始階層 ───────────────────────────
_level_schedule = list(GATE_CFG.level_schedule) if GATE_CFG.level_schedule else (
    (int(GATE_CFG.simulations_per_decision), 0.0004, float(SELFPLAY_CFG.value_q_weight)),
)
# 夾在合法範圍內（避免命令列參數超出階級表），並確保為 3 元組
_max_level = len(_level_schedule) - 1
START_LEVEL = max(0, min(START_LEVEL, _max_level))
_level_schedule = [
    (row[0], row[1], row[2] if len(row) >= 3 else 0.0)
    for row in _level_schedule
]

print()
print("階級表（模擬次數 → 學習率 → 混合權重 vq）：")
for i, (sims, lr, vq) in enumerate(_level_schedule):
    marker = "   <- 本階開始" if i == START_LEVEL else ""
    print(f"  level {i}: sims={sims:<6} lr={lr:.4f} vq={vq:.2f}{marker}")

print("\n訓練參數：")
print(f"  total_updates = {TRAINING_CFG.total_updates}")
print(f"  games_per_update = {TRAINING_CFG.games_per_update}")
print(f"  mcts_simulations = {MCTS_CFG.simulations}")
print(f"  cpp_backend = {os.environ.get('TZAAR_STATE_BACKEND', 'cpp')}")
print(f"  elo_opponent_pool = {ELO_CFG.enabled} | pool_size={ELO_CFG.pool_size}")
print(
    f"  起始階層 = level {START_LEVEL} "
    f"(sims={_level_schedule[START_LEVEL][0]}, "
    f"lr={_level_schedule[START_LEVEL][1]}, "
    f"vq={_level_schedule[START_LEVEL][2]})"
)

print("\n正在啟動訓練...\n")

# 匯入主訓練循環
from training.loop import run

try:
    run("cpp_restructure_test", start_level=START_LEVEL)
    print("\n✅ 測試完成！新 C++ 架構訓練正常運作。")
except Exception as e:
    print(f"\n❌ 測試失敗：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
