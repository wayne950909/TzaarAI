"""
core/board.py — 棋盤常數與輔助函式

提供訓練循環所需但不在 core/constants.py 中的棋盤常數。
這些常數原本分散在各舊版腳本中，統一集中至此。
"""

from __future__ import annotations

from typing import Dict

# 棋盤大小
BOARD_SIZE = (9, 9)

# 高度正規化最大值（與 config.MAX_HEIGHT_NORM 連動）
MAX_HEIGHT_NORM = 8.0

# 每方初始棋子數量分布
PIECE_COUNTS: Dict[int, int] = {
    1: 3,   # tzaar (white/black)
    2: 5,   # tzarra
    3: 7,   # tott
}

# 每方初始分組數量（用於某些策略計算）
N_GROUPS_INITIAL_PER_PLAYER = sum(PIECE_COUNTS.values())

# 玩家代號別名（與 core.constants 一致）
# 注意：核心常數在 core/constants.py 中定義
# 此處僅提供別名方便從 core.board import
PLAYER_WHITE = 1
PLAYER_BLACK = -1

# PASS 動作的虛擬獎勵值（供某些啟發式使用）
PASS_REWARD = 0.0

# 訓練對局的最大步數
TRAINING_GAME_STEPS = 200
