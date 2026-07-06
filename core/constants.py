"""
core/constants.py — 遊戲常數定義

包含棋盤佈局、方向、玩家代號、階段名稱、棋子代碼等所有遊戲層級常數。
不依賴任何其他模組（除了 typing 和 enum）。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple


# ══════════════════════════════════════════════════════════════════════
# 棋盤佈局
# ══════════════════════════════════════════════════════════════════════

BOARD_LAYOUT: List[List[int]] = [
    [3, 3, 3, 3, 6, -1, -1, -1, -1],
    [6, 2, 2, 2, 5, 6, -1, -1, -1],
    [6, 5, 1, 1, 4, 5, 6, -1, -1],
    [6, 5, 4, 3, 6, 4, 5, 6, -1],
    [6, 5, 4, 6, -1, 3, 1, 2, 3],
    [-1, 3, 2, 1, 3, 6, 1, 2, 3],
    [-1, -1, 3, 2, 1, 4, 4, 2, 3],
    [-1, -1, -1, 3, 2, 5, 5, 5, 3],
    [-1, -1, -1, -1, 3, 6, 6, 6, 6],
]

# 棋盤有效格數（固定不變）
N_POSITIONS = 60

# Board 的 _valid_positions 固定順序，以便 lookup
STATE_INDEX_TO_BOARD_COORDS: Tuple[Tuple[int, int], ...] = tuple(
    (r, c)
    for r in range(len(BOARD_LAYOUT))
    for c in range(len(BOARD_LAYOUT[0]))
    if BOARD_LAYOUT[r][c] != -1
)

if len(STATE_INDEX_TO_BOARD_COORDS) != N_POSITIONS:
    raise ValueError(
        f"State coordinate mapping mismatch: expected {N_POSITIONS}, "
        f"got {len(STATE_INDEX_TO_BOARD_COORDS)}"
    )


# ══════════════════════════════════════════════════════════════════════
# 方向定義
# ══════════════════════════════════════════════════════════════════════

DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),   # 0: Up
    (1, 0),    # 1: Down
    (0, -1),   # 2: Left
    (0, 1),    # 3: Right
    (-1, -1),  # 4: UpLeft
    (1, 1),    # 5: DnRight
)

N_DIRECTIONS = 6

DIRECTION_TO_INDEX: Dict[Tuple[int, int], int] = {
    d: i for i, d in enumerate(DIRECTIONS)
}

DIRECTION_LABELS = [
    "Up     (-1, 0)",
    "Down   ( 1, 0)",
    "Left   ( 0,-1)",
    "Right  ( 0, 1)",
    "UpLeft (-1,-1)",
    "DnRight( 1, 1)",
]


# ══════════════════════════════════════════════════════════════════════
# 玩家與棋子
# ══════════════════════════════════════════════════════════════════════

WHITE = 1
BLACK = -1

# 棋子代碼範圍
#   白方：1=tzaar, 2=tzarra, 3=tott
#   黑方：4=tzaar, 5=tzarra, 6=tott
PIECE_CODE_RANGE = range(1, 7)


def piece_owner(piece: int) -> int:
    """棋子代碼轉換成陣營 (WHITE / BLACK)。"""
    if piece in (1, 2, 3):
        return WHITE
    if piece in (4, 5, 6):
        return BLACK
    raise ValueError(f"Unknown piece code: {piece}")


def piece_type(piece: int) -> int:
    """棋子代碼轉換成種類 (1/2/3)。"""
    if piece in (1, 4):
        return 1
    if piece in (2, 5):
        return 2
    if piece in (3, 6):
        return 3
    raise ValueError(f"Unknown piece code: {piece}")


# ══════════════════════════════════════════════════════════════════════
# 回合階段名稱
# ══════════════════════════════════════════════════════════════════════

PHASE_STEP_1 = "step1_capture"
PHASE_STEP_2 = "step2_action"

TRAINING_PHASE_ORDER = (PHASE_STEP_1, PHASE_STEP_2)
TRAINING_PHASE_TO_IDX: Dict[str, int] = {
    name: i for i, name in enumerate(TRAINING_PHASE_ORDER)
}


# ══════════════════════════════════════════════════════════════════════
# 枚舉
# ══════════════════════════════════════════════════════════════════════

class MoveKind(str, Enum):
    """回合中的行動種類。"""
    CAPTURE = "capture"
    REINFORCE = "reinforce"
    PASS = "pass"


class WinReason(str, Enum):
    """遊戲結束原因。"""
    EXTINCTION = "extinction"
    NO_MANDATORY_CAPTURE = "no_mandatory_capture"
