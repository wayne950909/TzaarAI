"""
core — Tzaar 遊戲核心（無 ML 依賴）

提供棋盤規則、動作空間、常數定義。
"""

from core.constants import (
    BLACK,
    BOARD_LAYOUT,
    WHITE,
    DIRECTIONS,
    DIRECTION_LABELS,
    DIRECTION_TO_INDEX,
    N_DIRECTIONS,
    N_POSITIONS,
    PHASE_STEP_1,
    PHASE_STEP_2,
    PIECE_CODE_RANGE,
    STATE_INDEX_TO_BOARD_COORDS,
    TRAINING_PHASE_ORDER,
    TRAINING_PHASE_TO_IDX,
    MoveKind,
    WinReason,
    piece_owner,
    piece_type,
)

from core.game import (
    Board,
    GameResult,
    Move,
    MovePair,
    TzaarGame,
)

from core.action import (
    ACTION_EDGE_TABLE,
    CAPTURE_OFFSET,
    EDGE_TO_IDX,
    N_ACTIONS,
    N_EDGE_ACTIONS,
    PASS_ACTION_IDX,
    REINFORCE_OFFSET,
    decode_action_idx,
    encode_action_idx,
)

__all__ = [
    # constants
    "BLACK", "BOARD_LAYOUT", "WHITE",
    "DIRECTIONS", "DIRECTION_LABELS", "DIRECTION_TO_INDEX",
    "N_DIRECTIONS", "N_POSITIONS",
    "PHASE_STEP_1", "PHASE_STEP_2",
    "PIECE_CODE_RANGE",
    "STATE_INDEX_TO_BOARD_COORDS",
    "TRAINING_PHASE_ORDER", "TRAINING_PHASE_TO_IDX",
    "MoveKind", "WinReason",
    "piece_owner", "piece_type",
    # game
    "Board", "GameResult", "Move", "MovePair", "TzaarGame",
    # action
    "ACTION_EDGE_TABLE", "CAPTURE_OFFSET", "EDGE_TO_IDX",
    "N_ACTIONS", "N_EDGE_ACTIONS", "PASS_ACTION_IDX",
    "REINFORCE_OFFSET",
    "decode_action_idx", "encode_action_idx",
]
