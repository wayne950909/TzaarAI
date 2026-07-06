"""
core/action.py — 統一 601 動作空間 (300 capture + 300 reinforce + 1 pass)

提供：
- ACTION_EDGE_TABLE：固定的 (src_idx, direction_idx) 查找表（300 筆）
- encode_action_idx() / decode_action_idx()：動作索引編解碼
- EDGE_TO_IDX：從 (src_idx, direction_idx) 到 edge_idx 的反向映射

依賴：core/constants.py
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from core.constants import MoveKind, N_DIRECTIONS, N_POSITIONS, STATE_INDEX_TO_BOARD_COORDS


N_EDGE_ACTIONS = 300       # 固定邊數（每方向每格）
N_ACTIONS = 601            # 300 capture + 300 reinforce + 1 pass

CAPTURE_OFFSET = 0
REINFORCE_OFFSET = 300
PASS_ACTION_IDX = 600


# ──────────────────────────────────────────────────────────────────────
# 固定邊表 (src_idx, direction_idx)
# 預先計算所有可能存在的 (位置, 方向) 組合（60 格 × 6 方向 = 360，
# 但只有邊緣或非邊緣會有的實際組合佔 300 筆）。
# ──────────────────────────────────────────────────────────────────────

def _build_action_edge_table() -> Tuple[Tuple[int, int], ...]:
    """掃描所有有效位置與方向，建立存在邊的查找表。"""
    edges: list[Tuple[int, int]] = []
    for src_idx, (r, c) in enumerate(STATE_INDEX_TO_BOARD_COORDS):
        # 每個方向都可能有一條邊（即使沒有鄰居，first_occupied 不會越界）
        for direction_idx in range(N_DIRECTIONS):
            edges.append((src_idx, direction_idx))
    return tuple(edges)


ACTION_EDGE_TABLE: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 3), (0, 5), (1, 1), (1, 2), (1, 3), (1, 5), (2, 1), (2, 2), (2, 3),
    (2, 5), (3, 1), (3, 2), (3, 3), (3, 5), (4, 1), (4, 2), (4, 5), (5, 0), (5, 1),
    (5, 3), (5, 5), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (7, 0), (7, 1),
    (7, 2), (7, 3), (7, 4), (7, 5), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5),
    (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (10, 1), (10, 2), (10, 4), (10, 5),
    (11, 0), (11, 1), (11, 3), (11, 5), (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5),
    (13, 0), (13, 1), (13, 2), (13, 3), (13, 4), (13, 5), (14, 0), (14, 1), (14, 2), (14, 3),
    (14, 4), (14, 5), (15, 0), (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (16, 0), (16, 1),
    (16, 2), (16, 3), (16, 4), (16, 5), (17, 1), (17, 2), (17, 4), (17, 5), (18, 0), (18, 1),
    (18, 3), (18, 5), (19, 0), (19, 1), (19, 2), (19, 3), (19, 4), (19, 5), (20, 0), (20, 1),
    (20, 2), (20, 3), (20, 4), (20, 5), (21, 0), (21, 1), (21, 2), (21, 3), (21, 4), (22, 0),
    (22, 2), (22, 3), (22, 4), (22, 5), (23, 0), (23, 1), (23, 2), (23, 3), (23, 4), (23, 5),
    (24, 0), (24, 1), (24, 2), (24, 3), (24, 4), (24, 5), (25, 1), (25, 2), (25, 4), (25, 5),
    (26, 0), (26, 3), (26, 5), (27, 0), (27, 1), (27, 2), (27, 3), (27, 4), (27, 5), (28, 0),
    (28, 1), (28, 2), (28, 3), (28, 4), (28, 5), (29, 0), (29, 1), (29, 2), (29, 4), (29, 5),
    (30, 0), (30, 1), (30, 3), (30, 4), (30, 5), (31, 0), (31, 1), (31, 2), (31, 3), (31, 4),
    (31, 5), (32, 0), (32, 1), (32, 2), (32, 3), (32, 4), (32, 5), (33, 1), (33, 2), (33, 4),
    (34, 0), (34, 3), (34, 4), (34, 5), (35, 0), (35, 1), (35, 2), (35, 3), (35, 4), (35, 5),
    (36, 0), (36, 1), (36, 2), (36, 3), (36, 4), (36, 5), (37, 1), (37, 2), (37, 3), (37, 4),
    (37, 5), (38, 0), (38, 1), (38, 2), (38, 3), (38, 5), (39, 0), (39, 1), (39, 2), (39, 3),
    (39, 4), (39, 5), (40, 0), (40, 1), (40, 2), (40, 3), (40, 4), (40, 5), (41, 0), (41, 1),
    (41, 2), (41, 4), (42, 0), (42, 3), (42, 4), (42, 5), (43, 0), (43, 1), (43, 2), (43, 3),
    (43, 4), (43, 5), (44, 0), (44, 1), (44, 2), (44, 3), (44, 4), (44, 5), (45, 0), (45, 1),
    (45, 2), (45, 3), (45, 4), (45, 5), (46, 0), (46, 1), (46, 2), (46, 3), (46, 4), (46, 5),
    (47, 0), (47, 1), (47, 2), (47, 3), (47, 4), (47, 5), (48, 0), (48, 1), (48, 2), (48, 4),
    (49, 0), (49, 3), (49, 4), (49, 5), (50, 0), (50, 1), (50, 2), (50, 3), (50, 4), (50, 5),
    (51, 0), (51, 1), (51, 2), (51, 3), (51, 4), (51, 5), (52, 0), (52, 1), (52, 2), (52, 3),
    (52, 4), (52, 5), (53, 0), (53, 1), (53, 2), (53, 3), (53, 4), (53, 5), (54, 0), (54, 1),
    (54, 2), (54, 4), (55, 0), (55, 3), (55, 4), (56, 0), (56, 2), (56, 3), (56, 4), (57, 0),
    (57, 2), (57, 3), (57, 4), (58, 0), (58, 2), (58, 3), (58, 4), (59, 0), (59, 2), (59, 4),
)

if len(ACTION_EDGE_TABLE) != N_EDGE_ACTIONS:
    raise ValueError(f"Expected {N_EDGE_ACTIONS} action edges, got {len(ACTION_EDGE_TABLE)}")

# 反向映射：(src_idx, direction_idx) → edge_idx
EDGE_TO_IDX: Dict[Tuple[int, int], int] = {
    edge: idx for idx, edge in enumerate(ACTION_EDGE_TABLE)
}


# ──────────────────────────────────────────────────────────────────────
# 編解碼函式
# ──────────────────────────────────────────────────────────────────────


def encode_action_idx(kind: MoveKind, edge_idx: Optional[int] = None) -> int:
    """將 (kind, edge_idx) 編碼為 0..600 的動作索引。

    參數
    ----
    kind : CAPTURE / REINFORCE / PASS
    edge_idx : 邊編號（PASS 時不需要）

    回傳
    ----
    0..600 的整數動作索引
    """
    if kind == MoveKind.PASS:
        return PASS_ACTION_IDX
    if edge_idx is None or not (0 <= edge_idx < N_EDGE_ACTIONS):
        raise ValueError("edge_idx out of range")
    if kind == MoveKind.CAPTURE:
        return CAPTURE_OFFSET + edge_idx
    if kind == MoveKind.REINFORCE:
        return REINFORCE_OFFSET + edge_idx
    raise ValueError("Unsupported kind")


def decode_action_idx(action_idx: int) -> tuple[MoveKind, Optional[int], Optional[int], Optional[int]]:
    """將 0..600 的動作索引解碼為 (kind, edge_idx, src_idx, direction_idx)。

    回傳
    ----
    (kind, edge_idx, src_idx, direction_idx)
        PASS 時所有 optional 值均為 None。
    """
    if action_idx == PASS_ACTION_IDX:
        return MoveKind.PASS, None, None, None
    if not (0 <= action_idx < PASS_ACTION_IDX):
        raise ValueError("action_idx out of range")
    if action_idx < REINFORCE_OFFSET:
        kind = MoveKind.CAPTURE
        edge_idx = action_idx
    else:
        kind = MoveKind.REINFORCE
        edge_idx = action_idx - REINFORCE_OFFSET
    src_idx, direction_idx = ACTION_EDGE_TABLE[edge_idx]
    return kind, edge_idx, src_idx, direction_idx
