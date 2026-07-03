"""
mcts/state_builder.py — MCTS 葉節點特徵建構

提供從 TzaarGame 建構 CNN 輸入的工廠函式：
- build_cnn_state_12x9x9：棋盤特徵（12 通道）
- build_cnn_global_features：全域特徵
- terminal_value_for_current_player：終局價值
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from core.constants import (
    BLACK,
    TRAINING_PHASE_ORDER,
    TRAINING_PHASE_TO_IDX,
    WHITE,
)
from core.board import MAX_HEIGHT_NORM


def build_cnn_state_12x9x9(game: object, player: int) -> torch.Tensor:
    """建構 12 通道的棋盤特徵張量（從當前玩家視角）。

    通道配置
    --------
    0-2 : 己方佔據（依棋子種類）
    3-5 : 對方佔據（依棋子種類）
    6-8 : 己方高度（依棋子種類，正規化）
    9-11: 對方高度（依棋子種類，正規化）

    參數
    ----
    game : TzaarGame 實例（須有 board._piece_np / board._height_np）
    player : 視角玩家（WHITE=1 / BLACK=-1）

    回傳
    ----
    (12, 9, 9) float32 torch.Tensor
    """
    p = game.board._piece_np   # (9,9) int8
    h = game.board._height_np  # (9,9) float32

    h_norm = np.minimum(h, MAX_HEIGHT_NORM) * (1.0 / MAX_HEIGHT_NORM)

    if player == WHITE:
        own1 = p == 1; own2 = p == 2; own3 = p == 3
        opp1 = p == 4; opp2 = p == 5; opp3 = p == 6
    else:
        own1 = p == 4; own2 = p == 5; own3 = p == 6
        opp1 = p == 1; opp2 = p == 2; opp3 = p == 3

    state = np.stack([
        own1.astype(np.float32),
        own2.astype(np.float32),
        own3.astype(np.float32),
        opp1.astype(np.float32),
        opp2.astype(np.float32),
        opp3.astype(np.float32),
        h_norm * own1,
        h_norm * own2,
        h_norm * own3,
        h_norm * opp1,
        h_norm * opp2,
        h_norm * opp3,
    ], axis=0)

    return torch.from_numpy(state)


def build_cnn_global_features(game: object, player: int, phase: str) -> torch.Tensor:
    """建構全域特徵向量。

    特徵配置
    --------
    0  : turn_norm（回合數正規化到 [0, 1]）
    1  : player_sign（WHITE=1.0 / BLACK=-1.0）
    2-4: 己方 tzaar/tzarra/tott 疊數（正規化到 0~1）
    5-7: 對方 tzaar/tzarra/tott 疊數（正規化到 0~1）
    8  : 己方總疊數（正規化到 0~1）
    9  : 對方總疊數（正規化到 0~1）
    10+: phase one-hot（視特徵數量而定）

    參數
    ----
    game : TzaarGame 實例（須有 turn_number, _counts）
    player : 視角玩家
    phase : 階段名稱

    回傳
    ----
    (global_feature_dim,) float32 torch.Tensor
    """
    phase_one_hot = [0.0] * len(TRAINING_PHASE_ORDER)
    phase_one_hot[TRAINING_PHASE_TO_IDX[phase]] = 1.0

    opp = WHITE if player == BLACK else BLACK
    own_counts = game._counts[player]
    opp_counts = game._counts[opp]

    own_t1 = float(own_counts[1])
    own_t2 = float(own_counts[2])
    own_t3 = float(own_counts[3])
    opp_t1 = float(opp_counts[1])
    opp_t2 = float(opp_counts[2])
    opp_t3 = float(opp_counts[3])

    own_total = own_t1 + own_t2 + own_t3
    opp_total = opp_t1 + opp_t2 + opp_t3

    turn_norm = min(float(game.turn_number), 200.0) / 200.0
    player_sign = 1.0 if player == WHITE else -1.0

    vec = [
        turn_norm,
        player_sign,
        own_t1 / 15.0,
        own_t2 / 15.0,
        own_t3 / 15.0,
        opp_t1 / 15.0,
        opp_t2 / 15.0,
        opp_t3 / 15.0,
        own_total / 45.0,
        opp_total / 45.0,
        *phase_one_hot,
    ]
    return torch.tensor(vec, dtype=torch.float32)


def terminal_value_for_current_player(
    winner: Optional[int], current_player: int
) -> float:
    """從當前玩家視角計算終局價值。

    參數
    ----
    winner : 勝者（WHITE=1 / BLACK=-1），None 表示平局
    current_player : 當前輪到的玩家

    回傳
    ----
    1.0（當前玩家勝）/ -1.0（當前玩家敗）/ 0.0（平局）
    """
    if winner is None:
        return 0.0
    return 1.0 if winner == current_player else -1.0
