"""
training/sample.py — PolicySample 資料結構

為訓練提供的樣本資料容器。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PolicySample:
    """訓練樣本，包含棋盤狀態與 MCTS 輸出目標。

    屬性
    ----
    state : (12, 9, 9) 棋盤特徵張量
    global_features : (global_feature_dim,) 全域特徵
    action_dim : 動作空間維度
    legal_mask_padded : (N_ACTIONS,) 合法動作遮罩（已 padding 到統一大小）
    target_pi_padded : (N_ACTIONS,) MCTS 訪問次數正規化後的目標策略
    player : 走棋玩家（WHITE=1 / BLACK=-1）
    winner_sign : 最終勝者（0=平局）
    value_target : 價值目標（1.0 / -1.0 / 0.0）     
    """
    state: torch.Tensor
    global_features: torch.Tensor
    action_dim: int
    legal_mask_padded: torch.Tensor
    target_pi_padded: torch.Tensor
    player: int
    winner_sign: int = 0
    value_target: float = 0.0
