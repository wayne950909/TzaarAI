"""

mcts/heuristic.py — MCTS 啟發式先驗與評分

提供基於材質、機動性、稀缺性的啟發式評估函式。
可與策略網路先驗混合使用（HEURISTIC_PRIOR_WEIGHT 控制比例）。
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from core.constants import BLACK, WHITE, piece_owner
from config import MCTS_CFG

# 從 MCTS_CFG 取用啟發式參數
HEURISTIC_SOFTMAX_TEMPERATURE = MCTS_CFG.heuristic_softmax_temperature
HEURISTIC_PRIOR_WEIGHT = MCTS_CFG.heuristic_prior_weight
MATERIAL_BASE_TZAAR = MCTS_CFG.material_base_tzaar
MATERIAL_BASE_TZARRA = MCTS_CFG.material_base_tzarra
MATERIAL_BASE_TOTT = MCTS_CFG.material_base_tott
MOBILITY_CAPTURE_WEIGHT = MCTS_CFG.mobility_capture_weight
SCARCITY_BONUS_TWO_OR_LESS = MCTS_CFG.scarcity_bonus_two_or_less
SCARCITY_BONUS_ONE_OR_LESS = MCTS_CFG.scarcity_bonus_one_or_less
from mcts.node import MCTSNode


def _piece_type_from_code(piece_code: int) -> int:
    """棋子代碼轉換成種類 (1/2/3)。"""
    if piece_code in (1, 4):
        return 1
    if piece_code in (2, 5):
        return 2
    if piece_code in (3, 6):
        return 3
    raise ValueError(f"Unknown piece code: {piece_code}")


def _material_base_for_piece_type(piece_type: int) -> float:
    """每疊材質基礎分數。"""
    if piece_type == 1:
        return MATERIAL_BASE_TZAAR
    if piece_type == 2:
        return MATERIAL_BASE_TZARRA
    if piece_type == 3:
        return MATERIAL_BASE_TOTT
    raise ValueError(f"Unknown piece type: {piece_type}")


def material_score(game: object, player: int) -> float:
    """計算玩家的材質總分。"""
    score = 0.0
    for pos in game.board.valid_positions():
        if game.board.is_empty(pos):
            continue
        top = game.board.top_piece(pos)
        if piece_owner(top) != player:
            continue
        pt = _piece_type_from_code(top)
        height = float(game.board.height(pos))
        score += _material_base_for_piece_type(pt) * height
    return score


def mobility_capture_score(game: object, player: int) -> float:
    """計算玩家的吃子機動性分數。"""
    return float(len(game._capture_pairs_for_player(player))) * MOBILITY_CAPTURE_WEIGHT


def scarcity_bonus(game: object, player: int) -> float:
    """計算玩家棋種稀缺性的額外加分（棋種越少分數越高）。"""
    counts = game._counts[player]
    bonus = 0.0
    for pt in (1, 2, 3):
        remaining = int(counts[pt])
        if remaining <= 1:
            bonus += SCARCITY_BONUS_ONE_OR_LESS
        elif remaining <= 2:
            bonus += SCARCITY_BONUS_TWO_OR_LESS
    return bonus


def player_eval_score(game: object, player: int) -> float:
    """玩家的綜合評估分數。"""
    return (
        material_score(game, player)
        + mobility_capture_score(game, player)
        + scarcity_bonus(game, player)
    )


def heuristic_action_value(after_state: object, perspective_player: int) -> float:
    """給定套用動作後的狀態，從指定視角計算啟發式價值（self - opp）。"""
    opp = WHITE if perspective_player == BLACK else BLACK
    self_v = player_eval_score(after_state.game, perspective_player)
    opp_v = player_eval_score(after_state.game, opp)
    return self_v - opp_v


def heuristic_priors_for_actions(
    node: MCTSNode,
    legal_idxs: List[int],
    action_dim: int,
) -> torch.Tensor:
    """從啟發式評估計算動作先驗機率（softmax over heuristic values）。

    若節點無狀態或分數異常，回傳均勻分佈。
    """
    priors = torch.zeros(action_dim, dtype=torch.float32)
    if not legal_idxs:
        return priors
    if node.state is None:
        uniform = 1.0 / float(len(legal_idxs))
        for a in legal_idxs:
            priors[a] = uniform
        return priors

    perspective_player = node.state.current_player()
    scores: List[float] = []
    for action in legal_idxs:
        child_state = node.state.clone()
        child_state.apply_action(action, legal_mask=node.legal_mask)
        scores.append(heuristic_action_value(child_state, perspective_player))

    logits = torch.tensor(scores, dtype=torch.float32)
    logits = logits / float(HEURISTIC_SOFTMAX_TEMPERATURE)
    logits = logits - logits.max()
    exps = torch.exp(logits)
    sum_exps = float(exps.sum().item())
    if not math.isfinite(sum_exps) or sum_exps <= 0.0:
        uniform = 1.0 / float(len(legal_idxs))
        for a in legal_idxs:
            priors[a] = uniform
        return priors

    probs = exps / sum_exps
    if not torch.isfinite(probs).all():
        uniform = 1.0 / float(len(legal_idxs))
        for a in legal_idxs:
            priors[a] = uniform
        return priors

    for i, action in enumerate(legal_idxs):
        priors[action] = probs[i]
    return priors


def blend_policy_and_heuristic_priors(
    policy_priors: torch.Tensor,
    heuristic_priors: torch.Tensor,
    legal_idxs: List[int],
    action_dim: int,
) -> torch.Tensor:
    """混合策略網路先驗與啟發式先驗。

    混合比率由 HEURISTIC_PRIOR_WEIGHT 控制：
        mixed = (1 - w) * policy + w * heuristic
    """
    mixed = torch.zeros(action_dim, dtype=torch.float32)
    if not legal_idxs:
        return mixed

    legal_tensor = torch.tensor(legal_idxs, dtype=torch.long)
    policy_legal = policy_priors[:action_dim].to(
        device="cpu", dtype=torch.float32
    )[legal_tensor]
    heuristic_legal = heuristic_priors[:action_dim].to(dtype=torch.float32)[
        legal_tensor
    ]
    mixed_legal = (
        (1.0 - HEURISTIC_PRIOR_WEIGHT) * policy_legal
        + HEURISTIC_PRIOR_WEIGHT * heuristic_legal
    )

    sum_mixed = float(mixed_legal.sum().item())
    if not math.isfinite(sum_mixed) or sum_mixed <= 0.0:
        sum_heur = float(heuristic_legal.sum().item())
        if math.isfinite(sum_heur) and sum_heur > 0.0:
            mixed_legal = heuristic_legal / sum_heur
        else:
            uniform = 1.0 / float(len(legal_idxs))
            mixed_legal = torch.full(
                (len(legal_idxs),), uniform, dtype=torch.float32
            )
    else:
        mixed_legal = mixed_legal / sum_mixed

    mixed[legal_tensor] = mixed_legal
    return mixed
