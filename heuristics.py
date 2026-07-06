"""
heuristics.py — 啟發式策略與先驗計算

提供 HeuristicPlayoutPolicy 類別，用於與策略網路先驗混合。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from core.constants import WHITE, BLACK, PHASE_STEP_1, PHASE_STEP_2, TRAINING_PHASE_TO_IDX, piece_owner, piece_type
from core.action import N_ACTIONS, PASS_ACTION_IDX
from config import MCTS_CFG


@dataclass
class HeuristicConfig:
    """啟發式策略的設定。"""
    softmax_temperature: float = 1.0
    prior_weight: float = 0.0

    # 材質基礎分數
    material_base_tzaar: float = 100.0
    material_base_tzarra: float = 40.0
    material_base_tott: float = 15.0

    # 機動性權重
    mobility_capture_weight: float = 10.0

    # 稀缺性加分
    scarcity_bonus_two_or_less: float = 200.0
    scarcity_bonus_one_or_less: float = 500.0


class HeuristicPlayoutPolicy:
    """啟發式策略類別。

    提供啟發式評估和先驗計算，可與神經網路先驗混合使用。
    """

    @dataclass
    class Config:
        """相容舊版的設定 dataclass。"""
        softmax_temperature: float = 1.0
        prior_weight: float = 0.0

    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            self.config = HeuristicConfig()
        else:
            self.config = HeuristicConfig(
                softmax_temperature=config.softmax_temperature,
                prior_weight=config.prior_weight,
            )

    def evaluate_state(self, state: object, perspective_player: int) -> float:
        """給定狀態和視角，計算啟發式估值（self - opp）。"""
        return heuristic_action_value(state, perspective_player)

    def compute_action_priors(
        self,
        node: object,
        legal_idxs: List[int],
        action_dim: int,
    ) -> torch.Tensor:
        """從啟發式評估計算動作先驗機率。"""
        return heuristic_priors_for_actions(node, legal_idxs, action_dim)

    def blend(
        self,
        policy_priors: torch.Tensor,
        heuristic_priors: torch.Tensor,
        legal_idxs: List[int],
        action_dim: int,
    ) -> torch.Tensor:
        """混合策略網路先驗與啟發式先驗。"""
        return blend_policy_and_heuristic_priors(
            policy_priors, heuristic_priors, legal_idxs, action_dim,
        )


# ──────────────────────────────────────────────────────────────────────
# 啟發式評分函式（與 mcts/heuristic.py 同步）
# ──────────────────────────────────────────────────────────────────────

def _piece_type_from_code(piece_code: int) -> int:
    if piece_code in (1, 4):
        return 1
    if piece_code in (2, 5):
        return 2
    if piece_code in (3, 6):
        return 3
    raise ValueError(f"Unknown piece code: {piece_code}")


def _material_base_for_piece_type(piece_type_id: int) -> float:
    cfg = MCTS_CFG
    if piece_type_id == 1:
        return cfg.material_base_tzaar
    if piece_type_id == 2:
        return cfg.material_base_tzarra
    if piece_type_id == 3:
        return cfg.material_base_tott
    raise ValueError(f"Unknown piece type: {piece_type_id}")


def material_score(game: object, player: int) -> float:
    """計算指定玩家的材質分數。"""
    from core.game import TzaarGame

    if not isinstance(game, TzaarGame):
        return 0.0

    score = 0.0
    for row, col in game.board.valid_positions():
        pos = (row, col)
        if game.board.is_empty(pos):
            continue
        top = game.board.top_piece(pos)
        if piece_owner(top) != player:
            continue
        ptype = _piece_type_from_code(top)
        height = float(game.board.height(pos))
        score += _material_base_for_piece_type(ptype) * height
    return score


def mobility_capture_score(game: object, player: int) -> float:
    """計算指定玩家的機動性分數。"""
    from core.game import TzaarGame

    if not isinstance(game, TzaarGame):
        return 0.0

    cfg = MCTS_CFG
    return float(len(game._capture_pairs_for_player(player))) * cfg.mobility_capture_weight


def scarcity_bonus(game: object, player: int) -> float:
    """計算指定玩家的稀缺性加分。"""
    from core.game import TzaarGame

    if not isinstance(game, TzaarGame):
        return 0.0

    cfg = MCTS_CFG
    counts = game._counts[player]
    bonus = 0.0
    for ptype in (1, 2, 3):
        remaining = int(counts[ptype])
        if remaining <= 1:
            bonus += cfg.scarcity_bonus_one_or_less
        elif remaining <= 2:
            bonus += cfg.scarcity_bonus_two_or_less
    return bonus


def player_eval_score(game: object, player: int) -> float:
    """綜合評估指定玩家的局面分數。"""
    return (
        material_score(game, player)
        + mobility_capture_score(game, player)
        + scarcity_bonus(game, player)
    )


def heuristic_action_value(after_state: object, perspective_player: int) -> float:
    """給定套用動作後的狀態，從指定視角計算啟發式價值。"""
    from core.game import TzaarGame

    if not isinstance(after_state, TzaarGame):
        return 0.0

    opp = WHITE if perspective_player == BLACK else BLACK
    self_v = player_eval_score(after_state, perspective_player)
    opp_v = player_eval_score(after_state, opp)
    return self_v - opp_v


def heuristic_priors_for_actions(
    node: object,
    legal_idxs: List[int],
    action_dim: int,
) -> torch.Tensor:
    """從啟發式評估計算動作先驗機率（softmax over heuristic values）。"""
    import math

    cfg = MCTS_CFG
    priors = torch.zeros(action_dim, dtype=torch.float32)
    if not legal_idxs:
        return priors

    # 嘗試從節點取得狀態
    state = getattr(node, 'state', None) if hasattr(node, 'state') else None
    if state is None:
        uniform = 1.0 / float(len(legal_idxs))
        for a in legal_idxs:
            priors[a] = uniform
        return priors

    perspective_player = getattr(state, 'current_player', lambda: WHITE)()
    scores: List[float] = []
    for action in legal_idxs:
        child_state = state.clone()
        child_state.apply_action(action, legal_mask=getattr(node, 'legal_mask', None))
        # 從 child_state 取得 game 物件進行評估
        game = getattr(child_state, 'game', None)
        if game is not None:
            scores.append(heuristic_action_value(game, perspective_player))
        else:
            scores.append(0.0)

    logits = torch.tensor(scores, dtype=torch.float32)
    logits = logits / float(cfg.heuristic_softmax_temperature)
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

    mixed = (1 - w) * policy + w * heuristic
    """
    cfg = MCTS_CFG
    mixed = torch.zeros(action_dim, dtype=torch.float32)
    if not legal_idxs:
        return mixed

    legal_tensor = torch.tensor(legal_idxs, dtype=torch.long)
    policy_legal = policy_priors[:action_dim].to(device="cpu", dtype=torch.float32)[legal_tensor]
    heuristic_legal = heuristic_priors[:action_dim].to(dtype=torch.float32)[legal_tensor]
    mixed_legal = (1.0 - cfg.heuristic_prior_weight) * policy_legal + cfg.heuristic_prior_weight * heuristic_legal

    sum_mixed = float(mixed_legal.sum().item())
    if not math.isfinite(sum_mixed) or sum_mixed <= 0.0:
        sum_heur = float(heuristic_legal.sum().item())
        if math.isfinite(sum_heur) and sum_heur > 0.0:
            mixed_legal = heuristic_legal / sum_heur
        else:
            uniform = 1.0 / float(len(legal_idxs))
            mixed_legal = torch.full((len(legal_idxs),), uniform, dtype=torch.float32)
    else:
        mixed_legal = mixed_legal / sum_mixed

    mixed[legal_tensor] = mixed_legal
    return mixed
