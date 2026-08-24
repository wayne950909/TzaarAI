"""
mcts/evaluate.py — MCTS 葉節點評估

提供：
- evaluate_leaf：單個葉節點的策略/價值評估
- evaluate_leaf_batch：批次評估多個葉節點
- expand_node：展開單個節點
- expand_nodes_from_eval：從評估結果批量展開節點
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from config import HEAD_ACTION
from core.constants import BLACK, WHITE, TRAINING_PHASE_TO_IDX, PHASE_STEP_2
from core.action import N_ACTIONS, PASS_ACTION_IDX
from mcts.node import MCTSNode
from mcts.state_builder import (
    build_cnn_state_12x9x9,
    build_cnn_global_features,
)
from mcts.heuristic import (
    heuristic_priors_for_actions,
    blend_policy_and_heuristic_priors,
)
from config import MCTS_CFG

# 方便取用
HEURISTIC_PRIOR_WEIGHT = MCTS_CFG.heuristic_prior_weight


def legal_action_dim_and_head(state: object) -> Tuple[str, int, torch.Tensor]:
    """取得當前狀態的動作頭名稱、動作維度、合法遮罩。

    回傳
    ----
    (head: str, action_dim: int, mask: torch.Tensor)
    """
    head = HEAD_ACTION
    action_dim = N_ACTIONS
    mask = state.legal_mask()
    if mask is None:
        raise ValueError("legal mask is undefined for terminal state")
    return head, action_dim, mask


def evaluate_leaf(
    policy: torch.nn.Module,
    state: object,
    device: torch.device,
) -> Tuple[str, int, torch.Tensor, torch.Tensor, float]:
    """單個葉節點評估。

    參數
    ----
    policy : 策略網路（須有 encode, head_logits, forward_value 方法）
    state : PhaseGameState 或 CppPhaseGameStateAdapter
    device : torch 裝置

    回傳
    ----
    (head, action_dim, mask, priors, value)
    """
    phase = state.phase()
    if phase is None:
        raise ValueError("Cannot evaluate terminal leaf")

    head, action_dim, mask = legal_action_dim_and_head(state)

    board = (
        build_cnn_state_12x9x9(state.game, state.current_player())
        .unsqueeze(0)
        .to(device)
    )
    global_features = (
        build_cnn_global_features(state.game, state.current_player(), phase)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        vh, ph = policy.encode(board, global_features)
        logits = policy.head_logits(ph, HEAD_ACTION)[0]
        value = float(policy.forward_value(vh).squeeze().item())

    mask = mask.to(device=device)
    masked_logits = logits[:action_dim].masked_fill(~mask[:action_dim], -1e9)
    priors = torch.softmax(masked_logits, dim=-1)
    return head, action_dim, mask, priors, value


def evaluate_leaf_batch(
    policy: torch.nn.Module,
    states: List[object],
    device: torch.device,
) -> List[Tuple[str, int, torch.Tensor, torch.Tensor, float]]:
    """批次評估多個葉節點。"""
    if not states:
        return []

    phases: List[str] = []
    action_dims: List[int] = []
    masks: List[torch.Tensor] = []
    boards: List[torch.Tensor] = []
    globals_: List[torch.Tensor] = []

    for state in states:
        phase = state.phase()
        if phase is None:
            raise ValueError("Cannot evaluate terminal leaf")

        head, action_dim, mask = legal_action_dim_and_head(state)
        phases.append(phase)
        action_dims.append(action_dim)
        masks.append(mask)
        boards.append(build_cnn_state_12x9x9(state.game, state.current_player()))
        globals_.append(
            build_cnn_global_features(state.game, state.current_player(), phase)
        )

    board_batch = torch.stack(boards).to(device)
    global_batch = torch.stack(globals_).to(device)

    with torch.no_grad():
        vh, ph = policy.encode(board_batch, global_batch)
        action_logits = policy.action_head(ph)
        values = policy.forward_value(vh).squeeze(-1)

    results: List[Tuple[str, int, torch.Tensor, torch.Tensor, float]] = []
    for i in range(len(states)):
        head = HEAD_ACTION
        action_dim = action_dims[i]
        mask = masks[i].to(device=device)
        logits = action_logits[i]

        masked_logits = logits[:action_dim].masked_fill(
            ~mask[:action_dim], -1e9
        )
        priors = torch.softmax(masked_logits, dim=-1)
        results.append(
            (
                head,
                action_dim,
                mask,
                priors,
                float(values[i].item()),
            )
        )

    return results


def expand_node(
    node: MCTSNode,
    policy: torch.nn.Module,
    state: object,
    device: torch.device,
) -> Tuple[str, int, torch.Tensor, float]:
    """展開單個 MCTS 節點。

    使用策略網路評估葉節點，建立子節點。

    回傳
    ----
    (head, action_dim, mask, value)
    """
    head, action_dim, mask, policy_priors, value = evaluate_leaf(
        policy, state, device
    )

    node.expanded = True
    node.action_dim = action_dim
    node.legal_mask = mask.to(device="cpu")
    legal_idxs = (
        torch.nonzero(mask[:action_dim], as_tuple=False).flatten().tolist()
    )

    # 混合啟發式先驗（若啟用）
    if HEURISTIC_PRIOR_WEIGHT > 0.0:
        heuristic_priors = heuristic_priors_for_actions(
            node, legal_idxs, action_dim
        )
        mixed_priors = blend_policy_and_heuristic_priors(
            policy_priors[:action_dim].to(device="cpu", dtype=torch.float32),
            heuristic_priors,
            legal_idxs,
            action_dim,
        )
    else:
        mixed_priors = policy_priors[:action_dim].to(
            device="cpu", dtype=torch.float32
        )

    for a in legal_idxs:
        node.children[a] = MCTSNode(
            prior=float(mixed_priors[a].item()), to_play=None
        )

    return head, action_dim, mask, value


def expand_nodes_from_eval(
    nodes: List[MCTSNode],
    evals: List[Tuple[str, int, torch.Tensor, torch.Tensor, float]],
) -> List[Tuple[str, int, torch.Tensor, float]]:
    """從批次評估結果展開多個節點。

    參數
    ----
    nodes : 待展開的節點列表
    evals : evaluate_leaf_batch 的回傳值

    回傳
    ----
    List of (head, action_dim, mask, value)
    """
    if len(nodes) != len(evals):
        raise ValueError("nodes and evals length mismatch")

    out: List[Tuple[str, int, torch.Tensor, float]] = []
    for node, (head, action_dim, mask, policy_priors, value) in zip(
        nodes, evals
    ):
        node.expanded = True
        node.action_dim = action_dim
        node.legal_mask = mask.to(device="cpu")
        node.children = {}
        legal_idxs = (
            torch.nonzero(mask[:action_dim], as_tuple=False).flatten().tolist()
        )

        mixed_priors = policy_priors[:action_dim].to(
            device="cpu", dtype=torch.float32
        )
        for a in legal_idxs:
            node.children[a] = MCTSNode(
                prior=float(mixed_priors[a].item()), to_play=None
            )
        out.append((head, action_dim, mask, value))

    return out
