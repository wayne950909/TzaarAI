"""
mcts/search.py — Python 版 MCTS 搜尋引擎

提供 PUCT 演算法的完整實作：
- select_child_action：PUCT 選擇
- backup：價值備份
- apply_root_dirichlet_noise：根節點 Dirichlet 雜訊
- run_mcts_python：完整 Python 搜尋管線
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from core.constants import BLACK, WHITE
from core.action import N_ACTIONS
from config import (
    HEAD_ACTION,
    MCTS_CFG,
)

# 方便取用的別名
USE_ROOT_DIRICHLET_NOISE = MCTS_CFG.use_root_dirichlet_noise
ROOT_DIRICHLET_EPS = MCTS_CFG.root_dirichlet_eps
ROOT_DIRICHLET_ALPHA_ACTION = MCTS_CFG.root_dirichlet_alpha
from mcts.node import MCTSNode
from mcts.evaluate import (
    evaluate_leaf_batch,
    expand_nodes_from_eval,
    legal_action_dim_and_head,
)
from mcts.state_builder import terminal_value_for_current_player


# 方便取用的別名
PUCT_C = MCTS_CFG.puct_c
MCTS_LEAF_BATCH_SIZE = MCTS_CFG.leaf_batch_size


def select_child_action(node: MCTSNode) -> int:
    """PUCT 選擇子節點。

    使用公式: score = Q(s, a) + C * P(s, a) * sqrt(N + 1) / (1 + n)

    其中 Q 已從子節點視角轉換為父節點視角（當 turn switch 時取反）。
    """
    if not node.children:
        raise ValueError("select_child_action called on leaf without children")

    total = sum(child.visit_count for child in node.children.values())
    sqrt_total = math.sqrt(float(total + 1))
    best_score = -1e30
    best_actions: List[int] = []
    parent_player = node.to_play

    for action, child in node.children.items():
        q_child = child.mean_value()
        child_player = child.to_play
        if child_player is None and child.state is not None:
            child_player = child.state.current_player()

        # 當回合切換時，反轉子節點的價值
        if (
            parent_player is not None
            and child_player is not None
            and child_player != parent_player
        ):
            q = -q_child
        else:
            q = q_child
        u = PUCT_C * child.prior * sqrt_total / float(1 + child.visit_count)
        score = q + u

        if score > best_score + 1e-12:
            best_score = score
            best_actions = [action]
        elif abs(score - best_score) <= 1e-12:
            best_actions.append(action)

    return random.choice(best_actions)


def backup(path: List[MCTSNode], leaf_value: float, leaf_to_play: int) -> None:
    """從葉節點到根節點回傳價值。

    價值儲存從每個節點「輪到玩家」的視角。
    """
    white_value = leaf_value if leaf_to_play == WHITE else -leaf_value
    for node in path:
        node_player = leaf_to_play if node.to_play is None else node.to_play
        value_for_node = (
            white_value if node_player == WHITE else -white_value
        )
        node.value_sum += value_for_node
        node.visit_count += 1


def apply_root_dirichlet_noise(root: MCTSNode, head: str) -> None:
    """對根節點的子節點先驗施加 Dirichlet 雜訊。"""
    if not USE_ROOT_DIRICHLET_NOISE:
        return
    if not root.children:
        return

    legal_actions = sorted(root.children.keys())
    alpha = dirichlet_alpha_for_head(head)
    noise = np.random.dirichlet([alpha] * len(legal_actions))
    eps = ROOT_DIRICHLET_EPS

    for i, action in enumerate(legal_actions):
        child = root.children[action]
        child.prior = (1.0 - eps) * child.prior + eps * float(noise[i])


def dirichlet_alpha_for_head(head: str) -> float:
    """根據 head 名稱回傳 Dirichlet alpha 值。"""
    if head == HEAD_ACTION:
        return ROOT_DIRICHLET_ALPHA_ACTION
    raise ValueError(f"Unknown head: {head}")


def temperature_for_decision(decision_idx: int) -> float:
    """根據決策步數回傳採樣溫度。"""
    from config import SELFPLAY_CFG

    if decision_idx < SELFPLAY_CFG.temp_switch_decision:
        return SELFPLAY_CFG.temp_high
    return SELFPLAY_CFG.temp_low


def run_mcts_python(
    policy: torch.nn.Module,
    root_state: object,
    device: torch.device,
    apply_dirichlet_noise: bool = True,
    simulations: int = 600,
) -> Tuple[
    str,
    int,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    float,
]:
    """純 Python MCTS 搜尋引擎。

    參數
    ----
    policy : 策略網路
    root_state : 根節點狀態（PhaseGameState 或 CppPhaseGameStateAdapter）
    device : torch 裝置
    apply_dirichlet_noise : 是否在根節點添加探索雜訊
    simulations : 模擬次數




        回傳
    ----
    (head, action_dim, legal_mask, visits, replay_board, replay_global, root_value)
    root_value : 根節點訪問加權 Q value（root 玩家視角）
    """
    if root_state.is_done():
        raise ValueError("Cannot run MCTS from terminal state")

    root = MCTSNode(
        prior=1.0,
        to_play=root_state.current_player(),
        state=root_state.clone(),
    )

    # 根節點評估與展開
    root_eval = evaluate_leaf_batch(policy, [root.state], device)
    head, action_dim, legal_mask, _ = expand_nodes_from_eval(
        [root], root_eval
    )[0]
    if apply_dirichlet_noise:
        apply_root_dirichlet_noise(root, head)

    processed = 0
    while processed < simulations:
        chunk = min(MCTS_LEAF_BATCH_SIZE, simulations - processed)
        processed += chunk

        pending_nodes: Dict[int, MCTSNode] = {}
        pending_paths: Dict[int, List[List[MCTSNode]]] = {}

        for _ in range(chunk):
            node = root
            path = [node]

            while True:
                state = node.state
                if state is None:
                    raise ValueError("MCTS node state is not initialized")

                if state.is_done():
                    leaf_to_play = state.current_player()
                    leaf_value = terminal_value_for_current_player(
                        state.winner(), leaf_to_play
                    )
                    backup(path, leaf_value, leaf_to_play)
                    break

                if not node.expanded:
                    nid = id(node)
                    pending_nodes[nid] = node
                    pending_paths.setdefault(nid, []).append(path)
                    break

                action = select_child_action(node)
                child = node.children[action]
                if child.state is None:
                    child_state = state.clone()
                    child_state.apply_action(
                        action, legal_mask=node.legal_mask
                    )
                    child.state = child_state
                    if not child_state.is_done():
                        child.to_play = child_state.current_player()
                elif child.to_play is None and not child.state.is_done():
                    child.to_play = child.state.current_player()

                node = child
                path.append(node)

        # 批次評估所有待展開節點
        if pending_nodes:
            nodes = list(pending_nodes.values())
            states = [n.state for n in nodes]
            if any(s is None for s in states):
                raise ValueError("pending node has empty state")
            evals = evaluate_leaf_batch(
                policy, [s for s in states if s is not None], device
            )
            expanded = expand_nodes_from_eval(nodes, evals)
            for node, (_, _, _, leaf_value) in zip(nodes, expanded):
                nid = id(node)
                node_state = node.state
                if node_state is None:
                    raise ValueError("expanded node lost state")
                leaf_to_play = node_state.current_player()
                for path in pending_paths.get(nid, []):
                    backup(path, leaf_value, leaf_to_play)

    # 收集根節點訪問次數
    visits = torch.zeros(action_dim, dtype=torch.float32)
    for action, child in root.children.items():
        visits[action] = float(child.visit_count)

    if visits.sum().item() <= 0:
        legal = legal_mask[:action_dim]
        n_legal = int(legal.sum().item())
        if n_legal > 0:
            visits[legal] = 1.0 / float(n_legal)

    root_value = root.mean_value()
    return (
        head,
        action_dim,
        legal_mask.to(device="cpu"),
        visits,
        None,
        None,
        float(root_value),
    )
