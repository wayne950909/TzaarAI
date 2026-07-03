"""
mcts/cpp_session.py — C++ SearchSession 包裝

封裝 C++ 模組的 SearchSession，提供同步的 Python 呼叫介面。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from core.action import N_ACTIONS
import config as _cfg
from config import (
    HEAD_ACTION,
    MCTS_CFG,
)

# 方便取用
USE_ROOT_DIRICHLET_NOISE = MCTS_CFG.use_root_dirichlet_noise
ROOT_DIRICHLET_EPS = MCTS_CFG.root_dirichlet_eps
ROOT_DIRICHLET_ALPHA_ACTION = MCTS_CFG.root_dirichlet_alpha
from mcts.state_builder import terminal_value_for_current_player


def run_mcts_cpp_session(
    policy: torch.nn.Module,
    root_state: Any,
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
]:
    """使用 C++ SearchSession 執行 MCTS 搜尋。

    注意：root_state 必須是 CppPhaseGameStateAdapter（有 _inner 屬性）。

    回傳
    ----
    (head, action_dim, legal_mask, visits, replay_board, replay_global)
    """
    module = _cfg._ACTIVE_CPP_MODULE
    if module is None:
        raise RuntimeError("cpp module is not available for SearchSession")

    cfg = module.SearchConfig()
    cfg.simulations = int(simulations)
    cfg.leaf_batch_size = int(MCTS_CFG.leaf_batch_size)
    cfg.puct_c = float(MCTS_CFG.puct_c)
    cfg.add_root_dirichlet_noise = bool(
        apply_dirichlet_noise and USE_ROOT_DIRICHLET_NOISE
    )
    cfg.root_dirichlet_eps = float(ROOT_DIRICHLET_EPS)
    cfg.root_dirichlet_alpha = float(ROOT_DIRICHLET_ALPHA_ACTION)

    session = module.SearchSession(root_state._inner, cfg)
    use_packed_api = bool(
        hasattr(session, "collect_pending_leaves_packed")
        and hasattr(session, "submit_leaf_eval_batch")
    )

    while True:
        if use_packed_api:
            packed = session.collect_pending_leaves_packed(
                int(MCTS_CFG.leaf_batch_size)
            )
            node_ids_np = (
                np.asarray(packed["node_ids"], dtype=np.int32).copy()
            )
            if node_ids_np.size == 0:
                break

            legal_masks_np = np.asarray(
                packed["legal_masks"], dtype=np.uint8
            ).copy()
            boards_np = np.asarray(
                packed["board_state_flat"], dtype=np.float32
            ).copy()
            globals_np = np.asarray(
                packed["global_features"], dtype=np.float32
            ).copy()

            board_batch = (
                torch.from_numpy(boards_np.reshape(-1, 12, 9, 9))
                .to(device)
            )
            global_batch = torch.from_numpy(globals_np).to(device)
            mask_batch = torch.from_numpy(
                legal_masks_np.astype(np.bool_, copy=False)
            ).to(device=device)

            with torch.no_grad():
                hidden = policy.encode(board_batch, global_batch)
                logits = policy.action_head(hidden)
                values = policy.forward_value(hidden).squeeze(-1)

            masked_logits = logits.masked_fill(~mask_batch, -1e9)
            priors = (
                torch.softmax(masked_logits, dim=-1)
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
            values_cpu = (
                values.to(device="cpu", dtype=torch.float32).contiguous()
            )
            session.submit_leaf_eval_batch(
                node_ids_np, priors.numpy(), values_cpu.numpy()
            )
            continue

        # fallback to old API
        leaves = session.collect_pending_leaves(
            int(MCTS_CFG.leaf_batch_size)
        )
        if not leaves:
            break

        node_ids: List[int] = []
        board_batch_items: List[torch.Tensor] = []
        global_batch_items: List[torch.Tensor] = []
        legal_masks: List[torch.Tensor] = []
        terminal_payloads: List[
            Tuple[int, List[float], float]
        ] = []

        for leaf in leaves:
            node_id = int(leaf.node_id)
            mask = torch.tensor(list(leaf.legal_mask), dtype=torch.bool)
            if bool(leaf.is_done):
                current_player = int(leaf.current_player)
                winner = int(leaf.winner)
                value = terminal_value_for_current_player(
                    None if winner == 0 else winner, current_player
                )
                terminal_payloads.append(
                    (node_id, [0.0] * N_ACTIONS, float(value))
                )
                continue

            node_ids.append(node_id)
            legal_masks.append(mask)
            board_batch_items.append(
                torch.tensor(
                    leaf.board_state_flat, dtype=torch.float32
                ).reshape(12, 9, 9)
            )
            global_batch_items.append(
                torch.tensor(leaf.global_features, dtype=torch.float32)
            )

        for node_id, priors, value in terminal_payloads:
            session.submit_leaf_eval(node_id, priors, float(value))

        if node_ids:
            board_batch = torch.stack(board_batch_items).to(device)
            global_batch = torch.stack(global_batch_items).to(device)

            with torch.no_grad():
                hidden = policy.encode(board_batch, global_batch)
                logits = policy.action_head(hidden)
                values = policy.forward_value(hidden).squeeze(-1)

            for i, node_id in enumerate(node_ids):
                mask = legal_masks[i].to(device=device)
                masked_logits = logits[i].masked_fill(~mask, -1e9)
                priors = torch.softmax(
                    masked_logits, dim=-1
                ).to(device="cpu", dtype=torch.float32)
                session.submit_leaf_eval(
                    node_id, priors.tolist(), float(values[i].item())
                )

    result = session.finish()
    if not bool(result.is_complete):
        raise RuntimeError(
            "SearchSession finished before all simulations were processed"
        )

    legal_mask = torch.from_numpy(
        np.asarray(result.legal_mask, dtype=np.bool_)
    ).clone()
    visits = torch.from_numpy(
        np.asarray(result.root_visits, dtype=np.float32)
    ).clone()
    action_dim = N_ACTIONS

    if visits.sum().item() <= 0:
        legal = legal_mask[:action_dim]
        n_legal = int(legal.sum().item())
        if n_legal > 0:
            visits[legal] = 1.0 / float(n_legal)

    root_leaf = session.root_snapshot()
    replay_board = (
        torch.from_numpy(
            np.asarray(root_leaf.board_state_flat, dtype=np.float32)
        )
        .reshape(12, 9, 9)
        .clone()
    )
    replay_global = torch.from_numpy(
        np.asarray(root_leaf.global_features, dtype=np.float32)
    ).clone()

    return (
        HEAD_ACTION,
        action_dim,
        legal_mask.to(device="cpu"),
        visits,
        replay_board,
        replay_global,
    )
