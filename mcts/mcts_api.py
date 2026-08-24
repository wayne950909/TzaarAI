"""
mcts/mcts_api.py — MCTS 統一 API

提供 run_mcts（單一搜尋）和 run_mcts_batch（批次搜尋）
作為 MCTS 引擎的頂層入口，自動選擇 Python / C++ / 非同步後端。

這裡是整個搜尋系統的 routing layer：
- `run_mcts()` 處理單局面
- `run_mcts_batch()` 處理多局面
- 呼叫者不需要自己知道目前是 python backend、cpp backend、
    還是 async batch 路徑
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

import config as _cfg
from config import (
    ASYNC_MCTS_CFG,
    CPP_BACKEND_REQUIRED,
    MCTS_CFG,
)
from mcts.search import run_mcts_python
from mcts.cpp_session import run_mcts_cpp_session
from mcts.async_worker import run_mcts_cpp_batch_async


def run_mcts(
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
    float,
]:
    """執行 MCTS 搜尋（自動選擇後端）。

    參數
    ----
    policy : 策略網路
    root_state : 根狀態（PhaseGameState 或 CppPhaseGameStateAdapter）
    device : torch 裝置
    apply_dirichlet_noise : 是否在根節點添加 Dirichlet 雜訊
    simulations : 模擬次數

        回傳
    ----
    (head, action_dim, legal_mask, visits, replay_board, replay_global, root_value)
    root_value : 根節點訪問加權 Q value（root 玩家視角）
    """
    if root_state.is_done():
        raise ValueError("Cannot run MCTS from terminal state")

    # 單局面時，優先嘗試 C++ SearchSession；
    # 若 backend / state 不符合條件才回退到 Python 搜尋。
    if _cfg._ACTIVE_STATE_BACKEND == "cpp":
        if _cfg._ACTIVE_CPP_MODULE is None:
            if CPP_BACKEND_REQUIRED:
                raise RuntimeError(
                    "cpp backend requested but module is not loaded"
                )
            print(
                "[mcts] cpp backend unavailable; falling back to python"
            )
            return run_mcts_python(
                policy,
                root_state,
                device,
                apply_dirichlet_noise,
                simulations,
            )

        if not hasattr(root_state, "_inner"):
            if CPP_BACKEND_REQUIRED:
                raise RuntimeError(
                    "cpp backend requires CppPhaseGameStateAdapter state"
                )
            print(
                "[mcts] non-cpp state received for cpp backend; "
                "falling back to python"
            )
            return run_mcts_python(
                policy,
                root_state,
                device,
                apply_dirichlet_noise,
                simulations,
            )

        return run_mcts_cpp_session(
            policy,
            root_state,
            device,
            apply_dirichlet_noise,
            simulations,
        )

    return run_mcts_python(
        policy, root_state, device, apply_dirichlet_noise, simulations
    )


def run_mcts_batch(
    policy: torch.nn.Module,
    root_states: List[Any],
    device: torch.device,
    apply_dirichlet_noise: bool = True,
    simulations: int = 600,
) -> List[
    Tuple[
        str,
        int,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        float,
    ]
]:
    """批次執行 MCTS 搜尋（自動選擇後端）。

    若條件允許，使用非同步 pipeline 加速。
    root_states : 根狀態列表
    device : torch 裝置
    apply_dirichlet_noise : 是否在根節點添加 Dirichlet 雜訊
    simulations : 模擬次數

        回傳
    ----
    List of (head, action_dim, legal_mask, visits, replay_board, replay_global, root_value)
    root_value : 根節點訪問加權 Q value（root 玩家視角）
    """
    if not root_states:
        return []

    # 目前的「批次」只表示呼叫端一次丟入多個 root states。
    # 真正是否走 async，要看 backend、module、state 型別是否同時滿足。
    can_use_async = (
        ASYNC_MCTS_CFG.enabled
        and _cfg._ACTIVE_STATE_BACKEND == "cpp"
        and _cfg._ACTIVE_CPP_MODULE is not None
        and len(root_states) > 1
        and all(hasattr(state, "_inner") for state in root_states)
    )

    if can_use_async:
        return run_mcts_cpp_batch_async(
            policy,
            root_states,
            device,
            apply_dirichlet_noise,
            simulations,
        )

    return [
        run_mcts(
            policy,
            state,
            device,
            apply_dirichlet_noise=apply_dirichlet_noise,
            simulations=simulations,
        )
        for state in root_states
    ]
