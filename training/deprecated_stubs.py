"""
training/deprecated_stubs.py — 向後相容的別名（供新版 training/loop.py 使用）

新版 training/loop.py 從舊版腳本引入了許多需要簡化或修正的引用。
這個檔案提供必要的向後相容別名和修正。
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

import config as _cfg
from config import TRAINING_CFG, MCTS_CFG, SELFPLAY_CFG, ASYNC_MCTS_CFG, GATE_CFG, REPLAY_CFG, CPP_BACKEND_REQUIRED
from core.action import PASS_ACTION_IDX
from mcts import run_mcts, run_mcts_batch, build_cnn_state_12x9x9, build_cnn_global_features
from mcts.search import temperature_for_decision, sample_action_and_target
from mcts.state_builder import terminal_value_for_current_player


def _make_phase_state() -> Any:
    """統一的狀態工廠函式（與舊版 update_cnn_mcts_no_heuristic.py 中的 _make_phase_state 同步）。"""
    from config import _ACTIVE_STATE_BACKEND, _ACTIVE_CPP_MODULE, CPP_BACKEND_REQUIRED

    if _cfg._ACTIVE_STATE_BACKEND == "cpp":
        if _cfg._ACTIVE_CPP_MODULE is None:
            if CPP_BACKEND_REQUIRED:
                raise RuntimeError("cpp backend is active but module handle is missing")
            return _make_python_phase_state()
        from state import create_cpp_phase_state
        return create_cpp_phase_state(_cfg._ACTIVE_CPP_MODULE)
    return _make_python_phase_state()


def _make_python_phase_state() -> Any:
    """建立 Python 版的 PhaseGameState。"""
    from state.phase_state import PythonPhaseGameState
    return PythonPhaseGameState()


def _sample_action_and_target(visits: torch.Tensor, legal_mask: torch.Tensor, temperature: float) -> Tuple[int, torch.Tensor]:
    """從 MCTS 訪問次數採樣動作並回傳目標策略。"""
    legal = legal_mask[:visits.shape[0]]
    legal_visits = visits.clone()
    legal_visits[~legal] = 0.0

    if legal_visits.sum().item() <= 0:
        probs = torch.zeros_like(legal_visits)
        n_legal = int(legal.sum().item())
        if n_legal <= 0:
            raise ValueError("No legal actions available when sampling from visits")
        probs[legal] = 1.0 / float(n_legal)
    elif temperature <= 1e-6:
        probs = torch.zeros_like(legal_visits)
        probs[int(torch.argmax(legal_visits).item())] = 1.0
    else:
        adjusted = torch.pow(legal_visits, 1.0 / temperature)
        adjusted[~legal] = 0.0
        probs = adjusted / adjusted.sum().clamp_min(1e-8)

    action = int(torch.multinomial(probs, num_samples=1).item())
    return action, probs
