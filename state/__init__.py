"""
state — 遊戲狀態管理

提供統一的 MCTS/自我對弈狀態介面：
- PythonPhaseGameState（純 Python 實作）
- CppPhaseGameStateAdapter（C++ pybind11 包裝）

後端選擇由 config.STATE_BACKEND 控制。
"""

from state.phase_state import (
    PythonPhaseGameState,
    Stage,
    STAGE_TO_PHASE,
)

from state.cpp_adapter import (
    CppPhaseGameStateAdapter,
    create_cpp_phase_state,
    resolve_backend_name,
    try_load_cpp_backend,
)

__all__ = [
    "PythonPhaseGameState",
    "Stage",
    "STAGE_TO_PHASE",
    "CppPhaseGameStateAdapter",
    "create_cpp_phase_state",
    "resolve_backend_name",
    "try_load_cpp_backend",
]
