"""
mcts — Monte Carlo Tree Search 引擎

提供完整 MCTS 搜尋管線：
- Python 版節點搜尋（_run_mcts_python）
- C++ 版 Session 搜尋（_run_mcts_cpp_session）
- 非同步多 session 批次搜尋（_run_mcts_cpp_batch_async）
- 啟發式先驗混合
- 統一的 _run_mcts / _run_mcts_batch 入口

使用方式
--------
    from mcts import run_mcts
    head, action_dim, legal_mask, visits, replay_board, replay_global = run_mcts(
        policy, root_state, device
    )
"""

from mcts.node import MCTSNode
from mcts.evaluate import (
    evaluate_leaf,
    evaluate_leaf_batch,
    expand_node,
    expand_nodes_from_eval,
)
from mcts.heuristic import (
    heuristic_action_value,
    heuristic_priors_for_actions,
    blend_policy_and_heuristic_priors,
    material_score,
    mobility_capture_score,
    scarcity_bonus,
    player_eval_score,
)
from mcts.search import (
    run_mcts_python,
    select_child_action,
    backup,
    apply_root_dirichlet_noise,
    legal_action_dim_and_head,
    temperature_for_decision,
    dirichlet_alpha_for_head,
)
from mcts.cpp_session import run_mcts_cpp_session
from mcts.async_worker import (
    InferenceRequest,
    InferenceResponse,
    run_mcts_cpp_batch_async,
)
from mcts.mcts_api import run_mcts, run_mcts_batch

# 狀態建構工具（葉節點用）
from mcts.state_builder import (
    build_cnn_state_12x9x9,
    build_cnn_global_features,
    terminal_value_for_current_player,
)

__all__ = [
    "MCTSNode",
    "evaluate_leaf",
    "evaluate_leaf_batch",
    "expand_node",
    "expand_nodes_from_eval",
    "heuristic_action_value",
    "heuristic_priors_for_actions",
    "blend_policy_and_heuristic_priors",
    "material_score",
    "mobility_capture_score",
    "scarcity_bonus",
    "player_eval_score",
    "run_mcts_python",
    "select_child_action",
    "backup",
    "apply_root_dirichlet_noise",
    "legal_action_dim_and_head",
    "temperature_for_decision",
    "dirichlet_alpha_for_head",
    "run_mcts_cpp_session",
    "InferenceRequest",
    "InferenceResponse",
    "run_mcts_cpp_batch_async",
    "run_mcts",
    "run_mcts_batch",
    "build_cnn_state_12x9x9",
    "build_cnn_global_features",
    "terminal_value_for_current_player",
]
