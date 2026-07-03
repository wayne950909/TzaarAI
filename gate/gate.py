"""
gate/gate.py — Gatekeeper 評估邏輯

在訓練過程中，使用 MCTS 讓 candidate 與 guard 進行多局對戰，
以勝率決定是否晉升 candidate 為新的 guard。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from core.action import PASS_ACTION_IDX
from core.board import TRAINING_GAME_STEPS
from core.env import TzaarEnv, EnvConfig
from core.constants import WHITE, BLACK, PHASE_STEP_1, PHASE_STEP_2
import config as _cfg

from mcts import run_mcts
from mcts.search import temperature_for_decision


@dataclass
class GateResult:
    """Gatekeeper 評估結果。"""
    candidate_wins: int = 0
    guard_wins: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.candidate_wins + self.guard_wins + self.draws

    @property
    def win_rate(self) -> float:
        """candidate 勝率（draw 算 0.5）。"""
        if self.total == 0:
            return 0.0
        return (self.candidate_wins + 0.5 * self.draws) / self.total


def gate_keeper(
    candidate: torch.nn.Module,
    guard: torch.nn.Module,
    n_games: int,
    device: torch.device,
    gate_cfg: Any,
    game_cfg: Any,
    env_cfg: Any,
    hp: Optional[Any] = None,
) -> GateResult:
    """執行 candidate vs guard 評估。

    candidate 和 guard 輪流先手，進行 n_games 局對戰。
    每步雙方均使用 MCTS 搜尋決定動作。

    參數
    ----
    candidate : 被評估的候選網路
    guard : 目前的守門員網路
    n_games : 對戰局數
    device : torch 裝置
    gate_cfg : GatekeeperConfig 實例
    game_cfg : 保留參數（未使用）
    env_cfg : EnvConfig 實例
    hp : 啟發式策略（未使用）

    回傳
    ----
    GateResult（包含勝負統計）
    """
    _ = game_cfg, hp  # 保留參數供未來擴展

    result = GateResult()
    simulations = int(gate_cfg.simulations_per_decision)
    temperature = float(gate_cfg.temperature)

    for game_idx in range(n_games):
        env = TzaarEnv(env_cfg)
        env.reset()

        candidate_player = WHITE if game_idx % 2 == 0 else BLACK

        while env.game_in_progress:
            current_player = env.current_player
            policy = candidate if current_player == candidate_player else guard

            obs = env.observe()
            root_state = _build_phase_state_from_env(env)

            with torch.no_grad():
                _, _, legal_mask, visits, _, _ = run_mcts(
                    policy,
                    root_state,
                    device,
                    apply_dirichlet_noise=False,
                    simulations=simulations,
                )

            # 根據訪問次數採樣（使用指定溫度）
            probs = visits.clone()
            legal = legal_mask[:visits.shape[0]]
            legal_visits = probs.clone()
            legal_visits[~legal] = 0.0

            if legal_visits.sum().item() <= 0:
                action_idx = int(torch.multinomial(legal.float(), 1).item())
            elif temperature <= 1e-6:
                action_idx = int(torch.argmax(legal_visits).item())
            else:
                adjusted = torch.pow(legal_visits, 1.0 / temperature)
                adjusted[~legal] = 0.0
                adjusted = adjusted / adjusted.sum().clamp_min(1e-8)
                action_idx = int(torch.multinomial(adjusted, 1).item())

            env.step(action_idx)

        gr = env.last_game_result
        if gr is not None:
            if gr.value == WHITE:
                winner = WHITE
            elif gr.value == BLACK:
                winner = BLACK
            else:
                winner = 0
        else:
            winner = 0

        if winner == 0:
            result.draws += 1
        elif winner == candidate_player:
            result.candidate_wins += 1
        else:
            result.guard_wins += 1

        if (game_idx + 1) % 10 == 0 or game_idx == n_games - 1:
            print(
                f"  [gate] game {game_idx + 1}/{n_games}: "
                f"candidate={result.candidate_wins} guard={result.guard_wins} "
                f"draw={result.draws} win_rate={result.win_rate:.3f}"
            )

    return result


def _build_phase_state_from_env(env: TzaarEnv) -> Any:
    """從 TzaarEnv 建立 PhaseGameState（供 MCTS 搜尋使用）。

    當使用 C++ 後端時，直接 clone env.cpp_state；
    當使用 Python 後端時，建立 PythonPhaseGameState。
    """
    if _cfg._ACTIVE_STATE_BACKEND == "cpp" and _cfg._ACTIVE_CPP_MODULE is not None:
        cpp = env.cpp_state
        if cpp is None:
            raise RuntimeError(
                "C++ backend active but env.cpp_state is None"
            )
        return cpp.clone()

    from state.phase_state import PythonPhaseGameState, Stage
    from TzaarAI import TzaarAIInterface

    game = env.game
    if game is None:
        raise RuntimeError("Environment not reset")

    ai = TzaarAIInterface(game)
    if game.is_waiting_second_step():
        stage = Stage.NEED_STEP2
    elif game.is_game_over():
        stage = Stage.DONE
    else:
        stage = Stage.NEED_STEP1

    return PythonPhaseGameState(game=game, stage=stage, ai=ai)
