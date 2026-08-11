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
from mcts.cpp_manager import CppSearchManager


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
    search_manager: Optional[CppSearchManager] = None,
) -> GateResult:
    """執行 candidate vs guard 評估。

    candidate 和 guard 輪流先手，進行 n_games 局對戰。
    每步雙方均使用 MCTS 搜尋決定動作。

    若傳入已建立的 search_manager（CppSearchManager，與 self-play 共用的
    常駐 C++ worker pool），則所有局數會一次平行推進，同一個 SearchManager
    的 num_threads 個 worker 執行緒同時搜尋多棵樹（reuse C++ 多執行緒），
    並由本函式依 current_player 把各局分成 candidate 組 / guard 組，
    分兩次 run_search（candidate 組餵 candidate、guard 組餵 guard）。
    gate 只統計勝負，不蒐集 samples。

    若 search_manager 為 None、或平行路徑執行失敗，則自動 fallback 到
    原本的同步（一次一局）迴圈。
    """
    _ = game_cfg, hp  # 保留參數供未來擴展

    simulations = int(gate_cfg.simulations_per_decision)
    temperature = float(gate_cfg.temperature)

    # ── 平行路徑：重用 C++ SearchManager 的常駐 worker pool ──────
    if search_manager is not None and _cfg._ACTIVE_STATE_BACKEND == "cpp":
        try:
            return _gate_keeper_parallel(
                candidate,
                guard,
                n_games,
                device,
                simulations,
                temperature,
                env_cfg,
                search_manager,
            )
        except Exception as exc:
            print(
                "[gate] CppSearchManager parallel gate failed; "
                f"falling back to sync. {type(exc).__name__}: {exc}"
            )

    # ── 同步 fallback 路徑（一次一局）──────────────────────────
    result = GateResult()

    for game_idx in range(n_games):
        env = TzaarEnv(env_cfg)
        env.reset()

        candidate_player = WHITE if game_idx % 2 == 0 else BLACK

        while env.game_in_progress:
            current_player = env.current_player
            policy = candidate if current_player == candidate_player else guard

            root_state = _build_phase_state_from_env(env)

            with torch.no_grad():
                _, _, legal_mask, visits, _, _ = run_mcts(
                    policy,
                    root_state,
                    device,
                    apply_dirichlet_noise=False,
                    simulations=simulations,
                )

            action_idx = _sample_action_from_visits(
                visits, legal_mask, temperature
            )

            env.step(action_idx)

        winner = _winner_from_result(env.last_game_result)

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


def _sample_action_from_visits(
    visits: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float,
) -> int:
    """根據 MCTS 訪問次數 + 溫度採樣一個動作索引。

    temperature <= 1e-6 時退化成貪婪選取最大訪問數的動作。
    """
    legal = legal_mask[: visits.shape[0]]
    legal_visits = visits.clone()
    legal_visits[~legal] = 0.0

    if legal_visits.sum().item() <= 0:
        return int(torch.multinomial(legal.float(), 1).item())
    if temperature <= 1e-6:
        return int(torch.argmax(legal_visits).item())

    adjusted = torch.pow(legal_visits, 1.0 / temperature)
    adjusted[~legal] = 0.0
    adjusted = adjusted / adjusted.sum().clamp_min(1e-8)
    return int(torch.multinomial(adjusted, 1).item())


def _winner_from_result(result: Any) -> int:
    """把環境結果轉成 winner（WHITE / BLACK / 0=draw）。"""
    if result is None:
        return 0
    if result.value == WHITE:
        return WHITE
    if result.value == BLACK:
        return BLACK
    return 0


def _gate_keeper_parallel(
    candidate: torch.nn.Module,
    guard: torch.nn.Module,
    n_games: int,
    device: torch.device,
    simulations: int,
    temperature: float,
    env_cfg: Any,
    search_manager: CppSearchManager,
) -> GateResult:
    """平行 gate：重用 C++ SearchManager 常駐 worker pool。

    所有 n_games 局同時 active，每一輪決策：
      1. 依 env.current_player 把 active 局分成 candidate 組 / guard 組。
      2. 對每組各呼叫 reset_trees() + run_search(policy)。
         - candidate 組餵 candidate，guard 組餵 guard。
      3. 對每局用回傳的 visits 採樣動作 → env.step()。
      4. 結束的局統計勝負並移除；未完的留在下一輪。

    gate 只統計勝負，不建構/回傳任何 PolicySample。
    """
    result = GateResult()

    # 建立 active 局池。candidate 與 guard 輪流先手。
    active: list = []
    for i in range(n_games):
        env = TzaarEnv(env_cfg)
        env.reset()
        active.append(
            {
                "env": env,
                "candidate_player": WHITE if i % 2 == 0 else BLACK,
            }
        )

    completed = 0
    while completed < n_games:
        # 依 current_player 分組
        candidate_entries: list = []
        guard_entries: list = []
        for entry in active:
            if entry["env"].current_player == entry["candidate_player"]:
                candidate_entries.append(entry)
            else:
                guard_entries.append(entry)

        # candidate 組搜尋
        candidate_outputs = _run_parallel_batch(
            candidate_entries, candidate, device, simulations, search_manager
        )
        # guard 組搜尋
        guard_outputs = _run_parallel_batch(
            guard_entries, guard, device, simulations, search_manager
        )
        all_outputs = {**candidate_outputs, **guard_outputs}

        survivors: list = []
        for entry in candidate_entries + guard_entries:
            out = all_outputs.get(id(entry))
            if out is None:
                raise RuntimeError("missing parallel search output for a game")
            visits = out["visits"]
            legal_mask = out["legal_mask"]

            action_idx = _sample_action_from_visits(
                visits, legal_mask, temperature
            )
            env = entry["env"]
            env.step(action_idx)

            if env.game_in_progress:
                survivors.append(entry)
            else:
                winner = _winner_from_result(env.last_game_result)
                if winner == 0:
                    result.draws += 1
                elif winner == entry["candidate_player"]:
                    result.candidate_wins += 1
                else:
                    result.guard_wins += 1
                completed += 1

                if completed % 10 == 0 or completed == n_games:
                    print(
                        f"  [gate][parallel] completed {completed}/{n_games}: "
                        f"candidate={result.candidate_wins} "
                        f"guard={result.guard_wins} "
                        f"draw={result.draws} win_rate={result.win_rate:.3f}"
                    )

        active = survivors

    return result


def _run_parallel_batch(
    entries: list,
    policy: torch.nn.Module,
    device: torch.device,
    simulations: int,
    search_manager: CppSearchManager,
) -> Dict[int, Any]:
    """對一組 root states 執行一次 C++ SearchManager 平行搜尋。

    entries 必須為同一組（全由 candidate 或全由 guard 掌控），
    因為 C++ SearchManager 一次 run_search 只能套單一 policy。

    回傳 { id(entry): output_dict }。
    """
    if not entries:
        return {}

    root_states = [_build_phase_state_from_env(entry["env"]) for entry in entries]

    search_manager.reset_trees(
        root_states,
        simulations=simulations,
        apply_dirichlet_noise=False,  # gate 不做探索雜訊
    )
    outputs = search_manager.run_search(policy, device)

    if len(outputs) != len(entries):
        raise RuntimeError(
            "SearchManager output count mismatch: "
            f"{len(outputs)} vs {len(entries)}"
        )

    return {id(entries[i]): outputs[i] for i in range(len(entries))}


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
