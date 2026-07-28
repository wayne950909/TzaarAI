"""Self-play scheduling and runtime mode helpers for training.

這個模組承接 training.loop 的 self-play 細節，目標是把「一個 update
如何產生自我對弈資料」集中在同一個地方。

目前它做四件事：
1. 判斷 C++ SearchManager self-play 條件是否成立
2. 維護 active game pool（多局同時推進）
3. 透過 CppSearchManager 執行批次 MCTS 搜尋（常駐 C++ worker pool + 雙 buffer）
4. 在 CppSearchManager 路徑出錯時自動降級成 sync-single

注意：async 路徑已從 Python thread-based 的 async_worker
切換到 C++ SearchManager 常駐 worker pool（CppSearchManager）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

import config as _cfg
from config import ASYNC_MCTS_CFG, MCTS_CFG, SELFPLAY_CFG, TRAINING_CFG
from core.action import N_ACTIONS
from core.env import EnvConfig, TzaarEnv
from core.types import GameResult
from training.metrics import record_exploration_stats
from training.sample import PolicySample
from mcts.cpp_manager import CppSearchManager


def winner_sign_from_result(result: Optional[GameResult]) -> int:
    """把環境結果轉成訓練 value target 需要的 winner sign。"""
    if result == GameResult.WHITE_WIN:
        return 1
    if result == GameResult.BLACK_WIN:
        return -1
    return 0


def assign_value_targets(samples: List[PolicySample], winner_sign: int) -> None:
    """在一局結束後，回填該局所有 sample 的 winner/value target。"""
    for sample in samples:
        sample.winner_sign = int(winner_sign)
        if winner_sign == 0:
            sample.value_target = 0.0
        elif winner_sign == sample.player:
            sample.value_target = 1.0
        else:
            sample.value_target = -1.0


def is_async_selfplay_ready() -> bool:
    """檢查目前執行期是否具備 C++ SearchManager self-play 條件。
    
    即使 parallel_games = 1 也能使用 SearchManager
    （1 CPU worker + 1 result_handler 的多執行緒架構）。
    """
    return (
        ASYNC_MCTS_CFG.enabled
        and int(ASYNC_MCTS_CFG.parallel_games) >= 1
        and int(TRAINING_CFG.games_per_update) >= 1
        and _cfg._ACTIVE_STATE_BACKEND == "cpp"
        and _cfg._ACTIVE_CPP_MODULE is not None
    )


def log_runtime_mode() -> None:
    """輸出目前訓練將走哪種 self-play / backend 路徑。"""
    async_requested = (
        ASYNC_MCTS_CFG.enabled
        and int(ASYNC_MCTS_CFG.parallel_games) >= 1
        and int(TRAINING_CFG.games_per_update) >= 1
    )
    async_ready = is_async_selfplay_ready()
    mode = "CppSearchManager" if async_ready else "sync-single"
    print(
        "[runtime] "
        f"backend={_cfg._ACTIVE_STATE_BACKEND} | "
                f"selfplay_mode={mode} | "
        f"async_enabled={ASYNC_MCTS_CFG.enabled} | "
        f"parallel_games={ASYNC_MCTS_CFG.parallel_games} | "
        f"num_threads={ASYNC_MCTS_CFG.num_threads} | "
        f"infer_max_batch={ASYNC_MCTS_CFG.infer_max_batch} | "
        f"infer_max_wait_ms={ASYNC_MCTS_CFG.infer_max_wait_ms} | "
        f"response_timeout_s={ASYNC_MCTS_CFG.response_timeout_s}"
    )
    if async_requested and not async_ready:
        reason = (
            "cpp backend unavailable"
            if _cfg._ACTIVE_STATE_BACKEND != "cpp" or _cfg._ACTIVE_CPP_MODULE is None
            else "invalid async settings"
        )
        print(f"[runtime] async requested but disabled: {reason}")


def build_phase_state_from_env(env: TzaarEnv) -> object:
    """把 Env 目前局面轉成供 MCTS 使用的 PhaseGameState。

這裡是 Python env 與 MCTS 搜尋之間的橋接點。
對 C++ backend 來說，這一步只需要 clone 已存在的 cpp_state；
對 Python backend 才需要重建 PythonPhaseGameState。
"""
    if _cfg._ACTIVE_STATE_BACKEND == "cpp" and _cfg._ACTIVE_CPP_MODULE is not None:
        cpp = env.cpp_state
        if cpp is None:
            raise RuntimeError(
                "C++ backend active but env.cpp_state is None. "
                "Ensure config.STATE_BACKEND matches the loaded backend."
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


def sample_action_and_target(
    visits: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float,
) -> tuple[int, torch.Tensor]:
    """由 visits + temperature 產生實際動作與對應訓練 target policy。"""
    legal = legal_mask[:visits.shape[0]]
    legal_visits = visits.clone()
    legal_visits[~legal] = 0.0

    n_legal = int(legal.sum().item())
    if n_legal <= 0:
        raise ValueError("No legal actions available")

    if legal_visits.sum().item() <= 0:
        probs = torch.zeros_like(legal_visits)
        probs[legal] = 1.0 / float(n_legal)
    elif temperature <= 1e-6:
        probs = torch.zeros_like(legal_visits)
        best_action = int(torch.argmax(legal_visits).item())
        probs[best_action] = 1.0
    else:
        adjusted = torch.pow(legal_visits, 1.0 / temperature)
        adjusted[~legal] = 0.0
        if adjusted.sum().item() > 1e-12:
            probs = adjusted / adjusted.sum()
        else:
            probs = torch.zeros_like(legal_visits)
            probs[legal] = 1.0 / float(n_legal)

    action = int(torch.multinomial(probs, num_samples=1).item())
    return action, probs


def collect_selfplay_samples(
    policy: torch.nn.Module,
    device: torch.device,
    env_cfg: EnvConfig,
    update_idx: int,
    stats: Dict[str, float],
    total_updates: int,
    rng_seed_offset: Any,
    search_manager: Optional[CppSearchManager] = None,
) -> tuple[List[PolicySample], int, int, float]:
    """執行一個 update 需要的 self-play，並回傳訓練樣本。

    如果 search_manager 不為 None 且 C++ backend 可用，則使用
    CppSearchManager（常駐 C++ worker pool + 雙 buffer）加速批次搜尋，
    否則使用 sync-single 搜尋作為降級路徑。

    回傳：
    fresh_samples      當前 update 產生的新樣本
    games_played       實際完成的局數
    total_game_samples 新增樣本總數
    last_temperature   最後一個決策使用的溫度（供 checkpoint 記錄）

    主要資料流：
    active envs -> root states -> MCTS -> sampled action -> env.step -> PolicySample
    terminal game -> assign_value_targets -> fresh_samples
"""
    from mcts import run_mcts as _mcts_search

    n_games = int(TRAINING_CFG.games_per_update)
    async_ready = (
        is_async_selfplay_ready()
        and search_manager is not None
    )
    async_active = async_ready

    fresh_samples: List[PolicySample] = []
    games_played = 0
    total_game_samples = 0
    last_temperature = float(SELFPLAY_CFG.temp_low)

    # active pool 中每個 entry 代表一局尚未結束的對局。
    # 它保留 env、本局累積樣本、目前 decision index。
    active: List[Dict[str, Any]] = []
    launched = 0

    while games_played < n_games:
                # 補滿 active pool 到 parallel_games 上限
        # 即使 async_active=True 且 parallel_games=1，仍走 CppSearchManager
        parallel_games = min(
            n_games - games_played,
            int(ASYNC_MCTS_CFG.parallel_games) if async_active else 1,
        )
        # 確保至少 1 局
        parallel_games = max(parallel_games, 1)
        while launched < n_games and len(active) < parallel_games:
            game_i = launched
            rand_seed = rng_seed_offset(
                update_idx * n_games + game_i,
                int(total_updates) * n_games,
                0,
                2**31 - 1,
            )
            _ = rand_seed

            env = TzaarEnv(env_cfg)
            env.reset()
            active.append({"env": env, "decision_step": 0, "samples": []})
            launched += 1

        if not active:
            break

        # prepared 保存這一輪 batch 推進每一局所需的觀測快照，
        # 避免 MCTS 完成後又重新 observe 一次導致樣本與搜尋局面對不齊。
        root_states: List[Any] = []
        prepared: List[Dict[str, Any]] = []
        next_active: List[Dict[str, Any]] = []

        for entry in active:
            env = entry["env"]
            if not env.game_in_progress:
                winner_sign = winner_sign_from_result(env.last_game_result)
                samples = entry["samples"]
                assign_value_targets(samples, winner_sign)
                fresh_samples.extend(samples)
                total_game_samples += len(samples)
                games_played += 1
                continue

            decision_step = int(entry["decision_step"])
            temperature = (
                float(SELFPLAY_CFG.temp_low)
                if decision_step >= SELFPLAY_CFG.temp_switch_decision
                else float(SELFPLAY_CFG.temp_high)
            )
            last_temperature = temperature

            current_player = int(env.current_player)
            obs = env.observe()
            obs_tensor = obs.to_tensor(device=device).unsqueeze(0)
            global_f = obs.to_global_tensor().unsqueeze(0)

            root_states.append(build_phase_state_from_env(env))
            prepared.append(
                {
                    "entry": entry,
                    "env": env,
                    "current_player": current_player,
                    "temperature": temperature,
                    "obs_tensor": obs_tensor,
                    "global_f": global_f,
                }
            )
            next_active.append(entry)

        active = next_active
        if not root_states:
            continue

                        # ── 執行 MCTS 批次搜尋 ─────────────────────────
        # 走 CppSearchManager 路徑或 sync-single 降級路徑
        apply_noise = bool(update_idx > 0)
        simulations = int(MCTS_CFG.simulations)
        use_cpp_manager = async_active

        if use_cpp_manager:
            try:
                # 使用 C++ SearchManager（常駐 worker pool + 雙 buffer）
                search_manager.reset_trees(root_states, simulations=simulations)
                search_outputs = search_manager.run_search(policy, device)
                # 將搜尋輸出轉為 _mcts_search 格式以便重用 action sampling 邏輯
                batch_outputs = []
                for output in search_outputs:
                    batch_outputs.append((
                        output["head"]
                        if "head" in output
                        else _cfg.HEAD_ACTION,
                        output["action_dim"],
                        output["legal_mask"],
                        output["visits"],
                        output["replay_board"],
                        output["replay_global"],
                    ))
            except Exception as exc:
                print(
                    "[runtime] CppSearchManager failed in update "
                    f"{update_idx}; falling back to sync-single. "
                    f"reason={type(exc).__name__}: {exc}"
                )
                async_active = False
                batch_outputs = [
                    _mcts_search(
                        policy,
                        state,
                        device,
                        apply_dirichlet_noise=apply_noise,
                        simulations=simulations,
                    )
                    for state in root_states
                ]
        else:
            # sync-single 降級路徑：逐局面呼叫 Python/C++ SearchSession
            batch_outputs = [
                _mcts_search(
                    policy,
                    state,
                    device,
                    apply_dirichlet_noise=apply_noise,
                    simulations=simulations,
                )
                for state in root_states
            ]

        # 根據 MCTS 結果逐局採樣動作，然後把尚未結束的局留在 survivors。
        survivors: List[Dict[str, Any]] = []
        for prep, (_, _, legal_mask, visits, _, _) in zip(prepared, batch_outputs):
            entry = prep["entry"]
            env = prep["env"]
            temperature = float(prep["temperature"])

            legal_mask_cpu = (
                legal_mask.to(device="cpu", dtype=torch.bool).clone()
                if legal_mask.device.type != "cpu"
                else legal_mask.clone().to(dtype=torch.bool)
            )

            record_exploration_stats(
                stats,
                visits / visits.sum().clamp_min(1e-12),
                legal_mask_cpu,
                temperature,
            )

            action, target_pi = sample_action_and_target(
                visits,
                legal_mask_cpu,
                temperature,
            )
            env.step(action)

            sample_state = prep["obs_tensor"].squeeze(0).detach().to("cpu", dtype=torch.float32)
            sample_global = prep["global_f"].squeeze(0).detach().to("cpu", dtype=torch.float32)

            entry["samples"].append(
                PolicySample(
                    state=sample_state,
                    global_features=sample_global,
                    action_dim=N_ACTIONS,
                    legal_mask_padded=legal_mask_cpu,
                    target_pi_padded=target_pi.detach().to("cpu", dtype=torch.float32).clone(),
                    player=int(prep["current_player"]),
                )
            )
            entry["decision_step"] = int(entry["decision_step"]) + 1

            if env.game_in_progress:
                survivors.append(entry)
            else:
                winner_sign = winner_sign_from_result(env.last_game_result)
                samples = entry["samples"]
                assign_value_targets(samples, winner_sign)
                fresh_samples.extend(samples)
                total_game_samples += len(samples)
                games_played += 1

        active = survivors

    return fresh_samples, games_played, total_game_samples, last_temperature