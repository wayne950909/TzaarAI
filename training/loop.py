"""
training/loop.py — 訓練主循環

提供 run() 和 main() 的實現，包含 self-play、MCTS、訓練、gate 等流程。
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import TzaarTrain as train_module
from config import (
    TITLE,
    GUARD_TITLE,
    TRAINING_CFG,
    SELFPLAY_CFG,
    MCTS_CFG,
    GATE_CFG,
    OPTIMIZER_CFG,
    REPLAY_CFG,
    NETWORK_CFG,
    ASYNC_MCTS_CFG,
    _ACTIVE_STATE_BACKEND,
    _ACTIVE_CPP_MODULE,
)
from core.action import N_ACTIONS, PASS_ACTION_IDX
from core.board import TRAINING_GAME_STEPS, MAX_HEIGHT_NORM
from core.constants import WHITE, BLACK, PHASE_STEP_2, TRAINING_PHASE_TO_IDX
from core.env import TzaarEnv, EnvConfig
from core.types import GameResult
from gate.gate import gate_keeper
from heuristics import HeuristicPlayoutPolicy as HP
from network import PolicyNetCNNMin17
from training.sample import PolicySample
from training.metrics import (
    accumulate_kind_stats,
    format_exploration_stats,
    record_exploration_stats,
)
from training.replay import (
    replay_extend,
    save_replay_snapshot_async,
    select_train_samples,
)
from training.resume import (
    load_or_init_guard_policy,
    load_or_init_policy,
    optimizer_to,
)
from training.train_step import train_on_samples
from training.validation import validate_constants

import debugpy  # type: ignore[import-untyped]


_LOG_FILE_PATH: Optional[Path] = None


class _TeeTextIO(io.TextIOBase):
    """同時寫入到 stdout 和一個日誌檔案的輸出流。"""

    def __init__(self, log_path: Path) -> None:
        super().__init__()
        self._log_file = log_path.open("a", encoding="utf-8")
        self._log_path = log_path

    def write(self, s: str) -> int:
        written = 0
        try:
            written = self._log_file.write(s)
            self._log_file.flush()
        except Exception:
            pass
        try:
            sys.__stdout__.write(s)
            sys.__stdout__.flush()
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._log_file.flush()
        except Exception:
            pass
        try:
            sys.__stdout__.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._log_file.close()
        except Exception:
            pass


def _create_log_file(title: str) -> Path:
    """建立日誌檔案並回傳路徑。"""
    global _LOG_FILE_PATH
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{title}.log"
    _LOG_FILE_PATH = log_path
    return log_path


def _rng_seed_offset(
    update: int,
    n_updates: int,
    min_seed: int,
    max_seed: int,
) -> int:
    """基於 update 進度，將 seed 在 [min_seed, max_seed] 範圍內循環位移。"""
    if n_updates <= 1 or max_seed <= min_seed:
        return min_seed
    segment = (max_seed - min_seed) // n_updates
    if segment < 1:
        return min_seed
    return min_seed + segment * update


def _resolve_state_backend() -> tuple[str, object]:
    """解析狀態後端並載入 C++ 模組（如需要）。"""
    from state import resolve_backend_name, try_load_cpp_backend, create_cpp_phase_state

    backend = resolve_backend_name("cpp")
    cpp_module = None
    if backend == "cpp":
        cpp_module = try_load_cpp_backend()
        if cpp_module is None:
            print("[backend] C++ module not found, falling back to Python")
            backend = "python"
        else:
            print("[backend] Using C++ backend")
    else:
        print("[backend] Using Python backend")
    return backend, cpp_module


def _make_phase_state() -> object:
    """建立新的階段狀態（根據活躍後端）。"""
    from state import create_cpp_phase_state
    from state.phase_state import PythonPhaseGameState

    if _ACTIVE_STATE_BACKEND == "cpp" and _ACTIVE_CPP_MODULE is not None:
        return create_cpp_phase_state(_ACTIVE_CPP_MODULE)
    return PythonPhaseGameState()


def _run_mcts(
    policy: torch.nn.Module,
    env: TzaarEnv,
    device: torch.device,
    simulations: int,
    temperature: float,
    apply_dirichlet_noise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在當前環境狀態上執行 MCTS 搜尋。

    回傳 (visits, legal_mask)，兩者皆在 CPU 上。
    """
    from mcts import run_mcts as _mcts_search

    root_state = _build_phase_state_from_env(env)
    with torch.no_grad():
        _, action_dim, legal_mask, visits, _, _ = _mcts_search(
            policy,
            root_state,
            device,
            apply_dirichlet_noise=apply_dirichlet_noise,
            simulations=simulations,
        )
    return visits, legal_mask


def _build_phase_state_from_env(env: TzaarEnv) -> object:
    """從 TzaarEnv 建立 PhaseGameState（供 MCTS 搜尋使用）。

    當使用 C++ 後端時，直接 clone env.cpp_state（已有正確的遊戲狀態）；
    當使用 Python 後端時，建立 PythonPhaseGameState。
    """
    if _ACTIVE_STATE_BACKEND == "cpp" and _ACTIVE_CPP_MODULE is not None:
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


def _sample_action_and_target(
    visits: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float,
) -> tuple[int, torch.Tensor]:
    """從 MCTS 訪問次數採樣動作並回傳目標策略。"""
    legal = legal_mask[:visits.shape[0]]
    legal_visits = visits.clone()
    legal_visits[~legal] = 0.0

    n_legal = int(legal.sum().item())
    if n_legal <= 0:
        raise ValueError("No legal actions available")

    if legal_visits.sum().item() <= 0:
        # 均勻分布（fallback）
        probs = torch.zeros_like(legal_visits)
        probs[legal] = 1.0 / float(n_legal)
    elif temperature <= 1e-6:
        # 貪婪採樣
        probs = torch.zeros_like(legal_visits)
        best_action = int(torch.argmax(legal_visits).item())
        probs[best_action] = 1.0
    else:
        # 溫度調整採樣
        adjusted = torch.pow(legal_visits, 1.0 / temperature)
        adjusted[~legal] = 0.0
        if adjusted.sum().item() > 1e-12:
            probs = adjusted / adjusted.sum()
        else:
            probs = torch.zeros_like(legal_visits)
            probs[legal] = 1.0 / float(n_legal)

    action = int(torch.multinomial(probs, num_samples=1).item())
    return action, probs


def run(title: str) -> None:
    """執行訓練流程。

    參數
    ----
    title : 訓練任務名稱（用於檢查點/日誌命名）
    """
    # ── 設定 ──────────────────────────────────────────────
    validate_constants()

    log_file_path = _create_log_file(title)
    sys.stdout = _TeeTextIO(log_file_path)  # type: ignore[assignment]

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")
    device = torch.device("cuda")

        # 解析後端（設定 _ACTIVE_* 全域變數）
    import config as _cfg
    state_backend, cpp_module = _resolve_state_backend()
    _cfg._ACTIVE_STATE_BACKEND = state_backend
    _cfg._ACTIVE_CPP_MODULE = cpp_module
    global _ACTIVE_STATE_BACKEND, _ACTIVE_CPP_MODULE
    _ACTIVE_STATE_BACKEND = state_backend
    _ACTIVE_CPP_MODULE = cpp_module

    # ── 載入/初始化政策網路 ────────────────────────────
    start_time = time.perf_counter()
    policy, optimizer, start_update, next_ckpt_idx, replay_buffer, replay_write_idx = (
        load_or_init_policy(device)
    )
    optimizer_to(optimizer, device)
    policy.train()

    guard_policy, guard_source = load_or_init_guard_policy(device, policy)
    guard_policy.eval()
    print(f"[guard] source = {guard_source}")

    env_cfg = EnvConfig(max_steps=TRAINING_GAME_STEPS)
    total_updates = TRAINING_CFG.total_updates
    hp_config = HP.Config(
        softmax_temperature=float(MCTS_CFG.heuristic_softmax_temperature),
        prior_weight=float(MCTS_CFG.heuristic_prior_weight),
    )
    hp = HP(hp_config)

    # ── 全域統計 ──────────────────────────────────────────
    stats: Dict[str, float] = {
        "steps": 0.0,
        "entropy_sum": 0.0,
        "entropy_norm_sum": 0.0,
        "top1_sum": 0.0,
        "n_legal_sum": 0.0,
        "eff_n_sum": 0.0,
        "temp_high_steps": 0.0,
        "temp_low_steps": 0.0,
        "near_greedy_steps": 0.0,
    }
    total_fresh_samples = 0

    # ── 主訓練循環 ──────────────────────────────────────
    for update_idx in range(start_update, total_updates):
        update_start = time.perf_counter()
        games_played = 0
        fresh_samples: List[PolicySample] = []

        # ── Self‑play ────────────────────────────────────
        policy.eval()
        with torch.no_grad():
            for game_i in range(TRAINING_CFG.games_per_update):
                rand_seed = _rng_seed_offset(
                    update_idx * TRAINING_CFG.games_per_update + game_i,
                    total_updates * TRAINING_CFG.games_per_update,
                    0,
                    2**31 - 1,
                )
                _ = rand_seed  # 可選用於 RNG 種子

                env = TzaarEnv(env_cfg)
                env.reset()
                game_sample_count = 0
                temperature = float(SELFPLAY_CFG.temp_high)
                decision_step = 0

                while env.game_in_progress:
                    current_player = env.current_player

                    # 溫度排程
                    if decision_step >= SELFPLAY_CFG.temp_switch_decision:
                        temperature = float(SELFPLAY_CFG.temp_low)

                    # 觀測
                    obs = env.observe()
                    obs_tensor = obs.to_tensor(device=device).unsqueeze(0)
                    global_f = obs.to_global_tensor().unsqueeze(0)
                    legal_mask_t = obs.to_legal_mask_tensor(device=device)

                    # MCTS 搜尋
                    visits, legal_mask = _run_mcts(
                        policy,
                        env,
                        device,
                        simulations=int(MCTS_CFG.simulations),
                        temperature=temperature,
                        apply_dirichlet_noise=(update_idx > 0),
                    )

                    # 探索統計
                    record_exploration_stats(
                        stats, visits / visits.sum().clamp_min(1e-12),
                        legal_mask_t.cpu(), temperature,
                    )

                    # 採樣動作
                    action, target_pi = _sample_action_and_target(
                        visits, legal_mask, temperature,
                    )
                    target_pi_cpu = target_pi.detach().to("cpu", dtype=torch.float32).clone()
                    legal_mask_cpu = legal_mask.to(device="cpu", dtype=torch.bool).clone() if legal_mask.device.type != "cpu" else legal_mask.clone()

                    env.step(action)

                    # 收集樣本
                    sample_state = obs_tensor.squeeze(0).detach().to("cpu", dtype=torch.float32)
                    sample_global = global_f.squeeze(0).detach().to("cpu", dtype=torch.float32)
                    fresh_samples.append(
                        PolicySample(
                            state=sample_state,
                            global_features=sample_global,
                            action_dim=N_ACTIONS,
                            legal_mask_padded=legal_mask_cpu,
                            target_pi_padded=target_pi_cpu,
                            player=int(current_player),
                        )
                    )
                    game_sample_count += 1
                    decision_step += 1

                # 遊戲結束 → 賦予價值目標
                gr = env.last_game_result
                if gr is not None:
                    winner_sign = 0
                    if gr == GameResult.WHITE_WIN:
                        winner_sign = 1
                    elif gr == GameResult.BLACK_WIN:
                        winner_sign = -1

                    for sample in fresh_samples[-game_sample_count:]:
                        sample.winner_sign = int(winner_sign)
                        player = sample.player
                        if winner_sign == 0:
                            sample.value_target = 0.0
                        elif winner_sign == player:
                            sample.value_target = 1.0
                        else:
                            sample.value_target = -1.0

                    total_fresh_samples += game_sample_count
                    games_played += 1

        # ── 訓練 ────────────────────────────────────────────
        policy.train()

        if REPLAY_CFG.enabled and fresh_samples:
            replay_write_idx = replay_extend(
                replay_buffer,
                replay_write_idx,
                REPLAY_CFG.max_samples,
                fresh_samples,
            )

        train_samples = select_train_samples(fresh_samples, replay_buffer)

        # 動作類型統計
        action_kind_mass: list[float] = [0.0, 0.0, 0.0]
        kind_legal_counts: list[int] = [0, 0, 0]
        phase2_count = accumulate_kind_stats(
            train_samples,
            action_kind_mass,
            kind_legal_counts,
        )

        metrics = train_on_samples(policy, optimizer, train_samples, device)

        # ── Gate ──────────────────────────────────────────
        update_accepted = True
        gate_passed = False
        if update_idx > 0 and update_idx % GATE_CFG.eval_every_updates == 0:
            policy.eval()
            guard_policy.eval()
            with torch.no_grad():
                gate_result = gate_keeper(
                    policy,
                    guard_policy,
                    n_games=GATE_CFG.eval_games,
                    device=device,
                    gate_cfg=GATE_CFG,
                    game_cfg=None,
                    env_cfg=env_cfg,
                    hp=hp,
                )
            gate_passed = gate_result.win_rate >= GATE_CFG.winrate_threshold
            if gate_passed:
                print(f"[gate] PASSED | win_rate={gate_result.win_rate:.3f}")
                guard_policy.load_state_dict(policy.state_dict(), strict=True)
                guard_policy.eval()
            else:
                print(
                    f"[gate] FAILED | win_rate={gate_result.win_rate:.3f} "
                    f"< threshold={GATE_CFG.winrate_threshold}"
                )
                policy.load_state_dict(guard_policy.state_dict(), strict=True)
                policy.train()
                update_accepted = False

        # ── Logging ──────────────────────────────────────
        samples_in_update = len(fresh_samples)
        total_buffer = len(replay_buffer) if REPLAY_CFG.enabled else 0

        if update_idx % TRAINING_CFG.log_every == 0:
            sps_total = total_fresh_samples / (
                time.perf_counter() - start_time
            ) if (time.perf_counter() - start_time) > 0 else 0.0

            print(
                f"update={update_idx}/{total_updates} | "
                f"games={games_played} | "
                f"samples={samples_in_update} | "
                f"buffer={total_buffer} | "
                f"loss={metrics['loss']:.4f} | "
                f"pl={metrics['policy_loss']:.4f} | "
                f"vl={metrics['value_loss']:.4f} | "
                f"ent={metrics['entropy']:.4f} | "
                f"phase2={phase2_count} | "
                f"accepted={update_accepted} | "
                f"gate={gate_passed if update_idx % GATE_CFG.eval_every_updates == 0 else 'N/A'} | "
                f"{format_exploration_stats(stats)} | "
                f"sps={sps_total:.0f}"
            )

        # ── Checkpoint ──────────────────────────────────
        ckpt_saved = False
        if (update_idx % TRAINING_CFG.checkpoint_every_updates == 0) or update_idx >= total_updates - 1:
            ckpt_path = train_module.save_checkpoint(
                title=title,
                policy=policy,
                optimizer=optimizer,
                update_idx=update_idx,
                samples_in_update=samples_in_update,
                inference_temperature=temperature,
                checkpoint_index=int(next_ckpt_idx),
            )
            next_ckpt_idx += 1
            ckpt_saved = True

            if REPLAY_CFG.enabled and fresh_samples:
                save_replay_snapshot_async(
                    title=title,
                    checkpoint_index=None,
                    update_idx=update_idx,
                    replay_buffer=replay_buffer,
                    write_idx=replay_write_idx,
                    max_samples=REPLAY_CFG.max_samples,
                    base_dir=ckpt_path.parent,
                )

        update_elapsed = time.perf_counter() - update_start
        print(
            f"  [{update_idx}] update took {update_elapsed:.2f}s "
            f"| samples={samples_in_update} | "
            f"n_fresh={len(fresh_samples)} | "
            f"n_train={len(train_samples)}"
        )

    # ── 訓練結束 ──────────────────────────────────────
    final_ckpt = train_module.save_checkpoint(
        title=title,
        policy=policy,
        optimizer=optimizer,
        update_idx=total_updates - 1,
        samples_in_update=0,
        inference_temperature=float(SELFPLAY_CFG.temp_low),
        checkpoint_index=int(next_ckpt_idx),
    )
    if REPLAY_CFG.enabled and replay_buffer:
        save_replay_snapshot_async(
            title=title,
            checkpoint_index=None,
            update_idx=total_updates - 1,
            replay_buffer=replay_buffer,
            write_idx=replay_write_idx,
            max_samples=REPLAY_CFG.max_samples,
            base_dir=final_ckpt.parent,
        )

    elapsed_total = time.perf_counter() - start_time
    print(f"[done] total time: {elapsed_total:.1f}s")
    print(f"[done] total fresh samples generated: {total_fresh_samples}")

    if _LOG_FILE_PATH is not None:
        try:
            if isinstance(sys.stdout, _TeeTextIO):
                sys.stdout.close()
        except Exception:
            pass
        sys.stdout = sys.__stdout__


def main() -> None:
    """訓練腳本入口點。"""
    # 若 DEBUG=1 則啟動偵錯伺服器
    _DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    if _DEBUG_MODE:
        try:
            debugpy.listen(("0.0.0.0", 5678))
            print("Waiting for debugger to attach on port 5678...")
            debugpy.wait_for_client()
            print("Debugger attached, starting training loop.")
        except Exception as e:
            print(f"Debugger setup failed: {e}")

    try:
        run(TITLE)
    except Exception as exc:
        print(f"Fatal error: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()
