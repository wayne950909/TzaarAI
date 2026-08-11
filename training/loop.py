"""
training/loop.py — 訓練主循環

提供 run() 和 main() 的實現，包含 self-play、MCTS、訓練、gate 等流程。

目前的設計原則：
- loop.py 只做 orchestration，不再承擔太多 self-play 細節。
- self-play 排程已抽到 training/selfplay_engine.py。
- MCTS 路徑的真正決策（sync / async / cpp / python）由 mcts/mcts_api.py 決定。

如果之後要手動改訓練流程，建議把 loop.py 當成「總控台」來看：
它負責決定每個 update 要先產生資料、再訓練、再 gate、最後存檔。
"""

from __future__ import annotations

import io
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import torch

import TzaarTrain as train_module
from config import (
    TITLE,
    TRAINING_CFG,
    SELFPLAY_CFG,
    MCTS_CFG,
    GATE_CFG,
    REPLAY_CFG,
    ASYNC_MCTS_CFG,
    _ACTIVE_STATE_BACKEND,
    _ACTIVE_CPP_MODULE,
)
from core.board import TRAINING_GAME_STEPS
from core.env import TzaarEnv, EnvConfig
from gate.gate import gate_keeper
from heuristics import HeuristicPlayoutPolicy as HP
from training.metrics import (
    accumulate_kind_stats,
    format_exploration_stats,
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
from training.selfplay_engine import (
    build_phase_state_from_env as _build_phase_state_from_env,
    collect_selfplay_samples as _collect_selfplay_samples,
    log_runtime_mode as _log_runtime_mode,
    is_async_selfplay_ready,
)
from training.validation import validate_constants
from mcts.cpp_manager import CppSearchManager

import torch.cuda.nvtx as nvtx
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

這個 helper 只負責「單局面」搜尋。
批次 self-play 時，不直接透過這個函式做並行，而是由
training/selfplay_engine.py 決定是否走 run_mcts_batch。
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
    _log_runtime_mode()

    # ── 建立 CppSearchManager（如適用） ──────────────
    search_manager: Optional[CppSearchManager] = None
    if is_async_selfplay_ready():
        try:
            search_manager = CppSearchManager(
                num_threads=int(ASYNC_MCTS_CFG.num_threads),
                max_batch=int(ASYNC_MCTS_CFG.infer_max_batch),
                response_timeout_s=float(ASYNC_MCTS_CFG.response_timeout_s),
            )
            print("[loop] CppSearchManager created successfully")
        except Exception as exc:
            print(
                "[loop] failed to create CppSearchManager, "
                f"falling back to sync-single. error: {exc}"
            )
            search_manager = None

        # ── 載入/初始化政策網路 ────────────────────────────
    # 這裡開始進入正式訓練 lifecycle：
    # backend 已決定、裝置已準備好，接下來所有流程都使用同一組 policy / optimizer。
    #
    # guard_title 由 candidate 的 title 推導而來，確保「訓練 + 存檔 + resume」
    # 使用一致的命名，避免 guard checkpoint 散落在不同 title 下而找不到。
    guard_title = f"{title}_guard"

    start_time = time.perf_counter()
    policy, optimizer, start_update, _, replay_buffer, replay_write_idx = (
        load_or_init_policy(device, guard_title=guard_title, title=title)
    )
    optimizer_to(optimizer, device)
    policy.train()

    guard_policy, guard_source = load_or_init_guard_policy(
        device, policy, guard_title=guard_title
    )
    guard_policy.eval()
    print(f"[guard] title = {guard_title} | source = {guard_source}")

        # 追蹤 guard checkpoint 的下一個可用 index。若目前還沒有任何 guard
    # checkpoint，先把目前（candidate）權重存成初始 guard，這樣後續 resume
    # 才能直接接續強化，而不會每次從隨機權重重開新局。
    existing_guard = train_module.list_checkpoints(title=guard_title)
    next_guard_ckpt_idx = (
        max(item.index for item in existing_guard) + 1 if existing_guard else 0
    )
    if not existing_guard:
        train_module.save_checkpoint(
            title=guard_title,
            policy=guard_policy,
            optimizer=optimizer,
            update_idx=max(0, start_update - 1),
            samples_in_update=0,
            inference_temperature=float(SELFPLAY_CFG.temp_low),
            checkpoint_index=int(next_guard_ckpt_idx),
        )
        next_guard_ckpt_idx += 1
        print(f"[guard] bootstrapped initial guard checkpoint under {guard_title}")

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
        inference_temperature = float(SELFPLAY_CFG.temp_low)

        # ── Self‑play ────────────────────────────────────
        # 這裡只負責呼叫 self-play engine 產生 fresh samples。
        # engine 內部會自行決定：
        # - 是否採用 async-batch
        # - 是否因錯誤而 fallback 到 sync-single
        nvtx.range_push("selfplay")
        policy.eval()
        with torch.no_grad():
                        fresh_samples, games_played, n_new_samples, inference_temperature = _collect_selfplay_samples(
                policy,
                device,
                env_cfg,
                update_idx,
                stats,
                total_updates,
                _rng_seed_offset,
                search_manager=search_manager,
            )
        total_fresh_samples += n_new_samples
        nvtx.range_pop()  # selfplay

        # ── 訓練 ────────────────────────────────────────────
        # 資料來源可能是：
        # - 當前 update 的 fresh_samples
        # - replay buffer 抽樣
        nvtx.range_push(f"update_{update_idx}")
        policy.train()

        if REPLAY_CFG.enabled and fresh_samples:
            replay_write_idx = replay_extend(
                replay_buffer,
                replay_write_idx,
                REPLAY_CFG.max_samples,
                fresh_samples,
            )

                # ── GRADIENT PASS 迴圈 ─────────────────────────────
        # 每次 pass 都用 select_train_samples 重新抽樣，再執行完整
        # (train_epochs_per_update × batch_size) 訓練。
        # metrics 取最後一次 pass 的結果作為該 update 的代表性 metrics。
        n_passes = max(1, int(TRAINING_CFG.optimization_passes_per_update))
        train_samples: list = []
        metrics: Dict[str, float] = {}
        action_kind_mass = [0.0, 0.0, 0.0]
        kind_legal_counts = [0, 0, 0]
        phase2_count = 0
        for pass_idx in range(n_passes):
            # 每次 pass 都重新抽樣（有放回/無放回取最近樣本）
            train_samples = select_train_samples(fresh_samples, replay_buffer)

            # 動作類型統計（每次重新抽樣後重新累計）
            action_kind_mass = [0.0, 0.0, 0.0]
            kind_legal_counts = [0, 0, 0]
            phase2_count = accumulate_kind_stats(
                train_samples,
                action_kind_mass,
                kind_legal_counts,
            )

            metrics = train_on_samples(policy, optimizer, train_samples, device)
        nvtx.range_pop()  # update_{idx}

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
                    search_manager=search_manager,
                                )
                gate_passed = gate_result.win_rate >= GATE_CFG.winrate_threshold
            if gate_passed:
                print(f"[gate] PASSED | win_rate={gate_result.win_rate:.3f}")
                guard_policy.load_state_dict(policy.state_dict(), strict=True)
                guard_policy.eval()
                                                                # candidate 通過 gate → 把目前的權重存成一個 guard checkpoint。
                # 這是「guard 唯一會寫入磁碟」的時機，讓後續 resume 能接續強化。
                train_module.save_checkpoint(
                    title=guard_title,
                    policy=guard_policy,
                    optimizer=optimizer,
                    update_idx=update_idx,
                    samples_in_update=len(fresh_samples),
                    inference_temperature=inference_temperature,
                    checkpoint_index=int(next_guard_ckpt_idx),
                )
                next_guard_ckpt_idx += 1
                print(f"[guard] saved promoted guard checkpoint: {guard_title}")
                # gate 通過時也一併保存 replay buffer，讓 resume 能還原
                # 當前 self-play 累積的資料，而不只是權重。
                if REPLAY_CFG.enabled and replay_buffer:
                    save_replay_snapshot_async(
                        title=guard_title,
                        checkpoint_index=None,
                        update_idx=update_idx,
                        replay_buffer=replay_buffer,
                        write_idx=replay_write_idx,
                        max_samples=REPLAY_CFG.max_samples,
                    )
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

        update_elapsed = time.perf_counter() - update_start
        print(
            f"  [{update_idx}] update took {update_elapsed:.2f}s "
            f"| samples={samples_in_update} | "
            f"n_fresh={len(fresh_samples)} | "
            f"n_train={len(train_samples)}"
        )

                # ── 訓練結束 ──────────────────────────────────────
    elapsed_total = time.perf_counter() - start_time
    print(f"[done] total time: {elapsed_total:.1f}s")
    print(f"[done] total fresh samples generated: {total_fresh_samples}")

    # ── 輸出「每棵樹節點數的最大值」 ─────────────────────
    # 訓練全程每個 update 都會用 CppSearchManager 平行搜尋多棵樹
    # （例如一次 200 棵）。這裡印出：
    #   - max_tree_nodes_any   整個訓練所有被搜尋的樹之中，單棵樹創建節點數的最大值
    #   - max_tree_nodes_last  最近一次搜尋批次（最後一批樹）的單棵樹節點數最大值
    if search_manager is not None:
        max_any = getattr(search_manager, "max_node_count", 0)
        max_last = getattr(search_manager, "last_batch_max_node_count", 0)
        print(
            f"[node-count] training finished | "
            f"max_tree_nodes_any={max_any} | "
            f"max_tree_nodes_last_batch={max_last}"
        )

                # 每棵樹「所有節點中的最大合法步數量」的全程/最近批次最大值
        max_legal_any = getattr(search_manager, "max_node_legal_moves", 0)
        max_legal_last = getattr(
            search_manager, "last_batch_max_node_legal_moves", 0
        )
        print(
            f"[node-legal-moves] training finished | "
            f"max_tree_node_legal_moves_any={max_legal_any} | "
            f"max_tree_node_legal_moves_last_batch={max_legal_last}"
        )

        # 所有樹節點數總和（最近批次／全程累積）
        total_last = getattr(search_manager, "last_batch_total_node_count", 0)
        total_any = getattr(search_manager, "total_node_count", 0)
        print(
            f"[node-total] training finished | "
            f"last_batch_total_tree_nodes={total_last} | "
            f"overall_total_tree_nodes={total_any}"
        )

    # ── 關閉 SearchManager（如已建立） ────────────────
    if search_manager is not None:
        print("[loop] shutting down SearchManager...")
        try:
            search_manager.shutdown()
        except Exception as e:
            print(f"[loop] error shutting down SearchManager: {e}")
        print("[loop] SearchManager shut down complete")

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
