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
from typing import Any, Dict, Optional

import torch

import TzaarTrain as train_module
from config import (
    TITLE,
    TRAINING_CFG,
    SELFPLAY_CFG,
    MCTS_CFG,
    GATE_CFG,
    OPTIMIZER_CFG,
    REPLAY_CFG,
    ELO_CFG,
    ASYNC_MCTS_CFG,
    _ACTIVE_STATE_BACKEND,
    _ACTIVE_CPP_MODULE,
)
from core.board import TRAINING_GAME_STEPS
from core.env import TzaarEnv, EnvConfig
from gate.gate import gate_keeper, GateEscalator
from heuristics import HeuristicPlayoutPolicy as HP
from training.metrics import (
    accumulate_kind_stats,
    format_exploration_stats,
)
from training.replay import (
    load_replay_snapshot,
    prune_replay_snapshots,
    remove_replay_snapshots_after,
    remove_samples_from_buffer,
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
    collect_vs_past_samples as _collect_vs_past_samples,
    log_runtime_mode as _log_runtime_mode,
    is_async_selfplay_ready,
)
from training.validation import validate_constants
from training.opponent_pool import HistoricalOpponentPool
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


def _delete_checkpoints_after(title: str, cutoff_index: int) -> int:
    """刪除指定 title 中 checkpoint index 嚴格大於 cutoff_index 的 .pt 檔。

    用於 ELO 回溯：當挑出最強模型 index K 後，把 K 之後產生的所有 guard
    checkpoint 從磁碟移除，避免後續 resume / 掃描把它們當成可用模型。

    回傳被刪除的檔案數量。
    """
    removed = 0
    for info in train_module.list_checkpoints(title=title):
        if info.index > cutoff_index:
            try:
                info.path.unlink()
                print(f"[elo] removed guard checkpoint after rollback: {info.path.name}")
                removed += 1
            except FileNotFoundError:
                pass
            except Exception as exc:  # pragma: no cover - I/O guard
                print(f"[elo] failed to remove {info.path.name}: {exc}")
    return removed


def _rollback_to_guard(
    *,
    guard_title: str,
    policy: torch.nn.Module,
    guard_policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    opponent_pool: Any,
    replay_buffer: list,
    replay_write_idx: int,
    strongest_idx: int,
) -> tuple[list, int]:
    """ELO 觸發時，把訓練回滾到 ELO 最強 guard checkpoint K。

    參數
    ----
    guard_title    : guard checkpoint 的 title
    policy         : 目前 candidate 網路（就地覆蓋為 K 的權重）
    guard_policy   : 目前 guard 網路（就地覆蓋為 K 的權重）
    optimizer      : 目前優化器（就地覆蓋為 K 的 optimizer state）
    device         : torch device
    opponent_pool  : HistoricalOpponentPool（回溯後 refresh，反映磁碟刪檔）
    replay_buffer  : 目前 replay buffer（會被覆寫為 K 的快照）
    replay_write_idx : 目前 write idx（會被覆寫為 K 的 write idx）
    strongest_idx  : ELO 最強 guard 的 checkpoint index K

    回傳
    ----
    (new_replay_buffer, new_write_idx) ELO 回溯後的 replay 狀態。
    """
    # 1) 找到最強 guard K 的 checkpoint 檔
    candidates = {
        info.index: info
        for info in train_module.list_checkpoints(title=guard_title)
    }
    info = candidates.get(int(strongest_idx))
    if info is None:
        print(
            f"[rollback] checkpoint idx={strongest_idx} not found on disk; "
            "skipping rollback"
        )
        return replay_buffer, replay_write_idx

    # 2) 載入最強模型 K 的權重 + optimizer state → policy & guard & optimizer
    ckpt = torch.load(str(info.path), map_location=str(device), weights_only=True)
    policy.load_state_dict(ckpt["policy_state"], strict=True)
    guard_policy.load_state_dict(ckpt["policy_state"], strict=True)
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    optimizer_to(optimizer, device)
    policy.train()
    guard_policy.eval()
    print(
        f"[rollback] loaded guard idx={strongest_idx} into policy/guard/optimizer "
        f"(source={info.path.name})"
    )

    # 3) 回溯 replay buffer 到 K 晉升時存的快照
    if REPLAY_CFG.enabled:
        rb, rw = load_replay_snapshot(
            title=guard_title,
            checkpoint_index=int(strongest_idx),
            max_samples=REPLAY_CFG.max_samples,
        )
        if rb:
            replay_buffer = rb
            replay_write_idx = rw
        else:
            print(
                f"[rollback] no replay snapshot for idx={strongest_idx}; "
                "starting with empty buffer"
            )
            replay_buffer = []
            replay_write_idx = 0

    # 4) 刪除 K 之後的所有 guard checkpoint
    removed_ckpt = _delete_checkpoints_after(guard_title, int(strongest_idx))
    # 5) 刪除 K 之後的所有 indexed replay 快照
    removed_rb = remove_replay_snapshots_after(
        title=guard_title, cutoff_index=int(strongest_idx)
    )
    # 6) 讓 opponent pool 重新掃描磁碟，反映刪除後的狀態
    try:
        opponent_pool.refresh()
    except Exception as exc:  # pragma: no cover - I/O guard
        print(f"[rollback] failed to refresh opponent pool: {exc}")

    print(
        f"[rollback] rollback complete | removed {removed_ckpt} checkpoints, "
        f"{len(removed_rb)} replay snapshots | pool_size={len(opponent_pool)}"
    )
    return replay_buffer, replay_write_idx


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
        _, action_dim, legal_mask, visits, _, _, _ = _mcts_search(
            policy,
            root_state,
            device,
            apply_dirichlet_noise=apply_dirichlet_noise,
            simulations=simulations,
        )
    return visits, legal_mask


def run(title: str, start_level: Optional[int] = None) -> None:
    """執行訓練流程。

    參數
    ----
    title       : 訓練任務名稱（用於檢查點/日誌命名）
    start_level : 從階級表的哪一階開始（0-based）。None = 預設從第 0 階開始。
                  每階對應「模擬次數 + 學習率 + 混合權重」，例如
                  GATE_CFG.level_schedule 的 level 0 = (128, 8e-4, 0.2)、
                  level 1 = (256, 6e-4, 0.3) 等。
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

    # ── ELO 歷史對手池（可選）────────────────────────────
    # 掃描 guard checkpoint 建立固定大小的 ELO 對手池，作為 self-play 對手。
    opponent_pool = None
    if ELO_CFG.enabled:
        opponent_pool = HistoricalOpponentPool(
            guard_title=guard_title,
            device=device,
            env_cfg=env_cfg,
            search_manager=search_manager,
        )
        print(
            f"[elo] opponent pool ready | size={len(opponent_pool)} "
            f"| ratings={opponent_pool.ratings_summary()}"
        )

    # ── Gate 階級表：每一階的「模擬次數 + 學習率 + 混合權重」一起切換──
    # 每升一階（連續失敗累積達門檻），模擬次數、學習率與 value_q_weight
    # 會『一併』跳到下一階對應的值；不再用固定 step 遞增模擬次數。
    # 若 escalate_enabled=False，只保留單一階（永遠用底數，不升級）。
    _level_schedule = (
        list(GATE_CFG.level_schedule) if GATE_CFG.level_schedule
        else [(int(GATE_CFG.simulations_per_decision),
               float(OPTIMIZER_CFG.default_lr),
               float(SELFPLAY_CFG.value_q_weight))]
    )
    if not GATE_CFG.escalate_enabled:
        first_sims, first_lr, first_vq = _level_schedule[0]
        _level_schedule = [(first_sims, first_lr, first_vq)]
    gate_escalator = GateEscalator(
        schedule=_level_schedule,
        escalator_after_failures=int(GATE_CFG.escalate_after_failures),
        start_level=int(start_level) if start_level is not None else 0,
    )

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

        # ── 依目前 gate 階層同步模擬數 ────────────────────
        # 升級後的模擬數套用於 self-play 產生對局。
        # gate keeper 則用「較低」的模擬數（解法一：打破對稱）。self-play
        # 讀 MCTS_CFG.simulations，gate 讀 GATE_CFG.simulations_per_decision。
        current_sims = gate_escalator.current_simulations()
        MCTS_CFG.simulations = current_sims
        # gate_sims = max(level0_sims, int(current_sims / gate_simulation_ratio))
        # 讓 candidate 以「更強網路 + 較弱搜尋」勝過 guard，先建立第一道通過門檻。
        _level0_sims = int(_level_schedule[0][0]) if _level_schedule else 1
        _gate_sims = max(
            _level0_sims,
            int(current_sims / max(1.0, float(GATE_CFG.gate_simulation_ratio))),
        )
        GATE_CFG.simulations_per_decision = _gate_sims

        # ── 依目前階級套用「模擬次數 + 學習率」（一併切換）──
        # GateEscalator 每升一階，sims 與 lr 都由該階表格一併提供。
        current_lr = gate_escalator.current_learning_rate()
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # ── 依目前階級套用「混合 value target 權重」────────
        # value_q_weight 也隨階級表階梯式上升；selfplay_engine 在產生樣本時
        # 直接讀取 SELFPLAY_CFG.value_q_weight，因此在 self-play 前覆寫即可。
        current_vq = gate_escalator.current_value_q_weight()
        SELFPLAY_CFG.value_q_weight = current_vq



        # ── Self‑play ────────────────────────────────────
        # 這裡只負責呼叫 self-play engine 產生 fresh samples。
        # engine 內部會自行決定：
        # - 是否採用 async-batch
        # - 是否因錯誤而 fallback 到 sync-single
        #
        # Self-play 基準 =「目前最新 accepted guard」（guard_policy）權重，
        # 也就是用目前最強/最新模型自己跟自己對弈產生資料。
        # （ELO 只在 escalation 觸發時用於回溯決策，不再於每個 update
        #   用 ELO 最強模型覆蓋 self-play 基準。)
        policy.load_state_dict(guard_policy.state_dict(), strict=True)
        nvtx.range_push("selfplay")
        policy.eval()
        sp_t0 = time.perf_counter()
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
        sp_elapsed = time.perf_counter() - sp_t0
        avg_steps = (n_new_samples / games_played) if games_played > 0 else 0.0
        print(
            f"[self-play] update {update_idx} done | "
            f"games={games_played} | "
            f"time={sp_elapsed:.2f}s | "
            f"samples={n_new_samples} | "
            f"avg_steps={avg_steps:.1f}"
        )
        total_fresh_samples += n_new_samples
        nvtx.range_pop()  # selfplay

        # ── 最新模型 vs 隨機近期歷史模型（資料多樣性）────────
        # 除了自我對弈（games_per_update 局）之外，額外讓「最新模型
        # （guard_policy 權重，目前位於 policy 中）」對上「從最近
        # vs_past_recent_n 個歷史 guard 模型隨機選出的一個」，再產生
        # vs_past_games 局的訓練樣本。同樣使用 searchManager。
        # 若歷史模型不足（只有最新模型）則自動省略（回傳空）。
        if getattr(SELFPLAY_CFG, "vs_past_enabled", False):
            vs_t0 = time.perf_counter()
            with torch.no_grad():
                (
                    vs_fresh_samples,
                    vs_games_played,
                    vs_n_new_samples,
                    vs_opponent_idx,
                ) = _collect_vs_past_samples(
                    policy,
                    guard_title=guard_title,
                    device=device,
                    env_cfg=env_cfg,
                    update_idx=update_idx,
                    search_manager=search_manager,
                )
            vs_elapsed = time.perf_counter() - vs_t0
            if vs_n_new_samples > 0:
                fresh_samples = fresh_samples + vs_fresh_samples
                games_played += vs_games_played
                n_new_samples += vs_n_new_samples
                total_fresh_samples += vs_n_new_samples
            print(
                f"[vs-past] update {update_idx} done | "
                f"opponent_idx={vs_opponent_idx} | "
                f"games={vs_games_played} | "
                f"time={vs_elapsed:.2f}s | "
                f"samples={vs_n_new_samples}"
            )

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

                # ── 記錄本次用的模擬階層（供 log） ──────────
            # gate_sims = 實際用於 gate 評估的模擬數（解法一：低於 self-play）
            gate_sims = GATE_CFG.simulations_per_decision
            gate_level = gate_escalator.level
            gate_consecutive_failures = gate_escalator.consecutive_failures

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
                gate_escalator.record_success()
                print(
                    f"[gate] PASSED | win_rate={gate_result.win_rate:.3f} "
                    f"| sims={gate_sims}"
                )
                guard_policy.load_state_dict(policy.state_dict(), strict=True)
                guard_policy.eval()
                                # candidate 通過 gate → 把目前的權重存成一個 guard checkpoint。
                # 這是「guard 唯一會寫入磁碟」的時機，讓後續 resume 能接續強化。
                promoted_ckpt_idx = int(next_guard_ckpt_idx)
                train_module.save_checkpoint(
                    title=guard_title,
                    policy=guard_policy,
                    optimizer=optimizer,
                    update_idx=update_idx,
                    samples_in_update=len(fresh_samples),
                    inference_temperature=inference_temperature,
                    checkpoint_index=promoted_ckpt_idx,
                )
                next_guard_ckpt_idx += 1
                print(f"[guard] saved promoted guard checkpoint: {guard_title}")
                # gate 通過 → 以該 guard checkpoint index 另外存一份 indexed
                # replay 快照，供 ELO 回溯時 rollback 使用；並把快照數修剪回
                # REPLAY_CFG.max_snapshots（保留最近 6 份，刪除最舊）。
                # 注意：ELO round-robin「不在」這裡觸發——ELO 只在 escalation
                # （連續失敗、即將升級模擬次數/學率）時才評估。
                if REPLAY_CFG.enabled and replay_buffer:
                    save_replay_snapshot_async(
                        title=guard_title,
                        checkpoint_index=promoted_ckpt_idx,
                        update_idx=update_idx,
                        replay_buffer=replay_buffer,
                        write_idx=replay_write_idx,
                        max_samples=REPLAY_CFG.max_samples,
                    )
                    prune_replay_snapshots(
                        title=guard_title,
                        keep=REPLAY_CFG.max_snapshots,
                    )
            else:
                # 記錄升級前的 level（gate_level 已在本 update 開頭捕捉 =
                # gate_escalator.level，與 record_failure() 前相同）
                prev_level = gate_level
                gate_escalator.record_failure()
                print(
                    f"[gate] FAILED | win_rate={gate_result.win_rate:.3f} "
                    f"< threshold={GATE_CFG.winrate_threshold} "
                    f"| sims={gate_sims} | level={gate_level} | "
                    f"consecutive_failures={gate_consecutive_failures} | "
                    f"next_sims={gate_escalator.current_simulations()}"
                )
                # 先把權重載回目前 guard（現行行為）
                policy.load_state_dict(guard_policy.state_dict(), strict=True)
                policy.train()
                update_accepted = False

                # ── ELO 觸發 + 回溯（僅在 escalation 時）─────────
                # 當「正要切換模擬次數與學率」（level 升級）時，才對最新
                # pool_size 個 guard 做 ELO 評分，挑出最強模型 K，把
                # policy / guard / optimizer / replay 回溯到 K 的快照，
                # 並刪除 K 之後的所有 model 與 replay 快照，再以新的
                # sims / lr 繼續訓練。
                escalated = gate_escalator.level > prev_level
                if escalated and ELO_CFG.enabled and opponent_pool is not None:
                    print(
                        "[elo] gate failed enough times; triggering ELO "
                        f"evaluation at escalation level {gate_escalator.level}"
                    )
                    strongest_idx = opponent_pool.evaluate_topk(
                        k=int(ELO_CFG.pool_size),
                        # 與 gate 評估使用相同的模擬次數（GATE_CFG.simulations_per_decision）
                        simulations=int(GATE_CFG.simulations_per_decision),
                    )
                    if strongest_idx is not None:
                        replay_buffer, replay_write_idx = _rollback_to_guard(
                            guard_title=guard_title,
                            policy=policy,
                            guard_policy=guard_policy,
                            optimizer=optimizer,
                            device=device,
                            opponent_pool=opponent_pool,
                            replay_buffer=replay_buffer,
                            replay_write_idx=replay_write_idx,
                            strongest_idx=strongest_idx,
                        )
                        # 回溯後：K 之後的 guard 已被刪除，下一個可用的
                        # guard checkpoint index 應接續在 K 之後（避免空洞）。
                        next_guard_ckpt_idx = int(strongest_idx) + 1
                        # 回溯後：policy/guard/optimizer 已載回最強模型 K 的
                        # 權重與 optimizer state（等同「用最強模型當新基準」）。
                        print(
                            "[rollback] resumed training from strongest guard "
                            f"idx={strongest_idx} | sims="
                            f"{gate_escalator.current_simulations()} | lr="
                            f"{gate_escalator.current_learning_rate():.6g}"
                        )

                # 此次被拒 update 產生的 fresh samples 品質不佳。
                # 若 keep_replay_on_reject=False（預設），從 replay buffer 移除
                # 這批資料，避免被往後的 update 重複抽樣；若為 True 則保留。
                if not GATE_CFG.keep_replay_on_reject:
                    if REPLAY_CFG.enabled and replay_buffer and fresh_samples:
                        replay_write_idx = remove_samples_from_buffer(
                            replay_buffer,
                            replay_write_idx,
                            REPLAY_CFG.max_samples,
                            fresh_samples,
                        )
                    elif fresh_samples:
                        print(
                            "[replay] reject: replay disabled or buffer empty, "
                            "skipping removal"
                        )

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

        # ── 所有樹節點數等統計（已移除；見 git history）──

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
