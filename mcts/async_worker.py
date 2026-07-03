"""
mcts/async_worker.py — 非同步 MCTS 批次推論

提供 InferenceRequest / InferenceResponse 資料結構，
以及 run_mcts_cpp_batch_async 函式，用於多 CPU worker +
共享 GPU worker 的非同步 MCTS 批次搜尋。
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.action import N_ACTIONS
import config as _cfg
from config import (
    ASYNC_MCTS_CFG,
    HEAD_ACTION,
    MCTS_CFG,
)

# 方便取用
USE_ROOT_DIRICHLET_NOISE = MCTS_CFG.use_root_dirichlet_noise
ROOT_DIRICHLET_EPS = MCTS_CFG.root_dirichlet_eps
ROOT_DIRICHLET_ALPHA_ACTION = MCTS_CFG.root_dirichlet_alpha


@dataclass
class InferenceRequest:
    """非同步推論請求（從 CPU worker 發送到 GPU worker）。"""
    session_id: int
    batch_id: int
    node_ids: np.ndarray
    board_state_flat: np.ndarray
    global_features: np.ndarray
    legal_masks: np.ndarray
    model_id: int = 0  # 0 = primary policy, 1 = secondary policy (guard)


@dataclass
class InferenceResponse:
    """非同步推論回應（從 GPU worker 回傳到 CPU worker）。"""
    session_id: int
    batch_id: int
    node_ids: np.ndarray
    priors: np.ndarray
    values: np.ndarray
    error: Optional[Exception] = None


def _run_model_forward(
    model: torch.nn.Module,
    reqs: List[InferenceRequest],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """對一批請求執行模型 forward 並回傳 priors + values。"""
    boards_np = np.concatenate([r.board_state_flat for r in reqs], axis=0)
    globals_np = np.concatenate([r.global_features for r in reqs], axis=0)
    masks_np = np.concatenate([r.legal_masks for r in reqs], axis=0)

    board_batch = (
        torch.from_numpy(boards_np.reshape(-1, 12, 9, 9)).to(device)
    )
    global_batch = torch.from_numpy(globals_np).to(device)
    mask_batch = torch.from_numpy(
        masks_np.astype(np.bool_, copy=False)
    ).to(device=device)

    with torch.no_grad():
        hidden = model.encode(board_batch, global_batch)
        logits = model.action_head(hidden)
        values = model.forward_value(hidden).squeeze(-1)

    masked_logits = logits.masked_fill(~mask_batch, -1e9)
    priors_np = (
        torch.softmax(masked_logits, dim=-1)
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
    )
    values_np = (
        values.to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
    )
    return priors_np, values_np


def _gpu_worker_main(
    policy: torch.nn.Module,
    device: torch.device,
    request_queue: queue.Queue,
    response_queues: Dict[int, queue.Queue],
    stop_event: threading.Event,
    errors: queue.Queue,
    max_batch: int,
    max_wait_ms: float,
    request_queue_size: int,
    secondary_policy: Optional[torch.nn.Module] = None,
) -> None:
    """GPU worker 主迴圈。

    從請求佇列收集批次，執行模型推論，將結果送回對應的回應佇列。
    若 request.model_id 為 0 使用 primary policy，為 1 使用 secondary_policy。
    """
    try:
        while not stop_event.is_set():
            try:
                first_req = request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if first_req is None:
                break

            batch_reqs: List[InferenceRequest] = [first_req]
            total_rows = int(first_req.node_ids.shape[0])
            deadline = time.perf_counter() + (max_wait_ms / 1000.0)

            while (
                total_rows < max_batch and time.perf_counter() < deadline
            ):
                try:
                    next_req = request_queue.get_nowait()
                except queue.Empty:
                    break
                if next_req is None:
                    request_queue.put(None)
                    break
                batch_reqs.append(next_req)
                total_rows += int(next_req.node_ids.shape[0])

            # 按 model_id 分組
            reqs_primary = [r for r in batch_reqs if r.model_id == 0]
            reqs_secondary = [r for r in batch_reqs if r.model_id == 1]

            if reqs_primary:
                priors_p, values_p = _run_model_forward(
                    policy, reqs_primary, device
                )
                offset = 0
                for req in reqs_primary:
                    count = int(req.node_ids.shape[0])
                    response_queues[req.session_id].put(
                        InferenceResponse(
                            session_id=req.session_id,
                            batch_id=req.batch_id,
                            node_ids=req.node_ids,
                            priors=priors_p[offset : offset + count],
                            values=values_p[offset : offset + count],
                        )
                    )
                    offset += count

            if reqs_secondary and secondary_policy is not None:
                priors_s, values_s = _run_model_forward(
                    secondary_policy, reqs_secondary, device
                )
                offset = 0
                for req in reqs_secondary:
                    count = int(req.node_ids.shape[0])
                    response_queues[req.session_id].put(
                        InferenceResponse(
                            session_id=req.session_id,
                            batch_id=req.batch_id,
                            node_ids=req.node_ids,
                            priors=priors_s[offset : offset + count],
                            values=values_s[offset : offset + count],
                        )
                    )
                    offset += count

    except Exception as exc:
        errors.put(exc)
        stop_event.set()
        for q in response_queues.values():
            try:
                q.put_nowait(
                    InferenceResponse(
                        session_id=-1,
                        batch_id=-1,
                        node_ids=np.empty((0,), dtype=np.int32),
                        priors=np.empty((0, N_ACTIONS), dtype=np.float32),
                        values=np.empty((0,), dtype=np.float32),
                        error=exc,
                    )
                )
            except queue.Full:
                pass


def _cpu_worker_main(
    session_id: int,
    root_state: Any,
    module: Any,
    request_queue: queue.Queue,
    response_queues: Dict[int, queue.Queue],
    stop_event: threading.Event,
    errors: queue.Queue,
    results: List[Optional[Tuple]],
    simulations: int,
    response_timeout_s: float,
) -> None:
    """CPU worker 主迴圈。

    建立 SearchSession，收集葉節點、發送推論請求、接收回應。
    """
    try:
        cfg = module.SearchConfig()
        cfg.simulations = int(simulations)
        cfg.leaf_batch_size = int(MCTS_CFG.leaf_batch_size)
        cfg.puct_c = float(MCTS_CFG.puct_c)
        cfg.add_root_dirichlet_noise = bool(USE_ROOT_DIRICHLET_NOISE)
        cfg.root_dirichlet_eps = float(ROOT_DIRICHLET_EPS)
        cfg.root_dirichlet_alpha = float(ROOT_DIRICHLET_ALPHA_ACTION)

        session = module.SearchSession(root_state._inner, cfg)
        batch_id = 0

        while not stop_event.is_set():
            packed = session.collect_pending_leaves_packed(
                int(MCTS_CFG.leaf_batch_size)
            )
            node_ids_np = np.asarray(
                packed["node_ids"], dtype=np.int32
            ).copy()
            if node_ids_np.size == 0:
                break

            req = InferenceRequest(
                session_id=session_id,
                batch_id=batch_id,
                node_ids=node_ids_np,
                board_state_flat=np.asarray(
                    packed["board_state_flat"], dtype=np.float32
                ).copy(),
                global_features=np.asarray(
                    packed["global_features"], dtype=np.float32
                ).copy(),
                legal_masks=np.asarray(
                    packed["legal_masks"], dtype=np.uint8
                ).copy(),
            )
            request_queue.put(req)

            try:
                resp = response_queues[session_id].get(
                    timeout=response_timeout_s
                )
            except queue.Empty as exc:
                raise TimeoutError(
                    f"async inference timeout for session {session_id}"
                ) from exc

            if resp.error is not None:
                raise resp.error
            if (
                resp.session_id != session_id
                or resp.batch_id != batch_id
            ):
                raise RuntimeError(
                    f"stale/mismatched async response: expected "
                    f"({session_id}, {batch_id}), "
                    f"got ({resp.session_id}, {resp.batch_id})"
                )

            session.submit_leaf_eval_batch(
                resp.node_ids, resp.priors, resp.values
            )
            batch_id += 1

        result = session.finish()
        if not bool(result.is_complete):
            raise RuntimeError(
                "SearchSession finished before all simulations "
                "were processed"
            )

        legal_mask = torch.from_numpy(
            np.asarray(result.legal_mask, dtype=np.bool_)
        ).clone()
        visits = torch.from_numpy(
            np.asarray(result.root_visits, dtype=np.float32)
        ).clone()
        action_dim = N_ACTIONS

        if visits.sum().item() <= 0:
            legal = legal_mask[:action_dim]
            n_legal = int(legal.sum().item())
            if n_legal > 0:
                visits[legal] = 1.0 / float(n_legal)

        root_leaf = session.root_snapshot()
        replay_board = (
            torch.from_numpy(
                np.asarray(
                    root_leaf.board_state_flat, dtype=np.float32
                )
            )
            .reshape(12, 9, 9)
            .clone()
        )
        replay_global = torch.from_numpy(
            np.asarray(
                root_leaf.global_features, dtype=np.float32
            )
        ).clone()

        results[session_id] = (
            HEAD_ACTION,
            action_dim,
            legal_mask.to(device="cpu"),
            visits,
            replay_board,
            replay_global,
        )

    except Exception as exc:
        errors.put(exc)
        stop_event.set()


def run_mcts_cpp_batch_async(
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
    ]
]:
    """非同步批次 MCTS 搜尋。

    使用多個 CPU worker + 共享 GPU worker 同時搜尋多個根狀態。

    參數
    ----
    policy : 策略網路
    root_states : 根狀態列表（須為 CppPhaseGameStateAdapter）
    device : torch 裝置
    apply_dirichlet_noise : 是否在根節點添加探索雜訊
    simulations : 模擬次數

    回傳
    ----
    List of (head, action_dim, legal_mask, visits, replay_board, replay_global)
    """
    from config import _ACTIVE_CPP_MODULE as _cpp_mod
    module = _cpp_mod
    if module is None or not root_states:
        return []

    for state in root_states:
        if not hasattr(state, "_inner"):
            raise RuntimeError(
                "cpp SearchSession async batch expects "
                "CppPhaseGameStateAdapter input"
            )

    n_sessions = len(root_states)
    request_queue: queue.Queue = queue.Queue(
        maxsize=int(ASYNC_MCTS_CFG.request_queue_size)
    )
    response_queues: Dict[int, queue.Queue] = {
        i: queue.Queue(maxsize=1) for i in range(n_sessions)
    }
    errors: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    results: List[Optional[Tuple]] = [None] * n_sessions

    gpu_thread = threading.Thread(
        target=_gpu_worker_main,
        args=(
            policy,
            device,
            request_queue,
            response_queues,
            stop_event,
            errors,
            int(ASYNC_MCTS_CFG.infer_max_batch),
            float(ASYNC_MCTS_CFG.infer_max_wait_ms),
            int(ASYNC_MCTS_CFG.request_queue_size),
        ),
        name="mcts-gpu-worker",
        daemon=True,
    )
    gpu_thread.start()

    workers: List[threading.Thread] = []
    for idx, state in enumerate(root_states):
        thread = threading.Thread(
            target=_cpu_worker_main,
            args=(
                idx,
                state,
                module,
                request_queue,
                response_queues,
                stop_event,
                errors,
                results,
                simulations,
                float(ASYNC_MCTS_CFG.response_timeout_s),
            ),
            name=f"mcts-cpu-worker-{idx}",
            daemon=True,
        )
        workers.append(thread)
        thread.start()

    for thread in workers:
        thread.join()

    stop_event.set()
    request_queue.put(None)
    gpu_thread.join(timeout=5.0)

    if not errors.empty():
        raise errors.get()

    out: List[Tuple] = []
    for idx, item in enumerate(results):
        if item is None:
            raise RuntimeError(
                f"missing async batch result for session {idx}"
            )
        out.append(item)
    return out
