"""
mcts/cpp_manager.py — C++ SearchManager Python 包裝

提供 CppSearchManager 類別，作為 C++ SearchManager 的 Python 包裝。
這是取代 mcts/async_worker.py 中「每次搜尋建立 thread pool」的常駐方案。

核心設計
--------
- CppSearchManager 是常駐物件（跨多個 update 重複使用）
- 內部持有一個 C++ SearchManager 實例
- 每次搜尋時重置其樹（保留 worker pool），重用既有 worker threads
- Python 端只做：get_ready_batch → model forward → submit_eval_batch

使用方式
--------
    manager = CppSearchManager(num_threads=10, max_batch=480)

    # 每次 self-play
    manager.reset_trees(root_states_list, config)
    results = manager.run_search(policy, device)

    # 結束時
    manager.shutdown()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import config as _cfg
from config import ASYNC_MCTS_CFG, MCTS_CFG


class CppSearchManager:
    """C++ SearchManager 的 Python 包裝（常駐 worker pool）。

    職責：
    1. 建立並持有 C++ SearchManager 實例
    2. 每次 self-play 時呼叫 reset_trees() 重置搜尋樹
    3. 在 run_search() 中與 C++ worker threads 協同：
       - get_ready_batch() 取得待評估葉節點
       - Python 端做 GPU forward
       - submit_eval_batch() 回寫結果
    4. 訓練結束時 shutdown()
    """

    def __init__(
        self,
        num_threads: int = 10,
        max_batch: int = 480,
        response_timeout_s: float = 30.0,
    ):
        self._module = _cfg._ACTIVE_CPP_MODULE
        if self._module is None:
            raise RuntimeError(
                "C++ module not loaded; cannot create SearchManager"
            )

        self._num_threads = num_threads
        self._max_batch = max_batch
        self._response_timeout_s = response_timeout_s
        self._manager = None
        self._config = None
        self._state_list = []

        print(
            f"[SearchManager] __init__: threads={num_threads}, "
            f"max_batch={max_batch}, timeout={response_timeout_s}s"
        )

    def reset_trees(
        self,
        root_states: List[Any],
        simulations: int = 600,
    ) -> None:
        """重置所有搜尋樹（保留 worker pool）。

        參數
        ----
        root_states : 每棵樹的根狀態（CppPhaseGameStateAdapter 列表）
        simulations : 每棵樹的模擬次數
        """
        if not root_states:
            raise ValueError("root_states must not be empty")

        for state in root_states:
            if not hasattr(state, "_inner"):
                raise TypeError(
                    "SearchManager requires CppPhaseGameStateAdapter "
                    "with _inner attribute"
                )

        n_trees = len(root_states)

        # 建立 SearchConfig
        cfg = self._module.SearchConfig()
        cfg.simulations = int(simulations)
        cfg.leaf_batch_size = int(MCTS_CFG.leaf_batch_size)
        cfg.puct_c = float(MCTS_CFG.puct_c)
        cfg.add_root_dirichlet_noise = bool(MCTS_CFG.use_root_dirichlet_noise)
        cfg.root_dirichlet_eps = float(MCTS_CFG.root_dirichlet_eps)
        cfg.root_dirichlet_alpha = float(MCTS_CFG.root_dirichlet_alpha)
        cfg.min_batch_for_swap = int(MCTS_CFG.leaf_batch_size)
        cfg.flush_timeout_ms = int(ASYNC_MCTS_CFG.infer_max_wait_ms)
        self._config = cfg

        # 如果有舊的 manager，先 shutdown 確保 threads 被 join
        if self._manager is not None:
            try:
                self._manager.finish_all()
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None

        inner_states = [s._inner for s in root_states]
        t0 = time.perf_counter()
        self._manager = self._module.SearchManager(
            inner_states,
            cfg,
            self._num_threads,
            self._max_batch,
        )
        elapsed = time.perf_counter() - t0
        self._state_list = list(root_states)
        self._n_trees = len(root_states)
        self._workers_started = False

    def run_search(
        self,
        policy: torch.nn.Module,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        """與 C++ SearchManager 協同執行 MCTS 搜尋。

        這是 Python 端的「事件迴圈」：
        1. 啟動 C++ worker threads（第一次呼叫時）
        2. 輪詢 get_ready_batch()，直到所有樹完成
        3. 對每個 batch 做 GPU forward
        4. submit_eval_batch() 回寫結果
        5. 回傳所有樹的搜尋結果

        參數
        ----
        policy : 策略網路
        device : torch 裝置

        回傳
        ----
        List of dict，每個 dict 包含：
            - "head": str
            - "action_dim": int
            - "legal_mask": Tensor (CPU)
            - "visits": Tensor (CPU)
            - "replay_board": Tensor (CPU, 12x9x9)
            - "replay_global": Tensor (CPU)
            - "result": SearchResult (原始 C++ 結果)
        """
        if self._manager is None:
            raise RuntimeError(
                "SearchManager not initialized; call reset_trees() first"
            )

        # 啟動 worker threads
        # start_workers 若 thread 已存在會拋異常，所以用旗標保護
        if not self._workers_started:
            print("[SearchManager] run_search: starting worker threads...")
            t0 = time.perf_counter()
            self._manager.start_workers()
            self._workers_started = True
            elapsed = time.perf_counter() - t0
            print(f"[SearchManager]   workers started in {elapsed:.3f}s")
        else:
            print("[SearchManager] run_search: workers already started, reusing")

        # 主事件迴圈
        loop_iter = 0
        total_batches_processed = 0
        total_leaves_evaluated = 0
        t_start = time.perf_counter()
        next_progress_log = 10  # 每處理 10 個 batch 印一次進度

        while not self._manager.is_complete():
            packed = self._manager.get_ready_batch()

            buffer_id = int(packed["buffer_id"])
            batch_size = int(packed["batch_size"])

            if batch_size == 0 or buffer_id < 0:
                loop_iter += 1
                time.sleep(0.001)
                continue

            loop_iter += 1
            total_batches_processed += 1
            total_leaves_evaluated += batch_size

            # ── 從 packed buffer 取出資料 ──────────────
            node_ids_np = np.asarray(packed["node_ids"], dtype=np.int32).copy()
            tree_ids_np = np.asarray(packed["tree_ids"], dtype=np.int32).copy()
            boards_np = np.asarray(packed["board_state_flat"], dtype=np.float32).copy()
            globals_np = np.asarray(packed["global_features"], dtype=np.float32).copy()
            masks_np = np.asarray(packed["legal_masks"], dtype=np.uint8).copy()

            # ── GPU forward ────────────────────────────
            board_batch = (
                torch.from_numpy(boards_np.reshape(-1, 12, 9, 9))
                .to(device)
            )
            global_batch = torch.from_numpy(globals_np).to(device)
            mask_batch = torch.from_numpy(
                masks_np.astype(np.bool_, copy=False)
            ).to(device)

            with torch.no_grad():
                hidden = policy.encode(board_batch, global_batch)
                logits = policy.action_head(hidden)
                values = policy.forward_value(hidden).squeeze(-1)

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

            # ── 回寫結果 ──────────────────────────────
            self._manager.submit_eval_batch(
                int(buffer_id),
                node_ids_np,
                priors_np,
                values_np,
            )

            # 每處理 10 個 batch 或最後幾次時印進度
            if total_batches_processed >= next_progress_log:
                remaining = self._manager.total_remaining_simulations()
                total_sims = self._n_trees * self._config.simulations
                elapsed = time.perf_counter() - t_start
                print(
                    f"  [SearchManager]  batches={total_batches_processed}, "
                    f"leaves={total_leaves_evaluated}, "
                    f"remaining={remaining}/{total_sims}, "
                    f"elapsed={elapsed:.1f}s"
                )
                next_progress_log = total_batches_processed * 2  # 等比級數遞增

        elapsed_total = time.perf_counter() - t_start
        print(
            f"[SearchManager] search done: "
            f"{total_batches_processed} batches, "
            f"{total_leaves_evaluated} leaves, "
            f"elapsed={elapsed_total:.3f}s"
        )

        # ── 所有樹已完成，取回結果 ─────────────────────
        t0 = time.perf_counter()
        raw_results = self._manager.finish_all()
        finish_elapsed = time.perf_counter() - t0
        print(
            f"[SearchManager] finish_all(): {len(raw_results)} results "
            f"in {finish_elapsed:.3f}s"
        )

        outputs: List[Dict[str, Any]] = []
        for i, result in enumerate(raw_results):
            if not bool(result.is_complete):
                raise RuntimeError(
                    f"SearchManager tree {i} finished but not complete"
                )

            legal_mask = torch.from_numpy(
                np.asarray(result.legal_mask, dtype=np.bool_)
            ).clone()
            visits = torch.from_numpy(
                np.asarray(result.root_visits, dtype=np.float32)
            ).clone()
            action_dim = _cfg._ACTIVE_CPP_MODULE.N_ACTIONS

            visits_sum = visits.sum().item()
            if visits_sum <= 0:
                legal = legal_mask[:action_dim]
                n_legal = int(legal.sum().item())
                if n_legal > 0:
                    visits[legal] = 1.0 / float(n_legal)

            replay_board = torch.zeros(12, 9, 9, dtype=torch.float32)
            replay_global = torch.zeros(
                _cfg.NETWORK_CFG.global_feature_dim, dtype=torch.float32
            )

            outputs.append(
                {
                    "head": _cfg.HEAD_ACTION,
                    "action_dim": action_dim,
                    "legal_mask": legal_mask.to(device="cpu"),
                    "visits": visits,
                    "replay_board": replay_board,
                    "replay_global": replay_global,
                    "root_value": result.root_value,
                    "root_player": result.root_player,
                    "is_done": result.is_done,
                    "winner": result.winner,
                }
            )

        return outputs

    def shutdown(self) -> None:
        """關閉 SearchManager 及其 worker threads。"""
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._workers_started = False
        self._state_list = []
        self._config = None

    @property
    def is_active(self) -> bool:
        return self._manager is not None

    @property
    def n_trees(self) -> int:
        return self._n_trees

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
        return False
