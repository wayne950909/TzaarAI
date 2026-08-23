"""training/opponent_pool.py — ELO 歷史對手池管理

讓 self-play 不再讓最新模型自己跟自己对弈，而是從「歷史上通過 gate 的
guard 模型」組成的固定大小對手池中，選出 ELO 最強的模型作為對手。

核心設計
--------
1. 對手池成員 = guard checkpoint（training.loop 在 gate 通過時寫入磁碟，
   並帶有 in_opponent_pool=True 標記）。
2. 每個成員有一個 ELO 評分；新成員加入時設為 initial_rating。
3. 當新模型加入後：
   - 先擴充到 current pool ∪ {new}
   - 對池內所有成員兩兩對戰（round-robin），以對戰結果更新 ELO
   - 若超過 pool_size，剔除 ELO 最低的成員（固定大小）
   - 之後 self-play 改用 ELO 最強的模型作為對手
4. ELO 評分表以 JSON 持久化在 checkpoints/ 資料夾，resume 時得以還原。

round-robin 對戰是「不公平」的主客場：每對模型輪流先手 / 後手進行
round_robin_pairs_games 局，並以 gate_keeper（MCTS）執行——它內部會
處理 CppSearchManager 的「依 current_player 分組、分兩次 run_search」
以及 sync fallback，與 gate 評估共用同一套搜尋路徑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

import TzaarTrain as train_module
from config import ASYNC_MCTS_CFG, ELO_CFG
from gate.gate import gate_keeper
from heuristics import HeuristicPlayoutPolicy as HP


def _elo_ratings_path(guard_title: str) -> Path:
    """回傳 ELO 評分表的持久化路徑。"""
    return Path(train_module.CHECKPOINT_DIR) / f"{guard_title}_{ELO_CFG.ratings_filename}"


def _pool_list_path(guard_title: str) -> Path:
    """回傳「池內活躍名單」的持久化路徑。

    這份名單記錄「目前在池內」的 checkpoint index，與 ELO 表分開保存。
    它是池的「身份」：名單外的模型（即使 checkpoint 檔還在磁碟上）不算
    池成員，因此被踢出的模型不會因重新掃描而復活。
    """
    return Path(train_module.CHECKPOINT_DIR) / f"{guard_title}_pool.json"


def _expected_score(rating_a: float, rating_b: float) -> float:
    """標準 ELO 期望勝率（含平局折半）。"""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class EloTracker:
    """追蹤每個 guard checkpoint index 的 ELO 評分。"""

    def __init__(self, initial_rating: float, k_factor: float) -> None:
        self.initial_rating = float(initial_rating)
        self.k_factor = float(k_factor)
        # index -> rating（對所有「已知」的 guard checkpoint）
        self._ratings: Dict[int, float] = {}

    def get(self, index: int) -> float:
        return self._ratings.get(int(index), self.initial_rating)

    def set_init(self, index: int) -> None:
        self._ratings.setdefault(int(index), self.initial_rating)

    def reset_all(self, indices) -> None:
        """把指定成員的 ELO 全部重設回初始分（完全重新開始）。"""
        for index in indices:
            self._ratings[int(index)] = self.initial_rating

    def record_match(
        self,
        winner_index: int,
        loser_index: int,
        winner_win_rate: float,
    ) -> None:
        """以「winner_index 對 loser_index 的勝率」更新雙方 ELO。

        winner_win_rate 為 winner_index 在雙方對戰中的勝率（0~1，和局算 0.5）。
        這允許一次對戰含多局（round_robin_pairs_games）時，用實際勝率做
        一次性的 ELO 更新，而不是逐局更新。
        """
        wi, li = int(winner_index), int(loser_index)
        r_win = self.get(wi)
        r_lose = self.get(li)

        expected_win = _expected_score(r_win, r_lose)
        expected_lose = 1.0 - expected_win

        # 用「觀測勝率 - 期望勝率」的偏差更新
        self._ratings[wi] = r_win + self.k_factor * (winner_win_rate - expected_win)
        self._ratings[li] = r_lose + self.k_factor * (
            (1.0 - winner_win_rate) - expected_lose
        )

    @property
    def ratings(self) -> Dict[int, float]:
        return dict(self._ratings)

    def load_raw(self, data: Dict[str, float]) -> None:
        self._ratings = {int(k): float(v) for k, v in data.items()}

    def to_raw(self) -> Dict[str, float]:
        return {str(k): float(v) for k, v in self._ratings.items()}


class HistoricalOpponentPool:
    """固定大小的 ELO 歷史對手池（成員為 guard checkpoint）。"""

    def __init__(
        self,
        guard_title: str,
        device: torch.device,
        env_cfg: Any,
        search_manager: Optional[Any] = None,
    ) -> None:
        self.guard_title = guard_title
        self.device = device
        self.device_str = str(device)
        self.env_cfg = env_cfg
        self.search_manager = search_manager

        self.elo = EloTracker(ELO_CFG.initial_rating, ELO_CFG.k_factor)
        self._policy_cache: Dict[int, torch.nn.Module] = {}

        # 目前「活躍」的對手池成員 index 集合（固定大小 pool_size）。
        # 這份「名單」是池的身份：只有名單內的模型才算池成員、參與評估。
        # 名單外即使 checkpoint 檔還在磁碟上，也不算池成員。
        self._active_indices: Set[int] = set()

        self._load_ratings()
        self._load_pool_list()
        self.refresh()

    # ── 掃描與持久化 ─────────────────────────────────────
    def refresh(self) -> None:
        """重掃磁碟，把名單與磁碟現況對齊。

        以「已持久化的名單」為準；把名單中「checkpoint 檔已被刪除」的
        index 過濾掉；冷啟動（名單為空）時依掃描順序建立初始名單填到
        pool_size。名單外的 guard 不會被拉進池。
        """
        members = train_module.list_pool_checkpoints(self.guard_title)
        on_disk = {m.index for m in members}
        for info in members:
            self.elo.set_init(info.index)

        if not self._active_indices:
            # 冷啟動：名單為空 → 依磁碟掃描順序建立初始名單（最多 pool_size 個）
            initial = sorted(on_disk)
            if ELO_CFG.pool_size > 0:
                initial = initial[: ELO_CFG.pool_size]
            self._active_indices = set(initial)

        # 過濾掉已被刪檔的成員；並強制不超過 pool_size
        self._active_indices = {i for i in self._active_indices if i in on_disk}
        self._prune_to_size()
        # 清掉不再需要的 policy 快取
        valid_indices = {i for i in self._active_indices}
        self._policy_cache = {
            idx: m for idx, m in self._policy_cache.items() if idx in valid_indices
        }
        # 名單有實質內容就立刻持久化，避免建池後、首次 gate 前崩潰導致名單遺失
        if self._active_indices:
            self.save_pool_list()

    def _active_checkpoints(self) -> List[train_module.CheckpointInfo]:
        """回傳目前活躍成員的 CheckpointInfo（依 index 排序）。"""
        members = train_module.list_pool_checkpoints(self.guard_title)
        active = [m for m in members if m.index in self._active_indices]
        active.sort(key=lambda m: m.index)
        return active

    def _prune_to_size(self) -> None:
        """把名單修剪到 pool_size 以內（依目前 ELO 高低）。

        只針對「已在名單內」的成員操作，不從磁碟重新拉入。被踢出的
        成員離開名單，之後不會因重新掃描而復活。
        """
        if ELO_CFG.pool_size <= 0:
            self._active_indices = set()
            return
        if len(self._active_indices) <= ELO_CFG.pool_size:
            return
        ranked = sorted(self._active_indices, key=lambda i: self.elo.get(i), reverse=True)
        self._active_indices = set(ranked[: ELO_CFG.pool_size])

    def _load_ratings(self) -> None:
        path = _elo_ratings_path(self.guard_title)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.elo.load_raw(data)
        except Exception as exc:  # pragma: no cover - I/O guard
            print(f"[elo] failed to load ratings from {path}: {exc}")

    def save_ratings(self) -> None:
        path = _elo_ratings_path(self.guard_title)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.elo.to_raw(), fh, indent=2)
        except Exception as exc:  # pragma: no cover - I/O guard
            print(f"[elo] failed to save ratings to {path}: {exc}")

    def _load_pool_list(self) -> None:
        """載入池內活躍名單（persist index 集合）。"""
        path = _pool_list_path(self.guard_title)
        if not path.exists():
            self._active_indices = set()
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            indices = data.get("indices", []) if isinstance(data, dict) else data
            self._active_indices = {int(i) for i in indices}
        except Exception as exc:  # pragma: no cover - I/O guard
            print(f"[elo] failed to load pool list from {path}: {exc}")
            self._active_indices = set()

    def save_pool_list(self) -> None:
        """持久化池內活躍名單。"""
        path = _pool_list_path(self.guard_title)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"indices": sorted(self._active_indices)}, fh, indent=2)
        except Exception as exc:  # pragma: no cover - I/O guard
            print(f"[elo] failed to save pool list to {path}: {exc}")

    # ── 模型載入 ─────────────────────────────────────────
    def get_policy(self, index: int) -> torch.nn.Module:
        """回傳指定 checkpoint index 的凍結策略網路（含快取）。"""
        cached = self._policy_cache.get(int(index))
        if cached is not None:
            return cached
        members = {m.index: m for m in train_module.list_pool_checkpoints(self.guard_title)}
        info = members.get(int(index))
        if info is None:
            raise KeyError(f"checkpoint index {index} not in guard pool")
        policy = train_module.load_policy_from_checkpoint(info.path, self.device_str)
        self._policy_cache[int(index)] = policy
        return policy

    # ── 選取最強 ─────────────────────────────────────────
    def strongest(self) -> Optional[int]:
        """回傳目前 ELO 最高的活躍成員 index；池為空時回傳 None。"""
        if not self._active_indices:
            return None
        ranked = sorted(self._active_indices, key=lambda i: self.elo.get(i), reverse=True)
        return ranked[0]

    def strongest_policy(self) -> Optional[torch.nn.Module]:
        """回傳 ELO 最強模型的凍結策略網路；池為空時回傳 None。"""
        idx = self.strongest()
        if idx is None:
            return None
        return self.get_policy(idx)

    def sync_to_strongest(self, target: torch.nn.Module) -> bool:
        """把 ELO 最強 checkpoint 的權重載入到 target 模組（就地覆蓋）。

        這讓 self-play 的基準回到「目前最強模型」，並以它自己的對弈產生
        新的訓練資料。回傳 True 表示有成功載入，False 表示池為空（無法載入）。
        """
        idx = self.strongest()
        if idx is None:
            return False
        strongest = self.get_policy(idx)
        target.load_state_dict(strongest.state_dict(), strict=True)
        return True

    # ── 新成員加入 + 內戰 + 修剪 ─────────────────────────
    def on_new_guard(self, new_index: int, simulations: Optional[int] = None) -> None:
        """新 guard checkpoint 加入對手池，並觸發整池 round-robin + 修剪。

        流程：
          1. 把 new_index 加入 ELO 追蹤（若尚未加入設初始分）。
          2. 暫時活躍集合 = 目前活躍 ∪ {new}。
          3. 對該集合做 round-robin 內戰，更新所有成員 ELO。
          4. 依 ELO 修剪回 pool_size，剔除最低者。
          5. 持久化 ELO。

        參數
        ----
        simulations : 本次 round-robin 內戰使用的模擬次數。若為 None，
            回退到 ELO_CFG.round_robin_simulations。訓練迴圈會傳入「目前
            高模擬下限」，與 gate 評估使用同一套邏輯。
        """
        self.elo.set_init(new_index)
        working = self._active_indices | {int(new_index)}
        # 每次評估都重新開始：先把手池成員的 ELO 全部重設回初始分，
        # 再打整套 round-robin，確保每一輪評比獨立、不被歷史分數污染。
        self.elo.reset_all(working)
        self._play_round_robin(sorted(working), simulations=simulations)
        self._active_indices = set(working)
        self._prune_to_size()
        self._policy_cache = {
            idx: m for idx, m in self._policy_cache.items() if idx in self._active_indices
        }
        self.save_ratings()
        self.save_pool_list()
        top = self.strongest()
        print(
            f"[elo] pool updated | size={len(self._active_indices)} "
            f"| strongest={top} rating={self.elo.get(top) if top is not None else '-'}"
        )

    def _play_round_robin(
        self,
        indices: List[int],
        simulations: Optional[int] = None,
    ) -> None:
        """對指定成員集合做全對全內戰，更新 ELO。

        參數
        ----
        simulations : 內戰使用的模擬次數；若為 None 則回退到
            ELO_CFG.round_robin_simulations。
        """
        if len(indices) < 2:  # noqa: PLR2004
            return
        n_games = int(ELO_CFG.round_robin_pairs_games)
        sims = (
            int(simulations)
            if simulations is not None
            else int(ELO_CFG.round_robin_simulations)
        )
        temperature = float(ELO_CFG.round_robin_temperature)

        # 建立 gate_cfg 相容的 namespace（gate_keeper 讀 simulations 與 temperature）
        import types
        gate_cfg = types.SimpleNamespace(
            simulations_per_decision=sims,
            temperature=temperature,
        )

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_a = indices[i]
                idx_b = indices[j]
                policy_a = self.get_policy(idx_a)
                policy_b = self.get_policy(idx_b)
                try:
                    result = gate_keeper(
                        policy_a,
                        policy_b,
                        n_games=n_games,
                        device=self.device,
                        gate_cfg=gate_cfg,
                        game_cfg=None,
                        env_cfg=self.env_cfg,
                        hp=HP(HP.Config(
                            softmax_temperature=1.0,
                            prior_weight=0.0,
                        )),
                        search_manager=self.search_manager,
                    )
                except Exception as exc:  # 單對內戰失敗不中斷整池
                    print(
                        f"[elo] round-robin {idx_a} vs {idx_b} skipped: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

                total = result.total
                if total <= 0:
                    continue
                # a 的勝率（和局算 0.5）
                a_win_rate = (result.candidate_wins + 0.5 * result.draws) / total
                self.elo.record_match(idx_a, idx_b, a_win_rate)

    # ── 唯讀資訊 ─────────────────────────────────────────
    def ratings_summary(self) -> str:
        """輸出活躍成員的 ELO 摘要字串（供日誌顯示）。"""
        parts = []
        for idx in sorted(self._active_indices):
            parts.append(f"{idx}:{self.elo.get(idx):.0f}")
        return "[" + ", ".join(parts) + "]"

    def __len__(self) -> int:
        return len(self._active_indices)
