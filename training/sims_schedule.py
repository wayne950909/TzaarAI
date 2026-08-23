"""training/sims_schedule.py — 隨機模擬次數排程器

self-play 每個 run_search 呼叫時抽樣一次模擬次數，且該次呼叫內所有
局面共用同一個次數。

機制：
    1. 依當前 update 進度 t = update_idx / total_updates ∈ [0, 1]
       對「低模擬範圍」與「高模擬範圍」的端點做線性插值。
    2. 以 high_probability 機率選「高模擬範圍」，否則選「低模擬範圍」。
    3. 在選定的範圍內均勻抽一個整數作為本次 run_search 的模擬次數。

設定集中於 config.SIMS_CFG。本排程器僅用於 self-play 資料產生；
gate 評估與 ELO round-robin 維持固定模擬次數，不受此處影響。
"""

from __future__ import annotations

import random
from typing import Optional, Sequence, Tuple

from config import SIMS_CFG

# 模組層級的 RNG；若 config.SIMS_CFG.seed 為 0 則每次執行不固定，
# 否則可重現。同一 RNG 被同一次訓練的所有抽樣共用（依序消耗）。
_RNG = random.Random(
    SIMS_CFG.seed if SIMS_CFG.seed else None
)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _int_range_at_progress(
    start: Sequence[int],
    end: Sequence[int],
    t: float,
) -> Tuple[int, int]:
    """把 (start_min, start_max) ~ (end_min, end_max) 依 t∈[0,1] 線性插值。

    回傳排序後的 (lo, hi)。
    """
    lo = round(_lerp(float(start[0]), float(end[0]), t))
    hi = round(_lerp(float(start[1]), float(end[1]), t))
    return min(lo, hi), max(lo, hi)


def current_ranges(
    update_idx: int,
    total_updates: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """回傳當前 update 進度下的 (低模擬範圍, 高模擬範圍)。"""
    if total_updates <= 1:
        t = 1.0
    else:
        t = min(max(update_idx / total_updates, 0.0), 1.0)
    low = _int_range_at_progress(SIMS_CFG.low_start, SIMS_CFG.low_end, t)
    high = _int_range_at_progress(SIMS_CFG.high_start, SIMS_CFG.high_end, t)
    return low, high


def sample_simulations(
    update_idx: int,
    total_updates: int,
    rng: Optional[random.Random] = None,
) -> Tuple[int, str]:
    """依進度抽樣本次 run_search 的模擬次數。

    回傳 (模擬次數, 所選範圍標籤)：
        - "low"  → 低模擬範圍
        - "high" → 高模擬範圍

    若未提供 rng，使用模組層級 RNG（跨呼叫共用，維持隨機流一致性）。
    """
    if rng is None:
        rng = _RNG
    low, high = current_ranges(update_idx, total_updates)
    if rng.random() < SIMS_CFG.high_probability:
        lo, hi = high
        tag = "high"
    else:
        lo, hi = low
        tag = "low"
    return int(rng.randint(lo, hi)), tag
