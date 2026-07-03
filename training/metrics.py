"""
training/metrics.py — 訓練度量統計

提供探索統計、動作類型統計等輔助函式。
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from core.action import PASS_ACTION_IDX
from core.constants import PHASE_STEP_2, TRAINING_PHASE_TO_IDX
from training.sample import PolicySample


def record_exploration_stats(
    acc: Dict[str, float],
    probs: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float,
) -> None:
    """累積探索統計數據。

    probs 須為已正規化的動作機率向量。
    """
    from config import SELFPLAY_CFG
    TEMP_LOW = SELFPLAY_CFG.temp_low

    legal = legal_mask[: probs.shape[0]]
    n_legal = int(legal.sum().item())
    if n_legal <= 0:
        return

    legal_probs = probs[legal].clamp_min(1e-12)
    entropy = float((-(legal_probs * torch.log(legal_probs))).sum().item())
    entropy_norm = entropy / math.log(float(max(n_legal, 2)))
    top1 = float(legal_probs.max().item())
    eff_n = 1.0 / float((legal_probs * legal_probs).sum().item())

    acc["steps"] += 1.0
    acc["entropy_sum"] += entropy
    acc["entropy_norm_sum"] += entropy_norm
    acc["top1_sum"] += top1
    acc["n_legal_sum"] += float(n_legal)
    acc["eff_n_sum"] += eff_n

    if temperature > TEMP_LOW + 1e-8:
        acc["temp_high_steps"] += 1.0
    else:
        acc["temp_low_steps"] += 1.0

    if top1 >= 0.90:
        acc["near_greedy_steps"] += 1.0


def format_exploration_stats(acc: Dict[str, float]) -> str:
    """格式化探索統計為可讀字串。"""
    steps = int(acc.get("steps", 0.0))
    if steps <= 0:
        return "explore=N/A"

    s = float(max(steps, 1))
    entropy = acc["entropy_sum"] / s
    entropy_norm = acc["entropy_norm_sum"] / s
    top1 = acc["top1_sum"] / s
    n_legal = acc["n_legal_sum"] / s
    eff_n = acc["eff_n_sum"] / s
    high = int(acc["temp_high_steps"])
    low = int(acc["temp_low_steps"])
    greedy_ratio = acc["near_greedy_steps"] / s

    return (
        f"explore(H={entropy:.2f},Hn={entropy_norm:.2f},top1={top1:.2f},"
        f"legal={n_legal:.1f},effN={eff_n:.1f},T_hi/lo={high}/{low},"
        f"greedy90={greedy_ratio:.2f})"
    )


def accumulate_kind_stats(
    samples: List[PolicySample],
    action_kind_mass: List[float],
    kind_legal_counts: List[int],
) -> int:
    """累積 phase-2 樣本的動作類型統計（capture/reinforce/pass）。

    回傳處理的 phase-2 樣本數量。
    """
    phase2_idx = 10 + TRAINING_PHASE_TO_IDX[PHASE_STEP_2]
    phase2_samples = [
        s
        for s in samples
        if s.global_features[phase2_idx].item() > 0.5
    ]
    if not phase2_samples:
        return 0

    all_pi = torch.stack(
        [s.target_pi_padded for s in phase2_samples]
    )  # (N, A)
    all_legal = torch.stack(
        [s.legal_mask_padded for s in phase2_samples]
    )  # (N, A)

    action_kind_mass[0] += float(all_pi[:, :300].sum().item())
    action_kind_mass[1] += float(all_pi[:, 300:600].sum().item())
    action_kind_mass[2] += float(all_pi[:, PASS_ACTION_IDX].sum().item())

    kind_legal_counts[0] += int(
        all_legal[:, :300].any(dim=1).sum().item()
    )
    kind_legal_counts[1] += int(
        all_legal[:, 300:600].any(dim=1).sum().item()
    )
    kind_legal_counts[2] += int(
        all_legal[:, PASS_ACTION_IDX].sum().item()
    )

    return len(phase2_samples)
