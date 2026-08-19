"""
training/resume.py — 檢查點載入與初始化

提供策略網路的載入/初始化、guard 載入等功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from config import (
    GUARD_TITLE,
    TITLE,
    TRAINING_CFG,
    OPTIMIZER_CFG,
    REPLAY_CFG,
    NETWORK_CFG,
)
from network import PolicyNetCNNMin17
from training.replay import load_replay_snapshot
from training.sample import PolicySample

import TzaarTrain as train_module


def scan_resume_checkpoint(
    device: str,
) -> Optional[Tuple[Dict[str, object], Path]]:
    """掃描可恢復的檢查點（按 update_idx 降序排列取最新）。"""
    candidates = train_module.list_checkpoints(
        title=None if TRAINING_CFG.resume_any_title else TRAINING_CFG.resume_title
    )
    if not candidates:
        return None

    ranked: List[Tuple[float, Dict[str, object], Path]] = []
    for ckpt_info in candidates:
        path = ckpt_info.path
        try:
            ckpt = torch.load(str(path), map_location=device, weights_only=True)
        except Exception:
            continue
        arch = str(ckpt.get("architecture", "")).lower()
        if arch != "cnn_min17":
            continue
        update_idx = float(ckpt.get("update_idx", -1))
        ranked.append((update_idx, ckpt, path))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[2].stat().st_mtime), reverse=True)
    _, ckpt, path = ranked[0]
    return ckpt, path


def scan_latest_checkpoint_for_title(
    title: str, device: str
) -> Optional[Tuple[Dict[str, object], Path]]:
    """掃描指定 title 的最新檢查點。"""
    candidates = train_module.list_checkpoints(title=title)
    if not candidates:
        return None

    ranked: List[Tuple[float, Dict[str, object], Path]] = []
    for ckpt_info in candidates:
        path = ckpt_info.path
        try:
            ckpt = torch.load(str(path), map_location=device, weights_only=True)
        except Exception:
            continue
        arch = str(ckpt.get("architecture", "")).lower()
        if arch != "cnn_min17":
            continue
        update_idx = float(ckpt.get("update_idx", -1))
        ranked.append((update_idx, ckpt, path))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[2].stat().st_mtime), reverse=True)
    _, ckpt, path = ranked[0]
    return ckpt, path


def load_or_init_policy(
    device: torch.device,
    guard_title: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[
    PolicyNetCNNMin17,
    torch.optim.Optimizer,
    int,
    int,
    List[PolicySample],
    int,
]:
    """載入或初始化策略網路與優化器。

    從 guard title 的最新檢查點載入權重（若存在），
    否則使用隨機初始化。

    參數
    ----
    device : torch 裝置
    guard_title : 守門員 checkpoint 的 title。
        若為 None，使用 config.GUARD_TITLE。
    title : candidate 的 checkpoint title（用於計算下一個可用的
        checkpoint 索引）。若為 None，使用 config.TITLE。

    回傳
    ----
    (policy, optimizer, start_update, next_checkpoint_index,
     replay_buffer, replay_write_idx)
    """
    guard_title = guard_title or GUARD_TITLE
    title = title or TITLE

    policy = PolicyNetCNNMin17(
        global_feature_dim=NETWORK_CFG.global_feature_dim,
        dropout=NETWORK_CFG.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=OPTIMIZER_CFG.default_lr,
        weight_decay=OPTIMIZER_CFG.weight_decay,
    )

    start_update = 0
    next_checkpoint_index = next_checkpoint_index_for_title(title)
    replay_buffer: List[PolicySample] = []
    replay_write_idx = 0

    resume = scan_latest_checkpoint_for_title(guard_title, str(device))
    if resume is None:
        if TRAINING_CFG.require_resume:
            raise RuntimeError(
                f"No guard checkpoint found for title '{guard_title}'."
            )
        print(
            "[resume] no guard checkpoint found, "
            "bootstrapping candidate from fresh weights"
        )
        return policy, optimizer, start_update, next_checkpoint_index, replay_buffer, replay_write_idx

    ckpt, ckpt_path = resume
    policy_state = ckpt.get("policy_state")
    if not isinstance(policy_state, dict):
        raise KeyError(
            f"Invalid checkpoint: missing policy_state in {ckpt_path}"
        )

    policy.load_state_dict(policy_state, strict=True)
    optimizer_state = ckpt.get("optimizer_state")
    if isinstance(optimizer_state, dict):
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception:
            pass

    start_update = int(ckpt.get("update_idx", -1)) + 1
    replay_buffer, replay_write_idx = load_replay_snapshot(
        title=guard_title,
        checkpoint_index=None,
        max_samples=REPLAY_CFG.max_samples,
        base_dir=ckpt_path.parent,
    )
    print(
        f"[resume] candidate loaded from guard {ckpt_path.name} "
        f"| start_update={start_update} | replay={len(replay_buffer)}"
    )
    return policy, optimizer, start_update, next_checkpoint_index, replay_buffer, replay_write_idx


def load_or_init_guard_policy(
    device: torch.device,
    fallback_policy: PolicyNetCNNMin17,
    guard_title: Optional[str] = None,
) -> Tuple[PolicyNetCNNMin17, str]:
    """載入或初始化 guard 策略網路。

    若無 guard 檢查點，從 fallback_policy 複製權重。

    參數
    ----
    device : torch 裝置
    fallback_policy : 當沒有 guard checkpoint 時，以其權重 bootstrap。
    guard_title : 守門員 checkpoint 的 title。
        若為 None，使用 config.GUARD_TITLE。
    """
    guard_title = guard_title or GUARD_TITLE

    guard = PolicyNetCNNMin17(
        global_feature_dim=NETWORK_CFG.global_feature_dim,
        dropout=NETWORK_CFG.dropout,
    ).to(device)
    resume = scan_latest_checkpoint_for_title(guard_title, str(device))
    if resume is None:
        guard.load_state_dict(fallback_policy.state_dict(), strict=True)
        guard.eval()
        return guard, "bootstrap_from_candidate"

    ckpt, ckpt_path = resume
    policy_state = ckpt.get("policy_state")
    if not isinstance(policy_state, dict):
        raise KeyError(
            f"Invalid guard checkpoint: missing policy_state in {ckpt_path}"
        )
    guard.load_state_dict(policy_state, strict=True)
    guard.eval()
    return guard, f"loaded_{ckpt_path.name}"


def next_checkpoint_index_for_title(title: str) -> int:
    """回傳指定 title 的下一個可用 checkpoint 索引。"""
    by_title = train_module.list_checkpoints(title=title)
    if not by_title:
        return 0
    return max(item.index for item in by_title) + 1


def optimizer_to(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """移動優化器狀態張量到指定裝置。"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
