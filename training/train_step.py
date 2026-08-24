"""
training/train_step.py — 訓練步驟

提供 train_on_samples 的實現。
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from config import TRAINING_CFG, OPTIMIZER_CFG
from training.sample import PolicySample


def train_on_samples(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: List[PolicySample],
    device: torch.device,
) -> Dict[str, float]:
    """對一批樣本執行單次訓練（包含多個 epoch 的隨機 mini-batch）。

    參數
    ----
    policy : 策略網路
    optimizer : 優化器
    samples : 訓練樣本列表
    device : torch 裝置

    回傳
    ----
    metrics dict（loss, policy_loss, value_loss, entropy, n_samples）
    """
    batch_size = TRAINING_CFG.batch_size
    train_epochs = TRAINING_CFG.train_epochs_per_update
    policy_loss_weight = OPTIMIZER_CFG.policy_loss_weight
    value_loss_weight = OPTIMIZER_CFG.value_loss_weight
    entropy_weight = OPTIMIZER_CFG.entropy_weight
    grad_clip = OPTIMIZER_CFG.grad_clip

    if not samples:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "n_samples": 0,
        }

    pack_start = time.perf_counter()
    states_cpu = torch.stack([s.state for s in samples])
    globals_cpu = torch.stack([s.global_features for s in samples])
    masks_cpu = torch.stack([s.legal_mask_padded for s in samples])
    policy_targets_cpu = torch.stack([s.target_pi_padded for s in samples])
    value_targets_cpu = torch.tensor(
        [s.value_target for s in samples], dtype=torch.float32
    )
    pack_time_s = time.perf_counter() - pack_start

    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    total_count = 0
    transfer_time_s = 0.0
    compute_time_s = 0.0

    policy.train()
    n = states_cpu.shape[0]
    if batch_size > n:
        batch_size = n
    batch_count = n // batch_size

    for _ in range(train_epochs):
        for _ in range(batch_count):
            sample_ids = np.random.randint(n, size=batch_size)
            idx_cpu = torch.from_numpy(sample_ids.astype(np.int64, copy=False))

            transfer_start = time.perf_counter()
            b_states = states_cpu[idx_cpu].to(device, non_blocking=True)
            b_globals = globals_cpu[idx_cpu].to(device, non_blocking=True)
            b_masks = masks_cpu[idx_cpu].to(device, non_blocking=True)
            b_policy_targets = policy_targets_cpu[idx_cpu].to(
                device, non_blocking=True
            )
            b_value_targets = value_targets_cpu[idx_cpu].to(
                device, non_blocking=True
            )
            transfer_time_s += time.perf_counter() - transfer_start

            compute_start = time.perf_counter()

            vh, ph = policy.encode(b_states, b_globals)
            action_logits = policy.action_head(ph)
            value_pred = policy.forward_value(vh).squeeze(-1)

            masked_logits = action_logits.masked_fill(~b_masks, -1e9)
            logp = F.log_softmax(masked_logits, dim=-1)
            probs = torch.exp(logp)
            policy_loss = -(b_policy_targets * logp).sum(dim=-1).mean()
            entropy = -(probs * logp).sum(dim=-1).mean()

            value_loss = F.mse_loss(value_pred, b_value_targets)
            loss = (
                policy_loss_weight * policy_loss
                + value_loss_weight * value_loss
                + entropy_weight * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), grad_clip
                )
            optimizer.step()
            compute_time_s += time.perf_counter() - compute_start

            bs = batch_size
            total_loss += float(loss.item()) * bs
            total_policy += float(policy_loss.item()) * bs
            total_value += float(value_loss.item()) * bs
            total_entropy += float(entropy.item()) * bs
            total_count += bs

    denom = max(total_count, 1)
    return {
        "loss": total_loss / denom,
        "policy_loss": total_policy / denom,
        "value_loss": total_value / denom,
        "entropy": total_entropy / denom,
        "n_samples": float(n),
        "pack_time_s": pack_time_s,
        "transfer_time_s": transfer_time_s,
        "compute_time_s": compute_time_s,
    }
