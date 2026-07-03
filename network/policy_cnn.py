"""
network/policy_cnn.py — PolicyNetCNNMin17：政策/價值雙頭 CNN

Min17 架構（2017 MiniGo 風格）：
- Board branch: player-relative 12x9x9 tensor
    -> stem conv (12->64) -> N residual blocks (64ch) -> flatten
- Global branch: scalar features -> small MLP (->64)
- Fusion trunk: concat -> FC 512 -> FC 256
- Heads: action(601), value(1)

可透過 config.NETWORK_CFG 調整超參數（channels, num_res_blocks,
fc_hidden, fc_hidden_2, global_feature_dim, dropout）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import NETWORK_CFG


_NET_CFG = NETWORK_CFG


class _ResBlock(nn.Module):
    """標準殘差區塊: Conv -> BN -> ReLU -> Conv -> BN + skip -> ReLU。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class PolicyNetCNNMin17(nn.Module):
    """Min17 policy/value network with residual CNN backbone.

    參數
    ----
    global_feature_dim : 全域特徵維度（default: NETWORK_CFG.global_feature_dim = 12）
    dropout : dropout 比率（default: NETWORK_CFG.dropout = 0.3）
    channels : CNN 通道數（default: NETWORK_CFG.channels = 64）
    num_res_blocks : 殘差區塊數（default: NETWORK_CFG.num_res_blocks = 4）
    fc_hidden : fusion trunk 第一層隱藏維度（default: NETWORK_CFG.fc_hidden = 512）
    fc_hidden_2 : fusion trunk 第二層隱藏維度（default: NETWORK_CFG.fc_hidden_2 = 256）

    注意
    ----
    保留 backward compatibility：若建立時不帶參數會使用 config.NETWORK_CFG。
    若從舊 checkpoint 載入，可傳入相應的 legacy 參數值。
    """

    def __init__(
        self,
        global_feature_dim: int | None = None,
        dropout: float | None = None,
        channels: int | None = None,
        num_res_blocks: int | None = None,
        fc_hidden: int | None = None,
        fc_hidden_2: int | None = None,
    ) -> None:
        super().__init__()

        self.global_feature_dim = (
            global_feature_dim if global_feature_dim is not None
            else _NET_CFG.global_feature_dim
        )
        self.dropout_rate = dropout if dropout is not None else _NET_CFG.dropout
        ch = channels if channels is not None else _NET_CFG.channels
        n_res = num_res_blocks if num_res_blocks is not None else _NET_CFG.num_res_blocks
        f1 = fc_hidden if fc_hidden is not None else _NET_CFG.fc_hidden
        f2 = fc_hidden_2 if fc_hidden_2 is not None else _NET_CFG.fc_hidden_2

        # Stem: project 12 input channels to `ch`
        self.stem_conv = nn.Conv2d(12, ch, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(ch)

        # Residual tower
        self.res_blocks = nn.Sequential(*[_ResBlock(ch) for _ in range(n_res)])

        # Global feature branch: scalar features -> 64
        self.global_proj = nn.Sequential(
            nn.Linear(self.global_feature_dim, 64),
            nn.ReLU(inplace=True),
        )

        # Fusion trunk
        board_flat_dim = ch * 9 * 9
        self.fc1 = nn.Linear(board_flat_dim + 64, f1)
        self.drop = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(f1, f2)

        # Unified action head: 0-299 capture, 300-599 reinforce, 600 pass
        self.action_head = nn.Linear(f2, 601)
        self.value_head = nn.Linear(f2, 1)

    def encode(self, states: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """CNN backbone encode → fusion trunk output。

        參數
        ----
        states : (batch, 12, 9, 9) board 特徵張量
        global_features : (batch, global_feature_dim) 全域特徵

        回傳
        ----
        hidden : (batch, fc_hidden_2) fusion trunk 輸出
        """
        x = F.relu(self.stem_bn(self.stem_conv(states)), inplace=True)
        x = self.res_blocks(x)
        x = x.flatten(start_dim=1)

        g = self.global_proj(global_features)
        h = torch.cat([x, g], dim=1)
        h = F.relu(self.fc1(h), inplace=True)
        h = self.drop(h)
        h = F.relu(self.fc2(h), inplace=True)
        return h

    def forward_value(self, hidden: torch.Tensor) -> torch.Tensor:
        """從 fusion trunk 輸出計算價值預測 ([-1, 1])。"""
        return torch.tanh(self.value_head(hidden))

    def head_logits(self, hidden: torch.Tensor, head: str) -> torch.Tensor:
        """從 fusion trunk 輸出計算指定 head 的 logits。

        目前僅支援 head="action"。
        """
        if head == "action":
            return self.action_head(hidden)
        raise ValueError(f"Unknown head: {head}")

    def forward(self, states: torch.Tensor, global_features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(states, global_features)
        return {
            "action": self.action_head(hidden),
            "value": self.forward_value(hidden),
        }
