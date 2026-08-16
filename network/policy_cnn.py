"""
network/policy_cnn.py — PolicyNetCNNMin17：政策 / 價值雙頭 CNN

修正後架構（壓縮特徵 + 分離 head）：
- Board branch: player-relative 12x9x9 tensor
    -> stem conv (12 -> ch) -> N residual blocks (ch -> ch)
    -> channel compression (ch -> compress_channels, 預設 2) -> flatten -> 2*9*9 = 162
- Policy head: 直接從 162 -> 601（不再經過共享 trunk）
- Value network: 162 -> 隱藏層 (fc_hidden_2=256) -> 1

移除的多餘架構：
1. 原本的 fusion trunk（fc1 512、fc2 256、drop）——因為 policy 現在
   直接從 162 輸出、value 自帶單一隱藏層，共用 trunk 已無意義。
2. global_proj（全域特徵投影分支）——head 不再使用全域特徵，
   因此該投影分支為冗餘。
3. fc_hidden（512）超參數——已隨 fusion trunk 移除。

ResNet 殘差區塊本身已內建 skip connection，這裡保留標準寫法，
沒有額外多餘的殘差包裝。

可透過 config.NETWORK_CFG 調整超參數（channels, num_res_blocks,
fc_hidden_2, compress_channels, global_feature_dim, dropout）。
加大 num_res_blocks 即可加深 ResNet。
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


class _CompressBlock(nn.Module):
    """通道壓縮區塊: 1x1 Conv，把 ch 通道壓縮成 compress_channels。

    用途：把殘差塔輸出的高維 (ch, 9, 9) 特徵，
    壓縮成 (compress_channels, 9, 9)，攤平後維度極小（預設 2*9*9 = 162），
    讓 policy/value head 直接在此低維特徵上運算。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class PolicyNetCNNMin17(nn.Module):
    """Min17 policy/value network with residual CNN backbone + channel compression.

    參數
    ----
    global_feature_dim : 全域特徵維度（保留介面相容，head 不再使用）
    dropout : value 隱藏層 dropout 比率（default: NETWORK_CFG.dropout）
    channels : CNN 通道數（default: NETWORK_CFG.channels = 64）
    num_res_blocks : 殘差區塊數（default: NETWORK_CFG.num_res_blocks = 4）
    fc_hidden_2 : value 隱藏層維度（default: NETWORK_CFG.fc_hidden_2 = 256）
    compress_channels : 壓縮後通道數（default: NETWORK_CFG.compress_channels = 2）

    注意
    ----
    此版本架構與先前 fusion trunk 版本不同，舊 checkpoint 狀態字典
    無法直接載入，需重新訓練。
    """

    def __init__(
        self,
        global_feature_dim: int | None = None,
        dropout: float | None = None,
        channels: int | None = None,
        num_res_blocks: int | None = None,
        fc_hidden_2: int | None = None,
        compress_channels: int | None = None,
    ) -> None:
        super().__init__()

        self.global_feature_dim = (
            global_feature_dim if global_feature_dim is not None
            else _NET_CFG.global_feature_dim
        )
        self.dropout_rate = dropout if dropout is not None else _NET_CFG.dropout
        ch = channels if channels is not None else _NET_CFG.channels
        n_res = num_res_blocks if num_res_blocks is not None else _NET_CFG.num_res_blocks
        f2 = fc_hidden_2 if fc_hidden_2 is not None else _NET_CFG.fc_hidden_2
        cc = compress_channels if compress_channels is not None else _NET_CFG.compress_channels

        # Stem: project 12 input channels to `ch`
        self.stem_conv = nn.Conv2d(12, ch, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(ch)

        # Residual tower
        self.res_blocks = nn.Sequential(*[_ResBlock(ch) for _ in range(n_res)])

        # Channel compression：ch -> compress_channels，攤平後 board_flat_dim。
        self.compress = _CompressBlock(ch, cc)
        board_flat_dim = cc * 9 * 9

        # Policy head：直接從 162 -> 601（不再經過共享 trunk）
        # unified action：0-299 capture, 300-599 reinforce, 600 pass
        self.action_head = nn.Linear(board_flat_dim, 601)

        # Value network：162 -> (dropout) -> ReLU -> 256 -> 1
        self.value_fc = nn.Linear(board_flat_dim, f2)
        self.value_drop = nn.Dropout(self.dropout_rate)
        self.value_head = nn.Linear(f2, 1)

    def encode(self, states: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """CNN backbone encode → 壓縮後攤平的低維特徵。

        參數
        ----
        states : (batch, 12, 9, 9) board 特徵張量
        global_features : (batch, global_feature_dim) 全域特徵（保留參數但不使用）

        回傳
        ----
        hidden : (batch, compress_channels * 9 * 9) 低維特徵（預設 162）
        """
        x = F.relu(self.stem_bn(self.stem_conv(states)), inplace=True)
        x = self.res_blocks(x)
        x = self.compress(x)                 # (batch, cc, 9, 9)
        x = x.flatten(start_dim=1)           # (batch, cc*9*9)
        return x

    def forward_value(self, hidden: torch.Tensor) -> torch.Tensor:
        """從低維特徵計算價值預測 ([-1, 1])。"""
        v = F.relu(self.value_drop(self.value_fc(hidden)), inplace=True)
        return torch.tanh(self.value_head(v))

    def head_logits(self, hidden: torch.Tensor, head: str) -> torch.Tensor:
        """從低維特徵計算指定 head 的 logits。

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
