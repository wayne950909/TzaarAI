from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#python .\train_supervised_cnn_min17.py --epochs 20 --val-shard-name train_supervised_0007.pt --batch-size 512 --lr 2e-4
#python .\train_supervised_cnn_min17.py --resume .\checkpoints\supervised_cnn_min17\supervised_policy_cnn_min17.pt --epochs 2


class _ResBlock(nn.Module):
    """Standard residual block: Conv->BN->ReLU->Conv->BN + skip -> ReLU."""

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

    - Board branch: player-relative 12x9x9 tensor
        -> stem conv (12->64) -> 4 residual blocks (64ch) -> flatten
    - Global branch: scalar features -> small MLP (->64)
    - Fusion trunk: concat -> FC 512 -> FC 256
    - Heads: action(601), value(1)
    """

    CNN_CHANNELS = 64
    NUM_RES_BLOCKS = 4
    TRUNK_DIM = 256

    def __init__(self, global_feature_dim: int = 17, dropout: float = 0.10) -> None:
        super().__init__()
        self.global_feature_dim = int(global_feature_dim)
        ch = self.CNN_CHANNELS

        # Stem: project input channels to CNN_CHANNELS
        self.stem_conv = nn.Conv2d(12, ch, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(ch)

        # Residual tower
        self.res_blocks = nn.Sequential(*[_ResBlock(ch) for _ in range(self.NUM_RES_BLOCKS)])

        # Global feature branch
        self.global_proj = nn.Sequential(
            nn.Linear(self.global_feature_dim, 64),
            nn.ReLU(inplace=True),
        )

        # Fusion trunk
        self.fc1 = nn.Linear(ch * 9 * 9 + 64, 512)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, self.TRUNK_DIM)

        # Unified action head: 0-299 capture, 300-599 reinforce, 600 pass.
        self.action_head = nn.Linear(self.TRUNK_DIM, 601)
        self.value_head = nn.Linear(self.TRUNK_DIM, 1)

    def encode(self, states: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
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
        # Keep value prediction in [-1, 1] to match win/loss training targets.
        return torch.tanh(self.value_head(hidden))

    def head_logits(self, hidden: torch.Tensor, head: str) -> torch.Tensor:
        if head == "action":
            return self.action_head(hidden)
        raise ValueError(f"Unknown head: {head}")

    def forward(self, states: torch.Tensor, global_features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(states, global_features)
        return {
            "action": self.action_head(hidden),
            "value": self.forward_value(hidden),
        }
