"""
network — 神經網路架構

提供：
- PolicyNetCNNMin17：政策/價值雙頭 CNN（主要訓練用）
- _ResBlock：標準殘差區塊（內部使用）
"""

from network.policy_cnn import PolicyNetCNNMin17, _ResBlock

__all__ = [
    "PolicyNetCNNMin17",
    "_ResBlock",
]
