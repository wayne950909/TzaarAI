"""
[DEPRECATED] policy_net_cnn_min17.py

此檔案已遷移至 network/policy_cnn.py，此處僅為向後相容的重新匯出。
請更新所有 import 為：
    from network import PolicyNetCNNMin17
"""

import warnings

from network.policy_cnn import PolicyNetCNNMin17, _ResBlock  # noqa: F401

warnings.warn(
    "policy_net_cnn_min17 is deprecated; use 'from network import PolicyNetCNNMin17'",
    DeprecationWarning,
    stacklevel=2,
)
