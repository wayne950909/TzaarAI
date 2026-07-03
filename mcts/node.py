"""
mcts/node.py — MCTSNode 定義

提供 MCTS 樹搜尋的節點資料結構。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class MCTSNode:
    """MCTS 樹節點。

    屬性
    ----
    prior : 來自策略網路的先驗機率
    to_play : 輪到操作的玩家（WHITE=1 / BLACK=-1），None 表示未設定
    visit_count : 節點被訪問次數
    value_sum : 價值總和（用於計算平均價值）
    expanded : 是否已展開
    action_dim : 動作空間維度
    legal_mask : 合法動作遮罩
    state : 節點對應的遊戲狀態
    children : 子節點字典 {action_idx: MCTSNode}
    """

    prior: float
    to_play: Optional[int] = None
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    action_dim: int = 0
    legal_mask: Optional[torch.Tensor] = None
    state: Optional[object] = None
    children: Dict[int, MCTSNode] = field(default_factory=dict)

    def mean_value(self) -> float:
        """節點的平均價值（value_sum / visit_count）。"""
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / float(self.visit_count)
