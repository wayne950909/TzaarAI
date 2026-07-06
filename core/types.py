"""
core/types.py — 遊戲結果型別定義

提供 GameResult 列舉，定義對局結果類型。
"""

from __future__ import annotations

from enum import Enum


class GameResult(Enum):
    """對局結果。"""
    WHITE_WIN = 1
    BLACK_WIN = -1
    DRAW = 0
