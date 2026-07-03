"""
gate — Gatekeeper 評估模組

提供 candidate vs guard 的勝率評估功能。
"""

from gate.gate import GateResult, gate_keeper

__all__ = [
    "GateResult",
    "gate_keeper",
]
