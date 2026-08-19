"""
gate — Gatekeeper 評估模組

提供 candidate vs guard 的勝率評估功能。
"""

from gate.gate import GateResult, GateEscalator, gate_keeper

__all__ = [
    "GateResult",
    "GateEscalator",
    "gate_keeper",
]

