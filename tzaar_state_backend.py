from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from Tzaar import BOARD_LAYOUT, DIRECTIONS, Move, MoveKind, piece_owner
from TzaarAI import PASS_ACTION_IDX, PHASE_STEP_1


def _maybe_add_default_cpp_build_path() -> None:
    base = Path(__file__).resolve().parent
    candidates = [
        base / "cpp" / "build" / "Release",
        base / "build" / "Release",
    ]
    for module_dir in candidates:
        if module_dir.exists():
            module_dir_s = str(module_dir)
            if module_dir_s not in sys.path:
                sys.path.insert(0, module_dir_s)


def try_load_cpp_backend() -> Optional[Any]:
    _maybe_add_default_cpp_build_path()
    try:
        import tzaar_cpp  # type: ignore

        return tzaar_cpp
    except Exception:
        return None


def resolve_backend_name(default_name: str = "python") -> str:
    backend = os.environ.get("TZAAR_STATE_BACKEND", default_name).strip().lower()
    if backend not in {"python", "cpp"}:
        return default_name
    return backend


_VALID_POSITIONS: Tuple[Tuple[int, int], ...] = tuple(
    (r, c)
    for r, row in enumerate(BOARD_LAYOUT)
    for c, value in enumerate(row)
    if value != -1
)
_POS_TO_IDX: Dict[Tuple[int, int], int] = {pos: i for i, pos in enumerate(_VALID_POSITIONS)}
_PHASE_TO_STAGE_NAME = {
    "step1_capture": "NEED_STEP1",
    "step2_action": "NEED_STEP2",
    "": "DONE",
}


class _CppStageName:
    def __init__(self, name: str) -> None:
        self.name = name


class _CppBoardSnapshot:
    def __init__(self, cells: List[Dict[str, int]]) -> None:
        self._occupied: Dict[Tuple[int, int], Tuple[int, int]] = {
            (int(cell["row"]), int(cell["col"])): (int(cell["top_piece"]), int(cell["height"]))
            for cell in cells
        }

    def valid_positions(self) -> Tuple[Tuple[int, int], ...]:
        return _VALID_POSITIONS

    def pos_to_index_map(self) -> Dict[Tuple[int, int], int]:
        return _POS_TO_IDX

    def is_empty(self, pos: Tuple[int, int]) -> bool:
        return pos not in self._occupied

    def top_piece(self, pos: Tuple[int, int]) -> int:
        if pos not in self._occupied:
            raise ValueError("No top piece on invalid/empty cell")
        return self._occupied[pos][0]

    def height(self, pos: Tuple[int, int]) -> int:
        item = self._occupied.get(pos)
        if item is None:
            return 0
        return item[1]

    def first_occupied_in_direction(
        self,
        start: Tuple[int, int],
        direction: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        r, c = start
        dr, dc = direction
        rr, cc = r + dr, c + dc
        while 0 <= rr < len(BOARD_LAYOUT) and 0 <= cc < len(BOARD_LAYOUT[0]) and BOARD_LAYOUT[rr][cc] != -1:
            pos = (rr, cc)
            if pos in self._occupied:
                return pos
            rr += dr
            cc += dc
        return None


class _CppGameSnapshot:
    def __init__(self, current_player: int, turn_number: int, piece_counts: Dict[int, Dict[int, int]], cells: List[Dict[str, int]]) -> None:
        self.current_player = int(current_player)
        self.turn_number = int(turn_number)
        self._counts = {
            int(player): {int(piece_type): int(count) for piece_type, count in counts.items()}
            for player, counts in piece_counts.items()
        }
        self.board = _CppBoardSnapshot(cells)

    def _capture_pairs_for_player(self, player: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for src in self.board.valid_positions():
            if self.board.is_empty(src):
                continue
            if piece_owner(self.board.top_piece(src)) != player:
                continue
            src_height = self.board.height(src)
            for direction in DIRECTIONS:
                dst = self.board.first_occupied_in_direction(src, direction)
                if dst is None:
                    continue
                if piece_owner(self.board.top_piece(dst)) == player:
                    continue
                if self.board.height(dst) <= src_height:
                    pairs.append((src, dst))
        return pairs


class _CppAIAdapter:
    def __init__(self, owner: "CppPhaseGameStateAdapter") -> None:
        self._owner = owner

    def reconstruct_unified_action(self, action_idx: int, phase: str) -> Move:
        decoded = self._owner._cpp_module.decode_action_idx(int(action_idx))
        kind_name = str(decoded["kind"])
        if phase == PHASE_STEP_1 and kind_name != "capture":
            raise ValueError("step1 only allows capture actions")
        if kind_name == "pass":
            return Move(kind=MoveKind.PASS)

        src_idx = int(decoded["src_idx"])
        direction_idx = int(decoded["direction_idx"])
        src = _VALID_POSITIONS[src_idx]
        dst = self._owner.game.board.first_occupied_in_direction(src, DIRECTIONS[direction_idx])
        if dst is None:
            raise ValueError("No occupied cell found in chosen direction")
        kind = MoveKind.CAPTURE if kind_name == "capture" else MoveKind.REINFORCE
        return Move(kind=kind, src=src, dst=dst)

    def encode_unified_action_from_move(self, move: Move) -> int:
        if move.kind == MoveKind.PASS:
            return PASS_ACTION_IDX
        if move.src is None or move.dst is None:
            raise ValueError("non-pass move requires src/dst")
        src_idx = _POS_TO_IDX[move.src]
        direction_idx = -1
        for i, direction in enumerate(DIRECTIONS):
            hit = self._owner.game.board.first_occupied_in_direction(move.src, direction)
            if hit == move.dst:
                direction_idx = i
                break
        if direction_idx < 0:
            raise ValueError("cannot reconstruct direction from move")

        edge_table = self._owner._cpp_module.action_edge_table()
        for edge_idx, edge in enumerate(edge_table):
            if int(edge[0]) == src_idx and int(edge[1]) == direction_idx:
                kind_name = "capture" if move.kind == MoveKind.CAPTURE else "reinforce"
                return int(self._owner._cpp_module.encode_action_idx(kind_name, edge_idx))
        raise ValueError("(src, direction) not present in fixed edge table")


class CppPhaseGameStateAdapter:
    def __init__(self, cpp_module: Any, inner: Optional[Any] = None) -> None:
        self._cpp_module = cpp_module
        self._inner = cpp_module.PhaseGameState() if inner is None else inner
        self.ai = _CppAIAdapter(self)

    @property
    def stage(self) -> _CppStageName:
        return _CppStageName(_PHASE_TO_STAGE_NAME.get(self.phase() or "", "DONE"))

    @property
    def game(self) -> _CppGameSnapshot:
        return _CppGameSnapshot(
            current_player=self.current_player(),
            turn_number=int(self._inner.turn_number()),
            piece_counts=self._inner.piece_counts(),
            cells=self._inner.board_cells(),
        )

    def clone(self) -> "CppPhaseGameStateAdapter":
        return CppPhaseGameStateAdapter(self._cpp_module, self._inner.clone())

    def is_done(self) -> bool:
        return bool(self._inner.is_done())

    def winner(self) -> Optional[int]:
        winner = int(self._inner.winner())
        return None if winner == 0 else winner

    def current_player(self) -> int:
        return int(self._inner.current_player())

    def phase(self) -> Optional[str]:
        phase = str(self._inner.phase())
        return None if phase == "" else phase

    def legal_mask(self) -> Optional[torch.Tensor]:
        if self.is_done():
            return None
        mask = self._inner.legal_mask()
        return torch.tensor(mask, dtype=torch.bool)

    def apply_action(self, action_idx: int, legal_mask: Optional[torch.Tensor] = None) -> None:
        _ = legal_mask
        self._inner.apply_action(int(action_idx))


def create_cpp_phase_state(cpp_module: Any) -> CppPhaseGameStateAdapter:
    return CppPhaseGameStateAdapter(cpp_module)
