from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch

from Tzaar import DIRECTIONS, Move, MoveKind, TzaarGame
from TzaarAI import PASS_ACTION_IDX, PHASE_STEP_1, PHASE_STEP_2
from strategy_v2_python import StrategyV2Python


Pos = Tuple[int, int]


class _NamedStage:
	def __init__(self, name: str) -> None:
		self.name = name


class _BridgeState:
	"""Synthetic state surface for StrategyV2Python's staged action API."""

	def __init__(self, phase_state, stage_name: str, legal_mask: torch.Tensor) -> None:
		self._phase_state = phase_state
		self.stage = _NamedStage(stage_name)
		self.game = phase_state.game
		self._legal_mask = legal_mask

	def is_done(self) -> bool:
		return self._phase_state.is_done()

	def legal_mask(self) -> torch.Tensor:
		return self._legal_mask


class StrategyV2UnifiedPython:
	"""Adapter strategy that lets StrategyV2Python operate on unified 2-stage actions."""

	def __init__(self) -> None:
		self._base = StrategyV2Python()

	def choose_action(self, state) -> int:
		if state.is_done():
			raise ValueError("strategy called on terminal state")
		legal_mask = state.legal_mask()
		if legal_mask is None:
			raise ValueError("legal mask is undefined")

		stage_name = str(state.stage.name)
		if stage_name == "NEED_STEP1":
			return self._choose_step1(state, legal_mask)
		if stage_name == "NEED_STEP2":
			return self._choose_step2(state, legal_mask)

		legal_idxs = self._legal_indices(legal_mask)
		if not legal_idxs:
			raise ValueError("strategy found no legal action")
		return legal_idxs[0]

	def _choose_step1(self, state, legal_mask: torch.Tensor) -> int:
		legal_idxs = self._legal_indices(legal_mask)
		if not legal_idxs:
			raise ValueError("strategy found no legal action")

		legal_moves: List[Move] = [state.ai.reconstruct_unified_action(i, PHASE_STEP_1) for i in legal_idxs]
		src_mask, idx_to_pos, by_src = self._build_source_mask(state.game, legal_moves)
		if not by_src:
			return legal_idxs[0]

		src_state = _BridgeState(state, "NEED_FIRST_SRC", src_mask)
		src_idx = int(self._base.choose_action(src_state))
		src = idx_to_pos.get(src_idx)
		src_moves = by_src.get(src) if src is not None else None
		if not src_moves:
			src_moves = next(iter(by_src.values()))

		dir_mask = self._direction_mask_for_source(state.game, src_moves)
		dir_state = _BridgeState(state, "NEED_FIRST_DIR", dir_mask)
		dir_idx = int(self._base.choose_action(dir_state))
		chosen = self._pick_move_by_direction(state.game, src_moves, dir_idx)
		if chosen is None:
			return legal_idxs[0]

		action_idx = int(state.ai.encode_unified_action_from_move(chosen))
		if self._is_legal(action_idx, legal_mask):
			return action_idx
		return legal_idxs[0]

	def _choose_step2(self, state, legal_mask: torch.Tensor) -> int:
		legal_idxs = self._legal_indices(legal_mask)
		if not legal_idxs:
			raise ValueError("strategy found no legal action")

		capture_moves: List[Move] = []
		reinforce_moves: List[Move] = []
		for action_idx in legal_idxs:
			if action_idx == PASS_ACTION_IDX:
				continue
			move = state.ai.reconstruct_unified_action(action_idx, PHASE_STEP_2)
			if move.kind == MoveKind.CAPTURE:
				capture_moves.append(move)
			elif move.kind == MoveKind.REINFORCE:
				reinforce_moves.append(move)

		kind_mask = torch.zeros(3, dtype=torch.bool)
		kind_mask[0] = len(capture_moves) > 0
		kind_mask[1] = len(reinforce_moves) > 0
		kind_mask[2] = self._is_legal(PASS_ACTION_IDX, legal_mask)

		kind_state = _BridgeState(state, "NEED_SECOND_KIND", kind_mask)
		kind_idx = int(self._base.choose_action(kind_state))
		if not (0 <= kind_idx < 3 and bool(kind_mask[kind_idx].item())):
			legal_kind = [i for i in range(3) if bool(kind_mask[i].item())]
			kind_idx = legal_kind[0] if legal_kind else 2

		if kind_idx == 2:
			if bool(kind_mask[2].item()):
				return PASS_ACTION_IDX
			if capture_moves:
				kind_idx = 0
			elif reinforce_moves:
				kind_idx = 1
			else:
				return legal_idxs[0]

		target_moves = capture_moves if kind_idx == 0 else reinforce_moves
		if not target_moves:
			return legal_idxs[0]

		src_stage = "NEED_SECOND_CAP_SRC" if kind_idx == 0 else "NEED_SECOND_REIN_SRC"
		dir_stage = "NEED_SECOND_CAP_DIR" if kind_idx == 0 else "NEED_SECOND_REIN_DIR"
		src_mask, idx_to_pos, by_src = self._build_source_mask(state.game, target_moves)
		if not by_src:
			return legal_idxs[0]

		src_state = _BridgeState(state, src_stage, src_mask)
		src_idx = int(self._base.choose_action(src_state))
		src = idx_to_pos.get(src_idx)
		src_moves = by_src.get(src) if src is not None else None
		if not src_moves:
			src_moves = next(iter(by_src.values()))

		dir_mask = self._direction_mask_for_source(state.game, src_moves)
		dir_state = _BridgeState(state, dir_stage, dir_mask)
		dir_idx = int(self._base.choose_action(dir_state))
		chosen = self._pick_move_by_direction(state.game, src_moves, dir_idx)
		if chosen is None:
			return legal_idxs[0]

		action_idx = int(state.ai.encode_unified_action_from_move(chosen))
		if self._is_legal(action_idx, legal_mask):
			return action_idx
		return legal_idxs[0]

	@staticmethod
	def _legal_indices(legal_mask: torch.Tensor) -> List[int]:
		return [i for i in range(int(legal_mask.shape[0])) if bool(legal_mask[i].item())]

	@staticmethod
	def _is_legal(action_idx: int, legal_mask: torch.Tensor) -> bool:
		return 0 <= action_idx < int(legal_mask.shape[0]) and bool(legal_mask[action_idx].item())

	@staticmethod
	def _build_source_mask(game: TzaarGame, moves: List[Move]) -> Tuple[torch.Tensor, Dict[int, Pos], Dict[Pos, List[Move]]]:
		pos_to_idx = game.board.pos_to_index_map()
		idx_to_pos: Dict[int, Pos] = {int(idx): pos for pos, idx in pos_to_idx.items()}
		by_src: Dict[Pos, List[Move]] = {}
		for move in moves:
			if move.src is None:
				continue
			by_src.setdefault(move.src, []).append(move)
		max_idx = max(idx_to_pos.keys()) if idx_to_pos else -1
		mask = torch.zeros(max_idx + 1, dtype=torch.bool)
		for src in by_src:
			idx = pos_to_idx.get(src)
			if idx is not None:
				mask[int(idx)] = True
		return mask, idx_to_pos, by_src

	@staticmethod
	def _direction_mask_for_source(game: TzaarGame, src_moves: List[Move]) -> torch.Tensor:
		mask = torch.zeros(len(DIRECTIONS), dtype=torch.bool)
		if not src_moves:
			return mask
		src = src_moves[0].src
		if src is None:
			return mask
		for i, direction in enumerate(DIRECTIONS):
			hit = game.board.first_occupied_in_direction(src, direction)
			if any(move.dst == hit for move in src_moves):
				mask[i] = True
		return mask

	@staticmethod
	def _pick_move_by_direction(game: TzaarGame, src_moves: List[Move], direction_idx: int) -> Optional[Move]:
		if not src_moves:
			return None
		if 0 <= direction_idx < len(DIRECTIONS):
			direction = DIRECTIONS[direction_idx]
			hit = game.board.first_occupied_in_direction(src_moves[0].src, direction)
			for move in src_moves:
				if move.dst == hit:
					return move
		return random.choice(src_moves)
