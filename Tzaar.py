from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


BOARD_LAYOUT: List[List[int]] = [
	[3, 3, 3, 3, 6, -1, -1, -1, -1],
	[6, 2, 2, 2, 5, 6, -1, -1, -1],
	[6, 5, 1, 1, 4, 5, 6, -1, -1],
	[6, 5, 4, 3, 6, 4, 5, 6, -1],
	[6, 5, 4, 6, -1, 3, 1, 2, 3],
	[-1, 3, 2, 1, 3, 6, 1, 2, 3],
	[-1, -1, 3, 2, 1, 4, 4, 2, 3],
	[-1, -1, -1, 3, 2, 5, 5, 5, 3],
	[-1, -1, -1, -1, 3, 6, 6, 6, 6],
]

# Rule text defines six line directions.
DIRECTIONS: Tuple[Tuple[int, int], ...] = (
	(-1, 0),
	(1, 0),
	(0, -1),
	(0, 1),
	(-1, -1),
	(1, 1),
)

DIRECTION_TO_INDEX: Dict[Tuple[int, int], int] = {
	direction: i for i, direction in enumerate(DIRECTIONS)
}

WHITE = 1
BLACK = -1
MovePair = Tuple[Tuple[int, int], Tuple[int, int]]

PHASE_STEP_1 = "step1_capture"
PHASE_STEP_2 = "step2_action"


class MoveKind(str, Enum):
	"""回合中的行動種類。"""

	CAPTURE = "capture"
	REINFORCE = "reinforce"
	PASS = "pass"


class WinReason(str, Enum):
	"""遊戲結束原因。"""

	EXTINCTION = "extinction"
	NO_MANDATORY_CAPTURE = "no_mandatory_capture"


@dataclass(frozen=True)
class Move:
	"""一個完整動作：種類 + 起點 + 終點。"""

	kind: MoveKind
	src: Optional[Tuple[int, int]] = None
	dst: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class GameResult:
	"""勝負結果。"""

	winner: int
	reason: WinReason


class Board:
	"""Board stores each valid cell as a stack of piece codes (1..6)."""

	def __init__(self, layout: Optional[List[List[int]]] = None) -> None:
		layout = layout or BOARD_LAYOUT
		self.grid: List[List[Optional[Tuple[int, int]]]] = []
		for row in layout:
			row_data: List[Optional[Tuple[int, int]]] = []
			for value in row:
				if value == -1:
					row_data.append(None)
				elif value == 0:
					row_data.append((0, 0))
				else:
					row_data.append((value, 1))
			self.grid.append(row_data)

		self._valid_positions: Tuple[Tuple[int, int], ...] = tuple(
			(r, c)
			for r in range(len(self.grid))
			for c in range(len(self.grid[0]))
			if self.grid[r][c] is not None
		)
		self._pos_to_idx: Dict[Tuple[int, int], int] = {
			pos: i for i, pos in enumerate(self._valid_positions)
		}
		self._rays: Dict[Tuple[int, int], Tuple[Tuple[Tuple[int, int], ...], ...]] = {}
		for pos in self._valid_positions:
			rays_for_pos: List[Tuple[Tuple[int, int], ...]] = []
			for dr, dc in DIRECTIONS:
				r, c = pos[0] + dr, pos[1] + dc
				ray_cells: List[Tuple[int, int]] = []
				while 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] is not None:
					ray_cells.append((r, c))
					r += dr
					c += dc
				rays_for_pos.append(tuple(ray_cells))
			self._rays[pos] = tuple(rays_for_pos)

		nrows = len(self.grid)
		ncols = len(self.grid[0])
		self._piece_np: np.ndarray = np.zeros((nrows, ncols), dtype=np.int8)
		self._height_np: np.ndarray = np.zeros((nrows, ncols), dtype=np.float32)
		for pos in self._valid_positions:
			cell = self.grid[pos[0]][pos[1]]
			if cell is not None:
				self._piece_np[pos[0], pos[1]] = cell[0]
				self._height_np[pos[0], pos[1]] = cell[1]

	def clone(self) -> Board:
		"""深拷貝棋盤，避免模擬時污染原始狀態。"""

		copied = Board.__new__(Board)
		copied.grid = [[cell for cell in row] for row in self.grid]
		copied._valid_positions = self._valid_positions
		copied._pos_to_idx = self._pos_to_idx
		copied._rays = self._rays
		copied._piece_np = self._piece_np.copy()
		copied._height_np = self._height_np.copy()
		return copied

	@property
	def rows(self) -> int:
		return len(self.grid)

	@property
	def cols(self) -> int:
		return len(self.grid[0])

	def is_inside(self, pos: Tuple[int, int]) -> bool:
		"""檢查座標是否在矩形邊界內。"""

		r, c = pos
		return 0 <= r < self.rows and 0 <= c < self.cols

	def is_valid_cell(self, pos: Tuple[int, int]) -> bool:
		"""檢查座標是否為可用棋格（非 -1 洞位）。"""

		if not self.is_inside(pos):
			return False
		r, c = pos
		return self.grid[r][c] is not None

	def get_cell(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
		"""讀取指定位置的格子資訊 (top_piece, height)，空格回傳 (0, 0)，洞位回傳 None。"""

		r, c = pos
		return self.grid[r][c]

	def set_cell(self, pos: Tuple[int, int], piece_code: int, height: int) -> None:
		"""覆寫指定位置的格子資訊。清空格子用 set_cell(pos, 0, 0)。"""

		r, c = pos
		self.grid[r][c] = (piece_code, height)
		self._piece_np[r, c] = piece_code
		self._height_np[r, c] = height

	def is_empty(self, pos: Tuple[int, int]) -> bool:
		"""檢查是否為有效空格。"""

		cell = self.get_cell(pos)
		return cell is not None and cell[1] == 0

	def top_piece(self, pos: Tuple[int, int]) -> int:
		"""取得棋堆頂端棋子代碼。"""

		cell = self.get_cell(pos)
		if cell is None or cell[1] == 0:
			raise ValueError("No top piece on invalid/empty cell")
		return cell[0]

	def height(self, pos: Tuple[int, int]) -> int:
		"""取得棋堆高度（無效格回傳 0）。"""

		cell = self.get_cell(pos)
		if cell is None:
			return 0
		return cell[1]

	def iter_valid_positions(self) -> Iterable[Tuple[int, int]]:
		"""走訪所有有效棋格座標。"""
		return iter(self._valid_positions)

	def valid_positions(self) -> Tuple[Tuple[int, int], ...]:
		"""回傳有效棋格座標（固定順序）。"""
		return self._valid_positions

	def pos_to_index_map(self) -> Dict[Tuple[int, int], int]:
		"""回傳有效棋格座標到索引的映射。"""
		return self._pos_to_idx

	def iter_player_positions(self, player: int) -> Iterable[Tuple[int, int]]:
		"""走訪指定玩家目前可操作的棋堆座標。"""

		for pos in self.iter_valid_positions():
			if self.is_empty(pos):
				continue
			if piece_owner(self.top_piece(pos)) == player:
				yield pos

	def first_occupied_in_direction(
		self, start: Tuple[int, int], direction: Tuple[int, int]
	) -> Optional[Tuple[int, int]]:
		"""沿給定方向找到第一個非空棋格。"""

		direction_idx = DIRECTION_TO_INDEX.get(direction)
		if direction_idx is not None:
			ray_map = self._rays.get(start)
			if ray_map is not None:
				for pos in ray_map[direction_idx]:
					if not self.is_empty(pos):
						return pos
				return None

		r, c = start
		dr, dc = direction
		cur = (r + dr, c + dc)
		while self.is_valid_cell(cur):
			if not self.is_empty(cur):
				return cur
			cur = (cur[0] + dr, cur[1] + dc)
		return None


def piece_owner(piece: int) -> int:
	"""棋子代碼轉換成陣營（WHITE / BLACK）。"""

	if piece in (1, 2, 3):
		return WHITE
	if piece in (4, 5, 6):
		return BLACK
	raise ValueError(f"Unknown piece code: {piece}")


def piece_type(piece: int) -> int:
	"""棋子代碼轉換成種類（1/2/3）。"""

	if piece in (1, 4):
		return 1
	if piece in (2, 5):
		return 2
	if piece in (3, 6):
		return 3
	raise ValueError(f"Unknown piece code: {piece}")


class TzaarGame:
	"""Tzaar 規則引擎，包含白方首回合只下一步的特例。"""

	def __init__(self) -> None:
		self.board = Board()
		self.current_player = WHITE
		self.turn_number = 1
		self.result: Optional[GameResult] = None
		self.move_log: List[Move] = []
		self._waiting_second_step = False
		self._counts: Dict[int, Dict[int, int]] = {
			WHITE: {1: 0, 2: 0, 3: 0},
			BLACK: {1: 0, 2: 0, 3: 0},
		}
		for pos in self.board.iter_valid_positions():
			cell = self.board.get_cell(pos)
			if cell is None or cell[1] == 0:
				continue
			self._counts[piece_owner(cell[0])][piece_type(cell[0])] += 1


	def clone(self) -> TzaarGame:
		"""深拷貝整個對局狀態。"""

		# Hot path for MCTS: avoid running __init__ and rebuilding counts.
		copied = TzaarGame.__new__(TzaarGame)
		copied.board = self.board.clone()
		copied.current_player = self.current_player
		copied.turn_number = self.turn_number
		copied.result = self.result
		copied.move_log = list(self.move_log)
		copied._waiting_second_step = self._waiting_second_step
		copied._counts = {
			WHITE: dict(self._counts[WHITE]),
			BLACK: dict(self._counts[BLACK]),
		}
		return copied

	def is_waiting_second_step(self) -> bool:
		"""是否處於第一步完成、等待第二步的狀態。"""

		return self._waiting_second_step

	def is_white_first_turn(self) -> bool:
		"""是否為白方第一回合（第二步強制略過）。"""

		return self.turn_number == 1 and self.current_player == WHITE

	def is_game_over(self) -> bool:
		"""是否已產生勝負結果。"""

		return self.result is not None

	def get_piece_counts(self) -> Dict[int, Dict[int, int]]:
		"""統計雙方三種棋子的總數。"""
		"""統計雙方三種棋子的疊數（每疊以頂端棋子種類計）。"""

		return {
			WHITE: dict(self._counts[WHITE]),
			BLACK: dict(self._counts[BLACK]),
		}

	def _extinction_winner(self) -> Optional[int]:
		"""若有任一方任一棋種滅絕，回傳勝方；否則回傳 None。"""

		counts = self.get_piece_counts()
		for player in (WHITE, BLACK):
			if any(counts[player][ptype] == 0 for ptype in (1, 2, 3)):
				return -player
		return None

	def _capture_pairs_for_player(self, player: int) -> List[MovePair]:
		"""列舉指定玩家所有合法吃子 (src, dst)。"""

		pairs: List[MovePair] = []
		for src in self.board.iter_player_positions(player):
			src_height = self.board.height(src)
			for direction in DIRECTIONS:
				dst = self.board.first_occupied_in_direction(src, direction)
				if dst is None:
					continue
				dst_piece = self.board.top_piece(dst)
				if piece_owner(dst_piece) == player:
					continue
				if self.board.height(dst) <= src_height:
					pairs.append((src, dst))
		return pairs

	def _reinforce_pairs_for_player(self, player: int) -> List[MovePair]:
		"""列舉指定玩家所有合法疊子 (src, dst)。"""

		pairs: List[MovePair] = []
		for src in self.board.iter_player_positions(player):
			for direction in DIRECTIONS:
				dst = self.board.first_occupied_in_direction(src, direction)
				if dst is None:
					continue
				dst_piece = self.board.top_piece(dst)
				if piece_owner(dst_piece) == player:
					pairs.append((src, dst))
		return pairs

	def _has_capture_for_player(self, player: int) -> bool:
		"""檢查指定玩家是否至少存在一手合法吃子。"""

		for src in self.board.iter_player_positions(player):
			src_height = self.board.height(src)
			for direction in DIRECTIONS:
				dst = self.board.first_occupied_in_direction(src, direction)
				if dst is None:
					continue
				dst_piece = self.board.top_piece(dst)
				if piece_owner(dst_piece) == player:
					continue
				if self.board.height(dst) <= src_height:
					return True
		return False

	def _has_reinforce_for_player(self, player: int) -> bool:
		"""檢查指定玩家是否至少存在一手合法疊子。"""

		for src in self.board.iter_player_positions(player):
			for direction in DIRECTIONS:
				dst = self.board.first_occupied_in_direction(src, direction)
				if dst is None:
					continue
				dst_piece = self.board.top_piece(dst)
				if piece_owner(dst_piece) == player:
					return True
		return False

	def get_legal_first_moves(self) -> List[Move]:
		"""取得第一步合法動作（只能是吃子）。"""

		if self.is_game_over() or self.is_waiting_second_step():
			return []
		return [
			Move(kind=MoveKind.CAPTURE, src=src, dst=dst)
			for src, dst in self._capture_pairs_for_player(self.current_player)
		]

	def get_legal_second_captures(self) -> List[Move]:
		"""取得第二步合法吃子列表。"""

		if self.is_game_over() or not self.is_waiting_second_step():
			return []
		return [
			Move(kind=MoveKind.CAPTURE, src=src, dst=dst)
			for src, dst in self._capture_pairs_for_player(self.current_player)
		]

	def get_legal_second_reinforces(self) -> List[Move]:
		"""取得第二步合法疊子列表。"""

		if self.is_game_over() or not self.is_waiting_second_step():
			return []
		return [
			Move(kind=MoveKind.REINFORCE, src=src, dst=dst)
			for src, dst in self._reinforce_pairs_for_player(self.current_player)
		]

	def can_second_step_pass(self) -> bool:
		"""第二步是否可選擇 PASS。"""

		return (not self.is_game_over()) and self.is_waiting_second_step()

	def has_legal_second_capture(self) -> bool:
		"""第二步是否至少存在一手合法吃子。"""

		if self.is_game_over() or not self.is_waiting_second_step():
			return False
		return self._has_capture_for_player(self.current_player)

	def has_legal_second_reinforce(self) -> bool:
		"""第二步是否至少存在一手合法疊子。"""

		if self.is_game_over() or not self.is_waiting_second_step():
			return False
		return self._has_reinforce_for_player(self.current_player)

	def _apply_capture(self, move: Move, player: int) -> None:
		"""實際套用吃子，並做所有權/高度合法性檢查。"""

		if move.kind != MoveKind.CAPTURE or move.src is None or move.dst is None:
			raise ValueError("Invalid capture move")
		src_cell = self.board.get_cell(move.src)
		dst_cell = self.board.get_cell(move.dst)
		if (
			src_cell is None or dst_cell is None
			or src_cell[1] == 0 or dst_cell[1] == 0
			or piece_owner(src_cell[0]) != player
			or piece_owner(dst_cell[0]) == player
		):
			raise ValueError("Capture move violates ownership constraints")
		if dst_cell[1] > src_cell[1]:
			raise ValueError("Cannot capture stronger stack")

		# 被吃的棋疊消失：其頂端種類從快取扣除
		self._counts[piece_owner(dst_cell[0])][piece_type(dst_cell[0])] -= 1
		self.board.set_cell(move.src, 0, 0)
		self.board.set_cell(move.dst, src_cell[0], src_cell[1])

	def _apply_reinforce(self, move: Move, player: int) -> None:
		"""實際套用疊子，並做所有權合法性檢查。"""

		if move.kind != MoveKind.REINFORCE or move.src is None or move.dst is None:
			raise ValueError("Invalid reinforce move")
		src_cell = self.board.get_cell(move.src)
		dst_cell = self.board.get_cell(move.dst)
		if (
			src_cell is None or dst_cell is None
			or src_cell[1] == 0 or dst_cell[1] == 0
			or piece_owner(src_cell[0]) != player
			or piece_owner(dst_cell[0]) != player
		):
			raise ValueError("Reinforce move violates ownership constraints")

		# dst 的頂端種類被壓入底層，改由 src 頂端代表：扣除 dst 原有種類
		self._counts[piece_owner(dst_cell[0])][piece_type(dst_cell[0])] -= 1
		self.board.set_cell(move.src, 0, 0)
		self.board.set_cell(move.dst, src_cell[0], src_cell[1] + dst_cell[1])

	def _check_immediate_extinction(self) -> bool:
		"""每步後立即檢查滅絕勝利條件。"""

		winner = self._extinction_winner()
		if winner is not None:
			self.result = GameResult(winner=winner, reason=WinReason.EXTINCTION)
			return True
		return False

	def _end_turn_and_check_next_player(self) -> None:
		"""結束當前回合，切換到下一位玩家。"""

		self.current_player *= -1
		self.turn_number += 1

	def resolve_no_mandatory_capture_if_needed(self) -> bool:
		"""若目前輪到的玩家第一步無法吃子，直接判負並回傳 True。"""

		if self.is_game_over() or self.is_waiting_second_step():
			return False
		if self._has_capture_for_player(self.current_player):
			return False
		self.result = GameResult(
			winner=-self.current_player,
			reason=WinReason.NO_MANDATORY_CAPTURE,
		)
		return True

	def play_first_step(self, first_move: Move) -> None:
		"""執行第一步（必吃）並立即更新真實盤面。"""

		if self.is_game_over():
			raise ValueError("Game is already over")
		if self.is_waiting_second_step():
			raise ValueError("Cannot play first step while waiting for second step")

		if self.resolve_no_mandatory_capture_if_needed():
			return

		is_white_first_turn = self.is_white_first_turn()
		self._apply_capture(first_move, self.current_player)
		self.move_log.append(first_move)
		if self._check_immediate_extinction():
			return

		if is_white_first_turn:
			self._waiting_second_step = False
			self._end_turn_and_check_next_player()
			return

		self._waiting_second_step = True

	def play_second_step(self, second_move: Optional[Move] = None) -> None:
		"""執行第二步（吃/疊/過），完成後結束回合。"""

		if self.is_game_over():
			raise ValueError("Game is already over")
		if not self.is_waiting_second_step():
			raise ValueError("Cannot play second step before first step")

		if second_move is None:
			second_move = Move(kind=MoveKind.PASS)

		player = self.current_player
		if second_move.kind == MoveKind.CAPTURE:
			self._apply_capture(second_move, player)
			self.move_log.append(second_move)
			if self._check_immediate_extinction():
				self._waiting_second_step = False
				return
		elif second_move.kind == MoveKind.REINFORCE:
			self._apply_reinforce(second_move, player)
			self.move_log.append(second_move)
			if self._check_immediate_extinction():
				self._waiting_second_step = False
				return
		else:
			if second_move.kind != MoveKind.PASS or not self.can_second_step_pass():
				raise ValueError("Second move is not legal")
			self.move_log.append(second_move)

		self._waiting_second_step = False
		self._end_turn_and_check_next_player()

