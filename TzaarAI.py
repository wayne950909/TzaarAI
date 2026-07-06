from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from Tzaar import (
	BLACK,
	Board,
	DIRECTIONS,
	WHITE,
	GameResult,
	Move,
	MoveKind,
	TzaarGame,
	WinReason,
	piece_owner,
	piece_type,
)


N_POSITIONS = 60   # 棋盤有效格數（固定不變）
N_DIRECTIONS = 6   # 移動方向數
N_KINDS = 3        # second step 種類：capture / reinforce / pass
KIND_ORDER = (MoveKind.CAPTURE, MoveKind.REINFORCE, MoveKind.PASS)
KIND_TO_IDX = {k: i for i, k in enumerate(KIND_ORDER)}

PHASE_STEP_1 = "step1_capture"
PHASE_STEP_2 = "step2_action"

# Backward compatibility aliases for legacy imports.
PHASE_FIRST_CAPTURE_SOURCE = PHASE_STEP_1
PHASE_FIRST_CAPTURE_DIRECTION = PHASE_STEP_1
PHASE_SECOND_KIND = PHASE_STEP_2
PHASE_SECOND_CAPTURE_SOURCE = PHASE_STEP_2
PHASE_SECOND_CAPTURE_DIRECTION = PHASE_STEP_2
PHASE_SECOND_REINFORCE_SOURCE = PHASE_STEP_2
PHASE_SECOND_REINFORCE_DIRECTION = PHASE_STEP_2

N_EDGE_ACTIONS = 300
N_ACTIONS = 601
CAPTURE_OFFSET = 0
REINFORCE_OFFSET = 300
PASS_ACTION_IDX = 600

# Fixed table: each item is (source_position_index, direction_index).
ACTION_EDGE_TABLE: Tuple[Tuple[int, int], ...] = (
	(0, 1), (0, 3), (0, 5), (1, 1), (1, 2), (1, 3), (1, 5), (2, 1), (2, 2), (2, 3),
	(2, 5), (3, 1), (3, 2), (3, 3), (3, 5), (4, 1), (4, 2), (4, 5), (5, 0), (5, 1),
	(5, 3), (5, 5), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (7, 0), (7, 1),
	(7, 2), (7, 3), (7, 4), (7, 5), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5),
	(9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (10, 1), (10, 2), (10, 4), (10, 5),
	(11, 0), (11, 1), (11, 3), (11, 5), (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5),
	(13, 0), (13, 1), (13, 2), (13, 3), (13, 4), (13, 5), (14, 0), (14, 1), (14, 2), (14, 3),
	(14, 4), (14, 5), (15, 0), (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (16, 0), (16, 1),
	(16, 2), (16, 3), (16, 4), (16, 5), (17, 1), (17, 2), (17, 4), (17, 5), (18, 0), (18, 1),
	(18, 3), (18, 5), (19, 0), (19, 1), (19, 2), (19, 3), (19, 4), (19, 5), (20, 0), (20, 1),
	(20, 2), (20, 3), (20, 4), (20, 5), (21, 0), (21, 1), (21, 2), (21, 3), (21, 4), (22, 0),
	(22, 2), (22, 3), (22, 4), (22, 5), (23, 0), (23, 1), (23, 2), (23, 3), (23, 4), (23, 5),
	(24, 0), (24, 1), (24, 2), (24, 3), (24, 4), (24, 5), (25, 1), (25, 2), (25, 4), (25, 5),
	(26, 0), (26, 3), (26, 5), (27, 0), (27, 1), (27, 2), (27, 3), (27, 4), (27, 5), (28, 0),
	(28, 1), (28, 2), (28, 3), (28, 4), (28, 5), (29, 0), (29, 1), (29, 2), (29, 4), (29, 5),
	(30, 0), (30, 1), (30, 3), (30, 4), (30, 5), (31, 0), (31, 1), (31, 2), (31, 3), (31, 4),
	(31, 5), (32, 0), (32, 1), (32, 2), (32, 3), (32, 4), (32, 5), (33, 1), (33, 2), (33, 4),
	(34, 0), (34, 3), (34, 4), (34, 5), (35, 0), (35, 1), (35, 2), (35, 3), (35, 4), (35, 5),
	(36, 0), (36, 1), (36, 2), (36, 3), (36, 4), (36, 5), (37, 1), (37, 2), (37, 3), (37, 4),
	(37, 5), (38, 0), (38, 1), (38, 2), (38, 3), (38, 5), (39, 0), (39, 1), (39, 2), (39, 3),
	(39, 4), (39, 5), (40, 0), (40, 1), (40, 2), (40, 3), (40, 4), (40, 5), (41, 0), (41, 1),
	(41, 2), (41, 4), (42, 0), (42, 3), (42, 4), (42, 5), (43, 0), (43, 1), (43, 2), (43, 3),
	(43, 4), (43, 5), (44, 0), (44, 1), (44, 2), (44, 3), (44, 4), (44, 5), (45, 0), (45, 1),
	(45, 2), (45, 3), (45, 4), (45, 5), (46, 0), (46, 1), (46, 2), (46, 3), (46, 4), (46, 5),
	(47, 0), (47, 1), (47, 2), (47, 3), (47, 4), (47, 5), (48, 0), (48, 1), (48, 2), (48, 4),
	(49, 0), (49, 3), (49, 4), (49, 5), (50, 0), (50, 1), (50, 2), (50, 3), (50, 4), (50, 5),
	(51, 0), (51, 1), (51, 2), (51, 3), (51, 4), (51, 5), (52, 0), (52, 1), (52, 2), (52, 3),
	(52, 4), (52, 5), (53, 0), (53, 1), (53, 2), (53, 3), (53, 4), (53, 5), (54, 0), (54, 1),
	(54, 2), (54, 4), (55, 0), (55, 3), (55, 4), (56, 0), (56, 2), (56, 3), (56, 4), (57, 0),
	(57, 2), (57, 3), (57, 4), (58, 0), (58, 2), (58, 3), (58, 4), (59, 0), (59, 2), (59, 4),
)

if len(ACTION_EDGE_TABLE) != N_EDGE_ACTIONS:
	raise ValueError(f"Expected {N_EDGE_ACTIONS} action edges, got {len(ACTION_EDGE_TABLE)}")

EDGE_TO_IDX: Dict[Tuple[int, int], int] = {
	edge: idx for idx, edge in enumerate(ACTION_EDGE_TABLE)
}

TRAINING_PHASE_ORDER = (PHASE_STEP_1, PHASE_STEP_2)
TRAINING_PHASE_TO_IDX = {name: i for i, name in enumerate(TRAINING_PHASE_ORDER)}
TRAINING_STATE_DIM = 2 + N_POSITIONS * 3 + N_POSITIONS * 4 + N_POSITIONS + len(TRAINING_PHASE_ORDER)

# State vector layout constants（進行增量更新時使用）
STATE_TURN_PLAYER_BASE = 0          # [0:2] turn_number, current_player
STATE_COLOR_BASE = 2                # [2:182] color × 60 (3 dims per position)
STATE_TYPE_BASE = STATE_COLOR_BASE + N_POSITIONS * 3  # [182:422] type × 60 (4 dims per position)
STATE_HEIGHT_BASE = STATE_TYPE_BASE + N_POSITIONS * 4 # [422:482] height × 60 (1 dim per position)
STATE_PHASE_BASE = STATE_HEIGHT_BASE + N_POSITIONS

# Human-readable labels matching DIRECTIONS order in Tzaar.py
DIRECTION_LABELS = [
	"Up     (-1, 0)",
	"Down   ( 1, 0)",
	"Left   ( 0,-1)",
	"Right  ( 0, 1)",
	"UpLeft (-1,-1)",
	"DnRight( 1, 1)",
]

# ---------------------------------------------------------------------------
# 合法性底層工具（直接掃棋盤）
# ---------------------------------------------------------------------------


def _iter_legal_direction_targets(
	board: Board,
	src: Tuple[int, int],
	player: int,
	kind: MoveKind,
) -> List[Tuple[int, Tuple[int, int]]]:
	"""直接掃描 src 的六個方向，回傳合法 (direction_idx, dst)。

	這是所有合法判定的核心，第一步/第二步都共用同一套規則來源。
	"""
	if board.is_empty(src) or piece_owner(board.top_piece(src)) != player:
		return []

	src_height = board.height(src)
	result: List[Tuple[int, Tuple[int, int]]] = []
	for i, direction in enumerate(DIRECTIONS):
		# 規則：每個方向只看「第一個非空格」。
		dst = board.first_occupied_in_direction(src, direction)
		if dst is None:
			continue
		dst_is_self = piece_owner(board.top_piece(dst)) == player
		if kind == MoveKind.CAPTURE:
			# 吃子：目標必須是敵方，且目標高度不能大於 src。
			if dst_is_self:
				continue
			if board.height(dst) <= src_height:
				result.append((i, dst))
		elif kind == MoveKind.REINFORCE:
			# 疊子：目標必須是己方。
			if dst_is_self:
				result.append((i, dst))
		else:
			raise ValueError("kind must be CAPTURE or REINFORCE")
	return result


def _direction_mask_from_board(
	board: Board,
	src: Tuple[int, int],
	player: int,
	kind: MoveKind,
	device: Optional[torch.device] = None,
) -> torch.Tensor:
	"""直接掃棋盤一步產生 6 維方向 mask。"""
	mask = torch.zeros(N_DIRECTIONS, dtype=torch.bool, device=device)
	for direction_idx, _ in _iter_legal_direction_targets(board, src, player, kind):
		mask[direction_idx] = True
	return mask

def _source_mask_from_board(
	board: Board,
	player: int,
	kind: MoveKind,
	pos_to_idx: dict[Tuple[int, int], int],
	device: Optional[torch.device] = None,
) -> torch.Tensor:
	"""直接掃棋盤一步產生 60 維來源 mask。"""
	mask = torch.zeros(N_POSITIONS, dtype=torch.bool, device=device)
	for src in board.iter_player_positions(player):
		if _iter_legal_direction_targets(board, src, player, kind):
			mask[pos_to_idx[src]] = True
	return mask

def _has_legal_move_from_board(board: Board, player: int, kind: MoveKind) -> bool:
	"""直接掃棋盤檢查是否存在至少一手合法動作。"""
	for src in board.iter_player_positions(player):
		if _iter_legal_direction_targets(board, src, player, kind):
			return True
	return False

def _reconstruct_non_pass_move(
	game: TzaarGame,
	kind: MoveKind,
	src: Tuple[int, int],
	direction_idx: int,
) -> Move:
	"""依據 (src, direction_idx) 還原非 PASS 動作。

	依你的設定，這裡不做 legal-list membership 驗證，只做結構性檢查。
	"""

	if not (0 <= direction_idx < N_DIRECTIONS):
		raise ValueError("direction_idx out of range")
	dst = game.board.first_occupied_in_direction(src, DIRECTIONS[direction_idx])
	if dst is None:
		raise ValueError("No occupied cell found in chosen direction")
	return Move(kind=kind, src=src, dst=dst)


def encode_action_idx(kind: MoveKind, edge_idx: Optional[int] = None) -> int:
	if kind == MoveKind.PASS:
		return PASS_ACTION_IDX
	if edge_idx is None or not (0 <= edge_idx < N_EDGE_ACTIONS):
		raise ValueError("edge_idx out of range")
	if kind == MoveKind.CAPTURE:
		return CAPTURE_OFFSET + edge_idx
	if kind == MoveKind.REINFORCE:
		return REINFORCE_OFFSET + edge_idx
	raise ValueError("Unsupported kind")


def decode_action_idx(action_idx: int) -> Tuple[MoveKind, Optional[int], Optional[int], Optional[int]]:
	if action_idx == PASS_ACTION_IDX:
		return MoveKind.PASS, None, None, None
	if not (0 <= action_idx < PASS_ACTION_IDX):
		raise ValueError("action_idx out of range")
	if action_idx < REINFORCE_OFFSET:
		kind = MoveKind.CAPTURE
		edge_idx = action_idx
	else:
		kind = MoveKind.REINFORCE
		edge_idx = action_idx - REINFORCE_OFFSET
	src_idx, direction_idx = ACTION_EDGE_TABLE[edge_idx]
	return kind, edge_idx, src_idx, direction_idx


class TzaarAIInterface:
	"""AI 查詢介面：集中合法動作判定、mask 生成與狀態編碼。"""

	def __init__(self, game: Optional[TzaarGame] = None) -> None:
		self.game = game or TzaarGame()
		self._valid_positions: Tuple[Tuple[int, int], ...] = self.game.board.valid_positions()
		self._pos_to_idx = self.game.board.pos_to_index_map()

	# -----------------------------------------------------------------------
	# 第一步（必吃）
	# -----------------------------------------------------------------------

	def can_second_step_pass(self) -> bool:
		"""第二步是否可選擇 PASS。"""
		return self.game.can_second_step_pass()

	def has_legal_second_capture(self) -> bool:
		"""第二步是否至少存在一手合法吃子。"""
		return _has_legal_move_from_board(self.game.board, self.game.current_player, MoveKind.CAPTURE)

	def has_legal_second_reinforce(self) -> bool:
		"""第二步是否至少存在一手合法疊子。"""
		return _has_legal_move_from_board(self.game.board, self.game.current_player, MoveKind.REINFORCE)

	def resolve_no_mandatory_capture_if_needed(self) -> bool:
		"""若目前輪到玩家第一步無法吃子，直接判負。

		這個檢查屬於「第一步開始前」檢查，避免進入無解局面。
		"""
		if self.game.is_game_over() or self.can_second_step_pass():
			return False
		if _has_legal_move_from_board(self.game.board, self.game.current_player, MoveKind.CAPTURE):
			return False
		self.game.result = GameResult(
			winner=-self.game.current_player,
			reason=WinReason.NO_MANDATORY_CAPTURE,
		)
		return True

	def capture_source_mask(self) -> torch.Tensor:
		"""第一步（必吃）來源 mask，固定 60 維。

		一步到位：直接掃棋盤產生 1D mask，不先建其他中介結構。
		"""
		return _source_mask_from_board(
			self.game.board,
			self.game.current_player,
			MoveKind.CAPTURE,
			self._pos_to_idx,
		)

	def capture_direction_mask(self, src: Tuple[int, int]) -> torch.Tensor:
		"""第一步固定來源後的方向 mask，固定 6 維。"""
		return _direction_mask_from_board(
			self.game.board,
			src,
			self.game.current_player,
			MoveKind.CAPTURE,
		)

	def first_step_reconstruct_move(self, src: Tuple[int, int], direction_idx: int) -> Move:
		"""把第一步選擇 (src, direction_idx) 還原成 Move。"""
		return _reconstruct_non_pass_move(
			game=self.game,
			kind=MoveKind.CAPTURE,
			src=src,
			direction_idx=direction_idx,
		)

	# -----------------------------------------------------------------------
	# 第二步（capture / reinforce / pass）
	# -----------------------------------------------------------------------

	def second_step_kind_mask(self) -> torch.Tensor:
		"""回傳第二步 kind mask：[capture, reinforce, pass]。
		第二步必須在第一步套用後的盤面上計算，確保和真實規則同步。
		"""
		mask = torch.zeros(N_KINDS, dtype=torch.bool)
		if self.has_legal_second_capture():
			mask[KIND_TO_IDX[MoveKind.CAPTURE]] = True
		if self.has_legal_second_reinforce():
			mask[KIND_TO_IDX[MoveKind.REINFORCE]] = True
		if self.can_second_step_pass():
			mask[KIND_TO_IDX[MoveKind.PASS]] = True
		return mask

	def reinforce_source_mask(self) -> torch.Tensor:
		"""第二步選 reinforce 時的 source mask（60 維）。"""
		return _source_mask_from_board(
			self.game.board,
			self.game.current_player,
			MoveKind.REINFORCE,
			self._pos_to_idx,
		)

	def reinforce_direction_mask(self, src: Tuple[int, int]) -> torch.Tensor:
		"""第二步選 reinforce 且固定 src 後的 direction mask（6 維）。"""
		return _direction_mask_from_board(
			self.game.board,
			src,
			self.game.current_player,
			MoveKind.REINFORCE,
		)

	def second_step_reconstruct_move(
		self,
		kind: MoveKind,
		src: Optional[Tuple[int, int]] = None,
		direction_idx: Optional[int] = None,
	) -> Move:
		"""把第二步選擇結果還原成 Move。"""

		if kind == MoveKind.PASS:
			if not self.can_second_step_pass():
				raise ValueError("Hierarchical decision does not produce a legal second-step move")
			return Move(kind=MoveKind.PASS)

		if src is None or direction_idx is None:
			raise ValueError("src and direction_idx are required for non-pass moves")

		if kind not in (MoveKind.CAPTURE, MoveKind.REINFORCE):
			raise ValueError("Unsupported move kind for second-step reconstruction")

		return _reconstruct_non_pass_move(
			game=self.game,
			kind=kind,
			src=src,
			direction_idx=direction_idx,
		)

	# -----------------------------------------------------------------------
	# 統一 601 動作空間（300 capture + 300 reinforce + 1 pass）
	# -----------------------------------------------------------------------

	def unified_action_mask_step1(self) -> torch.Tensor:
		"""第一步 mask：只允許 capture 子空間 [0, 299]。"""
		mask = torch.zeros(N_ACTIONS, dtype=torch.bool)
		board = self.game.board
		player = self.game.current_player
		for src in board.iter_player_positions(player):
			src_idx = self._pos_to_idx[src]
			src_height = board.height(src)
			for direction_idx, direction in enumerate(DIRECTIONS):
				dst = board.first_occupied_in_direction(src, direction)
				if dst is None:
					continue
				if piece_owner(board.top_piece(dst)) == player:
					continue
				if board.height(dst) > src_height:
					continue
				edge_idx = EDGE_TO_IDX.get((src_idx, direction_idx))
				if edge_idx is not None:
					mask[encode_action_idx(MoveKind.CAPTURE, edge_idx)] = True
		return mask

	def unified_action_mask_step2(self) -> torch.Tensor:
		"""第二步 mask：允許 capture/reinforce/pass。"""
		mask = torch.zeros(N_ACTIONS, dtype=torch.bool)
		player = self.game.current_player

		for src in self.game.board.iter_player_positions(player):
			src_idx = self._pos_to_idx[src]
			src_height = self.game.board.height(src)
			for direction_idx, direction in enumerate(DIRECTIONS):
				dst = self.game.board.first_occupied_in_direction(src, direction)
				if dst is None:
					continue
				edge_idx = EDGE_TO_IDX.get((src_idx, direction_idx))
				if edge_idx is None:
					continue
				dst_is_self = piece_owner(self.game.board.top_piece(dst)) == player
				if dst_is_self:
					mask[encode_action_idx(MoveKind.REINFORCE, edge_idx)] = True
				elif self.game.board.height(dst) <= src_height:
					mask[encode_action_idx(MoveKind.CAPTURE, edge_idx)] = True

		if self.can_second_step_pass():
			mask[PASS_ACTION_IDX] = True
		return mask

	def reconstruct_unified_action(self, action_idx: int, phase: str) -> Move:
		"""將 0..600 的統一索引還原成合法 Move。"""
		kind, _, src_idx, direction_idx = decode_action_idx(action_idx)
		if phase == PHASE_STEP_1 and kind != MoveKind.CAPTURE:
			raise ValueError("step1 only allows capture actions")
		if kind == MoveKind.PASS:
			return Move(kind=MoveKind.PASS)
		if src_idx is None or direction_idx is None:
			raise ValueError("invalid unified action index")
		src = self._valid_positions[src_idx]
		return _reconstruct_non_pass_move(self.game, kind, src, direction_idx)

	def encode_unified_action_from_move(self, move: Move) -> int:
		"""將 Move 映射到統一動作索引。"""
		if move.kind == MoveKind.PASS:
			return PASS_ACTION_IDX
		if move.src is None or move.dst is None:
			raise ValueError("non-pass move requires src/dst")
		src_idx = self._pos_to_idx[move.src]
		direction_idx = -1
		for i, direction in enumerate(DIRECTIONS):
			hit = self.game.board.first_occupied_in_direction(move.src, direction)
			if hit == move.dst:
				direction_idx = i
				break
		if direction_idx < 0:
			raise ValueError("cannot reconstruct direction from move")
		edge_idx = EDGE_TO_IDX.get((src_idx, direction_idx))
		if edge_idx is None:
			raise ValueError("(src, direction) not present in fixed edge table")
		return encode_action_idx(move.kind, edge_idx)

	# -----------------------------------------------------------------------
	# 狀態向量增量更新 helper（用於加速，避免每次都完整掃盤）
	# -----------------------------------------------------------------------

	def update_state_turn_player(self, vec: torch.Tensor) -> None:
		"""在 live buffer 上刷新 turn_number 和 current_player。"""
		vec[STATE_TURN_PLAYER_BASE] = float(self.game.turn_number)
		vec[STATE_TURN_PLAYER_BASE + 1] = float(self.game.current_player)

	def update_state_phase(self, vec: torch.Tensor, phase: str) -> None:
		"""在 live buffer 上刷新 phase one-hot。"""
		if phase not in TRAINING_PHASE_TO_IDX:
			raise ValueError(f"Unknown training phase: {phase}")
		# 先清除舊 phase
		for i in range(len(TRAINING_PHASE_ORDER)):
			vec[STATE_PHASE_BASE + i] = 0.0
		# 設置新 phase
		vec[STATE_PHASE_BASE + TRAINING_PHASE_TO_IDX[phase]] = 1.0

	def update_state_board_position(self, vec: torch.Tensor, pos: Tuple[int, int], pos_idx: int) -> None:
		"""在 live buffer 上更新單一棋盤位置的顏色、兵種、高度。
		
		參數
		----
		vec : 489 維 state buffer
		pos : 棋盤位置 (row, col)
		pos_idx : 該位置在 valid_positions 中的索引
		"""
		cell = self.game.board.get_cell(pos)
		
		# 清除舊值
		color_idx = STATE_COLOR_BASE + pos_idx * 3
		type_idx = STATE_TYPE_BASE + pos_idx * 4
		height_idx = STATE_HEIGHT_BASE + pos_idx
		
		for i in range(3):
			vec[color_idx + i] = 0.0
		for i in range(4):
			vec[type_idx + i] = 0.0
		vec[height_idx] = 0.0
		
		# 重新填充
		if cell is None or cell[1] == 0:
			# 空格
			vec[color_idx + 2] = 1.0
			vec[type_idx + 3] = 1.0
		else:
			# 有棋子
			top = cell[0]
			owner = piece_owner(top)
			if owner == BLACK:
				vec[color_idx] = 1.0
			else:
				vec[color_idx + 1] = 1.0
			
			ptype = piece_type(top)
			vec[type_idx + (ptype - 1)] = 1.0
			vec[height_idx] = float(cell[1])

	# -----------------------------------------------------------------------
	# 訓練狀態編碼
	# -----------------------------------------------------------------------


	def encode_training_state(
		self,
		phase: str,
		device: Optional[str] = None,
		out: Optional["torch.Tensor"] = None,
	) -> "torch.Tensor":
		"""輸出固定 489 維訓練狀態向量，支援覆寫既有 tensor buffer。"""
		if phase not in TRAINING_PHASE_TO_IDX:
			raise ValueError(f"Unknown training phase: {phase}")

		positions = self._valid_positions
		if len(positions) != N_POSITIONS:
			raise ValueError(f"Expected {N_POSITIONS} valid positions, got {len(positions)}")

		if out is None:
			vec = torch.zeros(TRAINING_STATE_DIM, dtype=torch.float32, device=device)
		else:
			if out.shape != (TRAINING_STATE_DIM,):
				raise ValueError(
					f"Expected out tensor shape ({TRAINING_STATE_DIM},), got {tuple(out.shape)}"
				)
			if out.dtype != torch.float32:
				raise ValueError(f"Expected out tensor dtype torch.float32, got {out.dtype}")
			if device is not None:
				requested = torch.device(device)
				if requested.type != out.device.type:
					raise ValueError(f"Expected out tensor on device {device}, got {out.device}")
				if requested.index is not None and requested.index != out.device.index:
					raise ValueError(f"Expected out tensor on device {device}, got {out.device}")
			vec = out
			vec.zero_()
		base = 2
		color_base = base
		type_base = color_base + N_POSITIONS * 3
		height_base = type_base + N_POSITIONS * 4
		phase_base = height_base + N_POSITIONS

		# 1) 回合數與當前玩家（2 維）
		vec[0] = float(self.game.turn_number)
		vec[1] = float(self.game.current_player)

		# 2)~4) 單次走訪每格，填入顏色/兵種/高度
		for idx, pos in enumerate(positions):
			color_idx = color_base + idx * 3
			type_idx = type_base + idx * 4
			height_idx = height_base + idx

			cell = self.game.board.get_cell(pos)
			if cell is None or cell[1] == 0:
				vec[color_idx + 2] = 1.0
				vec[type_idx + 3] = 1.0
				continue

			top = cell[0]
			owner = piece_owner(top)
			if owner == BLACK:
				vec[color_idx] = 1.0
			else:
				vec[color_idx + 1] = 1.0

			ptype = piece_type(top)
			vec[type_idx + (ptype - 1)] = 1.0
			vec[height_idx] = float(cell[1])

		# 5) 目前選擇階段 one-hot（7）
		vec[phase_base + TRAINING_PHASE_TO_IDX[phase]] = 1.0

		if phase_base + len(TRAINING_PHASE_ORDER) != TRAINING_STATE_DIM:
			raise ValueError(
				f"Training state length mismatch: expected {TRAINING_STATE_DIM}, "
				f"got {phase_base + len(TRAINING_PHASE_ORDER)}"
			)
		return vec


