from __future__ import annotations

import math
import queue
import random
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import TzaarTrain as train_module
from Tzaar import BLACK, WHITE, Move, MoveKind, TzaarGame, piece_owner
from TzaarAI import (
	N_ACTIONS,
	PASS_ACTION_IDX,
	PHASE_STEP_1,
	PHASE_STEP_2,
	TRAINING_PHASE_ORDER,
	TRAINING_PHASE_TO_IDX,
	TzaarAIInterface,
)
from policy_net_cnn_min17 import PolicyNetCNNMin17
from strategy_v2_unified_python import StrategyV2UnifiedPython
from tzaar_state_backend import create_cpp_phase_state, resolve_backend_name, try_load_cpp_backend


# ============================================================
# Training constants (tune these)
# ============================================================
#python update_cnn_mcts.py
TITLE = "mcts_cnn"
GUARD_TITLE = f"{TITLE}_guard"
LOG_DIR = "logs"
SEED = 42
DEVICE = "auto"  # auto / cpu / cuda / cuda:N
INFERENCE_DEVICE = "cuda:0"  # MCTS inference device (batch=1，CPU 通常比 GPU 快)

# State backend migration toggle.
# Keep default as python until C++ parity checks are complete.
STATE_BACKEND = "cpp"  # python / cpp
CPP_BACKEND_REQUIRED = True

# Resume behavior
REQUIRE_RESUME = False
RESUME_ANY_TITLE = True
RESUME_TITLE = TITLE

# Main loop
TOTAL_UPDATES = 500
GAMES_PER_UPDATE = 150
# Ratio of pure model self-play games in each update (0.0~1.0).
# Remaining games use model-vs-strategy.
# Set to 1.0 to temporarily disable model-vs-strategy games.
SELFPLAY_MODEL_GAME_RATIO = 1.0
TRAIN_EPOCHS_PER_UPDATE = 10
OPTIMIZATION_PASSES_PER_UPDATE = 1
BATCH_SIZE = 128
CHECKPOINT_EVERY_UPDATES = 20
GATE_EVERY_UPDATES = 2  # run gate evaluation every N updates (1 = every update)
LOG_EVERY = 10
# Self-play progress log interval in games.
# <= 0 disables per-game progress logs (only update-level summary remains).
SELFPLAY_PROGRESS_LOG_INTERVAL = 0

# Gatekeeper
GATE_EVAL_GAMES = 150
GATE_WINRATE_THRESHOLD = 0.55
GATE_TEMPERATURE = 0.1  # sampling temperature for gate evaluation (1.0=visit counts, 0.1=lower entropy, 1e-8=greedy)
GATE_SIMULATIONS_PER_DECISION = 600  # MCTS simulations per decision during gate evaluation (lower than self-play)
KEEP_OPTIMIZER_ON_REJECT = False
KEEP_REPLAY_ON_REJECT = True

# Optimizer / loss
LEARNING_RATE_START = 0.0007
LEARNING_RATE_END = 0.0007
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.0
ENTROPY_WEIGHT = 0.0

# CNN architecture
CNN_GLOBAL_FEATURE_DIM = 12
CNN_DROPOUT = 0.3

# MCTS
SIMULATIONS_PER_DECISION = 600
PUCT_C = 1.5
MCTS_LEAF_BATCH_SIZE = 16

# Async MCTS pipeline (multi-session CPU workers + shared GPU inference worker)
ENABLE_ASYNC_MCTS = True
ASYNC_PARALLEL_GAMES = 15
ASYNC_INFER_MAX_BATCH = 480
ASYNC_INFER_MAX_WAIT_MS = 2.0
ASYNC_REQUEST_QUEUE_SIZE = 30
ASYNC_RESPONSE_TIMEOUT_S = 30.0

# Heuristic + policy prior blend for PUCT
HEURISTIC_SOFTMAX_TEMPERATURE = 1.0
HEURISTIC_PRIOR_WEIGHT = 0.0
MATERIAL_BASE_TZAAR = 100.0
MATERIAL_BASE_TZARRA = 40.0
MATERIAL_BASE_TOTT = 15.0
MOBILITY_CAPTURE_WEIGHT = 10.0
SCARCITY_BONUS_TWO_OR_LESS = 200.0
SCARCITY_BONUS_ONE_OR_LESS = 500.0

# Root Dirichlet noise (enabled during self-play training)
USE_ROOT_DIRICHLET_NOISE = True
ROOT_DIRICHLET_EPS = 0.25
ROOT_DIRICHLET_ALPHA_ACTION = 0.05

# Temperature schedule for action sampling from root visit counts
TEMP_HIGH = 1.0
TEMP_LOW = 0.1
TEMP_SWITCH_DECISION = 6

# Replay buffer
USE_REPLAY_BUFFER = True
REPLAY_BUFFER_MAX_SAMPLES = 50000
REPLAY_TRAIN_SAMPLES_PER_UPDATE = 8192
REPLAY_MIN_TRAIN_SAMPLES = 256
REPLAY_SNAPSHOT_VERSION = 1
REPLAY_SNAPSHOT_TAG = "latest"


MAX_HEIGHT_NORM = 8.0
HEAD_ACTION = "action"

_ACTIVE_STATE_BACKEND = "cpp"
_ACTIVE_CPP_MODULE: Optional[Any] = None
_snapshot_save_thread: Optional[threading.Thread] = None

@dataclass
class InferenceRequest:
	session_id: int
	batch_id: int
	node_ids: np.ndarray
	board_state_flat: np.ndarray
	global_features: np.ndarray
	legal_masks: np.ndarray
	model_id: int = 0  # 0 = primary policy (candidate), 1 = secondary policy (guard)


@dataclass
class InferenceResponse:
	session_id: int
	batch_id: int
	node_ids: np.ndarray
	priors: np.ndarray
	values: np.ndarray
	error: Optional[Exception] = None


# _resolve_state_backend: function implementation.
def _resolve_state_backend() -> Tuple[str, Optional[object]]:
	backend_name = resolve_backend_name(default_name=STATE_BACKEND)
	if backend_name != "cpp":
		return "python", None

	cpp_module = try_load_cpp_backend()
	if cpp_module is None:
		if CPP_BACKEND_REQUIRED:
			raise RuntimeError("STATE_BACKEND=cpp requested but tzaar_cpp module is not available")
		print("[backend] cpp requested but module not found; falling back to python")
		return "python", None

	print("[backend] cpp module loaded (foundation mode)")
	return "cpp", cpp_module


# _make_phase_state: function implementation.
def _make_phase_state() -> Any:
	if _ACTIVE_STATE_BACKEND == "cpp":
		if _ACTIVE_CPP_MODULE is None:
			raise RuntimeError("cpp backend is active but module handle is missing")
		return create_cpp_phase_state(_ACTIVE_CPP_MODULE)
	return PhaseGameState()


class Stage(Enum):
	NEED_STEP1 = auto()
	NEED_STEP2 = auto()
	DONE = auto()


STAGE_TO_PHASE: Dict[Stage, str] = {
	Stage.NEED_STEP1: PHASE_STEP_1,
	Stage.NEED_STEP2: PHASE_STEP_2,
}


@dataclass
class PolicySample:
	state: torch.Tensor
	global_features: torch.Tensor
	action_dim: int
	legal_mask_padded: torch.Tensor
	target_pi_padded: torch.Tensor
	player: int
	winner_sign: int = 0
	value_target: float = 0.0


@dataclass
class MCTSNode:
	prior: float
	to_play: Optional[int]
	visit_count: int = 0
	value_sum: float = 0.0
	expanded: bool = False
	action_dim: int = 0
	legal_mask: Optional[torch.Tensor] = None
	state: Optional["PhaseGameState"] = None
	children: Dict[int, "MCTSNode"] = None

	# __post_init__: function implementation.
	def __post_init__(self) -> None:
		if self.children is None:
			self.children = {}

	# mean_value: function implementation.
	def mean_value(self) -> float:
		if self.visit_count <= 0:
			return 0.0
		return self.value_sum / float(self.visit_count)


class PhaseGameState:
	"""A lightweight stage machine used by MCTS and self-play."""

	# __init__: function implementation.
	def __init__(self, game: Optional[TzaarGame] = None, stage: Stage = Stage.NEED_STEP1) -> None:
		self.game = game or TzaarGame()
		self.ai = TzaarAIInterface(self.game)
		self.stage = stage
		self._cached_mask_stage: Optional[Stage] = None
		self._cached_mask: Optional[torch.Tensor] = None

	# clone: function implementation.
	def clone(self) -> "PhaseGameState":
		copied = PhaseGameState(self.game.clone(), self.stage)
		copied._cached_mask_stage = self._cached_mask_stage
		copied._cached_mask = None if self._cached_mask is None else self._cached_mask.clone()
		return copied

	# _invalidate_cache: function implementation.
	def _invalidate_cache(self) -> None:
		self._cached_mask_stage = None
		self._cached_mask = None

	# _advance_from_first_src: function implementation.
	def _advance_from_first_src(self) -> None:
		"""After any transition to NEED_STEP1, eagerly resolve the
		no-mandatory-capture check so is_done() is accurate before anyone
		observes the state."""
		if self.stage == Stage.NEED_STEP1:
			if self.ai.resolve_no_mandatory_capture_if_needed():
				self.stage = Stage.DONE

	# is_done: function implementation.
	def is_done(self) -> bool:
		return self.stage == Stage.DONE or self.game.is_game_over()

	# winner: function implementation.
	def winner(self) -> Optional[int]:
		return self.game.result.winner if self.game.result else None

	# current_player: function implementation.
	def current_player(self) -> int:
		return int(self.game.current_player)

	# phase: function implementation.
	def phase(self) -> Optional[str]:
		if self.is_done():
			return None
		return STAGE_TO_PHASE[self.stage]

	# legal_mask: function implementation.
	def legal_mask(self) -> Optional[torch.Tensor]:
		if self.is_done():
			return None
		if self._cached_mask is not None and self._cached_mask_stage == self.stage:
			return self._cached_mask

		if self.stage == Stage.NEED_STEP1:
			if self.ai.resolve_no_mandatory_capture_if_needed():
				self.stage = Stage.DONE
				self._invalidate_cache()
				return None
			mask = self.ai.unified_action_mask_step1()
			self._cached_mask_stage = self.stage
			self._cached_mask = mask
			return mask

		if self.stage == Stage.NEED_STEP2:
			mask = self.ai.unified_action_mask_step2()
			self._cached_mask_stage = self.stage
			self._cached_mask = mask
			return mask

		raise ValueError(f"Unsupported stage: {self.stage}")

	# apply_action: function implementation.
	def apply_action(self, action_idx: int, legal_mask: Optional[torch.Tensor] = None) -> None:
		if self.is_done():
			return
		mask = legal_mask if legal_mask is not None else self.legal_mask()
		if mask is None:
			return
		if not (0 <= action_idx < int(mask.shape[0])):
			raise ValueError("action index out of range")
		if not bool(mask[action_idx].item()):
			raise ValueError("illegal action")
		self._invalidate_cache()

		if self.stage == Stage.NEED_STEP1:
			move = self.ai.reconstruct_unified_action(action_idx, PHASE_STEP_1)
			self.game.play_first_step(move)
			if self.game.is_game_over():
				self.stage = Stage.DONE
			elif self.game.is_waiting_second_step():
				self.stage = Stage.NEED_STEP2
			else:
				self.stage = Stage.NEED_STEP1
				self._advance_from_first_src()
			return

		if self.stage == Stage.NEED_STEP2:
			if action_idx == PASS_ACTION_IDX:
				self.game.play_second_step()
				self.stage = Stage.DONE if self.game.is_game_over() else Stage.NEED_STEP1
				self._advance_from_first_src()
			else:
				move = self.ai.reconstruct_unified_action(action_idx, PHASE_STEP_2)
				self.game.play_second_step(move)
				self.stage = Stage.DONE if self.game.is_game_over() else Stage.NEED_STEP1
			self._advance_from_first_src()
			return

		raise ValueError(f"Unsupported stage for action application: {self.stage}")


# _piece_type_from_code: function implementation.
def _piece_type_from_code(piece_code: int) -> int:
	if piece_code in (1, 4):
		return 1
	if piece_code in (2, 5):
		return 2
	if piece_code in (3, 6):
		return 3
	raise ValueError(f"Unknown piece code: {piece_code}")


# _build_cnn_state_12x9x9: function implementation.
def _build_cnn_state_12x9x9(game: TzaarGame, player: int) -> torch.Tensor:
	import numpy as _np
	p = game.board._piece_np   # (9,9) int8: piece_code, 0=empty/hole
	h = game.board._height_np  # (9,9) float32: raw height, 0=empty/hole

	h_norm = _np.minimum(h, MAX_HEIGHT_NORM) * (1.0 / MAX_HEIGHT_NORM)

	# Piece codes: 1,2,3 = WHITE tzaar/tzarra/tott; 4,5,6 = BLACK tzaar/tzarra/tott
	if player == WHITE:
		own1 = p == 1; own2 = p == 2; own3 = p == 3
		opp1 = p == 4; opp2 = p == 5; opp3 = p == 6
	else:
		own1 = p == 4; own2 = p == 5; own3 = p == 6
		opp1 = p == 1; opp2 = p == 2; opp3 = p == 3

	own_mask = own1 | own2 | own3
	opp_mask = opp1 | opp2 | opp3

	# channels 0-2: own occupation by piece type; 3-5: opp occupation by piece type
	# channels 6-8: own height by piece type;     9-11: opp height by piece type
	state = _np.stack([
		own1.astype(_np.float32),
		own2.astype(_np.float32),
		own3.astype(_np.float32),
		opp1.astype(_np.float32),
		opp2.astype(_np.float32),
		opp3.astype(_np.float32),
		h_norm * own1,
		h_norm * own2,
		h_norm * own3,
		h_norm * opp1,
		h_norm * opp2,
		h_norm * opp3,
	], axis=0)  # (12,9,9)

	return torch.from_numpy(state)



# _build_cnn_global_features: function implementation.
def _build_cnn_global_features(game: TzaarGame, player: int, phase: str) -> torch.Tensor:
	phase_one_hot = [0.0] * len(TRAINING_PHASE_ORDER)
	phase_one_hot[TRAINING_PHASE_TO_IDX[phase]] = 1.0

	opp = WHITE if player == BLACK else BLACK
	own_counts = game._counts[player]
	opp_counts = game._counts[opp]

	own_t1 = float(own_counts[1])
	own_t2 = float(own_counts[2])
	own_t3 = float(own_counts[3])
	opp_t1 = float(opp_counts[1])
	opp_t2 = float(opp_counts[2])
	opp_t3 = float(opp_counts[3])

	own_total = own_t1 + own_t2 + own_t3
	opp_total = opp_t1 + opp_t2 + opp_t3

	turn_norm = min(float(game.turn_number), 200.0) / 200.0
	player_sign = 1.0 if player == WHITE else -1.0

	vec = [
		turn_norm,
		player_sign,
		own_t1 / 15.0,
		own_t2 / 15.0,
		own_t3 / 15.0,
		opp_t1 / 15.0,
		opp_t2 / 15.0,
		opp_t3 / 15.0,
		own_total / 45.0,
		opp_total / 45.0,
		*phase_one_hot,
	]
	return torch.tensor(vec, dtype=torch.float32)



# _terminal_value_for_current_player: function implementation.
def _terminal_value_for_current_player(winner: Optional[int], current_player: int) -> float:
	if winner is None:
		return 0.0
	return 1.0 if winner == current_player else -1.0


# _material_base_for_piece_type: function implementation.
def _material_base_for_piece_type(piece_type: int) -> float:
	if piece_type == 1:
		return MATERIAL_BASE_TZAAR
	if piece_type == 2:
		return MATERIAL_BASE_TZARRA
	if piece_type == 3:
		return MATERIAL_BASE_TOTT
	raise ValueError(f"Unknown piece type: {piece_type}")


# _material_score: function implementation.
def _material_score(game: TzaarGame, player: int) -> float:
	score = 0.0
	for row, col in game.board.valid_positions():
		pos = (row, col)
		if game.board.is_empty(pos):
			continue
		top = game.board.top_piece(pos)
		if piece_owner(top) != player:
			continue
		piece_type = _piece_type_from_code(top)
		height = float(game.board.height(pos))
		score += _material_base_for_piece_type(piece_type) * height
	return score


# _mobility_capture_score: function implementation.
def _mobility_capture_score(game: TzaarGame, player: int) -> float:
	return float(len(game._capture_pairs_for_player(player))) * MOBILITY_CAPTURE_WEIGHT


# _scarcity_bonus: function implementation.
def _scarcity_bonus(game: TzaarGame, player: int) -> float:
	counts = game._counts[player]
	bonus = 0.0
	for piece_type in (1, 2, 3):
		remaining = int(counts[piece_type])
		if remaining <= 1:
			bonus += SCARCITY_BONUS_ONE_OR_LESS
		elif remaining <= 2:
			bonus += SCARCITY_BONUS_TWO_OR_LESS
	return bonus


# _player_eval_score: function implementation.
def _player_eval_score(game: TzaarGame, player: int) -> float:
	return _material_score(game, player) + _mobility_capture_score(game, player) + _scarcity_bonus(game, player)


# _heuristic_action_value: function implementation.
def _heuristic_action_value(after_state: PhaseGameState, perspective_player: int) -> float:
	opp = WHITE if perspective_player == BLACK else BLACK
	self_v = _player_eval_score(after_state.game, perspective_player)
	opp_v = _player_eval_score(after_state.game, opp)
	return self_v - opp_v


# _heuristic_priors_for_actions: function implementation.
def _heuristic_priors_for_actions(
	node: MCTSNode,
	legal_idxs: List[int],
	action_dim: int,
) -> torch.Tensor:
	priors = torch.zeros(action_dim, dtype=torch.float32)
	if not legal_idxs:
		return priors
	if node.state is None:
		uniform = 1.0 / float(len(legal_idxs))
		for a in legal_idxs:
			priors[a] = uniform
		return priors

	perspective_player = node.state.current_player()
	scores: List[float] = []
	for action in legal_idxs:
		child_state = node.state.clone()
		child_state.apply_action(action, legal_mask=node.legal_mask)
		scores.append(_heuristic_action_value(child_state, perspective_player))

	logits = torch.tensor(scores, dtype=torch.float32)
	logits = logits / float(HEURISTIC_SOFTMAX_TEMPERATURE)
	logits = logits - logits.max()
	exps = torch.exp(logits)
	sum_exps = float(exps.sum().item())
	if not math.isfinite(sum_exps) or sum_exps <= 0.0:
		uniform = 1.0 / float(len(legal_idxs))
		for a in legal_idxs:
			priors[a] = uniform
		return priors

	probs = exps / sum_exps
	if not torch.isfinite(probs).all():
		uniform = 1.0 / float(len(legal_idxs))
		for a in legal_idxs:
			priors[a] = uniform
		return priors

	for i, action in enumerate(legal_idxs):
		priors[action] = probs[i]
	return priors


# _blend_policy_and_heuristic_priors: function implementation.
def _blend_policy_and_heuristic_priors(
	policy_priors: torch.Tensor,
	heuristic_priors: torch.Tensor,
	legal_idxs: List[int],
	action_dim: int,
) -> torch.Tensor:
	mixed = torch.zeros(action_dim, dtype=torch.float32)
	if not legal_idxs:
		return mixed

	legal_tensor = torch.tensor(legal_idxs, dtype=torch.long)
	policy_legal = policy_priors[:action_dim].to(device="cpu", dtype=torch.float32)[legal_tensor]
	heuristic_legal = heuristic_priors[:action_dim].to(dtype=torch.float32)[legal_tensor]
	mixed_legal = (1.0 - HEURISTIC_PRIOR_WEIGHT) * policy_legal + HEURISTIC_PRIOR_WEIGHT * heuristic_legal

	sum_mixed = float(mixed_legal.sum().item())
	if not math.isfinite(sum_mixed) or sum_mixed <= 0.0:
		sum_heur = float(heuristic_legal.sum().item())
		if math.isfinite(sum_heur) and sum_heur > 0.0:
			mixed_legal = heuristic_legal / sum_heur
		else:
			uniform = 1.0 / float(len(legal_idxs))
			mixed_legal = torch.full((len(legal_idxs),), uniform, dtype=torch.float32)
	else:
		mixed_legal = mixed_legal / sum_mixed

	mixed[legal_tensor] = mixed_legal
	return mixed


# _dirichlet_alpha_for_head: function implementation.
def _dirichlet_alpha_for_head(head: str) -> float:
	if head == HEAD_ACTION:
		return ROOT_DIRICHLET_ALPHA_ACTION
	raise ValueError(f"Unknown head: {head}")


# _temperature_for_decision: function implementation.
def _temperature_for_decision(decision_idx: int) -> float:
	if decision_idx < TEMP_SWITCH_DECISION:
		return TEMP_HIGH
	return TEMP_LOW


# _legal_action_dim_and_head: function implementation.
def _legal_action_dim_and_head(state: PhaseGameState) -> Tuple[str, int, torch.Tensor]:
	head = HEAD_ACTION
	action_dim = N_ACTIONS
	mask = state.legal_mask()
	if mask is None:
		raise ValueError("legal mask is undefined for terminal state")
	return head, action_dim, mask


# _evaluate_leaf: function implementation.
def _evaluate_leaf(
	policy: PolicyNetCNNMin17,
	state: PhaseGameState,
	device: torch.device,
) -> Tuple[str, int, torch.Tensor, torch.Tensor, float]:
	phase = state.phase()
	if phase is None:
		raise ValueError("Cannot evaluate terminal leaf")

	head, action_dim, mask = _legal_action_dim_and_head(state)

	board = _build_cnn_state_12x9x9(state.game, state.current_player()).unsqueeze(0).to(device)
	global_features = _build_cnn_global_features(state.game, state.current_player(), phase).unsqueeze(0).to(device)

	with torch.no_grad():
		hidden = policy.encode(board, global_features)
		logits = policy.head_logits(hidden, HEAD_ACTION)[0]
		value = float(policy.forward_value(hidden).squeeze().item())

	mask = mask.to(device=device)
	masked_logits = logits[:action_dim].masked_fill(~mask[:action_dim], -1e9)
	priors = torch.softmax(masked_logits, dim=-1)
	return head, action_dim, mask, priors, value


# _evaluate_leaf_batch: function implementation.
def _evaluate_leaf_batch(
	policy: PolicyNetCNNMin17,
	states: List[PhaseGameState],
	device: torch.device,
) -> List[Tuple[str, int, torch.Tensor, torch.Tensor, float]]:
	if not states:
		return []

	phases: List[str] = []
	action_dims: List[int] = []
	masks: List[torch.Tensor] = []
	boards: List[torch.Tensor] = []
	globals_: List[torch.Tensor] = []

	for state in states:
		phase = state.phase()
		if phase is None:
			raise ValueError("Cannot evaluate terminal leaf")

		head, action_dim, mask = _legal_action_dim_and_head(state)
		phases.append(phase)
		action_dims.append(action_dim)
		masks.append(mask)
		boards.append(_build_cnn_state_12x9x9(state.game, state.current_player()))
		globals_.append(_build_cnn_global_features(state.game, state.current_player(), phase))

	board_batch = torch.stack(boards).to(device)
	global_batch = torch.stack(globals_).to(device)

	with torch.no_grad():
		hidden = policy.encode(board_batch, global_batch)
		action_logits = policy.action_head(hidden)
		values = policy.forward_value(hidden).squeeze(-1)

	results: List[Tuple[str, int, torch.Tensor, torch.Tensor, float]] = []
	for i in range(len(states)):
		head = HEAD_ACTION
		action_dim = action_dims[i]
		mask = masks[i].to(device=device)
		logits = action_logits[i]

		masked_logits = logits[:action_dim].masked_fill(~mask[:action_dim], -1e9)
		priors = torch.softmax(masked_logits, dim=-1)
		results.append((head, action_dim, mask, priors, float(values[i].item())))

	return results


# _expand_node: function implementation.
def _expand_node(
	node: MCTSNode,
	policy: PolicyNetCNNMin17,
	state: PhaseGameState,
	device: torch.device,
) -> Tuple[str, int, torch.Tensor, float]:
	head, action_dim, mask, policy_priors, value = _evaluate_leaf(policy, state, device)

	node.expanded = True
	node.action_dim = action_dim
	node.legal_mask = mask.to(device="cpu")
	legal_idxs = torch.nonzero(mask[:action_dim], as_tuple=False).flatten().tolist()
	mixed_priors = policy_priors[:action_dim].to(device="cpu", dtype=torch.float32)
	for a in legal_idxs:
		node.children[a] = MCTSNode(prior=float(mixed_priors[a].item()), to_play=None)

	return head, action_dim, mask, value


# _expand_nodes_from_eval: function implementation.
def _expand_nodes_from_eval(
	nodes: List[MCTSNode],
	evals: List[Tuple[str, int, torch.Tensor, torch.Tensor, float]],
) -> List[Tuple[str, int, torch.Tensor, float]]:
	if len(nodes) != len(evals):
		raise ValueError("nodes and evals length mismatch")

	out: List[Tuple[str, int, torch.Tensor, float]] = []
	for node, (head, action_dim, mask, policy_priors, value) in zip(nodes, evals):
		node.expanded = True
		node.action_dim = action_dim
		node.legal_mask = mask.to(device="cpu")
		node.children = {}
		legal_idxs = torch.nonzero(mask[:action_dim], as_tuple=False).flatten().tolist()
		mixed_priors = policy_priors[:action_dim].to(device="cpu", dtype=torch.float32)
		for a in legal_idxs:
			node.children[a] = MCTSNode(prior=float(mixed_priors[a].item()), to_play=None)
		out.append((head, action_dim, mask, value))
	return out


# _apply_root_dirichlet_noise: function implementation.
def _apply_root_dirichlet_noise(root: MCTSNode, head: str) -> None:
	if not USE_ROOT_DIRICHLET_NOISE:
		return
	if not root.children:
		return

	legal_actions = sorted(root.children.keys())
	alpha = _dirichlet_alpha_for_head(head)
	noise = np.random.dirichlet([alpha] * len(legal_actions))
	eps = ROOT_DIRICHLET_EPS

	for i, action in enumerate(legal_actions):
		child = root.children[action]
		child.prior = (1.0 - eps) * child.prior + eps * float(noise[i])


# _select_child_action: function implementation.
def _select_child_action(node: MCTSNode) -> int:
	if not node.children:
		raise ValueError("select_child_action called on leaf without children")

	total = sum(child.visit_count for child in node.children.values())
	sqrt_total = math.sqrt(float(total + 1))
	best_score = -1e30
	best_actions: List[int] = []
	parent_player = node.to_play

	for action, child in node.children.items():
		q_child = child.mean_value()
		child_player = child.to_play
		if child_player is None and child.state is not None:
			child_player = child.state.current_player()

		# Values are stored from each node's to_play perspective.
		# Convert child value to parent perspective when turn switches.
		if parent_player is not None and child_player is not None and child_player != parent_player:
			q = -q_child
		else:
			q = q_child
		u = PUCT_C * child.prior * sqrt_total / float(1 + child.visit_count)
		score = q + u

		if score > best_score + 1e-12:
			best_score = score
			best_actions = [action]
		elif abs(score - best_score) <= 1e-12:
			best_actions.append(action)

	return random.choice(best_actions)


# _backup: function implementation.
def _backup(path: List[MCTSNode], leaf_value: float, leaf_to_play: int) -> None:
	white_value = leaf_value if leaf_to_play == WHITE else -leaf_value
	for node in path:
		node_player = leaf_to_play if node.to_play is None else node.to_play
		value_for_node = white_value if node_player == WHITE else -white_value
		node.value_sum += value_for_node
		node.visit_count += 1


# _run_mcts_python: function implementation.
def _run_mcts_python(
	policy: PolicyNetCNNMin17,
	root_state: PhaseGameState,
	device: torch.device,
	apply_dirichlet_noise: bool = True,
	simulations: int = SIMULATIONS_PER_DECISION,
) -> Tuple[str, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
	if root_state.is_done():
		# 根本不允許對終局狀態做 MCTS，直接回傳空資訊或 raise
		raise ValueError("Cannot run MCTS from terminal state")

	root = MCTSNode(prior=1.0, to_play=root_state.current_player(), state=root_state.clone())
	root_eval = _evaluate_leaf_batch(policy, [root.state], device)
	head, action_dim, legal_mask, _ = _expand_nodes_from_eval([root], root_eval)[0]
	if apply_dirichlet_noise:
		_apply_root_dirichlet_noise(root, head)

	processed = 0
	while processed < simulations:
		chunk = min(MCTS_LEAF_BATCH_SIZE, simulations - processed)
		processed += chunk

		pending_nodes: Dict[int, MCTSNode] = {}
		pending_paths: Dict[int, List[List[MCTSNode]]] = {}

		for _ in range(chunk):
			node = root
			path = [node]

			while True:
				state = node.state
				if state is None:
					raise ValueError("MCTS node state is not initialized")

				if state.is_done():
					leaf_to_play = state.current_player()
					leaf_value = _terminal_value_for_current_player(state.winner(), leaf_to_play)
					_backup(path, leaf_value, leaf_to_play)
					break

				if not node.expanded:
					nid = id(node)
					pending_nodes[nid] = node
					pending_paths.setdefault(nid, []).append(path)
					break

				action = _select_child_action(node)
				child = node.children[action]
				if child.state is None:
					child_state = state.clone()
					child_state.apply_action(action, legal_mask=node.legal_mask)
					child.state = child_state
					if not child_state.is_done():
						child.to_play = child_state.current_player()
				elif child.to_play is None and not child.state.is_done():
					child.to_play = child.state.current_player()

				node = child
				path.append(node)

		if pending_nodes:
			nodes = list(pending_nodes.values())
			states = [n.state for n in nodes]
			if any(s is None for s in states):
				raise ValueError("pending node has empty state")
			evals = _evaluate_leaf_batch(policy, [s for s in states if s is not None], device)
			expanded = _expand_nodes_from_eval(nodes, evals)
			for node, (_, _, _, leaf_value) in zip(nodes, expanded):
				nid = id(node)
				node_state = node.state
				if node_state is None:
					raise ValueError("expanded node lost state")
				leaf_to_play = node_state.current_player()
				for path in pending_paths.get(nid, []):
					_backup(path, leaf_value, leaf_to_play)

	visits = torch.zeros(action_dim, dtype=torch.float32)
	for action, child in root.children.items():
		visits[action] = float(child.visit_count)

	if visits.sum().item() <= 0:
		legal = legal_mask[:action_dim]
		n_legal = int(legal.sum().item())
		if n_legal > 0:
			visits[legal] = 1.0 / float(n_legal)
	return head, action_dim, legal_mask.to(device="cpu"), visits, None, None


# _run_mcts_cpp_session: function implementation.
def _run_mcts_cpp_session(
	policy: PolicyNetCNNMin17,
	root_state: Any,
	device: torch.device,
	apply_dirichlet_noise: bool,
	simulations: int,
) -> Tuple[str, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
	if _ACTIVE_CPP_MODULE is None:
		raise RuntimeError("cpp module is not available for SearchSession")
	if not hasattr(root_state, "_inner"):
		raise RuntimeError("cpp SearchSession expects CppPhaseGameStateAdapter input")

	module = _ACTIVE_CPP_MODULE
	cfg = module.SearchConfig()
	cfg.simulations = int(simulations)
	cfg.leaf_batch_size = int(MCTS_LEAF_BATCH_SIZE)
	cfg.puct_c = float(PUCT_C)
	cfg.add_root_dirichlet_noise = bool(apply_dirichlet_noise and USE_ROOT_DIRICHLET_NOISE)
	cfg.root_dirichlet_eps = float(ROOT_DIRICHLET_EPS)
	cfg.root_dirichlet_alpha = float(ROOT_DIRICHLET_ALPHA_ACTION)

	session = module.SearchSession(root_state._inner, cfg)
	use_packed_api = bool(
		hasattr(session, "collect_pending_leaves_packed") and hasattr(session, "submit_leaf_eval_batch")
	)

	while True:
		if use_packed_api:
			packed = session.collect_pending_leaves_packed(int(MCTS_LEAF_BATCH_SIZE))
			# Packed arrays are views into SearchSession-owned buffers.
			# Copy immediately so future simulate_chunk calls cannot invalidate them.
			node_ids_np = np.asarray(packed["node_ids"], dtype=np.int32).copy()
			if node_ids_np.size == 0:
				break

			legal_masks_np = np.asarray(packed["legal_masks"], dtype=np.uint8).copy()
			boards_np = np.asarray(packed["board_state_flat"], dtype=np.float32).copy()
			globals_np = np.asarray(packed["global_features"], dtype=np.float32).copy()

			board_batch = torch.from_numpy(boards_np.reshape(-1, 12, 9, 9)).to(device)
			global_batch = torch.from_numpy(globals_np).to(device)
			mask_batch = torch.from_numpy(legal_masks_np.astype(np.bool_, copy=False)).to(device=device)

			with torch.no_grad():
				hidden = policy.encode(board_batch, global_batch)
				logits = policy.action_head(hidden)
				values = policy.forward_value(hidden).squeeze(-1)

			masked_logits = logits.masked_fill(~mask_batch, -1e9)
			priors = torch.softmax(masked_logits, dim=-1).to(device="cpu", dtype=torch.float32).contiguous()
			values_cpu = values.to(device="cpu", dtype=torch.float32).contiguous()
			session.submit_leaf_eval_batch(node_ids_np, priors.numpy(), values_cpu.numpy())
			continue

		leaves = session.collect_pending_leaves(int(MCTS_LEAF_BATCH_SIZE))
		if not leaves:
			break

		node_ids: List[int] = []
		board_batch_items: List[torch.Tensor] = []
		global_batch_items: List[torch.Tensor] = []
		legal_masks: List[torch.Tensor] = []
		terminal_payloads: List[Tuple[int, List[float], float]] = []

		for leaf in leaves:
			node_id = int(leaf.node_id)
			mask = torch.tensor(list(leaf.legal_mask), dtype=torch.bool)
			if bool(leaf.is_done):
				current_player = int(leaf.current_player)
				winner = int(leaf.winner)
				value = _terminal_value_for_current_player(None if winner == 0 else winner, current_player)
				terminal_payloads.append((node_id, [0.0] * N_ACTIONS, float(value)))
				continue

			node_ids.append(node_id)
			legal_masks.append(mask)
			board_batch_items.append(torch.tensor(leaf.board_state_flat, dtype=torch.float32).reshape(12, 9, 9))
			global_batch_items.append(torch.tensor(leaf.global_features, dtype=torch.float32))

		for node_id, priors, value in terminal_payloads:
			session.submit_leaf_eval(node_id, priors, float(value))

		if node_ids:
			board_batch = torch.stack(board_batch_items).to(device)
			global_batch = torch.stack(global_batch_items).to(device)

			with torch.no_grad():
				hidden = policy.encode(board_batch, global_batch)
				logits = policy.action_head(hidden)
				values = policy.forward_value(hidden).squeeze(-1)

			for i, node_id in enumerate(node_ids):
				mask = legal_masks[i].to(device=device)
				masked_logits = logits[i].masked_fill(~mask, -1e9)
				priors = torch.softmax(masked_logits, dim=-1).to(device="cpu", dtype=torch.float32)
				session.submit_leaf_eval(node_id, priors.tolist(), float(values[i].item()))

	result = session.finish()
	if not bool(result.is_complete):
		raise RuntimeError("SearchSession finished before all simulations were processed")

	legal_mask = torch.from_numpy(np.asarray(result.legal_mask, dtype=np.bool_)).clone()
	visits = torch.from_numpy(np.asarray(result.root_visits, dtype=np.float32)).clone()
	action_dim = N_ACTIONS

	if visits.sum().item() <= 0:
		legal = legal_mask[:action_dim]
		n_legal = int(legal.sum().item())
		if n_legal > 0:
			visits[legal] = 1.0 / float(n_legal)

	root_leaf = session.root_snapshot()
	replay_board = torch.from_numpy(np.asarray(root_leaf.board_state_flat, dtype=np.float32)).reshape(12, 9, 9).clone()
	replay_global = torch.from_numpy(np.asarray(root_leaf.global_features, dtype=np.float32)).clone()

	return HEAD_ACTION, action_dim, legal_mask.to(device="cpu"), visits, replay_board, replay_global


def _run_mcts_cpp_batch_async(
	policy: PolicyNetCNNMin17,
	root_states: List[Any],
	device: torch.device,
	apply_dirichlet_noise: bool,
	simulations: int,
) -> List[Tuple[str, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
	if _ACTIVE_CPP_MODULE is None:
		raise RuntimeError("cpp module is not available for SearchSession")
	if not root_states:
		return []
	for state in root_states:
		if not hasattr(state, "_inner"):
			raise RuntimeError("cpp SearchSession async batch expects CppPhaseGameStateAdapter input")

	module = _ACTIVE_CPP_MODULE
	request_queue: "queue.Queue[Optional[InferenceRequest]]" = queue.Queue(maxsize=int(ASYNC_REQUEST_QUEUE_SIZE))
	response_queues: Dict[int, "queue.Queue[InferenceResponse]"] = {
		i: queue.Queue(maxsize=1) for i in range(len(root_states))
	}
	errors: "queue.Queue[Exception]" = queue.Queue()
	stop_event = threading.Event()
	results: List[Optional[Tuple[str, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]] = [None] * len(root_states)

	def _build_cfg() -> Any:
		cfg = module.SearchConfig()
		cfg.simulations = int(simulations)
		cfg.leaf_batch_size = int(MCTS_LEAF_BATCH_SIZE)
		cfg.puct_c = float(PUCT_C)
		cfg.add_root_dirichlet_noise = bool(apply_dirichlet_noise and USE_ROOT_DIRICHLET_NOISE)
		cfg.root_dirichlet_eps = float(ROOT_DIRICHLET_EPS)
		cfg.root_dirichlet_alpha = float(ROOT_DIRICHLET_ALPHA_ACTION)
		return cfg

	def _gpu_worker() -> None:
		try:
			while not stop_event.is_set():
				try:
					first_req = request_queue.get(timeout=0.1)
				except queue.Empty:
					continue

				if first_req is None:
					break

				batch_reqs: List[InferenceRequest] = [first_req]
				total_rows = int(first_req.node_ids.shape[0])
				deadline = time.perf_counter() + (float(ASYNC_INFER_MAX_WAIT_MS) / 1000.0)

				while total_rows < int(ASYNC_INFER_MAX_BATCH) and time.perf_counter() < deadline:
					try:
						next_req = request_queue.get_nowait()
					except queue.Empty:
						break
					if next_req is None:
						request_queue.put(None)
						break
					batch_reqs.append(next_req)
					total_rows += int(next_req.node_ids.shape[0])

				boards_np = np.concatenate([req.board_state_flat for req in batch_reqs], axis=0)
				globals_np = np.concatenate([req.global_features for req in batch_reqs], axis=0)
				masks_np = np.concatenate([req.legal_masks for req in batch_reqs], axis=0)

				board_batch = torch.from_numpy(boards_np.reshape(-1, 12, 9, 9)).to(device)
				global_batch = torch.from_numpy(globals_np).to(device)
				mask_batch = torch.from_numpy(masks_np.astype(np.bool_, copy=False)).to(device=device)

				with torch.no_grad():
					hidden = policy.encode(board_batch, global_batch)
					logits = policy.action_head(hidden)
					values = policy.forward_value(hidden).squeeze(-1)

				masked_logits = logits.masked_fill(~mask_batch, -1e9)
				priors_np = torch.softmax(masked_logits, dim=-1).to(device="cpu", dtype=torch.float32).contiguous().numpy()
				values_np = values.to(device="cpu", dtype=torch.float32).contiguous().numpy()

				offset = 0
				for req in batch_reqs:
					count = int(req.node_ids.shape[0])
					response_queues[req.session_id].put(
						InferenceResponse(
							session_id=req.session_id,
							batch_id=req.batch_id,
							node_ids=req.node_ids,
							priors=priors_np[offset: offset + count],
							values=values_np[offset: offset + count],
						)
					)
					offset += count
		except Exception as exc:
			errors.put(exc)
			stop_event.set()
			for q in response_queues.values():
				try:
					q.put_nowait(
						InferenceResponse(
							session_id=-1,
							batch_id=-1,
							node_ids=np.empty((0,), dtype=np.int32),
							priors=np.empty((0, N_ACTIONS), dtype=np.float32),
							values=np.empty((0,), dtype=np.float32),
							error=exc,
						)
					)
				except queue.Full:
					pass

	def _cpu_worker(session_id: int, root_state: Any) -> None:
		try:
			session = module.SearchSession(root_state._inner, _build_cfg())
			batch_id = 0
			while not stop_event.is_set():
				packed = session.collect_pending_leaves_packed(int(MCTS_LEAF_BATCH_SIZE))
				node_ids_np = np.asarray(packed["node_ids"], dtype=np.int32).copy()
				if node_ids_np.size == 0:
					break

				req = InferenceRequest(
					session_id=session_id,
					batch_id=batch_id,
					node_ids=node_ids_np,
					board_state_flat=np.asarray(packed["board_state_flat"], dtype=np.float32).copy(),
					global_features=np.asarray(packed["global_features"], dtype=np.float32).copy(),
					legal_masks=np.asarray(packed["legal_masks"], dtype=np.uint8).copy(),
				)
				request_queue.put(req)

				try:
					resp = response_queues[session_id].get(timeout=float(ASYNC_RESPONSE_TIMEOUT_S))
				except queue.Empty as exc:
					raise TimeoutError(f"async inference timeout for session {session_id}") from exc

				if resp.error is not None:
					raise resp.error
				if resp.session_id != session_id or resp.batch_id != batch_id:
					raise RuntimeError(
						f"stale/mismatched async response: expected ({session_id}, {batch_id}), "
						f"got ({resp.session_id}, {resp.batch_id})"
					)

				session.submit_leaf_eval_batch(resp.node_ids, resp.priors, resp.values)
				batch_id += 1

			result = session.finish()
			if not bool(result.is_complete):
				raise RuntimeError("SearchSession finished before all simulations were processed")

			legal_mask = torch.from_numpy(np.asarray(result.legal_mask, dtype=np.bool_)).clone()
			visits = torch.from_numpy(np.asarray(result.root_visits, dtype=np.float32)).clone()
			action_dim = N_ACTIONS

			if visits.sum().item() <= 0:
				legal = legal_mask[:action_dim]
				n_legal = int(legal.sum().item())
				if n_legal > 0:
					visits[legal] = 1.0 / float(n_legal)

			root_leaf = session.root_snapshot()
			replay_board = torch.from_numpy(np.asarray(root_leaf.board_state_flat, dtype=np.float32)).reshape(12, 9, 9).clone()
			replay_global = torch.from_numpy(np.asarray(root_leaf.global_features, dtype=np.float32)).clone()
			results[session_id] = (
				HEAD_ACTION,
				action_dim,
				legal_mask.to(device="cpu"),
				visits,
				replay_board,
				replay_global,
			)
		except Exception as exc:
			errors.put(exc)
			stop_event.set()

	gpu_thread = threading.Thread(target=_gpu_worker, name="mcts-gpu-worker", daemon=True)
	gpu_thread.start()

	workers: List[threading.Thread] = []
	for idx, state in enumerate(root_states):
		thread = threading.Thread(target=_cpu_worker, args=(idx, state), name=f"mcts-cpu-worker-{idx}", daemon=True)
		workers.append(thread)
		thread.start()

	for thread in workers:
		thread.join()

	stop_event.set()
	request_queue.put(None)
	gpu_thread.join(timeout=5.0)

	if not errors.empty():
		raise errors.get()

	out: List[Tuple[str, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]] = []
	for idx, item in enumerate(results):
		if item is None:
			raise RuntimeError(f"missing async batch result for session {idx}")
		out.append(item)
	return out


def _run_mcts_batch(
	policy: PolicyNetCNNMin17,
	root_states: List[Any],
	device: torch.device,
	apply_dirichlet_noise: bool = True,
	simulations: int = SIMULATIONS_PER_DECISION,
) -> List[Tuple[str, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
	if not root_states:
		return []

	can_use_async = (
		ENABLE_ASYNC_MCTS
		and _ACTIVE_STATE_BACKEND == "cpp"
		and _ACTIVE_CPP_MODULE is not None
		and len(root_states) > 1
		and all(hasattr(state, "_inner") for state in root_states)
	)

	if can_use_async:
		return _run_mcts_cpp_batch_async(policy, root_states, device, apply_dirichlet_noise, simulations)

	return [
		_run_mcts(policy, state, device, apply_dirichlet_noise=apply_dirichlet_noise, simulations=simulations)
		for state in root_states
	]


# _run_mcts: function implementation.
def _run_mcts(
	policy: PolicyNetCNNMin17,
	root_state: Any,
	device: torch.device,
	apply_dirichlet_noise: bool = True,
	simulations: int = SIMULATIONS_PER_DECISION,
) -> Tuple[str, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
	if root_state.is_done():
		raise ValueError("Cannot run MCTS from terminal state")

	if _ACTIVE_STATE_BACKEND == "cpp":
		if _ACTIVE_CPP_MODULE is None:
			if CPP_BACKEND_REQUIRED:
				raise RuntimeError("cpp backend requested but module is not loaded")
			print("[mcts] cpp backend unavailable; falling back to python")
			return _run_mcts_python(policy, root_state, device, apply_dirichlet_noise, simulations)

		if not hasattr(root_state, "_inner"):
			if CPP_BACKEND_REQUIRED:
				raise RuntimeError("cpp backend requires CppPhaseGameStateAdapter state")
			print("[mcts] non-cpp state received for cpp backend; falling back to python")
			return _run_mcts_python(policy, root_state, device, apply_dirichlet_noise, simulations)

		return _run_mcts_cpp_session(policy, root_state, device, apply_dirichlet_noise, simulations)

	return _run_mcts_python(policy, root_state, device, apply_dirichlet_noise, simulations)


# _record_exploration_stats: function implementation.
def _record_exploration_stats(
	acc: Dict[str, float],
	probs: torch.Tensor,
	legal_mask: torch.Tensor,
	temperature: float,
) -> None:
	"""Accumulate exploration statistics. `probs` must be the already-normalized
	action probability vector returned by _sample_action_and_target."""
	legal = legal_mask[: probs.shape[0]]
	n_legal = int(legal.sum().item())
	if n_legal <= 0:
		return

	legal_probs = probs[legal].clamp_min(1e-12)
	entropy = float((-(legal_probs * torch.log(legal_probs))).sum().item())
	entropy_norm = entropy / math.log(float(max(n_legal, 2)))
	top1 = float(legal_probs.max().item())
	eff_n = 1.0 / float((legal_probs * legal_probs).sum().item())

	acc["steps"] += 1.0
	acc["entropy_sum"] += entropy
	acc["entropy_norm_sum"] += entropy_norm
	acc["top1_sum"] += top1
	acc["n_legal_sum"] += float(n_legal)
	acc["eff_n_sum"] += eff_n

	if temperature > TEMP_LOW + 1e-8:
		acc["temp_high_steps"] += 1.0
	else:
		acc["temp_low_steps"] += 1.0

	if top1 >= 0.90:
		acc["near_greedy_steps"] += 1.0


# _format_exploration_stats: function implementation.
def _format_exploration_stats(acc: Dict[str, float]) -> str:
	steps = int(acc.get("steps", 0.0))
	if steps <= 0:
		return "explore=N/A"

	s = float(max(steps, 1))
	entropy = acc["entropy_sum"] / s
	entropy_norm = acc["entropy_norm_sum"] / s
	top1 = acc["top1_sum"] / s
	n_legal = acc["n_legal_sum"] / s
	eff_n = acc["eff_n_sum"] / s
	high = int(acc["temp_high_steps"])
	low = int(acc["temp_low_steps"])
	greedy_ratio = acc["near_greedy_steps"] / s

	return (
		f"explore(H={entropy:.2f},Hn={entropy_norm:.2f},top1={top1:.2f},"
		f"legal={n_legal:.1f},effN={eff_n:.1f},T_hi/lo={high}/{low},"
		f"greedy90={greedy_ratio:.2f})"
	)


# _sample_action_and_target: function implementation.
def _sample_action_and_target(visits: torch.Tensor, legal_mask: torch.Tensor, temperature: float) -> Tuple[int, torch.Tensor]:
	legal = legal_mask[: visits.shape[0]]
	legal_visits = visits.clone()
	legal_visits[~legal] = 0.0

	if legal_visits.sum().item() <= 0:
		probs = torch.zeros_like(legal_visits)
		n_legal = int(legal.sum().item())
		if n_legal <= 0:
			raise ValueError("No legal actions available when sampling from visits")
		probs[legal] = 1.0 / float(n_legal)
	elif temperature <= 1e-6:
		probs = torch.zeros_like(legal_visits)
		probs[int(torch.argmax(legal_visits).item())] = 1.0
	else:
		adjusted = torch.pow(legal_visits, 1.0 / temperature)
		adjusted[~legal] = 0.0
		probs = adjusted / adjusted.sum().clamp_min(1e-8)

	action = int(torch.multinomial(probs, num_samples=1).item())
	return action, probs


# _build_policy_sample: helper to construct a PolicySample from MCTS output.
def _build_policy_sample(
	action_dim: int,
	legal_mask: torch.Tensor,
	target_pi: torch.Tensor,
	replay_board: Optional[torch.Tensor],
	replay_global: Optional[torch.Tensor],
	state: Any,
	phase: Any,
) -> "PolicySample":
	if replay_board is not None and replay_global is not None:
		board = replay_board
		global_features = replay_global
	else:
		board = _build_cnn_state_12x9x9(state.game, state.current_player())
		global_features = _build_cnn_global_features(state.game, state.current_player(), phase)

	target_padded = target_pi.new_zeros(N_ACTIONS)
	target_padded[:action_dim] = target_pi[:action_dim]

	legal_padded = legal_mask.new_zeros(N_ACTIONS)
	legal_padded[:action_dim] = legal_mask[:action_dim]

	return PolicySample(
		state=board,
		global_features=global_features,
		action_dim=action_dim,
		legal_mask_padded=legal_padded,
		target_pi_padded=target_padded,
		player=state.current_player(),
	)


# _self_play_game: function implementation.
def _self_play_game(
	policy: PolicyNetCNNMin17,
	device: torch.device,
	explore_acc: Optional[Dict[str, float]] = None,
) -> Tuple[List[PolicySample], int, int]:
	state = _make_phase_state()
	decision_idx = 0
	samples: List[PolicySample] = []

	while not state.is_done():
		# 防呆：如果 state 已經是終局，直接 break，避免傳 terminal state 給 _run_mcts
		if state.is_done():
			break
		phase = state.phase()
		if phase is None:
			break

		head, action_dim, legal_mask, visits, replay_board, replay_global = _run_mcts(policy, state, device)
		temperature = _temperature_for_decision(decision_idx)
		action_idx, target_pi = _sample_action_and_target(visits, legal_mask, temperature)
		if explore_acc is not None:
			_record_exploration_stats(explore_acc, target_pi, legal_mask, temperature)

		samples.append(_build_policy_sample(action_dim, legal_mask, target_pi, replay_board, replay_global, state, phase))

		state.apply_action(action_idx)
		decision_idx += 1

	winner = state.winner()
	winner_sign = 0 if winner is None else int(winner)
	for sample in samples:
		sample.winner_sign = winner_sign
		if winner_sign == 0:
			sample.value_target = 0.0
		else:
			sample.value_target = 1.0 if sample.player == winner_sign else -1.0

	return samples, winner_sign, decision_idx


# _self_play_games_parallel: function implementation.
def _self_play_games_parallel(
	policy: PolicyNetCNNMin17,
	device: torch.device,
	n_games: int,
	parallel_games: int,
	explore_acc: Optional[Dict[str, float]] = None,
) -> List[Tuple[List[PolicySample], int, int]]:
	"""Run n_games self-play games in parallel using async batch MCTS.

	Returns list of (samples, winner_sign, n_decisions) in completion order.
	Requires cpp backend with ENABLE_ASYNC_MCTS=True and parallel_games > 1.
	"""
	parallel_start_time = time.perf_counter()
	results: List[Tuple[List[PolicySample], int, int]] = []
	active: List[dict] = []
	launched = 0

	while len(results) < n_games:
		while launched < n_games and len(active) < parallel_games:
			active.append({
				"state": _make_phase_state(),
				"samples": [],
				"decision_idx": 0,
			})
			launched += 1

		if not active:
			break

		batch_states = [entry["state"] for entry in active]
		batch_outputs = _run_mcts_batch(
			policy,
			batch_states,
			device,
			apply_dirichlet_noise=True,
			simulations=SIMULATIONS_PER_DECISION,
		)

		for entry, (head, action_dim, legal_mask, visits, replay_board, replay_global) in zip(active, batch_outputs):
			state = entry["state"]
			decision_idx = entry["decision_idx"]
			temperature = _temperature_for_decision(decision_idx)
			action_idx, target_pi = _sample_action_and_target(visits, legal_mask, temperature)
			if explore_acc is not None:
				_record_exploration_stats(explore_acc, target_pi, legal_mask, temperature)

			phase = state.phase()
			entry["samples"].append(_build_policy_sample(action_dim, legal_mask, target_pi, replay_board, replay_global, state, phase))
			state.apply_action(action_idx)
			entry["decision_idx"] += 1

		next_active: List[dict] = []
		for entry in active:
			if entry["state"].is_done():
				winner = entry["state"].winner()
				winner_sign = 0 if winner is None else int(winner)
				samples = entry["samples"]
				for s in samples:
					s.winner_sign = winner_sign
					if winner_sign == 0:
						s.value_target = 0.0
					else:
						s.value_target = 1.0 if s.player == winner_sign else -1.0
				results.append((samples, winner_sign, entry["decision_idx"]))
			else:
				next_active.append(entry)
		active = next_active

	parallel_elapsed = time.perf_counter() - parallel_start_time
	print(f"    [parallel games] completed {n_games} games in {parallel_elapsed:.2f}s (avg: {parallel_elapsed/n_games:.2f}s/game)")
	return results


# _model_vs_strategy_game: function implementation.
def _model_vs_strategy_game(
	policy: PolicyNetCNNMin17,
	device: torch.device,
	model_player: int,
	explore_acc: Optional[Dict[str, float]] = None,
) -> Tuple[List[PolicySample], int, int]:
	state = _make_phase_state()
	strategy = StrategyV2UnifiedPython()
	decision_idx = 0
	samples: List[PolicySample] = []

	while not state.is_done():
		if state.current_player() == model_player:
			phase = state.phase()
			if phase is None:
				break

			head, action_dim, legal_mask, visits, replay_board, replay_global = _run_mcts(policy, state, device)
			temperature = _temperature_for_decision(decision_idx)
			action_idx, target_pi = _sample_action_and_target(visits, legal_mask, temperature)
			if explore_acc is not None:
				_record_exploration_stats(explore_acc, target_pi, legal_mask, temperature)
			samples.append(_build_policy_sample(action_dim, legal_mask, target_pi, replay_board, replay_global, state, phase))
		else:
			phase = state.phase()
			if phase is None:
				break

			# 策略走棋，但同時用 MCTS 生成該局面的訓練目標
			head, action_dim, legal_mask, visits, replay_board, replay_global = _run_mcts(policy, state, device)
			temperature = _temperature_for_decision(decision_idx)
			_, target_pi = _sample_action_and_target(visits, legal_mask, temperature)
			if explore_acc is not None:
				_record_exploration_stats(explore_acc, target_pi, legal_mask, temperature)
			samples.append(_build_policy_sample(action_dim, legal_mask, target_pi, replay_board, replay_global, state, phase))

			# 實際走棋仍由策略決定
			action_idx = strategy.choose_action(state)
			if not (0 <= action_idx < int(legal_mask.shape[0])):
				raise ValueError("strategy action index out of range")
			if not bool(legal_mask[action_idx].item()):
				raise ValueError("strategy produced illegal action")

		state.apply_action(action_idx)
		decision_idx += 1

	winner = state.winner()
	winner_sign = 0 if winner is None else int(winner)
	for sample in samples:
		sample.winner_sign = winner_sign
		if winner_sign == 0:
			sample.value_target = 0.0
		else:
			sample.value_target = 1.0 if sample.player == winner_sign else -1.0

	return samples, winner_sign, decision_idx


# _model_vs_model_game: function implementation.
def _model_vs_model_game(
	policy_a: PolicyNetCNNMin17,
	policy_b: PolicyNetCNNMin17,
	device: torch.device,
	a_player: int,
	use_mcts: bool = True,
	mcts_simulations: int = GATE_SIMULATIONS_PER_DECISION,
	mcts_temperature: float = GATE_TEMPERATURE,
	mcts_apply_dirichlet_noise: bool = False,
) -> Tuple[int, int]:
	state = _make_phase_state()
	decision_idx = 0

	while not state.is_done():
		phase = state.phase()
		if phase is None:
			break

		acting_policy = policy_a if state.current_player() == a_player else policy_b
		if use_mcts:
			head, action_dim, legal_mask, visits, _, _ = _run_mcts(
				acting_policy,
				state,
				device,
				apply_dirichlet_noise=mcts_apply_dirichlet_noise,
				simulations=mcts_simulations,
			)
			_ = head, action_dim
			action_idx, _ = _sample_action_and_target(visits, legal_mask, temperature=mcts_temperature)
		else:
			# Direct sampling without MCTS: use model priors directly
			head, action_dim, legal_mask, priors, _ = _evaluate_leaf(acting_policy, state, device)
			action_idx, _ = _sample_action_and_target(priors, legal_mask, temperature=mcts_temperature)
		state.apply_action(action_idx)
		decision_idx += 1

	winner = state.winner()
	winner_sign = 0 if winner is None else int(winner)
	return winner_sign, decision_idx


# _evaluate_candidate_vs_guard_parallel: parallel gate evaluation using async MCTS.
def _evaluate_candidate_vs_guard_parallel(
	candidate: PolicyNetCNNMin17,
	guard: PolicyNetCNNMin17,
	device: torch.device,
	games: int,
	parallel_games: int,
) -> Dict[str, float]:
	"""Gate evaluation with parallel MCTS sessions.

	Uses the same async CPU-worker + GPU-worker pipeline as self-play.
	Each CPU worker claims games from a shared counter and runs them sequentially.
	The GPU worker routes leaves to candidate (model_id=0) or guard (model_id=1)
	based on which model owns that MCTS session's root player.
	"""
	if _ACTIVE_CPP_MODULE is None:
		raise RuntimeError("cpp module is not available for parallel gate evaluation")

	module = _ACTIVE_CPP_MODULE
	n_workers = min(parallel_games, games)

	request_queue: "queue.Queue[Optional[InferenceRequest]]" = queue.Queue(maxsize=int(ASYNC_REQUEST_QUEUE_SIZE))
	response_queues: Dict[int, "queue.Queue[InferenceResponse]"] = {
		i: queue.Queue(maxsize=1) for i in range(n_workers)
	}
	errors: "queue.Queue[Exception]" = queue.Queue()
	stop_event = threading.Event()

	# Shared counters (protected by result_lock)
	candidate_wins = 0
	guard_wins = 0
	draws = 0
	total_decisions = 0

	result_lock = threading.Lock()
	game_counter_lock = threading.Lock()
	game_counter = [0]

	def _build_cfg() -> Any:
		cfg = module.SearchConfig()
		cfg.simulations = int(GATE_SIMULATIONS_PER_DECISION)
		cfg.leaf_batch_size = int(MCTS_LEAF_BATCH_SIZE)
		cfg.puct_c = float(PUCT_C)
		cfg.add_root_dirichlet_noise = False
		cfg.root_dirichlet_eps = float(ROOT_DIRICHLET_EPS)
		cfg.root_dirichlet_alpha = float(ROOT_DIRICHLET_ALPHA_ACTION)
		return cfg

	def _run_model_forward(
		model: PolicyNetCNNMin17,
		reqs: List[InferenceRequest],
	) -> Tuple[np.ndarray, np.ndarray]:
		boards_np = np.concatenate([r.board_state_flat for r in reqs], axis=0)
		globals_np = np.concatenate([r.global_features for r in reqs], axis=0)
		masks_np = np.concatenate([r.legal_masks for r in reqs], axis=0)
		board_batch = torch.from_numpy(boards_np.reshape(-1, 12, 9, 9)).to(device)
		global_batch = torch.from_numpy(globals_np).to(device)
		mask_batch = torch.from_numpy(masks_np.astype(np.bool_, copy=False)).to(device=device)
		with torch.no_grad():
			hidden = model.encode(board_batch, global_batch)
			logits = model.action_head(hidden)
			values = model.forward_value(hidden).squeeze(-1)
		masked_logits = logits.masked_fill(~mask_batch, -1e9)
		priors_np = torch.softmax(masked_logits, dim=-1).to(device="cpu", dtype=torch.float32).contiguous().numpy()
		values_np = values.to(device="cpu", dtype=torch.float32).contiguous().numpy()
		return priors_np, values_np

	def _gpu_worker() -> None:
		try:
			while not stop_event.is_set():
				try:
					first_req = request_queue.get(timeout=0.1)
				except queue.Empty:
					continue

				if first_req is None:
					break

				batch_reqs: List[InferenceRequest] = [first_req]
				total_rows = int(first_req.node_ids.shape[0])
				deadline = time.perf_counter() + (float(ASYNC_INFER_MAX_WAIT_MS) / 1000.0)

				while total_rows < int(ASYNC_INFER_MAX_BATCH) and time.perf_counter() < deadline:
					try:
						next_req = request_queue.get_nowait()
					except queue.Empty:
						break
					if next_req is None:
						request_queue.put(None)
						break
					batch_reqs.append(next_req)
					total_rows += int(next_req.node_ids.shape[0])

				# Split by model_id and run the appropriate model
				reqs_a = [r for r in batch_reqs if r.model_id == 0]  # candidate
				reqs_b = [r for r in batch_reqs if r.model_id == 1]  # guard

				priors_a: Optional[np.ndarray] = None
				values_a: Optional[np.ndarray] = None
				priors_b: Optional[np.ndarray] = None
				values_b: Optional[np.ndarray] = None

				if reqs_a:
					priors_a, values_a = _run_model_forward(candidate, reqs_a)
				if reqs_b:
					priors_b, values_b = _run_model_forward(guard, reqs_b)

				offset_a = 0
				for req in reqs_a:
					count = int(req.node_ids.shape[0])
					response_queues[req.session_id].put(InferenceResponse(
						session_id=req.session_id,
						batch_id=req.batch_id,
						node_ids=req.node_ids,
						priors=priors_a[offset_a: offset_a + count],  # type: ignore[index]
						values=values_a[offset_a: offset_a + count],  # type: ignore[index]
					))
					offset_a += count

				offset_b = 0
				for req in reqs_b:
					count = int(req.node_ids.shape[0])
					response_queues[req.session_id].put(InferenceResponse(
						session_id=req.session_id,
						batch_id=req.batch_id,
						node_ids=req.node_ids,
						priors=priors_b[offset_b: offset_b + count],  # type: ignore[index]
						values=values_b[offset_b: offset_b + count],  # type: ignore[index]
					))
					offset_b += count

		except Exception as exc:
			errors.put(exc)
			stop_event.set()
			for q in response_queues.values():
				try:
					q.put_nowait(InferenceResponse(
						session_id=-1,
						batch_id=-1,
						node_ids=np.empty((0,), dtype=np.int32),
						priors=np.empty((0, N_ACTIONS), dtype=np.float32),
						values=np.empty((0,), dtype=np.float32),
						error=exc,
					))
				except queue.Full:
					pass

	def _cpu_worker(session_id: int) -> None:
		nonlocal candidate_wins, guard_wins, draws, total_decisions
		try:
			while not stop_event.is_set():
				with game_counter_lock:
					my_game_idx = game_counter[0]
					if my_game_idx >= games:
						break
					game_counter[0] += 1

				candidate_player = WHITE if (my_game_idx % 2 == 0) else BLACK
				state = _make_phase_state()
				if not hasattr(state, "_inner"):
					raise RuntimeError("parallel gate evaluation requires cpp state backend")

				decision_count = 0

				while not state.is_done() and not stop_event.is_set():
					current_player = state.current_player()
					model_id = 0 if current_player == candidate_player else 1

					session = module.SearchSession(state._inner, _build_cfg())
					batch_id = 0

					while not stop_event.is_set():
						packed = session.collect_pending_leaves_packed(int(MCTS_LEAF_BATCH_SIZE))
						node_ids_np = np.asarray(packed["node_ids"], dtype=np.int32).copy()
						if node_ids_np.size == 0:
							break

						req = InferenceRequest(
							session_id=session_id,
							batch_id=batch_id,
							node_ids=node_ids_np,
							board_state_flat=np.asarray(packed["board_state_flat"], dtype=np.float32).copy(),
							global_features=np.asarray(packed["global_features"], dtype=np.float32).copy(),
							legal_masks=np.asarray(packed["legal_masks"], dtype=np.uint8).copy(),
							model_id=model_id,
						)
						request_queue.put(req)

						try:
							resp = response_queues[session_id].get(timeout=float(ASYNC_RESPONSE_TIMEOUT_S))
						except queue.Empty as exc:
							raise TimeoutError(
								f"gate async inference timeout for session {session_id}"
							) from exc

						if resp.error is not None:
							raise resp.error
						if resp.session_id != session_id or resp.batch_id != batch_id:
							raise RuntimeError(
								f"stale/mismatched gate response: expected ({session_id}, {batch_id}), "
								f"got ({resp.session_id}, {resp.batch_id})"
							)

						session.submit_leaf_eval_batch(resp.node_ids, resp.priors, resp.values)
						batch_id += 1

					result = session.finish()
					if not bool(result.is_complete):
						raise RuntimeError("gate SearchSession finished before all simulations were processed")

					legal_mask = torch.from_numpy(np.asarray(result.legal_mask, dtype=np.bool_)).clone()
					visits = torch.from_numpy(np.asarray(result.root_visits, dtype=np.float32)).clone()
					action_idx, _ = _sample_action_and_target(visits, legal_mask, temperature=GATE_TEMPERATURE)
					state.apply_action(action_idx)
					decision_count += 1

				if stop_event.is_set():
					break

				winner = state.winner()
				winner_sign = 0 if winner is None else int(winner)

				with result_lock:
					if winner_sign == 0:
						draws += 1
					elif winner_sign == candidate_player:
						candidate_wins += 1
					else:
						guard_wins += 1
					total_decisions += decision_count

		except Exception as exc:
			errors.put(exc)
			stop_event.set()

	gpu_thread = threading.Thread(target=_gpu_worker, name="gate-gpu-worker", daemon=True)
	gpu_thread.start()

	workers: List[threading.Thread] = []
	for idx in range(n_workers):
		t = threading.Thread(target=_cpu_worker, args=(idx,), name=f"gate-cpu-worker-{idx}", daemon=True)
		workers.append(t)
		t.start()

	for t in workers:
		t.join()

	stop_event.set()
	request_queue.put(None)
	gpu_thread.join(timeout=5.0)

	if not errors.empty():
		raise errors.get()

	total_games = candidate_wins + guard_wins + draws
	games_f = float(total_games)
	winrate = candidate_wins / games_f if games_f > 0 else 0.0
	scorerate = (candidate_wins + 0.5 * draws) / games_f if games_f > 0 else 0.0
	return {
		"games": games_f,
		"candidate_wins": float(candidate_wins),
		"guard_wins": float(guard_wins),
		"draws": float(draws),
		"winrate": winrate,
		"scorerate": scorerate,
		"decisions": float(total_decisions),
	}


# _evaluate_candidate_vs_guard: function implementation.
def _evaluate_candidate_vs_guard(
	candidate: PolicyNetCNNMin17,
	guard: PolicyNetCNNMin17,
	device: torch.device,
	games: int,
) -> Dict[str, float]:
	if games <= 0:
		raise ValueError("GATE_EVAL_GAMES must be >= 1")

	candidate.eval()
	guard.eval()

	use_parallel = (
		ENABLE_ASYNC_MCTS
		and _ACTIVE_STATE_BACKEND == "cpp"
		and _ACTIVE_CPP_MODULE is not None
		and ASYNC_PARALLEL_GAMES > 1
	)

	if use_parallel:
		return _evaluate_candidate_vs_guard_parallel(
			candidate,
			guard,
			device,
			games,
			parallel_games=ASYNC_PARALLEL_GAMES,
		)

	candidate_wins = 0
	guard_wins = 0
	draws = 0
	total_decisions = 0

	for game_idx in range(games):
		candidate_player = WHITE if (game_idx % 2 == 0) else BLACK
		winner, n_decisions = _model_vs_model_game(
			candidate,
			guard,
			device,
			candidate_player,
			use_mcts=True,
			mcts_simulations=GATE_SIMULATIONS_PER_DECISION,
			mcts_temperature=GATE_TEMPERATURE,
			mcts_apply_dirichlet_noise=False,
		)
		total_decisions += n_decisions

		if winner == 0:
			draws += 1
		elif winner == candidate_player:
			candidate_wins += 1
		else:
			guard_wins += 1

	games_f = float(games)
	winrate = candidate_wins / games_f
	scorerate = (candidate_wins + 0.5 * draws) / games_f
	return {
		"games": games_f,
		"candidate_wins": float(candidate_wins),
		"guard_wins": float(guard_wins),
		"draws": float(draws),
		"winrate": winrate,
		"scorerate": scorerate,
		"decisions": float(total_decisions),
	}


# _train_on_samples: function implementation.
def _train_on_samples(
	policy: PolicyNetCNNMin17,
	optimizer: torch.optim.Optimizer,
	samples: List[PolicySample],
	device: torch.device,
) -> Dict[str, float]:
	if not samples:
		return {
			"loss": 0.0,
			"policy_loss": 0.0,
			"value_loss": 0.0,
			"entropy": 0.0,
			"n_samples": 0,
		}

	pack_start = time.perf_counter()
	states_cpu = torch.stack([s.state for s in samples])
	globals_cpu = torch.stack([s.global_features for s in samples])
	masks_cpu = torch.stack([s.legal_mask_padded for s in samples])
	policy_targets_cpu = torch.stack([s.target_pi_padded for s in samples])
	value_targets_cpu = torch.tensor([s.value_target for s in samples], dtype=torch.float32)
	pack_time_s = time.perf_counter() - pack_start

	total_loss = 0.0
	total_policy = 0.0
	total_value = 0.0
	total_entropy = 0.0
	total_count = 0
	transfer_time_s = 0.0
	compute_time_s = 0.0

	policy.train()
	n = states_cpu.shape[0]
	batch_count = n // BATCH_SIZE

	for _ in range(TRAIN_EPOCHS_PER_UPDATE):
		for _ in range(batch_count):
			sample_ids = np.random.randint(n, size=BATCH_SIZE)
			idx_cpu = torch.from_numpy(sample_ids.astype(np.int64, copy=False))

			transfer_start = time.perf_counter()
			b_states = states_cpu[idx_cpu].to(device, non_blocking=True)
			b_globals = globals_cpu[idx_cpu].to(device, non_blocking=True)
			b_masks = masks_cpu[idx_cpu].to(device, non_blocking=True)
			b_policy_targets = policy_targets_cpu[idx_cpu].to(device, non_blocking=True)
			b_value_targets = value_targets_cpu[idx_cpu].to(device, non_blocking=True)
			transfer_time_s += time.perf_counter() - transfer_start

			compute_start = time.perf_counter()

			hidden = policy.encode(b_states, b_globals)
			action_logits = policy.action_head(hidden)
			value_pred = policy.forward_value(hidden).squeeze(-1)

			masked_logits = action_logits.masked_fill(~b_masks, -1e9)
			logp = F.log_softmax(masked_logits, dim=-1)
			probs = torch.exp(logp)
			policy_loss = -(b_policy_targets * logp).sum(dim=-1).mean()
			entropy = -(probs * logp).sum(dim=-1).mean()

			value_loss = F.mse_loss(value_pred, b_value_targets)
			loss = (
				POLICY_LOSS_WEIGHT * policy_loss
				+ VALUE_LOSS_WEIGHT * value_loss
			)

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()
			compute_time_s += time.perf_counter() - compute_start

			bs = BATCH_SIZE
			total_loss += float(loss.item()) * bs
			total_policy += float(policy_loss.item()) * bs
			total_value += float(value_loss.item()) * bs
			total_entropy += float(entropy.item()) * bs
			total_count += bs

	denom = max(total_count, 1)
	return {
		"loss": total_loss / denom,
		"policy_loss": total_policy / denom,
		"value_loss": total_value / denom,
		"entropy": total_entropy / denom,
		"n_samples": float(n),
		"pack_time_s": pack_time_s,
		"transfer_time_s": transfer_time_s,
		"compute_time_s": compute_time_s,
	}


# _scan_resume_checkpoint: function implementation.
def _scan_resume_checkpoint(device: str) -> Optional[Tuple[Dict[str, object], Path]]:
	candidates = train_module.list_checkpoints(title=None if RESUME_ANY_TITLE else RESUME_TITLE)
	if not candidates:
		return None

	ranked: List[Tuple[float, Dict[str, object], Path]] = []
	for ckpt_info in candidates:
		path = ckpt_info.path
		try:
			ckpt = torch.load(str(path), map_location=device, weights_only=True)
		except Exception:
			continue
		arch = str(ckpt.get("architecture", "")).lower()
		if arch != "cnn_min17":
			continue
		update_idx = float(ckpt.get("update_idx", -1))
		ranked.append((update_idx, ckpt, path))

	if not ranked:
		return None

	ranked.sort(key=lambda item: (item[0], item[2].stat().st_mtime), reverse=True)
	_, ckpt, path = ranked[0]
	return ckpt, path


# _scan_latest_checkpoint_for_title: function implementation.
def _scan_latest_checkpoint_for_title(title: str, device: str) -> Optional[Tuple[Dict[str, object], Path]]:
	candidates = train_module.list_checkpoints(title=title)
	if not candidates:
		return None

	ranked: List[Tuple[float, Dict[str, object], Path]] = []
	for ckpt_info in candidates:
		path = ckpt_info.path
		try:
			ckpt = torch.load(str(path), map_location=device, weights_only=True)
		except Exception:
			continue
		arch = str(ckpt.get("architecture", "")).lower()
		if arch != "cnn_min17":
			continue
		update_idx = float(ckpt.get("update_idx", -1))
		ranked.append((update_idx, ckpt, path))

	if not ranked:
		return None

	ranked.sort(key=lambda item: (item[0], item[2].stat().st_mtime), reverse=True)
	_, ckpt, path = ranked[0]
	return ckpt, path


# _replay_snapshot_path: function implementation.
def _replay_snapshot_path(
	title: str,
	checkpoint_index: Optional[int] = None,
	base_dir: Optional[Path] = None,
) -> Path:
	directory = base_dir if base_dir is not None else Path(train_module.CHECKPOINT_DIR)
	if checkpoint_index is None:
		filename = f"{title}_replay_{REPLAY_SNAPSHOT_TAG}.pt"
	else:
		if checkpoint_index < 0:
			raise ValueError("checkpoint_index must be >= 0")
		filename = f"{title}_replay_{checkpoint_index:06d}.pt"
	return directory / filename


# _save_replay_snapshot: function implementation.
def _save_replay_snapshot(
	title: str,
	checkpoint_index: Optional[int],
	update_idx: int,
	replay_buffer: List[PolicySample],
	write_idx: int,
	max_samples: int,
	base_dir: Optional[Path] = None,
) -> Optional[Path]:
	if not USE_REPLAY_BUFFER:
		return None

	path = _replay_snapshot_path(title, checkpoint_index, base_dir=base_dir)
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_name(f"{path.name}.tmp")

	payload_samples: List[Dict[str, object]] = []
	for sample in replay_buffer:
		payload_samples.append(
			{
				"state": sample.state.detach().to(device="cpu", dtype=torch.float32),
				"global_features": sample.global_features.detach().to(device="cpu", dtype=torch.float32),
				"action_dim": int(sample.action_dim),
				"legal_mask_padded": sample.legal_mask_padded.detach().to(device="cpu", dtype=torch.bool),
				"target_pi_padded": sample.target_pi_padded.detach().to(device="cpu", dtype=torch.float32),
				"player": int(sample.player),
				"winner_sign": int(sample.winner_sign),
				"value_target": float(sample.value_target),
			}
		)

	payload: Dict[str, object] = {
		"version": int(REPLAY_SNAPSHOT_VERSION),
		"title": str(title),
		"checkpoint_index": None if checkpoint_index is None else int(checkpoint_index),
		"update_idx": int(update_idx),
		"buffer_size": int(len(payload_samples)),
		"max_samples": int(max_samples),
		"write_idx": int(write_idx),
		"samples": payload_samples,
	}

	try:
		torch.save(payload, str(tmp_path))
		tmp_path.replace(path)
		print(
			f"[replay] saved {path.name} | samples={len(payload_samples)} "
			f"write_idx={write_idx} update={update_idx}"
		)
		return path
	except Exception as exc:
		print(f"[replay] save failed for {path.name}: {exc}")
		try:
			if tmp_path.exists():
				tmp_path.unlink()
		except Exception:
			pass
		return None


# _save_replay_snapshot_async: fire-and-forget wrapper for _save_replay_snapshot.
def _save_replay_snapshot_async(
	title: str,
	checkpoint_index: Optional[int],
	update_idx: int,
	replay_buffer: "List[PolicySample]",
	write_idx: int,
	max_samples: int,
	base_dir: Optional[Path] = None,
) -> None:
	"""Launch _save_replay_snapshot in a background daemon thread so training
	resumes immediately.  Any previous snapshot thread is joined (with a short
	timeout) before starting a new one so at most one save is in flight."""
	global _snapshot_save_thread

	if _snapshot_save_thread is not None and _snapshot_save_thread.is_alive():
		_snapshot_save_thread.join(timeout=1.0)

	buffer_copy = list(replay_buffer)

	def _worker() -> None:
		_save_replay_snapshot(
			title=title,
			checkpoint_index=checkpoint_index,
			update_idx=update_idx,
			replay_buffer=buffer_copy,
			write_idx=write_idx,
			max_samples=max_samples,
			base_dir=base_dir,
		)

	_snapshot_save_thread = threading.Thread(target=_worker, name="replay-snapshot-saver", daemon=True)
	_snapshot_save_thread.start()


# _load_replay_snapshot: function implementation.
def _load_replay_snapshot(
	title: str,
	checkpoint_index: Optional[int],
	max_samples: int,
	base_dir: Optional[Path] = None,
) -> Tuple[List[PolicySample], int]:
	if not USE_REPLAY_BUFFER:
		return [], 0

	path = _replay_snapshot_path(title, checkpoint_index, base_dir=base_dir)
	if not path.exists():
		print(f"[replay] no snapshot found: {path.name}")
		return [], 0

	try:
		payload = torch.load(str(path), map_location="cpu", weights_only=True)
	except Exception as exc:
		print(f"[replay] failed to load {path.name}: {exc}")
		return [], 0

	if not isinstance(payload, dict):
		print(f"[replay] invalid payload type in {path.name}")
		return [], 0

	version = int(payload.get("version", -1))
	if version != REPLAY_SNAPSHOT_VERSION:
		print(
			f"[replay] version mismatch in {path.name}: "
			f"got={version} expected={REPLAY_SNAPSHOT_VERSION}"
		)
		return [], 0

	raw_samples = payload.get("samples")
	if not isinstance(raw_samples, list):
		print(f"[replay] invalid sample list in {path.name}")
		return [], 0

	replay_buffer: List[PolicySample] = []
	for raw in raw_samples:
		if not isinstance(raw, dict):
			continue
		state = raw.get("state")
		global_features = raw.get("global_features")
		legal_mask = raw.get("legal_mask_padded")
		target_pi = raw.get("target_pi_padded")
		if not all(torch.is_tensor(x) for x in (state, global_features, legal_mask, target_pi)):
			continue

		replay_buffer.append(
			PolicySample(
				state=state.to(dtype=torch.float32),
				global_features=global_features.to(dtype=torch.float32),
				action_dim=int(raw.get("action_dim", 0)),
				legal_mask_padded=legal_mask.to(dtype=torch.bool),
				target_pi_padded=target_pi.to(dtype=torch.float32),
				player=int(raw.get("player", 0)),
				winner_sign=int(raw.get("winner_sign", 0)),
				value_target=float(raw.get("value_target", 0.0)),
			)
		)

	if max_samples > 0 and len(replay_buffer) > max_samples:
		print(
			f"[replay] truncating loaded samples {len(replay_buffer)} -> {max_samples}"
		)
		replay_buffer = replay_buffer[:max_samples]

	if not replay_buffer:
		print(f"[replay] snapshot empty after parsing: {path.name}")
		return [], 0

	write_idx = int(payload.get("write_idx", 0))
	max_valid = max_samples if len(replay_buffer) >= max_samples and max_samples > 0 else len(replay_buffer)
	if write_idx < 0 or write_idx >= max(max_valid, 1):
		print(f"[replay] invalid write_idx={write_idx}, reset to 0")
		write_idx = 0

	print(
		f"[replay] loaded {path.name} | samples={len(replay_buffer)} write_idx={write_idx}"
	)
	return replay_buffer, write_idx


# _next_checkpoint_index_for_title: function implementation.
def _next_checkpoint_index_for_title(title: str) -> int:
	by_title = train_module.list_checkpoints(title=title)
	if not by_title:
		return 0
	return max(item.index for item in by_title) + 1


# _optimizer_to: function implementation.
def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
	"""Move optimizer state tensors to the specified device."""
	for state in optimizer.state.values():
		for k, v in state.items():
			if torch.is_tensor(v):
				state[k] = v.to(device)


# _load_or_init_policy: function implementation.
def _load_or_init_policy(
	device: torch.device,
	guard_title: str,
) -> Tuple[PolicyNetCNNMin17, torch.optim.Optimizer, int, int, List[PolicySample], int]:
	policy = PolicyNetCNNMin17(global_feature_dim=CNN_GLOBAL_FEATURE_DIM, dropout=CNN_DROPOUT).to(device)
	optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE_START, weight_decay=WEIGHT_DECAY)

	start_update = 0
	next_checkpoint_index = _next_checkpoint_index_for_title(TITLE)
	replay_buffer: List[PolicySample] = []
	replay_write_idx = 0

	resume = _scan_latest_checkpoint_for_title(guard_title, str(device))
	if resume is None:
		if REQUIRE_RESUME:
			raise RuntimeError(f"No guard checkpoint found for title '{guard_title}'.")
		print("[resume] no guard checkpoint found, bootstrapping candidate from fresh weights")
		return policy, optimizer, start_update, next_checkpoint_index, replay_buffer, replay_write_idx

	ckpt, ckpt_path = resume
	policy_state = ckpt.get("policy_state")
	if not isinstance(policy_state, dict):
		raise KeyError(f"Invalid checkpoint: missing policy_state in {ckpt_path}")

	policy.load_state_dict(policy_state, strict=True)
	optimizer_state = ckpt.get("optimizer_state")
	if isinstance(optimizer_state, dict):
		try:
			optimizer.load_state_dict(optimizer_state)
		except Exception:
			pass

	start_update = int(ckpt.get("update_idx", -1)) + 1
	replay_buffer, replay_write_idx = _load_replay_snapshot(
		title=guard_title,
		checkpoint_index=None,
		max_samples=REPLAY_BUFFER_MAX_SAMPLES,
		base_dir=ckpt_path.parent,
	)
	print(
		f"[resume] candidate loaded from guard {ckpt_path.name} "
		f"| start_update={start_update} | replay={len(replay_buffer)}"
	)
	return policy, optimizer, start_update, next_checkpoint_index, replay_buffer, replay_write_idx


# _load_or_init_guard_policy: function implementation.
def _load_or_init_guard_policy(
	device: torch.device,
	fallback_policy: PolicyNetCNNMin17,
	guard_title: str,
) -> Tuple[PolicyNetCNNMin17, str]:
	guard = PolicyNetCNNMin17(global_feature_dim=CNN_GLOBAL_FEATURE_DIM, dropout=CNN_DROPOUT).to(device)
	resume = _scan_latest_checkpoint_for_title(guard_title, str(device))
	if resume is None:
		guard.load_state_dict(fallback_policy.state_dict(), strict=True)
		guard.eval()
		return guard, "bootstrap_from_candidate"

	ckpt, ckpt_path = resume
	policy_state = ckpt.get("policy_state")
	if not isinstance(policy_state, dict):
		raise KeyError(f"Invalid guard checkpoint: missing policy_state in {ckpt_path}")
	guard.load_state_dict(policy_state, strict=True)
	guard.eval()
	return guard, f"loaded_{ckpt_path.name}"


# _validate_constants: function implementation.
def _validate_constants() -> None:
	if TOTAL_UPDATES <= 0:
		raise ValueError("TOTAL_UPDATES must be >= 1")
	if GAMES_PER_UPDATE <= 0:
		raise ValueError("GAMES_PER_UPDATE must be >= 1")
	if not (0.0 <= SELFPLAY_MODEL_GAME_RATIO <= 1.0):
		raise ValueError("SELFPLAY_MODEL_GAME_RATIO must be in [0, 1]")
	if TRAIN_EPOCHS_PER_UPDATE <= 0:
		raise ValueError("TRAIN_EPOCHS_PER_UPDATE must be >= 1")
	if BATCH_SIZE <= 0:
		raise ValueError("BATCH_SIZE must be >= 1")
	if SIMULATIONS_PER_DECISION <= 0:
		raise ValueError("SIMULATIONS_PER_DECISION must be >= 1")
	if CHECKPOINT_EVERY_UPDATES <= 0:
		raise ValueError("CHECKPOINT_EVERY_UPDATES must be >= 1")
	if GATE_EVERY_UPDATES <= 0:
		raise ValueError("GATE_EVERY_UPDATES must be >= 1")
	if LOG_EVERY <= 0:
		raise ValueError("LOG_EVERY must be >= 1")
	if OPTIMIZATION_PASSES_PER_UPDATE <= 0:
		raise ValueError("OPTIMIZATION_PASSES_PER_UPDATE must be >= 1")
	if GATE_EVAL_GAMES <= 0:
		raise ValueError("GATE_EVAL_GAMES must be >= 1")
	if not (0.0 < GATE_WINRATE_THRESHOLD < 1.0):
		raise ValueError("GATE_WINRATE_THRESHOLD must be in (0, 1)")
	if HEURISTIC_SOFTMAX_TEMPERATURE <= 0:
		raise ValueError("HEURISTIC_SOFTMAX_TEMPERATURE must be > 0")
	if not (0.0 <= HEURISTIC_PRIOR_WEIGHT <= 1.0):
		raise ValueError("HEURISTIC_PRIOR_WEIGHT must be in [0, 1]")
	if USE_REPLAY_BUFFER:
		if REPLAY_BUFFER_MAX_SAMPLES <= 0:
			raise ValueError("REPLAY_BUFFER_MAX_SAMPLES must be >= 1 when replay buffer is enabled")
		if REPLAY_TRAIN_SAMPLES_PER_UPDATE <= 0:
			raise ValueError("REPLAY_TRAIN_SAMPLES_PER_UPDATE must be >= 1 when replay buffer is enabled")
		if REPLAY_MIN_TRAIN_SAMPLES <= 0:
			raise ValueError("REPLAY_MIN_TRAIN_SAMPLES must be >= 1 when replay buffer is enabled")
		if REPLAY_MIN_TRAIN_SAMPLES > REPLAY_BUFFER_MAX_SAMPLES:
			raise ValueError("REPLAY_MIN_TRAIN_SAMPLES must be <= REPLAY_BUFFER_MAX_SAMPLES")


# _replay_extend: function implementation.
def _replay_extend(
	buffer: List[PolicySample],
	write_idx: int,
	max_samples: int,
	new_samples: List[PolicySample],
) -> int:
	if max_samples <= 0 or not new_samples:
		return write_idx

	for sample in new_samples:
		if len(buffer) < max_samples:
			buffer.append(sample)
			if len(buffer) == max_samples:
				write_idx = 0
		else:
			buffer[write_idx] = sample
			write_idx = (write_idx + 1) % max_samples

	return write_idx


# _select_train_samples: function implementation.
def _select_train_samples(
	fresh_samples: List[PolicySample],
	replay_buffer: List[PolicySample],
) -> List[PolicySample]:
	if USE_REPLAY_BUFFER:
		available = len(replay_buffer)
		if available <= 0:
			return []
		if available < REPLAY_MIN_TRAIN_SAMPLES:
			return replay_buffer.copy()
		n_draw = min(REPLAY_TRAIN_SAMPLES_PER_UPDATE, available)
		if n_draw >= available:
			return replay_buffer.copy()
		indices = np.random.choice(available, size=n_draw, replace=False)
		return [replay_buffer[int(i)] for i in indices]
	return fresh_samples


class _TeeTextIO:
	"""Mirror text output to both console and a log file."""

	# __init__: function implementation.
	def __init__(self, console_stream, file_stream) -> None:
		self._console = console_stream
		self._file = file_stream

	# write: function implementation.
	def write(self, data: str) -> int:
		self._console.write(data)
		self._file.write(data)
		return len(data)

	# flush: function implementation.
	def flush(self) -> None:
		self._console.flush()
		self._file.flush()


# _create_log_file: function implementation.
def _create_log_file(title: str) -> Path:
	log_dir = Path(LOG_DIR)
	log_dir.mkdir(parents=True, exist_ok=True)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	return log_dir / f"{title}_{ts}.log"


# _accumulate_kind_stats: batch accumulation of action-kind statistics.
def _accumulate_kind_stats(
	samples: "List[PolicySample]",
	action_kind_mass: "List[float]",
	kind_legal_counts: "List[int]",
) -> int:
	"""Accumulate capture/reinforce/pass statistics for phase-2 samples using
	batched tensor operations. Returns the number of phase-2 samples processed."""
	phase2_idx = 10 + TRAINING_PHASE_TO_IDX[PHASE_STEP_2]
	phase2_samples = [s for s in samples if s.global_features[phase2_idx].item() > 0.5]
	if not phase2_samples:
		return 0

	all_pi = torch.stack([s.target_pi_padded for s in phase2_samples])    # (N, A)
	all_legal = torch.stack([s.legal_mask_padded for s in phase2_samples])  # (N, A)

	action_kind_mass[0] += float(all_pi[:, :300].sum().item())
	action_kind_mass[1] += float(all_pi[:, 300:600].sum().item())
	action_kind_mass[2] += float(all_pi[:, PASS_ACTION_IDX].sum().item())

	kind_legal_counts[0] += int(all_legal[:, :300].any(dim=1).sum().item())
	kind_legal_counts[1] += int(all_legal[:, 300:600].any(dim=1).sum().item())
	kind_legal_counts[2] += int(all_legal[:, PASS_ACTION_IDX].sum().item())

	return len(phase2_samples)


# run: function implementation.
def run() -> None:
	global _ACTIVE_STATE_BACKEND, _ACTIVE_CPP_MODULE
	_validate_constants()
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	state_backend, cpp_module = _resolve_state_backend()
	_ACTIVE_STATE_BACKEND = state_backend
	_ACTIVE_CPP_MODULE = cpp_module

	train_device = torch.device(train_module.resolve_device(DEVICE))
	infer_device = torch.device(INFERENCE_DEVICE)
	title = train_module.sanitize_checkpoint_title(TITLE)
	guard_title = train_module.sanitize_checkpoint_title(GUARD_TITLE)

	(
		policy,
		optimizer,
		start_update,
		next_checkpoint_index,
		replay_buffer,
		replay_write_idx,
	) = _load_or_init_policy(infer_device, guard_title)
	guard_policy, guard_source = _load_or_init_guard_policy(infer_device, policy, guard_title)
	next_guard_checkpoint_index = _next_checkpoint_index_for_title(guard_title)

	print("mcts model-vs-strategy training started")
	print(
		f"  title={title} infer_device={infer_device} train_device={train_device} updates={TOTAL_UPDATES} start={start_update} "
		f"games_per_update={GAMES_PER_UPDATE} sims_per_decision={SIMULATIONS_PER_DECISION}"
	)
	print(f"  state_backend={state_backend}")
	print(
		f"  puct_c={PUCT_C:.3f} temp(high/low/switch)={TEMP_HIGH:.2f}/{TEMP_LOW:.2f}/{TEMP_SWITCH_DECISION} "
		f"dir(eps/alpha_action)={ROOT_DIRICHLET_EPS:.2f}/{ROOT_DIRICHLET_ALPHA_ACTION:.2f}"
	)
	print(
		f"  optimize_passes={OPTIMIZATION_PASSES_PER_UPDATE} gate_games={GATE_EVAL_GAMES} "
		f"gate_threshold={GATE_WINRATE_THRESHOLD:.2f} guard={guard_title} ({guard_source})"
	)
	print(f"  selfplay_model_ratio={SELFPLAY_MODEL_GAME_RATIO:.2f}")

	for update in range(start_update, TOTAL_UPDATES): #每一次update
		lr = train_module.linear_schedule_value(update, TOTAL_UPDATES, LEARNING_RATE_START, LEARNING_RATE_END)
		for group in optimizer.param_groups:
			group["lr"] = lr

		update_start = time.perf_counter()
		fresh_samples: List[PolicySample] = []
		wins = {WHITE: 0, BLACK: 0, 0: 0}
		model_vs_strategy = {"model": 0, "strategy": 0, "draw": 0}
		selfplay_stats = {"white": 0, "black": 0, "draw": 0}
		selfplay_games = 0
		vs_strategy_games = 0
		decision_steps = 0
		fresh_inserted = 0

		# second-step action 統計（capture/reinforce/pass）
		action_kind_mass = [0.0, 0.0, 0.0]
		kind_legal_counts = [0, 0, 0]
		total_kind = 0
		explore_acc: Dict[str, float] = {
			"steps": 0.0,
			"entropy_sum": 0.0,
			"entropy_norm_sum": 0.0,
			"top1_sum": 0.0,
			"n_legal_sum": 0.0,
			"eff_n_sum": 0.0,
			"temp_high_steps": 0.0,
			"temp_low_steps": 0.0,
			"near_greedy_steps": 0.0,
		}

		print(
			f"[self-play] update {update + 1}/{TOTAL_UPDATES} start: "
			f"simulating {GAMES_PER_UPDATE} game(s) | device={infer_device}"
		)

		policy.eval()
		self_play_log_interval = int(SELFPLAY_PROGRESS_LOG_INTERVAL)

		def _should_log_selfplay_progress(game_idx: int) -> bool:
			if self_play_log_interval <= 0:
				return False
			if game_idx == GAMES_PER_UPDATE:
				return True
			return (game_idx == 1) or (game_idx % self_play_log_interval == 0)

		use_parallel_selfplay = (
			ENABLE_ASYNC_MCTS
			and _ACTIVE_STATE_BACKEND == "cpp"
			and _ACTIVE_CPP_MODULE is not None
			and ASYNC_PARALLEL_GAMES > 1
		)

		def _process_selfplay_game(samples: List[PolicySample], winner: int, n_decisions: int, game_idx: int) -> None:
			nonlocal fresh_inserted, replay_write_idx, decision_steps, total_kind
			if winner == 0:
				selfplay_stats["draw"] += 1
			elif winner == WHITE:
				selfplay_stats["white"] += 1
			else:
				selfplay_stats["black"] += 1
			if not USE_REPLAY_BUFFER:
				fresh_samples.extend(samples)
			fresh_inserted += len(samples)
			if USE_REPLAY_BUFFER:
				replay_write_idx = _replay_extend(
					replay_buffer,
					replay_write_idx,
					REPLAY_BUFFER_MAX_SAMPLES,
					samples,
				)
			wins[winner] = wins.get(winner, 0) + 1
			decision_steps += n_decisions

			total_kind += _accumulate_kind_stats(samples, action_kind_mass, kind_legal_counts)

		# Timing for self-play phase
		selfplay_start_time = time.perf_counter()
		
		if use_parallel_selfplay:
			print(f"  [self-play] using parallel self-play: parallel_games={ASYNC_PARALLEL_GAMES}")
			parallel_results = _self_play_games_parallel(
				policy,
				infer_device,
				GAMES_PER_UPDATE,
				ASYNC_PARALLEL_GAMES,
				explore_acc=explore_acc,
			)
			for game_idx, (samples, winner, n_decisions) in enumerate(parallel_results, 1):
				selfplay_games += 1
				_process_selfplay_game(samples, winner, n_decisions, game_idx)
		else:
			for game_idx in range(1, GAMES_PER_UPDATE + 1):
				if random.random() < SELFPLAY_MODEL_GAME_RATIO:
					samples, winner, n_decisions = _self_play_game(policy, infer_device, explore_acc=explore_acc) #自我對弈
					selfplay_games += 1
					_process_selfplay_game(samples, winner, n_decisions, game_idx)
				else:
					model_player = WHITE if random.random() < 0.5 else BLACK
					samples, winner, n_decisions = _model_vs_strategy_game(
						policy,
						infer_device,
						model_player,
						explore_acc=explore_acc,
					)
					vs_strategy_games += 1
					if winner == 0:
						model_vs_strategy["draw"] += 1
					elif winner == model_player:
						model_vs_strategy["model"] += 1
					else:
						model_vs_strategy["strategy"] += 1
					if not USE_REPLAY_BUFFER:
						fresh_samples.extend(samples)
					fresh_inserted += len(samples)
					if USE_REPLAY_BUFFER:
						replay_write_idx = _replay_extend(
							replay_buffer,
							replay_write_idx,
							REPLAY_BUFFER_MAX_SAMPLES,
							samples,
						)
					wins[winner] = wins.get(winner, 0) + 1
					decision_steps += n_decisions

					total_kind += _accumulate_kind_stats(samples, action_kind_mass, kind_legal_counts)

					if _should_log_selfplay_progress(game_idx):
						total_vs_strategy = (
							model_vs_strategy["model"]
							+ model_vs_strategy["strategy"]
							+ model_vs_strategy["draw"]
						)
						model_wr = (model_vs_strategy["model"] / float(total_vs_strategy)) if total_vs_strategy > 0 else 0.0
						explore_now = _format_exploration_stats(explore_acc)
					# 進度打印已移除，只在最後打印總時間

		# Log total self-play time after all games completed
		selfplay_elapsed = time.perf_counter() - selfplay_start_time
		print(
			f"  [self-play] TOTAL TIME: {selfplay_elapsed:.2f}s for {GAMES_PER_UPDATE} game(s) "
			f"(avg: {selfplay_elapsed/GAMES_PER_UPDATE:.2f}s/game)"
		)

		print(
			f"[self-play] update {update + 1}/{TOTAL_UPDATES} done: "
			f"{GAMES_PER_UPDATE} game(s), decisions={decision_steps}, fresh_samples={fresh_inserted} | device={infer_device}"
		)

		_save_replay_snapshot_async(
			title=guard_title,
			checkpoint_index=None,
			update_idx=update,
			replay_buffer=replay_buffer,
			write_idx=replay_write_idx,
			max_samples=REPLAY_BUFFER_MAX_SAMPLES,
		)

		if USE_REPLAY_BUFFER:
			train_samples = replay_buffer.copy() if replay_buffer else []
		else:
			train_samples = fresh_samples

		if USE_REPLAY_BUFFER:
			print(f"  replay_size={len(replay_buffer)}/{REPLAY_BUFFER_MAX_SAMPLES} before training")

		last_metrics: Dict[str, float] = {
			"loss": 0.0,
			"policy_loss": 0.0,
			"value_loss": 0.0,
			"entropy": 0.0,
			"n_samples": 0.0,
		}
		agg_loss = 0.0
		agg_policy = 0.0
		agg_value = 0.0
		agg_entropy = 0.0
		agg_samples = 0.0
		optimize_wall_start = time.perf_counter()

		policy.to(train_device)
		_optimizer_to(optimizer, train_device)
		print(
			f"[optimize] update {update + 1}/{TOTAL_UPDATES} start: "
			f"{OPTIMIZATION_PASSES_PER_UPDATE} pass(es) | device={train_device}"
		)

		for pass_idx in range(1, OPTIMIZATION_PASSES_PER_UPDATE + 1):
			if not train_samples:
				print(f"  [optimize] pass {pass_idx}/{OPTIMIZATION_PASSES_PER_UPDATE}: skipped (no samples)")
				continue
			pass_start = time.perf_counter()
			metrics = _train_on_samples(policy, optimizer, train_samples, train_device)
			pass_wall = time.perf_counter() - pass_start
			last_metrics = metrics
			weight = float(max(metrics["n_samples"], 1.0))
			agg_loss += float(metrics["loss"]) * weight
			agg_policy += float(metrics["policy_loss"]) * weight
			agg_value += float(metrics["value_loss"]) * weight
			agg_entropy += float(metrics["entropy"]) * weight
			agg_samples += weight
			print(
				f"  [optimize] pass {pass_idx}/{OPTIMIZATION_PASSES_PER_UPDATE} "
				f"| loss={metrics['loss']:+.4f} p={metrics['policy_loss']:+.4f} "
				f"v={metrics['value_loss']:+.4f} ent={metrics['entropy']:+.4f} "
				f"| samples={int(metrics['n_samples'])}"
				f" | t(pack/xfer/compute/wall)="
				f"{metrics.get('pack_time_s', 0.0):.2f}/"
				f"{metrics.get('transfer_time_s', 0.0):.2f}/"
				f"{metrics.get('compute_time_s', 0.0):.2f}/"
				f"{pass_wall:.2f}s"
			)

		optimize_wall = time.perf_counter() - optimize_wall_start
		print(
			f"[optimize] update {update + 1}/{TOTAL_UPDATES} done "
			f"| device={train_device} | wall={optimize_wall:.2f}s"
		)
		policy.to(infer_device)

		if agg_samples > 0.0:
			metrics = {
				"loss": agg_loss / agg_samples,
				"policy_loss": agg_policy / agg_samples,
				"value_loss": agg_value / agg_samples,
				"entropy": agg_entropy / agg_samples,
				"n_samples": agg_samples,
			}
		else:
			metrics = last_metrics

		sampled_for_train = len(train_samples)

		if (update + 1) % GATE_EVERY_UPDATES == 0:
			print(f"[gate] update {update + 1}/{TOTAL_UPDATES} evaluating {GATE_EVAL_GAMES} game(s) | device={infer_device}")
			gate = _evaluate_candidate_vs_guard(policy, guard_policy, infer_device, GATE_EVAL_GAMES)
			gate_pass = bool(gate["winrate"] > GATE_WINRATE_THRESHOLD)
			print(
				f"[gate] update {update + 1}/{TOTAL_UPDATES} result: "
				f"wr={gate['winrate']:.3f} score={gate['scorerate']:.3f} "
				f"({int(gate['candidate_wins'])}W/{int(gate['guard_wins'])}L/{int(gate['draws'])}D) "
				f"threshold={GATE_WINRATE_THRESHOLD:.2f} -> {'PASS' if gate_pass else 'FAIL'}"
			)

			if gate_pass:
				guard_policy.load_state_dict(policy.state_dict(), strict=True)
				guard_policy.eval()
				guard_ckpt_index = next_guard_checkpoint_index
				train_module.save_checkpoint(
					policy=policy,
					optimizer=optimizer,
					title=guard_title,
					checkpoint_index=guard_ckpt_index,
					update_idx=update,
					in_opponent_pool=True,
					reason="gatekeeper_promote",
				)
				next_guard_checkpoint_index += 1
				gate_status = "promoted"
			else:
				# Reset candidate parameters to guard and reset optimizer on gate reject.
				policy.load_state_dict(guard_policy.state_dict(), strict=True)
				optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE_START, weight_decay=WEIGHT_DECAY)
				if not KEEP_REPLAY_ON_REJECT:
					replay_buffer.clear()
					replay_write_idx = 0
				gate_status = "rejected"
		else:
			gate = {"winrate": 0.0, "candidate_wins": 0.0, "guard_wins": 0.0, "draws": 0.0}
			gate_status = "skipped"

		elapsed = time.perf_counter() - update_start

		if (update + 1) % LOG_EVERY == 0 or (update + 1) == TOTAL_UPDATES:
			total_vs_strategy = (
				model_vs_strategy["model"]
				+ model_vs_strategy["strategy"]
				+ model_vs_strategy["draw"]
			)
			explore_str = _format_exploration_stats(explore_acc)
			if total_vs_strategy > 0:
				model_wr = model_vs_strategy["model"] / float(total_vs_strategy)
			else:
				model_wr = 0.0
			# 計算 kind 機率（C/R/P）與合法率（C/R/P）
			if total_kind > 0:
				c = action_kind_mass[0] / total_kind
				r = action_kind_mass[1] / total_kind
				p = action_kind_mass[2] / total_kind
				c_legal = kind_legal_counts[0] / total_kind
				r_legal = kind_legal_counts[1] / total_kind
				p_legal = kind_legal_counts[2] / total_kind
				kind_str = f"C/R/P={c:.2f}/{r:.2f}/{p:.2f}"
				legal_str = f"legal(C/R/P)={c_legal:.2f}/{r_legal:.2f}/{p_legal:.2f}"
			else:
				kind_str = "C/R/P=N/A"
				legal_str = "legal(C/R/P)=N/A"
			if USE_REPLAY_BUFFER:
				reuse_ratio = (sampled_for_train / float(max(fresh_inserted, 1)))
				replay_str = (
					f"replay={len(replay_buffer)}/{REPLAY_BUFFER_MAX_SAMPLES}"
					f" fresh={fresh_inserted} train={sampled_for_train}"
					f" reuse={reuse_ratio:.2f}x"
				)
			else:
				replay_str = f"replay=off fresh={fresh_inserted} train={sampled_for_train}"
			print(
				f"  update {update + 1:6d}/{TOTAL_UPDATES}"
				f" | loss={metrics['loss']:+8.4f}"
				f" p={metrics['policy_loss']:+7.4f}"
				f" v={metrics['value_loss']:+7.4f}"
				f" ent={metrics['entropy']:+7.4f}"
				f" | samples={int(metrics['n_samples'])}"
				f" opt={OPTIMIZATION_PASSES_PER_UPDATE}"
				f" decisions={decision_steps}"
				f" | W/B/D={wins.get(WHITE,0)}/{wins.get(BLACK,0)}/{wins.get(0,0)}"
				f" | M/S/D={model_vs_strategy['model']}/{model_vs_strategy['strategy']}/{model_vs_strategy['draw']}"
				f" model_wr={model_wr:.2f}"
				f" | {explore_str}"
				f" | {kind_str}"
				f" {legal_str}"
				f" | gate={gate_status} wr={gate['winrate']:.2f}"
				f" ({int(gate['candidate_wins'])}/{int(gate['guard_wins'])}/{int(gate['draws'])})"
				f" | {replay_str}"
				f" | lr={lr:.2e}"
				f" | t={elapsed:.1f}s"
			)

		if (update + 1) % CHECKPOINT_EVERY_UPDATES == 0:
			candidate_ckpt_index = next_checkpoint_index
			train_module.save_checkpoint(
				policy=policy,
				optimizer=optimizer,
				title=title,
				checkpoint_index=candidate_ckpt_index,
				update_idx=update,
				in_opponent_pool=False,
				reason="mcts_selfplay",
			)
			next_checkpoint_index += 1

	print("mcts self-play training completed")

	if _snapshot_save_thread is not None and _snapshot_save_thread.is_alive():
		print("[replay] waiting for final snapshot save to complete...")
		_snapshot_save_thread.join()

# main: function implementation.
def main() -> None:
	log_title = train_module.sanitize_checkpoint_title(TITLE)
	log_path = _create_log_file(log_title)
	with log_path.open("a", encoding="utf-8", buffering=65536) as log_file:
		stdout_tee = _TeeTextIO(sys.stdout, log_file)
		stderr_tee = _TeeTextIO(sys.stderr, log_file)
		with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
			print(f"[log] writing training log to {log_path}")
			run()


if __name__ == "__main__":
	main()
