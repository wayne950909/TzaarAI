"""
core/env.py — TzaarEnv 環境封裝（C++ 後端為主，Python 後端相容）

使用 C++ PhaseGameState 作為主要遊戲狀態。
Python 僅負責 GPU 推論、樣本收集、進度輸出。

這個模組的價值在於：
- 對訓練主流程隱藏後端差異
- 讓 loop / selfplay engine 不需要關心目前局面是 Python 還是 C++ 狀態
- 將「觀測」固定成訓練可直接使用的 tensor 形式

當 _ACTIVE_STATE_BACKEND 為 "python" 時，會 fallback 到原本的
Python TzaarGame + TzaarAIInterface 後端。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.constants import WHITE, BLACK, PHASE_STEP_1, PHASE_STEP_2, TRAINING_PHASE_TO_IDX, piece_owner, piece_type
from core.action import N_ACTIONS, PASS_ACTION_IDX
from core.types import GameResult
from core.board import MAX_HEIGHT_NORM
import config


@dataclass
class EnvConfig:
    """環境設定。"""
    max_steps: int = 200


class TzaarEnv:
    """Tzaar 棋類的 gym-like 環境封裝。

    支援 C++ 和 Python 兩種後端。
    - C++ 後端：使用 C++ PhaseGameState 管理所有遊戲邏輯
    - Python 後端：使用 TzaarGame + TzaarAIInterface（向下相容）

    提供 reset / observe / step / game_in_progress 等統一介面。

    未來如果你手動改訓練程式，這個類別就是
    「遊戲邏輯 <-> 訓練樣本」之間最重要的抽象邊界。
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        # C++ 後端專用
        self._cpp_state: Optional[Any] = None  # CppPhaseGameStateAdapter
        # Python 後端專用（向下相容）
        self.game: Optional[Any] = None
        self.ai: Optional[Any] = None
        # 通用
        self.current_player: int = WHITE
        self._current_step: int = 0
        self._game_in_progress: bool = False
        self._last_action: Optional[int] = None
        self._last_game_result: Optional[GameResult] = None

    @property
    def game_in_progress(self) -> bool:
        return self._game_in_progress

    @property
    def last_game_result(self) -> Optional[GameResult]:
        return self._last_game_result

    @property
    def cpp_state(self) -> Optional[Any]:
        """回傳 C++ PhaseGameStateAdapter（若使用 C++ 後端）。"""
        return self._cpp_state

    def reset(self) -> None:
        """重置環境為初始狀態。

        根據 _ACTIVE_STATE_BACKEND 選擇後端：
        - cpp: 建立 C++ PhaseGameState
        - python: 建立 Python TzaarGame + TzaarAIInterface
        """
        if config._ACTIVE_STATE_BACKEND == "cpp" and config._ACTIVE_CPP_MODULE is not None:
            from state import create_cpp_phase_state
            self._cpp_state = create_cpp_phase_state(config._ACTIVE_CPP_MODULE)
            self.game = None
            self.ai = None
            self.current_player = self._cpp_state.current_player()
        else:
            from core.game import TzaarGame
            from TzaarAI import TzaarAIInterface
            self._cpp_state = None
            self.game = TzaarGame()
            self.ai = TzaarAIInterface(self.game)
            self.current_player = WHITE

        self._current_step = 0
        self._game_in_progress = True
        self._last_action = None
        self._last_game_result = None

    def observe(self) -> TzaarObservation:
        """回傳當前狀態的觀測物件。"""
        if config._ACTIVE_STATE_BACKEND == "cpp" and self._cpp_state is not None:
            return self._observe_cpp()
        return self._observe_python()

    def _observe_cpp(self) -> TzaarObservation:
        """使用 C++ 後端建構觀測（透過 leaf_snapshot）。"""
        cpp = self._cpp_state
        if cpp is None:
            raise RuntimeError("Environment not reset (cpp)")

        phase = cpp.phase()
        if phase is None:
            phase = PHASE_STEP_1

        # 使用 C++ PhaseGameState::leaf_snapshot() 一次取得所有特徵
        snap = cpp._inner.leaf_snapshot(0)

        board_tensor = torch.tensor(
            list(snap.board_state_flat), dtype=torch.float32
        ).reshape(12, 9, 9)

        global_features = torch.tensor(
            list(snap.global_features), dtype=torch.float32
        )

        legal_mask = self._build_legal_mask_cpp()

        return TzaarObservation(
            board_tensor=board_tensor,
            global_features=global_features,
            legal_mask=legal_mask,
            current_player=cpp.current_player(),
            phase=phase,
        )

    def _observe_python(self) -> TzaarObservation:
        """使用 Python 後端建構觀測（原本的邏輯）。"""
        game = self.game
        if game is None:
            raise RuntimeError("Environment not reset")

        board_tensor = _build_cnn_state_12x9x9(game, self.current_player)
        phase = self._get_phase()
        global_features = _build_cnn_global_features(game, self.current_player, phase)
        legal_mask = self._build_legal_mask_python()

        return TzaarObservation(
            board_tensor=board_tensor,
            global_features=global_features,
            legal_mask=legal_mask,
            current_player=self.current_player,
            phase=phase,
        )

    def _get_phase(self) -> str:
        """回傳當前階段名稱（Python 後端用）。"""
        game = self.game
        if game is None:
            return PHASE_STEP_1
        if game.is_game_over():
            return PHASE_STEP_1
        if game.is_waiting_second_step():
            return PHASE_STEP_2
        return PHASE_STEP_1

    def _build_legal_mask_cpp(self) -> torch.Tensor:
        """從 C++ 後端取得合法動作遮罩。"""
        cpp = self._cpp_state
        if cpp is None or cpp.is_done():
            return torch.zeros(N_ACTIONS, dtype=torch.bool)
        mask = cpp.legal_mask()
        return mask if mask is not None else torch.zeros(N_ACTIONS, dtype=torch.bool)

    def _build_legal_mask_python(self) -> torch.Tensor:
        """從 Python 後端取得合法動作遮罩。"""
        from TzaarAI import TzaarAIInterface
        game = self.game
        ai = self.ai
        if game is None or ai is None:
            return torch.zeros(N_ACTIONS, dtype=torch.bool)
        if game.is_game_over():
            return torch.zeros(N_ACTIONS, dtype=torch.bool)
        if game.is_waiting_second_step():
            return ai.unified_action_mask_step2()
        if ai.resolve_no_mandatory_capture_if_needed():
            return torch.zeros(N_ACTIONS, dtype=torch.bool)
        return ai.unified_action_mask_step1()

    def step(self, action_idx: int) -> None:
        """執行動作，更新環境狀態。"""
        if config._ACTIVE_STATE_BACKEND == "cpp" and self._cpp_state is not None:
            self._step_cpp(action_idx)
        else:
            self._step_python(action_idx)

    def _step_cpp(self, action_idx: int) -> None:
        """C++ 後端：直接委派給 C++ PhaseGameState。"""
        cpp = self._cpp_state
        if cpp is None:
            raise RuntimeError("Environment not reset")
        if not self._game_in_progress:
            raise RuntimeError("Game is already over")

        cpp.apply_action(action_idx)
        self._last_action = action_idx
        self._current_step += 1

        if cpp.is_done():
            self._game_in_progress = False
            winner = cpp.winner()
            if winner == WHITE:
                self._last_game_result = GameResult.WHITE_WIN
            elif winner == BLACK:
                self._last_game_result = GameResult.BLACK_WIN
            else:
                self._last_game_result = GameResult.DRAW

        self.current_player = cpp.current_player()

    def _step_python(self, action_idx: int) -> None:
        """Python 後端：原本的 step 邏輯。"""
        from core.game import MoveKind

        game = self.game
        ai = self.ai
        if game is None or ai is None:
            raise RuntimeError("Environment not reset")
        if not self._game_in_progress:
            raise RuntimeError("Game is already over")

        phase = self._get_phase()
        if phase == PHASE_STEP_1:
            move = ai.reconstruct_unified_action(action_idx, PHASE_STEP_1)
            game.play_first_step(move)
        else:
            if action_idx == PASS_ACTION_IDX:
                game.play_second_step()
            else:
                move = ai.reconstruct_unified_action(action_idx, PHASE_STEP_2)
                game.play_second_step(move)

        self._last_action = action_idx
        self._current_step += 1

        if game.is_game_over():
            self._game_in_progress = False
            if game.result is not None:
                if game.result.winner == WHITE:
                    self._last_game_result = GameResult.WHITE_WIN
                elif game.result.winner == BLACK:
                    self._last_game_result = GameResult.BLACK_WIN
                else:
                    self._last_game_result = GameResult.DRAW
            self.current_player = game.current_player
        else:
            self.current_player = game.current_player

    def get_legal_moves(self) -> List[int]:
        """回傳合法動作索引列表。"""
        if config._ACTIVE_STATE_BACKEND == "cpp" and self._cpp_state is not None:
            mask = self._build_legal_mask_cpp()
        else:
            mask = self._build_legal_mask_python()
        return torch.nonzero(mask, as_tuple=False).flatten().tolist()


@dataclass
class TzaarObservation:
    """Tzaar 環境的觀測資料結構。"""
    board_tensor: torch.Tensor        # (12, 9, 9)
    global_features: torch.Tensor     # (global_feature_dim,)
    legal_mask: torch.Tensor          # (N_ACTIONS,) bool
    current_player: int               # WHITE=1 / BLACK=-1
    phase: str                        # "step1_capture" / "step2_action"

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """回傳棋盤張量。"""
        if device is not None:
            return self.board_tensor.to(device)
        return self.board_tensor

    def to_global_tensor(self) -> torch.Tensor:
        """回傳全域特徵張量。"""
        return self.global_features

    def to_legal_mask_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """回傳合法動作遮罩。"""
        if device is not None:
            return self.legal_mask.to(device)
        return self.legal_mask


# ──────────────────────────────────────────────────────────────────────
# CNN 特徵建構輔助函式（Python 後端用，與 mcts/state_builder.py 同步）
# ──────────────────────────────────────────────────────────────────────

def _build_cnn_state_12x9x9(game: Any, player: int) -> torch.Tensor:
    """建構 12 通道棋盤特徵張量（Python TzaarGame 用）。

    通道定義：
        0-2: 己方 tzaar/tzarra/tott 佔據（binary）
        3-5: 對方 tzaar/tzarra/tott 佔據（binary）
        6-8: 己方高度（正規化）× 佔據
        9-11: 對方高度（正規化）× 佔據
    """
    import numpy as _np

    p = game.board._piece_np   # (9,9) int8
    h = game.board._height_np  # (9,9) float32

    h_norm = _np.minimum(h, MAX_HEIGHT_NORM) * (1.0 / MAX_HEIGHT_NORM)

    opp = BLACK if player == WHITE else WHITE
    own1 = _np.zeros_like(p, dtype=_np.float32)
    own2 = _np.zeros_like(p, dtype=_np.float32)
    own3 = _np.zeros_like(p, dtype=_np.float32)
    opp1 = _np.zeros_like(p, dtype=_np.float32)
    opp2 = _np.zeros_like(p, dtype=_np.float32)
    opp3 = _np.zeros_like(p, dtype=_np.float32)

    for r in range(9):
        for c in range(9):
            code = p[r, c]
            if code == 0:
                continue
            owner = piece_owner(int(code))
            ptype = piece_type(int(code))
            if owner == player:
                if ptype == 1: own1[r, c] = 1.0
                elif ptype == 2: own2[r, c] = 1.0
                else: own3[r, c] = 1.0
            else:
                if ptype == 1: opp1[r, c] = 1.0
                elif ptype == 2: opp2[r, c] = 1.0
                else: opp3[r, c] = 1.0

    state = _np.stack([
        own1, own2, own3,
        opp1, opp2, opp3,
        h_norm * own1, h_norm * own2, h_norm * own3,
        h_norm * opp1, h_norm * opp2, h_norm * opp3,
    ], axis=0)

    return torch.from_numpy(state)


def _build_cnn_global_features(game: Any, player: int, phase: str) -> torch.Tensor:
    """建構全域特徵張量（Python TzaarGame 用）。

    特徵向量：
        0: 回合數正規化 min(turn,200)/200
        1: 玩家符號 (WHITE=1, BLACK=-1)
        2-4: 己方 tzaar/tzarra/tott 疊數 /15
        5-7: 對方 tzaar/tzarra/tott 疊數 /15
        8: 己方總疊數 /45
        9: 對方總疊數 /45
        10+: 階段 one-hot
    """
    phase_one_hot = [0.0] * len(TRAINING_PHASE_TO_IDX)
    phase_one_hot[TRAINING_PHASE_TO_IDX[phase]] = 1.0

    opp = WHITE if player == BLACK else BLACK
    own_counts = game._counts[player]
    opp_counts = game._counts[opp]

    turn_norm = min(float(game.turn_number), 200.0) / 200.0
    player_sign = 1.0 if player == WHITE else -1.0

    vec = [
        turn_norm,
        player_sign,
        float(own_counts[1]) / 15.0,
        float(own_counts[2]) / 15.0,
        float(own_counts[3]) / 15.0,
        float(opp_counts[1]) / 15.0,
        float(opp_counts[2]) / 15.0,
        float(opp_counts[3]) / 15.0,
        float(sum(own_counts.values())) / 45.0,
        float(sum(opp_counts.values())) / 45.0,
        *phase_one_hot,
    ]
    return torch.tensor(vec, dtype=torch.float32)
