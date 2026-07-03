"""
state/phase_state.py — Python 版 PhaseGameState（階段狀態機）

用於 MCTS 節點管理和自我對弈流程控制。
包裝 TzaarGame + TzaarAIInterface，提供統一的階段化遊戲介面。

當 STATE_BACKEND="cpp" 時，這個類別不會被使用；
取而代之的是 state/cpp_adapter.py 中的 CppPhaseGameStateAdapter。
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, Optional

import torch

from core.constants import PHASE_STEP_1, PHASE_STEP_2
from core.game import TzaarGame
from core.action import PASS_ACTION_IDX
from TzaarAI import N_ACTIONS, TzaarAIInterface


class Stage(Enum):
    """階段狀態機的階段。"""
    NEED_STEP1 = auto()
    NEED_STEP2 = auto()
    DONE = auto()


STAGE_TO_PHASE: Dict[Stage, str] = {
    Stage.NEED_STEP1: PHASE_STEP_1,
    Stage.NEED_STEP2: PHASE_STEP_2,
}


class PythonPhaseGameState:
    """Python 版階段狀態機，包裝 TzaarGame + TzaarAIInterface。

    提供 clone / is_done / winner / current_player / phase /
    legal_mask / apply_action 等統一介面，與 CppPhaseGameStateAdapter 對齊。
    """

    def __init__(
        self,
        game: Optional[TzaarGame] = None,
        stage: Stage = Stage.NEED_STEP1,
        ai: Optional[object] = None,
    ) -> None:
        """初始化階段狀態機。

        參數
        ----
        game : 可傳入現有對局（用於 clone）
        stage : 起始階段
        ai : TzaarAIInterface 實例（若無則自動從 game 建立）
        """
        self.game = game or TzaarGame()
        self.ai = ai or TzaarAIInterface(self.game)
        self.stage = stage
        self._cached_mask_stage: Optional[Stage] = None
        self._cached_mask: Optional[torch.Tensor] = None

    def clone(self) -> PythonPhaseGameState:
        """深拷貝狀態。"""
        copied = PythonPhaseGameState(self.game.clone(), self.stage)
        copied._cached_mask_stage = self._cached_mask_stage
        copied._cached_mask = (
            None if self._cached_mask is None else self._cached_mask.clone()
        )
        return copied

    def _invalidate_cache(self) -> None:
        self._cached_mask_stage = None
        self._cached_mask = None

    def _advance_from_first_src(self) -> None:
        """當轉為 NEED_STEP1 時，立即檢查無步可吃的情況。"""
        if self.stage == Stage.NEED_STEP1:
            if self.ai.resolve_no_mandatory_capture_if_needed():
                self.stage = Stage.DONE

    def is_done(self) -> bool:
        """對局是否已結束。"""
        return self.stage == Stage.DONE or self.game.is_game_over()

    def winner(self) -> Optional[int]:
        """回傳勝者（WHITE=1 / BLACK=-1），無結果時為 None。"""
        return self.game.result.winner if self.game.result else None

    def current_player(self) -> int:
        """當前輪到的玩家。"""
        return int(self.game.current_player)

    def phase(self) -> Optional[str]:
        """回傳當前階段名稱（step1_capture / step2_action），終局回傳 None。"""
        if self.is_done():
            return None
        return STAGE_TO_PHASE[self.stage]

    def legal_mask(self) -> Optional[torch.Tensor]:
        """回傳 601 維合法動作遮罩。終局回傳 None。"""
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

    def apply_action(
        self,
        action_idx: int,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """套用統一動作索引（0..600），更新狀態。"""
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
