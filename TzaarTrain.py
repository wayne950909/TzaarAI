"""TzaarTrain.py — FC 策略網路自我對戰 REINFORCE 訓練

架構
----
* 一個共享的 PolicyNet（雙方都用同一組權重）
* FC backbone + 三個決策頭：kind(3)、source(60)、direction(6)
* mask 過濾非法動作後做 Categorical 採樣
* 終局 ±1 獎勵 + gamma 折扣回報
* 每 GAMES_PER_UPDATE 局蒐集完後一次 Adam 更新
* 同 stage 的多局 state 一次批次推論，減少 forward 次數

建議閱讀順序
------------
1. train()：先看一個 update 怎麼跑完
2. collect_batch()：再看 rollout 怎麼蒐集
3. compute_loss() / compute_returns()：最後看學習訊號怎麼算
4. _transition()：需要時才下探單步狀態機細節
5. _ActiveGame：需要查 shaping 或局內暫存時再看

使用方式
--------
    python TzaarTrain.py               # 啟動後輸入標題；同標題自動接續訓練
    python TzaarTrain.py --updates 500 # 指定更新次數
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from policy_net_cnn_min17 import PolicyNetCNNMin17
from Tzaar import BOARD_LAYOUT, Move, MoveKind, TzaarGame
from TzaarAI import (
    KIND_ORDER,
    N_DIRECTIONS,
    N_KINDS,
    N_POSITIONS,
    PHASE_FIRST_CAPTURE_DIRECTION,
    PHASE_FIRST_CAPTURE_SOURCE,
    PHASE_SECOND_CAPTURE_DIRECTION,
    PHASE_SECOND_CAPTURE_SOURCE,
    PHASE_SECOND_KIND,
    PHASE_SECOND_REINFORCE_DIRECTION,
    PHASE_SECOND_REINFORCE_SOURCE,
    TRAINING_STATE_DIM,
    TzaarAIInterface,
)

# ══════════════════════════════════════════════════════════════════════
# 超參數
# ══════════════════════════════════════════════════════════════════════

HIDDEN_SIZE_1     = 1024      # 第一層隱藏維度
HIDDEN_SIZE_2     = 768      # 第二層隱藏維度
LEARNING_RATE_START = 0.0003   # Adam 學習率起點
LEARNING_RATE_END   = 0.0001   # Adam 學習率終點
GAMMA             = 0.995     # 折扣因子
GAMES_PER_UPDATE  = 64       # 每次 optimizer step 蒐集的局數
DEFAULT_UPDATES   = 100     # 預設總更新次數
ENTROPY_BETA_START = 0.05    # 熵鼓勵係數起點（避免策略過早收斂）
ENTROPY_BETA_END   = 0.05    # 熵鼓勵係數終點
VALUE_LOSS_COEFF  = 0.5     # 價值函數損失權重係數
GRAD_CLIP         = 1.0      # 梯度裁剪上限
SHAPING_REWARD_SCALE_START = 1.0  # 非終局 reward（shaping）縮放係數起點
SHAPING_REWARD_SCALE_END   = 1.0  # 非終局 reward（shaping）縮放係數終點
CHECKPOINT_EVERY  = 100      # 每 N 次更新儲存一次 checkpoint
CHECKPOINT_DIR    = "checkpoints"
DEVICE            = "cpu"  # auto / cpu / cuda / cuda:N

# 單一 shaping reward（訓練方視角）
# k = 三種棋子數量最小值；對手 k 下降加分，自己 k 下降扣分。
REWARD_K_DELTA_SCALE = 0.2

# 訓練方法相關超參數
CHECKPOINT_INDEX_WIDTH = 6
RECENT_OPPONENT_PROB = 1.0
EARLY_OPPONENT_PROB = 0.0
RECENT_POOL_SIZE = 4
PROMOTION_CHECK_EVERY = 100
PROMOTION_WINRATE_THRESHOLD = 1.0
PROMOTION_MIN_UPDATES = 100

# 固定 reward 模式：True 表示只使用終局勝負，False 表示使用 shaping + 終局勝負
TERMINAL_ONLY = True

# 模型架構切換：True=CNN(min17), False=FC
USE_CNN_POLICY = True

ARCH_FC = "fc"
ARCH_CNN_MIN17 = "cnn_min17"
CNN_GLOBAL_FEATURE_DIM = 17
CNN_DROPOUT = 0.10

STATE_INDEX_TO_BOARD_COORDS: Tuple[Tuple[int, int], ...] = tuple(
    (r, c)
    for r in range(len(BOARD_LAYOUT))
    for c in range(len(BOARD_LAYOUT[0]))
    if BOARD_LAYOUT[r][c] != -1
)
if len(STATE_INDEX_TO_BOARD_COORDS) != N_POSITIONS:
    raise ValueError(
        f"State coordinate mapping mismatch: expected {N_POSITIONS}, got {len(STATE_INDEX_TO_BOARD_COORDS)}"
    )



_LEGACY_CHECKPOINT_RE = re.compile(r"^ckpt_(?P<index>\d+)\.pt$")
_TITLED_CHECKPOINT_RE = re.compile(r"^(?P<title>.+)_(?P<index>\d+)\.pt$")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def _format_duration(seconds: float) -> str:
    """把秒數格式化為 HH:MM:SS。"""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_device(requested: str) -> str:
    """把使用者指定 device 轉成實際可用裝置字串。"""

    device = requested.strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is False")
        return device
    raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")


def linear_schedule_value(
    update_idx: int,
    total_updates: int,
    start_value: float,
    end_value: float,
) -> float:
    """依 update 進度回傳線性插值後的係數值。"""
    if total_updates <= 1:
        return float(start_value)

    clamped_update = min(max(update_idx, 0), total_updates - 1)
    progress = clamped_update / float(total_updates - 1)
    return float(start_value + (end_value - start_value) * progress)


@dataclass(frozen=True)
class CheckpointInfo:
    """描述一個可解析的 checkpoint 檔案。"""

    path: Path
    title: Optional[str]
    index: int


@dataclass(frozen=True)
class OpponentSelection:
    """描述本局對手來源與對應 checkpoint。"""

    checkpoint: CheckpointInfo
    source: str


@dataclass
class PromotionWindow:
    """集中管理 promotion 視窗的累積勝率。"""

    wins: int = 0
    games: int = 0

    def observe_batch(self, batch_wins: int, batch_games: int) -> None:
        self.wins += batch_wins
        self.games += batch_games

    def snapshot(self) -> Tuple[int, int, float]:
        win_rate = self.wins / self.games if self.games > 0 else 0.0
        return self.wins, self.games, win_rate

    def should_check(self, update: int) -> bool:
        return (update + 1) % PROMOTION_CHECK_EVERY == 0

    def should_promote(self, update: int) -> bool:
        _, _, win_rate = self.snapshot()
        return (
            (update + 1) >= PROMOTION_MIN_UPDATES
            and self.games > 0
            and win_rate >= PROMOTION_WINRATE_THRESHOLD
        )

    def reset(self) -> None:
        self.wins = 0
        self.games = 0


def sanitize_checkpoint_title(raw_title: str) -> str:
    """將使用者輸入的標題轉為安全檔名。"""
    title = re.sub(r"\s+", "_", raw_title.strip())
    title = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", title)
    title = re.sub(r"_+", "_", title).strip(" ._")

    if not title:
        raise ValueError("標題不能為空，也不能只包含非法字元。")
    if title.upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("標題不能是 Windows 保留名稱。")
    return title


def prompt_checkpoint_title() -> str:
    """互動式讀取模型標題，直到輸入合法為止。"""
    while True:
        raw_title = input("請輸入這次訓練的模型標題: ").strip()
        try:
            title = sanitize_checkpoint_title(raw_title)
        except ValueError as exc:
            print(f"[title] {exc}")
            continue

        if title != raw_title:
            print(f"[title] 會使用安全檔名：{title}")
        return title


def _checkpoint_dirs() -> List[Path]:
    """回傳 checkpoint 可能所在的搜尋路徑。"""
    dirs: List[Path] = []
    for directory in (Path.cwd() / CHECKPOINT_DIR, Path(__file__).resolve().parent / CHECKPOINT_DIR):
        resolved = directory.resolve()
        if resolved not in dirs:
            dirs.append(resolved)
    return dirs


def _parse_checkpoint_filename(filename: str) -> Optional[Tuple[Optional[str], int]]:
    """解析 checkpoint 檔名；回傳 (title, index)。舊格式 title 會是 None。"""
    match = _LEGACY_CHECKPOINT_RE.fullmatch(filename)
    if match is not None:
        return None, int(match.group("index"))

    match = _TITLED_CHECKPOINT_RE.fullmatch(filename)
    if match is not None:
        return match.group("title"), int(match.group("index"))

    return None


def list_checkpoints(title: Optional[str] = None) -> List[CheckpointInfo]:
    """列出所有可識別的 checkpoint；若指定 title，僅回傳該標題。"""
    checkpoints: List[CheckpointInfo] = []
    seen_paths: set[Path] = set()

    for directory in _checkpoint_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pt")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue

            parsed = _parse_checkpoint_filename(path.name)
            if parsed is None:
                continue

            parsed_title, parsed_index = parsed
            if title is not None and parsed_title != title:
                continue

            checkpoints.append(
                CheckpointInfo(
                    path=resolved,
                    title=parsed_title,
                    index=parsed_index,
                )
            )
            seen_paths.add(resolved)

    return checkpoints


def resolve_checkpoint_path(checkpoint: str) -> Path:
    """解析 checkpoint 路徑，支援顯式路徑與 latest。"""
    if checkpoint == "latest":
        checkpoints = list_checkpoints()
        if not checkpoints:
            raise FileNotFoundError("No compatible checkpoint *.pt found in checkpoints/")
        return max(checkpoints, key=lambda item: item.path.stat().st_mtime).path

    raw = Path(checkpoint)
    path_candidates = [
        raw,
        Path.cwd() / raw,
        Path(__file__).resolve().parent / raw,
        Path.cwd() / CHECKPOINT_DIR / raw,
        Path(__file__).resolve().parent / CHECKPOINT_DIR / raw,
    ]
    for path in path_candidates:
        if path.exists() and path.is_file():
            return path.resolve()

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def _checkpoint_is_pool_member(path: Path, device: str = "cpu") -> bool:
    """判斷 checkpoint 是否標記為對手池成員。"""
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    return bool(ckpt.get("in_opponent_pool", True))


def list_pool_checkpoints(title: str) -> List[CheckpointInfo]:
    """列出同 title 中屬於對手池的 checkpoint。"""
    pool_members: List[CheckpointInfo] = []
    for ckpt in sorted(list_checkpoints(title=title), key=lambda item: item.index):
        if _checkpoint_is_pool_member(ckpt.path):
            pool_members.append(ckpt)
    return pool_members


def validate_opponent_probs(recent_prob: float, early_prob: float) -> float:
    """驗證對手抽樣機率，並回傳常駐對手機率。"""
    if recent_prob < 0.0 or early_prob < 0.0:
        raise ValueError("RECENT_OPPONENT_PROB and EARLY_OPPONENT_PROB must be >= 0")
    resident_prob = 1.0 - recent_prob - early_prob
    if resident_prob < 0.0:
        raise ValueError("RECENT_OPPONENT_PROB + EARLY_OPPONENT_PROB must be <= 1")
    return resident_prob


def load_policy_from_checkpoint(path: Path, device: str) -> nn.Module:
    """由 checkpoint 建立凍結的對手網路。"""
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    architecture = str(ckpt.get("architecture", ARCH_FC))

    if architecture == ARCH_CNN_MIN17:
        global_feature_dim = int(ckpt.get("global_feature_dim", CNN_GLOBAL_FEATURE_DIM))
        dropout = float(ckpt.get("dropout", CNN_DROPOUT))
        net = PolicyNetCNNMin17(global_feature_dim=global_feature_dim, dropout=dropout).to(device)
    else:
        hidden1 = int(ckpt.get("hidden_size_1", HIDDEN_SIZE_1))
        hidden2 = int(ckpt.get("hidden_size_2", HIDDEN_SIZE_2))
        net = PolicyNet(hidden1=hidden1, hidden2=hidden2).to(device)

    net.load_state_dict(ckpt["policy_state"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


class OpponentPool:
    """依新舊排序管理對手池，並提供常駐/近期/早期抽樣。"""

    def __init__(
        self,
        title: str,
        device: str,
        recent_prob: float,
        early_prob: float,
        recent_pool_size: int,
    ) -> None:
        if recent_pool_size <= 0:
            raise ValueError("RECENT_POOL_SIZE must be >= 1")
        self.title = title
        self.device = device
        self.recent_prob = recent_prob
        self.early_prob = early_prob
        self.recent_pool_size = recent_pool_size
        self.resident_prob = validate_opponent_probs(recent_prob, early_prob)
        self._cache: Dict[Path, nn.Module] = {}
        self._pool: List[CheckpointInfo] = []
        self.refresh()

    def refresh(self) -> None:
        """重新掃描對手池並保持索引排序。"""
        self._pool = list_pool_checkpoints(self.title)
        valid_paths = {item.path for item in self._pool}
        self._cache = {path: net for path, net in self._cache.items() if path in valid_paths}

    def __len__(self) -> int:
        return len(self._pool)

    @property
    def resident(self) -> CheckpointInfo:
        if not self._pool:
            raise ValueError("OpponentPool is empty")
        return self._pool[-1]

    def _recent_candidates(self) -> List[CheckpointInfo]:
        if len(self._pool) <= 1:
            return []
        pool_wo_resident = self._pool[:-1]
        return pool_wo_resident[-self.recent_pool_size :]

    def _early_candidates(self) -> List[CheckpointInfo]:
        if len(self._pool) <= 1:
            return []
        pool_wo_resident = self._pool[:-1]
        if len(pool_wo_resident) <= self.recent_pool_size:
            return []
        return pool_wo_resident[: -self.recent_pool_size]

    def select_checkpoint(self) -> OpponentSelection:
        """按常駐/近期/早期機率抽樣對手；分區不足時自動降級。"""
        resident = self.resident
        recent_candidates = self._recent_candidates()
        early_candidates = self._early_candidates()
        roll = random.random()

        if roll < self.resident_prob:
            return OpponentSelection(checkpoint=resident, source="resident")

        if roll < self.resident_prob + self.recent_prob:
            if recent_candidates:
                return OpponentSelection(checkpoint=random.choice(recent_candidates), source="recent")
            return OpponentSelection(checkpoint=resident, source="resident")

        if early_candidates:
            return OpponentSelection(checkpoint=random.choice(early_candidates), source="early")
        if recent_candidates:
            return OpponentSelection(checkpoint=random.choice(recent_candidates), source="recent")
        return OpponentSelection(checkpoint=resident, source="resident")

    def get_policy(self, checkpoint: CheckpointInfo) -> nn.Module:
        """回傳指定 checkpoint 對應的凍結對手網路（含快取）。"""
        cached = self._cache.get(checkpoint.path)
        if cached is not None:
            return cached
        policy = load_policy_from_checkpoint(checkpoint.path, self.device)
        self._cache[checkpoint.path] = policy
        return policy

# ══════════════════════════════════════════════════════════════════════
# 策略網路
# ══════════════════════════════════════════════════════════════════════

class PolicyNet(nn.Module):
    """
    FC 策略網路。

    輸入
    ----
    x : Tensor of shape (batch, 489)
        encode_training_state 產生的狀態向量（含 phase one-hot）。

    輸出
    ----
    dict with keys:
        'kind'      : (batch, 3)   第二步種類 logit
        'source'    : (batch, 60)  來源格 logit
        'direction' : (batch, 6)   方向 logit
    """

    def __init__(
        self,
        state_dim: int = TRAINING_STATE_DIM,
        hidden1: int = HIDDEN_SIZE_1,
        hidden2: int = HIDDEN_SIZE_2,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.kind_head      = nn.Linear(hidden2, N_KINDS)
        self.source_head    = nn.Linear(hidden2, N_POSITIONS)
        self.direction_head = nn.Linear(hidden2, N_DIRECTIONS)
        self.value_head     = nn.Linear(hidden2, 1)    # Value function head

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """只計算共享 backbone 表徵。"""
        return self.backbone(x)

    def head_logits(self, hidden: torch.Tensor, head: str) -> torch.Tensor:
        """由 backbone 表徵計算指定 head 的 logit。"""
        if head == "kind":
            return self.kind_head(hidden)
        if head == "source":
            return self.source_head(hidden)
        if head == "direction":
            return self.direction_head(hidden)
        raise ValueError(f"Unknown head: {head}")

    def forward_value(self, hidden: torch.Tensor) -> torch.Tensor:
        """由 backbone 表徵計算價值預測 V(s)。
        
        參數
        ----
        hidden : Tensor of shape (batch, hidden_size)
            backbone 輸出的隱藏表徵
        
        回傳
        ----
        Tensor of shape (batch, 1)
            每個狀態的價值預測
        """
        return self.value_head(hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """相容介面：一次輸出三個 head（供舊呼叫端）。"""
        h = self.encode(x)
        return {
            "kind":      self.kind_head(h),
            "source":    self.source_head(h),
            "direction": self.direction_head(h),
        }


def _policy_arch_from_instance(policy: nn.Module) -> str:
    if isinstance(policy, PolicyNetCNNMin17):
        return ARCH_CNN_MIN17
    if isinstance(policy, PolicyNet):
        return ARCH_FC
    raise TypeError(f"Unsupported policy type: {type(policy).__name__}")


def _decode_training_state_batch_for_cnn(states_489: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 (B, 489) 訓練狀態解碼成 CNN 所需的 (B,12,9,9) 與 (B,17)。"""
    if states_489.dim() != 2 or states_489.size(1) != TRAINING_STATE_DIM:
        raise ValueError(f"Expected states shape (B, {TRAINING_STATE_DIM}), got {tuple(states_489.shape)}")

    batch_size = states_489.size(0)
    device = states_489.device
    dtype = states_489.dtype

    turn_number = states_489[:, 0]
    current_player = states_489[:, 1]

    color = states_489[:, 2:182].view(batch_size, N_POSITIONS, 3)
    ptype = states_489[:, 182:422].view(batch_size, N_POSITIONS, 4)
    height = states_489[:, 422:482]
    phase = states_489[:, 482:489]

    is_black = color[:, :, 0] > 0.5
    is_white = color[:, :, 1] > 0.5

    piece_type_idx = ptype[:, :, :3].argmax(dim=-1)
    has_piece = ptype[:, :, 3] < 0.5

    idx_black = (is_black & has_piece).unsqueeze(-1)
    idx_white = (is_white & has_piece).unsqueeze(-1)
    idx_t0 = piece_type_idx == 0
    idx_t1 = piece_type_idx == 1
    idx_t2 = piece_type_idx == 2

    black_t1 = (idx_black.squeeze(-1) & idx_t0).to(dtype)
    black_t2 = (idx_black.squeeze(-1) & idx_t1).to(dtype)
    black_t3 = (idx_black.squeeze(-1) & idx_t2).to(dtype)
    white_t1 = (idx_white.squeeze(-1) & idx_t0).to(dtype)
    white_t2 = (idx_white.squeeze(-1) & idx_t1).to(dtype)
    white_t3 = (idx_white.squeeze(-1) & idx_t2).to(dtype)

    player_is_white = current_player > 0
    own_t1 = torch.where(player_is_white.unsqueeze(1), white_t1, black_t1)
    own_t2 = torch.where(player_is_white.unsqueeze(1), white_t2, black_t2)
    own_t3 = torch.where(player_is_white.unsqueeze(1), white_t3, black_t3)
    opp_t1 = torch.where(player_is_white.unsqueeze(1), black_t1, white_t1)
    opp_t2 = torch.where(player_is_white.unsqueeze(1), black_t2, white_t2)
    opp_t3 = torch.where(player_is_white.unsqueeze(1), black_t3, white_t3)

    h_norm = torch.clamp(height, min=0.0, max=8.0) / 8.0
    own_h_t1 = h_norm * own_t1
    own_h_t2 = h_norm * own_t2
    own_h_t3 = h_norm * own_t3
    opp_h_t1 = h_norm * opp_t1
    opp_h_t2 = h_norm * opp_t2
    opp_h_t3 = h_norm * opp_t3

    board = torch.zeros((batch_size, 12, 9, 9), dtype=dtype, device=device)
    rows = torch.tensor([pos[0] for pos in STATE_INDEX_TO_BOARD_COORDS], device=device, dtype=torch.long)
    cols = torch.tensor([pos[1] for pos in STATE_INDEX_TO_BOARD_COORDS], device=device, dtype=torch.long)

    channels = [
        own_t1,
        own_t2,
        own_t3,
        opp_t1,
        opp_t2,
        opp_t3,
        own_h_t1,
        own_h_t2,
        own_h_t3,
        opp_h_t1,
        opp_h_t2,
        opp_h_t3,
    ]
    for channel_idx, values in enumerate(channels):
        board[:, channel_idx, rows, cols] = values

    own_count_t1 = own_t1.sum(dim=1)
    own_count_t2 = own_t2.sum(dim=1)
    own_count_t3 = own_t3.sum(dim=1)
    opp_count_t1 = opp_t1.sum(dim=1)
    opp_count_t2 = opp_t2.sum(dim=1)
    opp_count_t3 = opp_t3.sum(dim=1)
    own_total = own_count_t1 + own_count_t2 + own_count_t3
    opp_total = opp_count_t1 + opp_count_t2 + opp_count_t3

    turn_norm = torch.clamp(turn_number, max=200.0) / 200.0
    player_sign = torch.where(player_is_white, torch.ones_like(turn_norm), -torch.ones_like(turn_norm))

    global_features = torch.cat(
        [
            turn_norm.unsqueeze(1),
            player_sign.unsqueeze(1),
            (own_count_t1 / 15.0).unsqueeze(1),
            (own_count_t2 / 15.0).unsqueeze(1),
            (own_count_t3 / 15.0).unsqueeze(1),
            (opp_count_t1 / 15.0).unsqueeze(1),
            (opp_count_t2 / 15.0).unsqueeze(1),
            (opp_count_t3 / 15.0).unsqueeze(1),
            (own_total / 45.0).unsqueeze(1),
            (opp_total / 45.0).unsqueeze(1),
            phase,
        ],
        dim=1,
    )

    return board, global_features


def _encode_hidden(policy: nn.Module, states_489: torch.Tensor) -> torch.Tensor:
    """統一 FC/CNN 的 backbone encode 呼叫。"""
    if isinstance(policy, PolicyNetCNNMin17):
        board, global_features = _decode_training_state_batch_for_cnn(states_489)
        return policy.encode(board, global_features)
    if isinstance(policy, PolicyNet):
        return policy.encode(states_489)
    raise TypeError(f"Unsupported policy type: {type(policy).__name__}")


# ══════════════════════════════════════════════════════════════════════
# Mask 採樣工具
# ══════════════════════════════════════════════════════════════════════

def masked_sample(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    在 mask=True 的合法位置採樣一個動作。

    參數
    ----
    logits : 1D Tensor，長度為動作空間大小
    mask   : 同長度的 bool Tensor

    回傳
    ----
    (action_idx, log_prob, entropy)
        action_idx : int，被選中的動作索引
        log_prob   : scalar Tensor，帶梯度
        entropy    : scalar Tensor，帶梯度（用於熵獎勵）

    例外
    ----
    ValueError : 若 mask 全為 0（無合法動作）
    """
    if mask.sum() == 0:
        raise ValueError("masked_sample: 所有動作均被遮蔽，遊戲狀態異常")

    masked_logits = logits.masked_fill(~mask, -1e9)
    dist = torch.distributions.Categorical(logits=masked_logits)
    action = dist.sample()
    return action.item(), dist.log_prob(action), dist.entropy()


# ══════════════════════════════════════════════════════════════════════
# 遊戲內部階段狀態機
# ══════════════════════════════════════════════════════════════════════

class _Stage(Enum):
    """一局遊戲中每個決策點的狀態。"""
    NEED_FIRST_SRC       = auto()
    NEED_FIRST_DIR       = auto()
    NEED_SECOND_KIND     = auto()
    NEED_SECOND_CAP_SRC  = auto()
    NEED_SECOND_CAP_DIR  = auto()
    NEED_SECOND_REIN_SRC = auto()
    NEED_SECOND_REIN_DIR = auto()
    DONE                 = auto()


_STAGE_TO_PHASE: Dict[_Stage, str] = {
    _Stage.NEED_FIRST_SRC:       PHASE_FIRST_CAPTURE_SOURCE,
    _Stage.NEED_FIRST_DIR:       PHASE_FIRST_CAPTURE_DIRECTION,
    _Stage.NEED_SECOND_KIND:     PHASE_SECOND_KIND,
    _Stage.NEED_SECOND_CAP_SRC:  PHASE_SECOND_CAPTURE_SOURCE,
    _Stage.NEED_SECOND_CAP_DIR:  PHASE_SECOND_CAPTURE_DIRECTION,
    _Stage.NEED_SECOND_REIN_SRC: PHASE_SECOND_REINFORCE_SOURCE,
    _Stage.NEED_SECOND_REIN_DIR: PHASE_SECOND_REINFORCE_DIRECTION,
}

_STAGE_TO_HEAD: Dict[_Stage, str] = {
    _Stage.NEED_FIRST_SRC: "source",
    _Stage.NEED_FIRST_DIR: "direction",
    _Stage.NEED_SECOND_KIND: "kind",
    _Stage.NEED_SECOND_CAP_SRC: "source",
    _Stage.NEED_SECOND_CAP_DIR: "direction",
    _Stage.NEED_SECOND_REIN_SRC: "source",
    _Stage.NEED_SECOND_REIN_DIR: "direction",
}

# ══════════════════════════════════════════════════════════════════════
# 單局遊戲包裝器
# ══════════════════════════════════════════════════════════════════════

# 每個決策步驟的完整記錄：
#   log_prob : 所選動作的 log 機率（有梯度，用於損失）
#   entropy  : 當前分佈的熵（有梯度，用於熵獎勵）
#   player   : 做決策的玩家（WHITE=1 / BLACK=-1）
#   state    : 決策時的狀態向量（用於價值函數訓練）
_StepRecord = Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]
_ActionRecord = Tuple[_Stage, int]

# ══════════════════════════════════════════════════════════════════════
# reward
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class _ShapingSnapshot:
    """單一步落子前的 shaping 快照（由訓練層持有）。"""

    move_kind: MoveKind
    actor: int
    actor_k_before: int
    actor_k_after: int
    opponent_k_before: int
    opponent_k_after: int


def _player_k(piece_counts: Dict[int, Dict[int, int]], player: int) -> int:
    """計算指定玩家 k 值（3 種棋子數量最小值）。"""
    player_counts = piece_counts[player]
    return min(player_counts[1], player_counts[2], player_counts[3])


def _capture_shaping_snapshot(
    ai: TzaarAIInterface,
    move: Optional[Move] = None,
    *,
    actor: Optional[int] = None,
    counts_before: Optional[Dict[int, Dict[int, int]]] = None,
    counts_after: Optional[Dict[int, Dict[int, int]]] = None,
) -> _ShapingSnapshot:
    """擷取落子 k 變化快照；move=None 時視為 PASS（k 不變）。"""
    game = ai.game
    actor_player = game.current_player if actor is None else actor

    if counts_before is None:
        counts_before = game.get_piece_counts()
    if counts_after is None:
        counts_after = counts_before

    if move is None or move.kind == MoveKind.PASS:
        return _ShapingSnapshot(
            move_kind=MoveKind.PASS,
            actor=actor_player,
            actor_k_before=_player_k(counts_before, actor_player),
            actor_k_after=_player_k(counts_after, actor_player),
            opponent_k_before=_player_k(counts_before, -actor_player),
            opponent_k_after=_player_k(counts_after, -actor_player),
        )

    if move.src is None or move.dst is None:
        raise ValueError("Non-pass move must include src/dst")

    return _ShapingSnapshot(
        move_kind=move.kind,
        actor=actor_player,
        actor_k_before=_player_k(counts_before, actor_player),
        actor_k_after=_player_k(counts_after, actor_player),
        opponent_k_before=_player_k(counts_before, -actor_player),
        opponent_k_after=_player_k(counts_after, -actor_player),
    )


def _compute_step_shaping_from_snapshot(
    snapshot: _ShapingSnapshot,
    trainable_player: int,
) -> float:
    """計算單步 shaping（trainable 視角，單一 k-delta 方案）。"""
    actor = snapshot.actor
    if actor == trainable_player:
        my_k_before = snapshot.actor_k_before
        my_k_after = snapshot.actor_k_after
        opp_k_before = snapshot.opponent_k_before
        opp_k_after = snapshot.opponent_k_after
    else:
        my_k_before = snapshot.opponent_k_before
        my_k_after = snapshot.opponent_k_after
        opp_k_before = snapshot.actor_k_before
        opp_k_after = snapshot.actor_k_after

    opp_k_drop = max(0, opp_k_before - opp_k_after)
    my_k_drop = max(0, my_k_before - my_k_after)
    return REWARD_K_DELTA_SCALE * float(opp_k_drop - my_k_drop)

#測試用的 trace_training_perspective
def _compute_step_shaping_breakdown_from_snapshot(
    snapshot: _ShapingSnapshot,
    trainable_player: int,
) -> Dict[str, object]:
    """回傳單一 k-delta reward 方案的 shaping 分解資訊。"""
    actor = snapshot.actor
    if actor == trainable_player:
        my_k_before = snapshot.actor_k_before
        my_k_after = snapshot.actor_k_after
        opp_k_before = snapshot.opponent_k_before
        opp_k_after = snapshot.opponent_k_after
    else:
        my_k_before = snapshot.opponent_k_before
        my_k_after = snapshot.opponent_k_after
        opp_k_before = snapshot.actor_k_before
        opp_k_after = snapshot.actor_k_after

    opp_k_drop = max(0, opp_k_before - opp_k_after)
    my_k_drop = max(0, my_k_before - my_k_after)
    total = REWARD_K_DELTA_SCALE * float(opp_k_drop - my_k_drop)
    assignment = "immediate" if actor == trainable_player else "delayed_backfill"

    return {
        "move_kind": snapshot.move_kind.value,
        "actor": actor,
        "k_scale": REWARD_K_DELTA_SCALE,
        "my_k_before": my_k_before,
        "my_k_after": my_k_after,
        "opp_k_before": opp_k_before,
        "opp_k_after": opp_k_after,
        "my_k_drop": my_k_drop,
        "opp_k_drop": opp_k_drop,
        "assignment": assignment,
        "total_shaping": total,
    }


class _ActiveGame:
    """包裝一局進行中遊戲的可變狀態與蒐集資料。"""

    __slots__ = (
        "game", "ai", "device",
        "stage", "steps", "trainable_actions",
        "trainable_player",
        "opponent_policy",
        "opponent_source",
        "opponent_checkpoint_index",
        "_state_buffer",
        "_valid_positions",
        "_first_src",
        "_second_kind",
        "_second_src",
        "step_shapings",                 # List[float] 每個 step 的累計 shaping
        "pending_second_step_idx",       # Optional[int] 延遲負分回填目標索引（保留舊欄位名）
        "pending_opening_first_step_idx", # Optional[int] 相容保留，不再主動使用
        "pending_shaping_snapshot",      # Optional[_ShapingSnapshot] 相容保留，不再主動使用
        "pending_opponent_penalty",      # float 對手回合累積的訓練方負分
    )

    def __init__(
        self,
        trainable_player: int,
        opponent_policy: nn.Module,
        opponent_source: str,
        opponent_checkpoint_index: int,
        device: Optional[str] = None,
    ) -> None:
        self.game   = TzaarGame()
        self.ai     = TzaarAIInterface(self.game)
        self.device = device
        self.stage: _Stage              = _Stage.NEED_FIRST_SRC
        self.steps: List[_StepRecord]   = []
        self.trainable_actions: List[_ActionRecord] = []
        self.trainable_player = trainable_player
        self.opponent_policy = opponent_policy
        self.opponent_source = opponent_source
        self.opponent_checkpoint_index = opponent_checkpoint_index
        self._state_buffer: Optional[torch.Tensor]              = None
        self._valid_positions: Optional[List[Tuple[int, int]]] = None
        self._first_src: Optional[Tuple[int, int]]             = None
        self._second_kind: Optional[MoveKind]                   = None
        self._second_src:  Optional[Tuple[int, int]]            = None
        self.step_shapings: List[float]                         = []
        self.pending_second_step_idx: Optional[int]             = None
        self.pending_opening_first_step_idx: Optional[int]      = None
        self.pending_shaping_snapshot: Optional[_ShapingSnapshot] = None
        self.pending_opponent_penalty: float                    = 0.0

    def valid_positions(self) -> List[Tuple[int, int]]:
        """取得棋盤所有有效位置（固定順序，只計算一次）。"""
        if self._valid_positions is None:
            self._valid_positions = list(self.game.board.iter_valid_positions())
        return self._valid_positions

    def encode(self, phase: str) -> torch.Tensor:
        """依指定 phase 編碼當前狀態向量。
        
        第一次調用時完整初始化 buffer，之後只更新 phase one-hot
        （turn/player 和棋盤位置已通過 patch_state_after_move 同步）。
        """
        if self._state_buffer is None:
            # 第一次：完整初始化
            self._state_buffer = torch.empty(TRAINING_STATE_DIM, dtype=torch.float32, device=self.device)
            result = self.ai.encode_training_state(phase, device=self.device, out=self._state_buffer)
            return result
        else:
            # 後續調用：只更新 phase one-hot（turn/player 和棋盤位置已由 patch 同步）
            self.ai.update_state_phase(self._state_buffer, phase)
            return self._state_buffer

    def is_done(self) -> bool:
        """遊戲是否已結束（stage=DONE 或 game.is_game_over()）。"""
        return self.stage == _Stage.DONE or self.game.is_game_over()

    def winner(self) -> Optional[int]:
        """回傳勝者（WHITE=1 / BLACK=-1），無結果時為 None。"""
        return self.game.result.winner if self.game.result else None

    def record_trainable_step(
        self,
        lp: torch.Tensor,
        ent: torch.Tensor,
        state_vector: torch.Tensor,
        stage: _Stage,
        action_idx: int,
    ) -> None:
        """為 trainable 方記錄一個步驟，初始化 shaping 為 0。"""
        # state_vector 來自可重用 buffer，這裡需 clone 保留該步的快照。
        self.steps.append((lp, ent, self.game.current_player, state_vector.detach().clone()))
        self.trainable_actions.append((stage, action_idx))
        self.step_shapings.append(0.0)

        # 若開局前尚無回填目標，但已有待結算負分，回填到第一個可用 step。
        if self.pending_second_step_idx is None and self.pending_opponent_penalty != 0.0:
            self.step_shapings[-1] += self.pending_opponent_penalty
            self.pending_opponent_penalty = 0.0

    def accumulate_shaping_to_latest(self, shaping_delta: float) -> None:
        """將 shaping 分數累加到最新記錄的步驟。"""
        if self.step_shapings:
            self.step_shapings[-1] += shaping_delta

    def accumulate_shaping_to_pending(self, shaping_delta: float) -> None:
        """將 shaping 分數回填到 pending_second_step_idx 所指的延遲步驟，並清除索引。"""
        assert self.pending_second_step_idx is not None, "no pending step to back-fill"
        self.step_shapings[self.pending_second_step_idx] += shaping_delta
        self.pending_shaping_snapshot = None

    def accumulate_shaping_to_pending_opening_first(self, shaping_delta: float) -> None:
        """將 shaping 分數回填到 pending_opening_first_step_idx 並清除索引。"""
        assert self.pending_opening_first_step_idx is not None, "no opening first-step pending to back-fill"
        self.step_shapings[self.pending_opening_first_step_idx] += shaping_delta
        self.pending_shaping_snapshot = None

    def set_pending_backfill_target_to_latest(self) -> None:
        """把延遲負分回填目標設為目前最後一個 trainable step。"""
        if self.step_shapings:
            self.pending_second_step_idx = len(self.step_shapings) - 1

    def add_pending_opponent_penalty(self, shaping_delta: float) -> None:
        """累積對手回合造成的負分，等待回填。"""
        self.pending_opponent_penalty += shaping_delta

    def flush_pending_opponent_penalty(self) -> None:
        """把待結算對手負分回填到目標 step（若可用）。"""
        if self.pending_opponent_penalty == 0.0:
            return

        if self.pending_second_step_idx is not None:
            self.step_shapings[self.pending_second_step_idx] += self.pending_opponent_penalty
            self.pending_opponent_penalty = 0.0
            return

        if self.step_shapings:
            self.step_shapings[-1] += self.pending_opponent_penalty
            self.pending_opponent_penalty = 0.0

    def patch_state_after_move(self, move_src: Optional[Tuple[int, int]], 
                               move_dst: Optional[Tuple[int, int]]) -> None:
        """在 move 應用後，patch live buffer 中受影響的棋盤位置與 turn/player。
        
        參數
        ----
        move_src : move 的來源位置，若為 None 表示無棋盤改變（如 PASS）
        move_dst : move 的目標位置，若為 None 表示無棋盤改變
        """
        if self._state_buffer is None:
            return

        # 更新受影響的棋盤位置
        if move_src is not None and move_dst is not None:
            positions = self.valid_positions()
            pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}
            
            if move_src in pos_to_idx:
                self.ai.update_state_board_position(self._state_buffer, move_src, pos_to_idx[move_src])
            if move_dst in pos_to_idx:
                self.ai.update_state_board_position(self._state_buffer, move_dst, pos_to_idx[move_dst])
        
        # move 應用後，turn/player 可能已改變，需要刷新
        self.ai.update_state_turn_player(self._state_buffer)


# ══════════════════════════════════════════════════════════════════════
# 單步轉移函式
# ══════════════════════════════════════════════════════════════════════

def _transition(
    ag:               _ActiveGame,
    decision_logits:  torch.Tensor,
    state_vector:     torch.Tensor,
    record_for_training: bool,
) -> None:
    """
    依 ag.stage 執行一次決策採樣、更新 ag.steps，並推進 ag.stage。
    同時在 trainable 方的決策及落子時計算 shaping。

    注意
    ----
    決策 logits 由呼叫方傳入（已經依 stage 選好對應 head）。
    state_vector：決策時的狀態向量（用於價值函數訓練）。
    """
    device = ag.device
    stage  = ag.stage
    game   = ag.game

    # ── 第一步 source ────────────────────────────────────────────────
    if stage == _Stage.NEED_FIRST_SRC:
        # 訓練方進入新回合前，先把對手上一回合造成的負分回填。
        if record_for_training:
            ag.flush_pending_opponent_penalty()
        if ag.ai.resolve_no_mandatory_capture_if_needed():
            ag.stage = _Stage.DONE
            return
        mask = ag.ai.capture_source_mask().to(device=device)
        idx, lp, ent = masked_sample(decision_logits, mask)
        if record_for_training:
            ag.record_trainable_step(lp, ent, state_vector, stage, idx)
        ag._first_src = ag.valid_positions()[idx]
        ag.stage = _Stage.NEED_FIRST_DIR

    # ── 第一步 direction ─────────────────────────────────────────────
    elif stage == _Stage.NEED_FIRST_DIR:
        assert ag._first_src is not None
        src  = ag._first_src
        mask = ag.ai.capture_direction_mask(src).to(device=device)
        idx, lp, ent = masked_sample(decision_logits, mask)
        if record_for_training:
            ag.record_trainable_step(lp, ent, state_vector, stage, idx)
        move = ag.ai.first_step_reconstruct_move(src, idx)
        actor = game.current_player
        counts_before = game.get_piece_counts()
        game.play_first_step(move)
        counts_after = game.get_piece_counts()
        
        # 落子後同步更新 live buffer
        ag.patch_state_after_move(move.src, move.dst)

        before_snapshot = _capture_shaping_snapshot(
            ag.ai,
            move,
            actor=actor,
            counts_before=counts_before,
            counts_after=counts_after,
        )
        
        shaping_delta = _compute_step_shaping_from_snapshot(
            before_snapshot,
            ag.trainable_player,
        )
        if record_for_training:
            ag.accumulate_shaping_to_latest(shaping_delta)
            if not game.is_waiting_second_step():
                ag.set_pending_backfill_target_to_latest()
        elif shaping_delta != 0.0:
            ag.add_pending_opponent_penalty(shaping_delta)

        if game.is_game_over():
            ag.stage = _Stage.DONE
        elif game.is_waiting_second_step():
            ag.stage = _Stage.NEED_SECOND_KIND
        else:
            # 白方首回合：play_first_step 已結束該回合，換黑方
            ag.stage = _Stage.NEED_FIRST_SRC

    # ── 第二步 kind ──────────────────────────────────────────────────
    elif stage == _Stage.NEED_SECOND_KIND:
        mask = ag.ai.second_step_kind_mask().to(device=device)
        idx, lp, ent = masked_sample(decision_logits, mask)
        if record_for_training:
            ag.record_trainable_step(lp, ent, state_vector, stage, idx)
        kind = KIND_ORDER[idx]
        ag._second_kind = kind
        if kind == MoveKind.PASS:
            game.play_second_step()
            
            # PASS 不改棋盤，但要刷新 turn/player
            ag.patch_state_after_move(None, None)
            
            if record_for_training:
                ag.set_pending_backfill_target_to_latest()
            ag.stage = _Stage.DONE if game.is_game_over() else _Stage.NEED_FIRST_SRC
        elif kind == MoveKind.CAPTURE:
            ag.stage = _Stage.NEED_SECOND_CAP_SRC
        else:  # REINFORCE
            ag.stage = _Stage.NEED_SECOND_REIN_SRC

    # ── 第二步 capture source ────────────────────────────────────────
    elif stage == _Stage.NEED_SECOND_CAP_SRC:
        mask = ag.ai.capture_source_mask().to(device=device)
        idx, lp, ent = masked_sample(decision_logits, mask)
        if record_for_training:
            ag.record_trainable_step(lp, ent, state_vector, stage, idx)
        ag._second_src = ag.valid_positions()[idx]
        ag.stage = _Stage.NEED_SECOND_CAP_DIR

    # ── 第二步 capture direction ─────────────────────────────────────
    elif stage == _Stage.NEED_SECOND_CAP_DIR:
        assert ag._second_src is not None
        src  = ag._second_src
        mask = ag.ai.capture_direction_mask(src).to(device=device)
        idx, lp, ent = masked_sample(decision_logits, mask)
        if record_for_training:
            ag.record_trainable_step(lp, ent, state_vector, stage, idx)
        move = ag.ai.second_step_reconstruct_move(MoveKind.CAPTURE, src, idx)
        actor = game.current_player
        counts_before = game.get_piece_counts()
        game.play_second_step(move)
        counts_after = game.get_piece_counts()
        
        # 落子後同步更新 live buffer
        ag.patch_state_after_move(move.src, move.dst)

        before_snapshot = _capture_shaping_snapshot(
            ag.ai,
            move,
            actor=actor,
            counts_before=counts_before,
            counts_after=counts_after,
        )
        
        shaping_delta = _compute_step_shaping_from_snapshot(
            before_snapshot,
            ag.trainable_player,
        )
        if record_for_training:
            ag.accumulate_shaping_to_latest(shaping_delta)
            ag.set_pending_backfill_target_to_latest()
        elif shaping_delta != 0.0:
            ag.add_pending_opponent_penalty(shaping_delta)

        ag.stage = _Stage.DONE if game.is_game_over() else _Stage.NEED_FIRST_SRC

    # ── 第二步 reinforce source ──────────────────────────────────────
    elif stage == _Stage.NEED_SECOND_REIN_SRC:
        mask = ag.ai.reinforce_source_mask().to(device=device)
        idx, lp, ent = masked_sample(decision_logits, mask)
        if record_for_training:
            ag.record_trainable_step(lp, ent, state_vector, stage, idx)
        ag._second_src = ag.valid_positions()[idx]
        ag.stage = _Stage.NEED_SECOND_REIN_DIR

    # ── 第二步 reinforce direction ───────────────────────────────────
    elif stage == _Stage.NEED_SECOND_REIN_DIR:
        assert ag._second_src is not None
        src  = ag._second_src
        mask = ag.ai.reinforce_direction_mask(src).to(device=device)
        idx, lp, ent = masked_sample(decision_logits, mask)
        if record_for_training:
            ag.record_trainable_step(lp, ent, state_vector, stage, idx)
        move = ag.ai.second_step_reconstruct_move(MoveKind.REINFORCE, src, idx)
        actor = game.current_player
        counts_before = game.get_piece_counts()
        game.play_second_step(move)
        counts_after = game.get_piece_counts()
        
        # 落子後同步更新 live buffer
        ag.patch_state_after_move(move.src, move.dst)

        before_snapshot = _capture_shaping_snapshot(
            ag.ai,
            move,
            actor=actor,
            counts_before=counts_before,
            counts_after=counts_after,
        )
        
        shaping_delta = _compute_step_shaping_from_snapshot(
            before_snapshot,
            ag.trainable_player,
        )
        if record_for_training:
            ag.accumulate_shaping_to_latest(shaping_delta)
            ag.set_pending_backfill_target_to_latest()
        elif shaping_delta != 0.0:
            ag.add_pending_opponent_penalty(shaping_delta)

        ag.stage = _Stage.DONE if game.is_game_over() else _Stage.NEED_FIRST_SRC
# ══════════════════════════════════════════════════════════════════════
# 批次蒐集
# ══════════════════════════════════════════════════════════════════════

def collect_batch(
    trainable_policy: nn.Module,
    opponent_pool: OpponentPool,
    n_games: int = GAMES_PER_UPDATE,
    device:  Optional[str] = None,
) -> Tuple[List[_ActiveGame], Dict[str, int]]:
    """
    同時跑 n_games 局，僅更新 trainable_policy；
    每局會隨機分配 trainable 方是白或黑，對手由 opponent_pool 抽樣。

    每輪 while 迴圈開始時，先快照各遊戲的 stage；
    同 stage 的遊戲堆疊成一個 batch 做一次 policy.forward()；
    各遊戲依各自 mask 採樣並推進 stage。
    本輪新轉移到其他 stage 的遊戲在「下一輪」才被處理（確保無重複推論）。

    參數
    ----
    trainable_policy : 需要更新的策略網路
    opponent_pool    : 對手池管理器（提供對手 checkpoint 與固定網路）
    n_games : 同時進行的局數
    device  : torch device 字串（None 代表 cpu）

    回傳
    ----
    (List[_ActiveGame], batch_stats)
        batch_stats 包含對手來源分布與 trainable 白黑局數。
    """
    active: List[_ActiveGame] = []
    batch_stats = {
        "games_total": 0,
        "source_resident": 0,
        "source_recent": 0,
        "source_early": 0,
        "trainable_white_games": 0,
        "trainable_black_games": 0,
    }
    
    for _ in range(n_games): #n_games 局同時進行
        trainable_player = 1 if random.random() < 0.5 else -1
        selection = opponent_pool.select_checkpoint()
        opponent_policy = opponent_pool.get_policy(selection.checkpoint)
        active.append(
            _ActiveGame(
                trainable_player=trainable_player,
                opponent_policy=opponent_policy,
                opponent_source=selection.source,
                opponent_checkpoint_index=selection.checkpoint.index,
                device=device,
            )
        )

        batch_stats["games_total"] += 1
        if trainable_player == 1:
            batch_stats["trainable_white_games"] += 1
        else:
            batch_stats["trainable_black_games"] += 1
        if selection.source == "resident":
            batch_stats["source_resident"] += 1
        elif selection.source == "recent":
            batch_stats["source_recent"] += 1
        else:
            batch_stats["source_early"] += 1

    while not all(g.is_done() for g in active):
        # 快照本輪所有待決策樣本。
        sample_game_idxs: List[int] = []
        sample_heads: List[str] = []
        sample_states: List[torch.Tensor] = []
        sample_is_trainable: List[bool] = []
        trainable_sample_idxs: List[int] = []
        opponent_sample_idxs: List[int] = []

        #把所有 active 遊戲中還沒結束的遊戲的當前 stage 和狀態向量收集起來，然後再分別對 trainable 方和對手方計算 logits。這樣可以確保同一輪中所有遊戲的決策都是基於同一個快照，避免在同一輪中重複推論或更新。
        for i, g in enumerate(active):
            if g.is_done():
                continue
            stage = g.stage
            is_trainable_turn = (g.game.current_player == g.trainable_player)
            sample_idx = len(sample_game_idxs)
            sample_game_idxs.append(i)
            sample_heads.append(_STAGE_TO_HEAD[stage])
            sample_states.append(g.encode(_STAGE_TO_PHASE[stage]))
            sample_is_trainable.append(is_trainable_turn)
            if is_trainable_turn:
                trainable_sample_idxs.append(sample_idx)
            else:
                opponent_sample_idxs.append(sample_idx)

        if not sample_states:
            break

        head_logits_by_sample: Dict[int, torch.Tensor] = {}
        # 先一次性計算 trainable 方的 logits（可能多於一個樣本），並分配回各樣本。
        if trainable_sample_idxs:
            trainable_states = torch.stack([sample_states[i] for i in trainable_sample_idxs])
            trainable_hidden = _encode_hidden(trainable_policy, trainable_states)
            head_to_local_idxs: Dict[str, List[int]] = {}
            for local_idx, sample_idx in enumerate(trainable_sample_idxs):
                head_to_local_idxs.setdefault(sample_heads[sample_idx], []).append(local_idx)

            for head, local_idxs in head_to_local_idxs.items():
                head_hidden = trainable_hidden[local_idxs]
                head_logits = trainable_policy.head_logits(head_hidden, head)
                for j, local_idx in enumerate(local_idxs):
                    sample_idx = trainable_sample_idxs[local_idx]
                    head_logits_by_sample[sample_idx] = head_logits[j]
        # 對手方按 policy 分組後批次推論，再按 head 分組取對應 logits。
        if opponent_sample_idxs:
            policy_to_sample_idxs: Dict[nn.Module, List[int]] = {}
            for sample_idx in opponent_sample_idxs:
                game_idx = sample_game_idxs[sample_idx]
                policy = active[game_idx].opponent_policy
                policy_to_sample_idxs.setdefault(policy, []).append(sample_idx)

            for opponent_policy, grouped_sample_idxs in policy_to_sample_idxs.items():
                opponent_states = torch.stack([sample_states[i] for i in grouped_sample_idxs])
                with torch.inference_mode():
                    opponent_hidden = _encode_hidden(opponent_policy, opponent_states)

                    head_to_local_idxs: Dict[str, List[int]] = {}
                    for local_idx, sample_idx in enumerate(grouped_sample_idxs):
                        head_to_local_idxs.setdefault(sample_heads[sample_idx], []).append(local_idx)

                    for head, local_idxs in head_to_local_idxs.items():
                        head_hidden = opponent_hidden[local_idxs]
                        head_logits = opponent_policy.head_logits(head_hidden, head)
                        for j, local_idx in enumerate(local_idxs):
                            sample_idx = grouped_sample_idxs[local_idx]
                            head_logits_by_sample[sample_idx] = head_logits[j]

        for sample_idx, game_idx in enumerate(sample_game_idxs):
            game = active[game_idx]
            _transition(
                ag=game,
                decision_logits=head_logits_by_sample[sample_idx],
                state_vector=sample_states[sample_idx],
                record_for_training=sample_is_trainable[sample_idx],
            )

    # 遊戲在對手回合結束時結算，trainable 方不一定回到 NEED_FIRST_SRC；
    # 此時把尚未回填的對手負分補回最後目標 step。
    for ag in active:
        ag.flush_pending_opponent_penalty()

    return active, batch_stats


# ══════════════════════════════════════════════════════════════════════
# 折扣回報計算
# ══════════════════════════════════════════════════════════════════════

def compute_returns(
    steps:  List[_StepRecord],
    winner: Optional[int],
    step_shapings: List[float],
    is_terminal_only: bool,
    gamma:  float = GAMMA,
    shaping_scale: float = SHAPING_REWARD_SCALE_START,
) -> List[float]:
    """
    為一局軌跡的每個決策步驟計算折扣回報。

    回報定義
    --------
    每個 step 先定義 immediate reward，再由後往前累積 discounted return。

    terminal-only 模式：
        只有最後一步帶終局勝負 ±1，其餘步 reward 為 0。

    shaping 模式：
        每一步 immediate reward = shaping_t；
        最後一步再額外加上終局勝負 ±1。

    遞推公式：
        G_t = r_t + gamma * G_{t+1}

    參數
    ----
    steps  : _StepRecord 列表
    winner : 勝方整數（WHITE=1 / BLACK=-1），None 代表未決（不應出現）
    step_shapings : 每個 step 的 shaping 分數
    is_terminal_only : 是否只用終局勝負，忽略 shaping
    gamma  : 折扣因子
    shaping_scale : 非終局 reward（shaping）縮放係數

    回傳
    ----
    長度與 steps 相同的 float 列表
    """
    n = len(steps)
    if n == 0:
        return []

    step_rewards: List[float] = [0.0] * n
    for t in range(n):
        if not is_terminal_only and t < len(step_shapings):
            step_rewards[t] += shaping_scale * step_shapings[t]

    final_player = steps[-1][2]
    terminal_reward = 1.0 if (winner is not None and final_player == winner) else -1.0
    step_rewards[-1] += terminal_reward

    returns: List[float] = [0.0] * n
    running_return = 0.0
    for t in range(n - 1, -1, -1):
        running_return = step_rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


# ══════════════════════════════════════════════════════════════════════
# 損失計算
# ══════════════════════════════════════════════════════════════════════

def compute_loss(
    games: List[_ActiveGame],
    policy: nn.Module,
    is_terminal_only: bool = False,
    gamma: float = GAMMA,
    shaping_scale: float = SHAPING_REWARD_SCALE_START,
    entropy_beta: float = ENTROPY_BETA_START,
) -> Tuple[torch.Tensor, dict]:
    """
    對一批已完成的遊戲計算 Actor-Critic 損失（包含價值函數）。

    損失公式
    --------
    L = Σ_t [ -log_prob_t × advantage_t ] + λ × Σ_t (return_t - V(s_t))^2 − β × Σ_t entropy_t

    其中：
        advantage_t = return_t - V(s_t)
        V(s_t) = value head 的預測
        若 is_terminal_only=True，return_t 僅含終局勝負
        否則 return_t = 終局勝負 + shaping

    流程
    ----
    1. 對每局計算折扣回報列表（含 shaping）
    2. 收集所有狀態向量、log_prob、entropy
    3. 使用 policy 計算每個狀態的價值預測 V(s)
    4. 計算 advantages = (returns - V(s)).detach()
    5. 計算 policy loss 使用 advantages（降低方差）
    6. 計算 value loss MSE
    7. 合併 policy loss + value loss + entropy loss

    參數
    ----
    games  : collect_batch 回傳的已結束遊戲列表
    policy : PolicyNet 網路（用於計算 value head）
    is_terminal_only : 是否忽略 shaping，只用終局勝負
    gamma  : 折扣因子
    shaping_scale : 非終局 reward（shaping）縮放係數

    回傳
    ----
    (loss_tensor, metrics_dict)
        loss_tensor : 可直接呼叫 .backward() 的純量 Tensor
        metrics_dict : 訓練指標字典（供日誌輸出）
    """
    raw_returns:  List[float]          = []
    all_log_probs: List[torch.Tensor]  = []
    all_entropies: List[torch.Tensor]  = []
    all_states:    List[torch.Tensor]  = []
    winners:       List[Optional[int]] = []
    trainable_wins = 0

    for g in games:
        winner = g.winner()
        winners.append(winner)
        if winner is not None and winner == g.trainable_player:
            trainable_wins += 1
        rets = compute_returns(
            g.steps,
            winner,
            g.step_shapings,
            is_terminal_only,
            gamma,
            shaping_scale,
        )
        raw_returns.extend(rets)
        # g.steps 只會記錄 trainable 方的決策步驟。
        for lp, ent, _, state in g.steps:
            all_log_probs.append(lp)
            all_entropies.append(ent)
            all_states.append(state)

    if not all_log_probs:
        loss = torch.zeros((), dtype=torch.float32)
        n = len(games)
        metrics = {
            "win_rate_white": 0.0,
            "win_rate_black": 0.0,
            "win_rate_trainable": (trainable_wins / n) if n > 0 else 0.0,
            "avg_steps":      0.0,
            "policy_loss":    0.0,
            "value_loss":     0.0,
            "entropy_loss":   0.0,
            "total_loss":     0.0,
            "mean_return":    0.0,
            "n_trainable_steps": 0,
        }
        return loss, metrics

    device = all_log_probs[0].device
    returns_tensor = torch.tensor(raw_returns, dtype=torch.float32, device=device)
    
    # 堆疊所有狀態並計算價值預測
    states_batch = torch.stack(all_states)                          # (N, 489)
    hidden_batch = _encode_hidden(policy, states_batch)             # (N, hidden_size)
    value_preds = policy.forward_value(hidden_batch).squeeze(-1)    # (N,)
    
    # 計算 advantages（停止梯度以穩定訓練）
    advantages = (returns_tensor - value_preds.detach()).detach()
    
    log_probs_tensor = torch.stack(all_log_probs)
    entropies_tensor = torch.stack(all_entropies)

    # Actor-Critic losses
    policy_loss = -(log_probs_tensor * advantages).mean()
    value_loss = ((returns_tensor - value_preds) ** 2).mean()
    entropy_loss = -entropy_beta * entropies_tensor.mean()
    
    loss = policy_loss + VALUE_LOSS_COEFF * value_loss + entropy_loss

    # 計算訓練指標
    n = len(games)
    metrics = {
        "win_rate_white": sum(1 for w in winners if w ==  1) / n,
        "win_rate_black": sum(1 for w in winners if w == -1) / n,
        "win_rate_trainable": trainable_wins / n,
        "avg_steps":      sum(len(g.steps) for g in games) / n,
        "policy_loss":    policy_loss.item(),
        "value_loss":     value_loss.item(),
        "entropy_loss":   entropy_loss.item(),
        "total_loss":     loss.item(),
        "mean_return":    returns_tensor.mean().item() if raw_returns else 0.0,
        "n_trainable_steps": len(all_log_probs),
    }
    return loss, metrics


# ══════════════════════════════════════════════════════════════════════
# Checkpoint 工具
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    policy:     nn.Module,
    optimizer:  torch.optim.Optimizer,
    title:      str,
    checkpoint_index: int,
    update_idx: int,
    in_opponent_pool: bool = True,
    reason: str = "training",
    **kwargs,
) -> Path:
    """將網路權重與優化器狀態存至檔案。

    支援額外關鍵字參數（如 samples_in_update, inference_temperature）
    會被儲存在 payload 中。

    回傳儲存的 checkpoint 路徑。
    """
    architecture = _policy_arch_from_instance(policy)

    payload: Dict[str, object] = {
        "update_idx": update_idx,
        "title": title,
        "checkpoint_index": checkpoint_index,
        "architecture": architecture,
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "in_opponent_pool": in_opponent_pool,
        "checkpoint_reason": reason,
    }

    # 加入額外資訊
    for extra_key in ("samples_in_update", "inference_temperature"):
        if extra_key in kwargs:
            payload[extra_key] = kwargs[extra_key]

    if architecture == ARCH_FC:
        payload["hidden_size_1"] = HIDDEN_SIZE_1
        payload["hidden_size_2"] = HIDDEN_SIZE_2
    elif architecture == ARCH_CNN_MIN17:
        payload["global_feature_dim"] = getattr(policy, "global_feature_dim", CNN_GLOBAL_FEATURE_DIM)
        payload["dropout"] = CNN_DROPOUT

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{title}_{checkpoint_index:0{CHECKPOINT_INDEX_WIDTH}d}.pt"
    path = checkpoint_dir / filename
    torch.save(payload, path)
    print(
        f"  [checkpoint saved → {path}]"
        f" [pool={'yes' if in_opponent_pool else 'no'} reason={reason}]"
    )
    return path


def load_latest_checkpoint(
    policy:    nn.Module,
    optimizer: torch.optim.Optimizer,
    title:     str,
    device:    str,
) -> Tuple[int, int]:
    """
    從同標題的最新 checkpoint 載入權重。

    回傳
    ----
    (下一個 update 的起始索引, 下一個 checkpoint index)；若無 checkpoint 則回傳 (0, 0)。
    """
    checkpoints = sorted(list_checkpoints(title=title), key=lambda item: item.index, reverse=True)
    if not checkpoints:
        return 0, 0

    expected_arch = _policy_arch_from_instance(policy)
    latest: Optional[CheckpointInfo] = None
    ckpt = None
    for candidate in checkpoints:
        candidate_ckpt = torch.load(str(candidate.path), map_location=device, weights_only=True)
        candidate_arch = str(candidate_ckpt.get("architecture", ARCH_FC))
        if candidate_arch != expected_arch:
            continue
        latest = candidate
        ckpt = candidate_ckpt
        break

    if latest is None or ckpt is None:
        return 0, 0

    policy.load_state_dict(ckpt["policy_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    next_update = ckpt["update_idx"] + 1
    next_checkpoint_index = int(ckpt.get("checkpoint_index", latest.index)) + 1
    print(
        f"  [resumed title={title} from {latest.path}, "
        f"next update={next_update}, next checkpoint_index={next_checkpoint_index}]"
    )
    return next_update, next_checkpoint_index


def _count_batch_trainable_wins(games: List[_ActiveGame]) -> int:
    """統計本批次中 trainable 方的勝場數。"""
    batch_wins = 0
    for game in games:
        winner = game.winner()
        if winner is not None and winner == game.trainable_player:
            batch_wins += 1
    return batch_wins


def _ratio(numer: int, denom: int) -> float:
    """避免除以 0 的比例工具。"""
    return (numer / denom) if denom > 0 else 0.0


def _format_kind_distribution(games: List[_ActiveGame]) -> str:
    """彙整 trainable 模型在本批次第二步 kind 的採樣分布。"""
    kind_counts = {MoveKind.PASS: 0, MoveKind.CAPTURE: 0, MoveKind.REINFORCE: 0}

    for game in games:
        for stage, action_idx in game.trainable_actions:
            if stage == _Stage.NEED_SECOND_KIND:
                kind_counts[KIND_ORDER[action_idx]] += 1

    kind_total = sum(kind_counts.values())

    return (
        f"kind(P/C/R)={_ratio(kind_counts[MoveKind.PASS], kind_total):.2f}/"
        f"{_ratio(kind_counts[MoveKind.CAPTURE], kind_total):.2f}/"
        f"{_ratio(kind_counts[MoveKind.REINFORCE], kind_total):.2f}"
    )


def _print_training_banner(
    title: str,
    architecture: str,
    runtime_device: str,
    resident_prob: float,
    opponent_pool: OpponentPool,
    n_updates: int,
    start: int,
    next_checkpoint_index: int,
    lr_start: float,
    lr_end: float,
    entropy_start: float,
    entropy_end: float,
    shaping_scale_start: float,
    shaping_scale_end: float,
) -> None:
    """輸出訓練開始時最重要的設定摘要。"""
    if architecture == ARCH_CNN_MIN17:
        model_summary = f"arch={architecture} channels=12 global={CNN_GLOBAL_FEATURE_DIM} dropout={CNN_DROPOUT}"
    else:
        model_summary = f"arch={architecture} hidden1={HIDDEN_SIZE_1} hidden2={HIDDEN_SIZE_2}"
    print(f"訓練開始：title={title}  device={runtime_device}  {model_summary}")
    print(f"  GAMES_PER_UPDATE={GAMES_PER_UPDATE}  GAMMA={GAMMA}")
    print(f"  lr_schedule={lr_start:.6g}->{lr_end:.6g}")
    print(f"  entropy_schedule={entropy_start:.6g}->{entropy_end:.6g}")
    print(f"  shaping_scale_schedule={shaping_scale_start:.6g}->{shaping_scale_end:.6g}")
    print(
        f"  opponent_prob: resident={resident_prob:.2f} "
        f"recent={RECENT_OPPONENT_PROB:.2f} early={EARLY_OPPONENT_PROB:.2f}"
    )
    print(
        f"  promotion: check_every={PROMOTION_CHECK_EVERY} "
        f"winrate>={PROMOTION_WINRATE_THRESHOLD:.2f} min_update={PROMOTION_MIN_UPDATES}"
    )
    print(f"  Reward Mode: {'TERMINAL_ONLY' if TERMINAL_ONLY else 'SHAPING'}")
    print(f"  updates={n_updates}  start_from={start}  next_checkpoint_index={next_checkpoint_index}")
    print(f"  opponent_pool_size={len(opponent_pool)}  resident={opponent_pool.resident.path.name}")
    print()


def _log_update_status(
    update: int,
    n_updates: int,
    log_every: int,
    metrics: dict,
    kind_distribution: str,
) -> None:
    """輸出單次 update 的摘要日誌。"""
    if (update + 1) % log_every != 0 and (update + 1) != n_updates:
        return

    print(
        f"  update {update + 1:6d}"
        f" | loss={metrics['total_loss']:+8.4f}"
        f" wr_T={metrics['win_rate_trainable']:.2f}"
        f" | {kind_distribution}"
    )


# ══════════════════════════════════════════════════════════════════════
# 訓練主迴圈
# ══════════════════════════════════════════════════════════════════════

def train(
    title: str,
    n_updates: int = DEFAULT_UPDATES,
    log_every: int = 5,
    lr_start: float = LEARNING_RATE_START,
    lr_end: float = LEARNING_RATE_END,
    entropy_start: float = ENTROPY_BETA_START,
    entropy_end: float = ENTROPY_BETA_END,
    shaping_scale_start: float = SHAPING_REWARD_SCALE_START,
    shaping_scale_end: float = SHAPING_REWARD_SCALE_END,
) -> None:
    """
    主訓練迴圈。

    流程
    ----
    1. 建立 PolicyNet 與 Adam optimizer
    2. 自動嘗試從同標題最新 checkpoint 載入
    3. 每輪：collect_batch → compute_loss → backward → optimizer.step
    4. 根據 TERMINAL_ONLY 決定是否加入 shaping
    5. 每 CHECKPOINT_EVERY 輪儲存 checkpoint

    參數
    ----
    title     : 本次實驗的模型標題
    n_updates : 總更新次數
    log_every : 每 N 次更新輸出一次訓練日誌
    """
    train_start_time = time.perf_counter()

    if log_every <= 0:
        raise ValueError("log_every must be >= 1")
    if lr_start < 0.0 or lr_end < 0.0:
        raise ValueError("learning-rate schedule values must be >= 0")
    if entropy_start < 0.0 or entropy_end < 0.0:
        raise ValueError("entropy schedule values must be >= 0")
    if shaping_scale_start < 0.0 or shaping_scale_end < 0.0:
        raise ValueError("shaping-scale schedule values must be >= 0")
    if PROMOTION_CHECK_EVERY <= 0:
        raise ValueError("PROMOTION_CHECK_EVERY must be >= 1")
    runtime_device = resolve_device(DEVICE)
    resident_prob = validate_opponent_probs(RECENT_OPPONENT_PROB, EARLY_OPPONENT_PROB)

    if USE_CNN_POLICY:
        policy = PolicyNetCNNMin17(
            global_feature_dim=CNN_GLOBAL_FEATURE_DIM,
            dropout=CNN_DROPOUT,
        ).to(runtime_device)
    else:
        policy = PolicyNet().to(runtime_device)

    policy_arch = _policy_arch_from_instance(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_start)

    title = sanitize_checkpoint_title(title)
    start, next_checkpoint_index = load_latest_checkpoint(policy, optimizer, title, runtime_device)

    # 若還沒有任何對手池成員，先把目前模型存成初始常駐對手。
    if not list_pool_checkpoints(title):
        save_checkpoint(
            policy,
            optimizer,
            title,
            next_checkpoint_index,
            update_idx=max(0, start - 1),
            in_opponent_pool=True,
            reason="bootstrap_resident",
        )
        next_checkpoint_index += 1

    opponent_pool = OpponentPool(
        title=title,
        device=runtime_device,
        recent_prob=RECENT_OPPONENT_PROB,
        early_prob=EARLY_OPPONENT_PROB,
        recent_pool_size=RECENT_POOL_SIZE,
    )

    promotion_window = PromotionWindow()
    _print_training_banner(
        title=title,
        architecture=policy_arch,
        runtime_device=runtime_device,
        resident_prob=resident_prob,
        opponent_pool=opponent_pool,
        n_updates=n_updates,
        start=start,
        next_checkpoint_index=next_checkpoint_index,
        lr_start=lr_start,
        lr_end=lr_end,
        entropy_start=entropy_start,
        entropy_end=entropy_end,
        shaping_scale_start=shaping_scale_start,
        shaping_scale_end=shaping_scale_end,
    )

    for update in range(start, n_updates):
        current_lr = linear_schedule_value(update, n_updates, lr_start, lr_end)
        current_entropy_beta = linear_schedule_value(update, n_updates, entropy_start, entropy_end)
        current_shaping_scale = linear_schedule_value(update, n_updates, shaping_scale_start, shaping_scale_end)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        policy.train()
        games, batch_stats = collect_batch(
            trainable_policy=policy,
            opponent_pool=opponent_pool,
            n_games=GAMES_PER_UPDATE,
            device=runtime_device,
        )

        batch_wins = _count_batch_trainable_wins(games)
        batch_games = len(games)
        kind_distribution = _format_kind_distribution(games)
        promotion_window.observe_batch(batch_wins=batch_wins, batch_games=batch_games)

        loss, metrics = compute_loss(
            games,
            policy=policy,
            is_terminal_only=TERMINAL_ONLY,
            gamma=GAMMA,
            shaping_scale=current_shaping_scale,
            entropy_beta=current_entropy_beta,
        )
        if metrics["n_trainable_steps"] > 0:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()

        promoted = False
        promotion_checked = promotion_window.should_check(update)
        if promotion_checked:
            if promotion_window.should_promote(update):
                save_checkpoint(
                    policy,
                    optimizer,
                    title,
                    next_checkpoint_index,
                    update,
                    in_opponent_pool=True,
                    reason="promotion",
                )
                next_checkpoint_index += 1
                opponent_pool.refresh()
                promoted = True

            _log_update_status(
                update=update,
                n_updates=n_updates,
                log_every=log_every,
                metrics=metrics,
                kind_distribution=kind_distribution,
            )
            promotion_window.reset()
        else:
            _log_update_status(
                update=update,
                n_updates=n_updates,
                log_every=log_every,
                metrics=metrics,
                kind_distribution=kind_distribution,
            )

        if (update + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                policy,
                optimizer,
                title,
                next_checkpoint_index,
                update,
                in_opponent_pool=False,
                reason="periodic",
            )
            next_checkpoint_index += 1

    elapsed = time.perf_counter() - train_start_time
    performed_updates = max(0, n_updates - start)

    print()
    print("訓練完成。")
    print(f"總耗時: {_format_duration(elapsed)} ({elapsed:.2f}s)")
    if performed_updates > 0:
        print(f"平均每次更新: {elapsed / performed_updates:.4f}s")


# ══════════════════════════════════════════════════════════════════════
# 命令列入口
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tzaar REINFORCE 自我對戰訓練")
    parser.add_argument(
        "--updates", type=int, default=DEFAULT_UPDATES, metavar="N",
        help=f"訓練更新次數（預設：{DEFAULT_UPDATES}）",
    )
    parser.add_argument(
        "--log-every", type=int, default=5, metavar="N",
        help="每 N 次更新輸出一次訓練日誌（預設：5）",
    )
    parser.add_argument(
        "--lr-start", type=float, default=LEARNING_RATE_START, metavar="LR",
        help=f"learning rate 起點（預設：{LEARNING_RATE_START}）",
    )
    parser.add_argument(
        "--lr-end", type=float, default=LEARNING_RATE_END, metavar="LR",
        help=f"learning rate 終點（預設：{LEARNING_RATE_END}）",
    )
    parser.add_argument(
        "--entropy-start", type=float, default=ENTROPY_BETA_START, metavar="E",
        help=f"entropy beta 起點（預設：{ENTROPY_BETA_START}）",
    )
    parser.add_argument(
        "--entropy-end", type=float, default=ENTROPY_BETA_END, metavar="E",
        help=f"entropy beta 終點（預設：{ENTROPY_BETA_END}）",
    )
    parser.add_argument(
        "--shaping-start", type=float, default=SHAPING_REWARD_SCALE_START, metavar="S",
        help=(
            "非終局 reward（shaping）縮放係數起點，最終回報為 "
            "terminal(±1) + S * shaping"
        ),
    )
    parser.add_argument(
        "--shaping-end", type=float, default=SHAPING_REWARD_SCALE_END, metavar="S",
        help="非終局 reward（shaping）縮放係數終點",
    )
    parser.add_argument(
        "--shaping-scale", type=float, default=None, metavar="S",
        help=(
            "相容舊介面：將 shaping 起點與終點同時設成此值"
        ),
    )
    args = parser.parse_args()

    shaping_scale_start = args.shaping_start
    shaping_scale_end = args.shaping_end
    if args.shaping_scale is not None:
        shaping_scale_start = args.shaping_scale
        shaping_scale_end = args.shaping_scale

    title = prompt_checkpoint_title()
    train(
        title=title,
        n_updates=args.updates,
        log_every=args.log_every,
        lr_start=args.lr_start,
        lr_end=args.lr_end,
        entropy_start=args.entropy_start,
        entropy_end=args.entropy_end,
        shaping_scale_start=shaping_scale_start,
        shaping_scale_end=shaping_scale_end,
    )
