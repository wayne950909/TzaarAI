"""
training/replay.py — 回放緩衝區管理

提供 replay buffer 的儲存、載入、擴充、採樣功能。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import REPLAY_CFG
from training.sample import PolicySample

import TzaarTrain as train_module

_snapshot_save_thread: Optional[threading.Thread] = None


def replay_snapshot_path(
    title: str,
    checkpoint_index: Optional[int] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """建構 replay snapshot 檔案路徑。

    若 checkpoint_index 為 None，使用"latest"標籤；
    否則使用 6 位數的 checkpoint 索引。
    """
    directory = base_dir if base_dir is not None else Path(train_module.CHECKPOINT_DIR)
    if checkpoint_index is None:
        filename = f"{title}_replay_{REPLAY_CFG.snapshot_tag}.pt"
    else:
        if checkpoint_index < 0:
            raise ValueError("checkpoint_index must be >= 0")
        filename = f"{title}_replay_{checkpoint_index:06d}.pt"
    return directory / filename


def save_replay_snapshot(
    title: str,
    checkpoint_index: Optional[int],
    update_idx: int,
    replay_buffer: List[PolicySample],
    write_idx: int,
    max_samples: int,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    """儲存 replay buffer 到磁碟。"""
    if not REPLAY_CFG.enabled:
        return None

    path = replay_snapshot_path(title, checkpoint_index, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")

    payload_samples: List[Dict[str, object]] = []
    for sample in replay_buffer:
        payload_samples.append(
            {
                "state": sample.state.detach()
                .to(device="cpu", dtype=torch.float32),
                "global_features": sample.global_features.detach()
                .to(device="cpu", dtype=torch.float32),
                "action_dim": int(sample.action_dim),
                "legal_mask_padded": sample.legal_mask_padded.detach()
                .to(device="cpu", dtype=torch.bool),
                "target_pi_padded": sample.target_pi_padded.detach()
                .to(device="cpu", dtype=torch.float32),
                "player": int(sample.player),
                "winner_sign": int(sample.winner_sign),
                "value_target": float(sample.value_target),
            }
        )

    payload: Dict[str, object] = {
        "version": int(REPLAY_CFG.snapshot_version),
        "title": str(title),
        "checkpoint_index": (
            None if checkpoint_index is None else int(checkpoint_index)
        ),
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


def save_replay_snapshot_async(
    title: str,
    checkpoint_index: Optional[int],
    update_idx: int,
    replay_buffer: List[PolicySample],
    write_idx: int,
    max_samples: int,
    base_dir: Optional[Path] = None,
) -> None:
    """非同步儲存 replay buffer（背景執行緒）。"""
    global _snapshot_save_thread

    if _snapshot_save_thread is not None and _snapshot_save_thread.is_alive():
        _snapshot_save_thread.join(timeout=1.0)

    buffer_copy = list(replay_buffer)

    def _worker() -> None:
        save_replay_snapshot(
            title=title,
            checkpoint_index=checkpoint_index,
            update_idx=update_idx,
            replay_buffer=buffer_copy,
            write_idx=write_idx,
            max_samples=max_samples,
            base_dir=base_dir,
        )

    _snapshot_save_thread = threading.Thread(
        target=_worker, name="replay-snapshot-saver", daemon=True
    )
    _snapshot_save_thread.start()


def load_replay_snapshot(
    title: str,
    checkpoint_index: Optional[int],
    max_samples: int,
    base_dir: Optional[Path] = None,
) -> Tuple[List[PolicySample], int]:
    """從磁碟載入 replay buffer。"""
    if not REPLAY_CFG.enabled:
        return [], 0

    path = replay_snapshot_path(title, checkpoint_index, base_dir=base_dir)
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
    if version != REPLAY_CFG.snapshot_version:
        print(
            f"[replay] version mismatch in {path.name}: "
            f"got={version} expected={REPLAY_CFG.snapshot_version}"
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
        if not all(
            torch.is_tensor(x)
            for x in (state, global_features, legal_mask, target_pi)
        ):
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
            f"[replay] truncating loaded samples {len(replay_buffer)} -> "
            f"{max_samples}"
        )
        replay_buffer = replay_buffer[:max_samples]

    if not replay_buffer:
        print(f"[replay] snapshot empty after parsing: {path.name}")
        return [], 0

    write_idx = int(payload.get("write_idx", 0))
    max_valid = (
        max_samples
        if len(replay_buffer) >= max_samples and max_samples > 0
        else len(replay_buffer)
    )
    if write_idx < 0 or write_idx >= max(max_valid, 1):
        print(f"[replay] invalid write_idx={write_idx}, reset to 0")
        write_idx = 0

    print(
        f"[replay] loaded {path.name} | samples={len(replay_buffer)} "
        f"write_idx={write_idx}"
    )
    return replay_buffer, write_idx


def replay_extend(
    buffer: List[PolicySample],
    write_idx: int,
    max_samples: int,
    new_samples: List[PolicySample],
) -> int:
    """將新樣本加入 replay buffer（環形緩衝區）。"""
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


def remove_samples_from_buffer(
    buffer: List[PolicySample],
    write_idx: int,
    max_samples: int,
    samples_to_remove: List[PolicySample],
) -> int:
    """從 replay buffer 中移除指定的一批樣本（依物件身分比對）。

    用途：gate 拒絕時，把「此次被拒絕的 update 所產生的 fresh samples」
    從 replay buffer 中剔除，避免這批品質不佳的學習資料被往後的 update
    重複抽樣。覆寫掉的部分不追蹤，能清多少清多少。

    語意
    ----
    - 以 object identity（id）比對，因為 fresh_samples 與 buffer 內存的是
      同一批 PolicySample 物件的參考。
    - buffer 是一個環形緩衝區；移除後會壓緊（compact）成連續的頭段。
    - 回傳的 write_idx 指向「下一個要覆寫的位置」 = len(buffer) % max_samples
      （因移除後 buffer 通常未滿 max_samples，屆時 replay_extend 會走 append
      分支，write_idx 僅在蓋滿後才被用到；這裡保守地依 max_samples 計算）。
    - buffer 為空、或不含任何要移除的樣本時，原地不動並回傳原 write_idx。
    """
    if not buffer or not samples_to_remove:
        return write_idx

    remove_ids = {id(s) for s in samples_to_remove}
    kept = [s for s in buffer if id(s) not in remove_ids]

    removed = len(buffer) - len(kept)
    if removed == 0:
        return write_idx

    # 就地覆寫，保留傳入 list 的物件身份（呼叫端持有的引用不受影響）
    buffer[:] = kept

    # 新 write_idx：下一個要被覆寫的位置。
    # 移除後 len(buffer) < max_samples 時 replay_extend 全走 append，
    # 此值不會被用到；若正好仍滿載則依環形語意重算。
    next_write_idx = len(buffer) % max_samples if max_samples > 0 else 0

    print(
        f"[replay] removed {removed} samples from buffer on reject "
        f"| buffer={len(buffer)}"
    )
    return next_write_idx


def select_train_samples(
    fresh_samples: List[PolicySample],
    replay_buffer: List[PolicySample],
) -> List[PolicySample]:
    """從 replay buffer（或 fresh samples）選擇訓練樣本。"""
    if REPLAY_CFG.enabled:
        available = len(replay_buffer)
        if available <= 0:
            return []
        if available < REPLAY_CFG.min_train_samples:
            return replay_buffer.copy()
        n_draw = min(REPLAY_CFG.train_samples_per_update, available)
        if n_draw >= available:
            return replay_buffer.copy()
        indices = np.random.choice(available, size=n_draw, replace=False)
        return [replay_buffer[int(i)] for i in indices]
    return fresh_samples
