"""

battle_server.py — TZAAR 模型對戰 / 重播 HTTP 服務


提供給 test.html 使用的一組 HTTP API（本端 127.0.0.1:8788）：

- GET  /api/checkpoints          列出 checkpoints/ 下的模型檔
- POST /api/battle               啟動「A 模型 vs B 模型」對戰（可各自設定模擬次數、場數）
- GET  /api/battle/<id>          查詢對戰進度 / 勝率結果
- GET  /api/games                列出已完成對局的遊戲紀錄
- GET  /api/games/<id>           取回單一對局完整紀錄（供網頁重播）
- POST /api/strategy-move        單手策略（AI 代走目前回合），回傳 move1 / move2

背景執行：
    Python 後端會嘗試載入原生 cpp 模組；本服務預設走 Python 狀態後端，
    以與 test.html 的 RAW_LAYOUT 完全對齊（較易產生可信的重播盤面）。
    對戰以背景執行緒執行，前端可輪詢進度。
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import config as _cfg
from config import NETWORK_CFG
from core.action import N_ACTIONS, PASS_ACTION_IDX
from core.constants import WHITE, BLACK
from core.env import TzaarEnv, EnvConfig
from Tzaar import BOARD_LAYOUT
from network import PolicyNetCNNMin17
from mcts import run_mcts

# ── 狀態後端固定為 Python，確保盤面與前端 RAW_LAYOUT 一致 ──────────
_cfg._ACTIVE_STATE_BACKEND = "python"
_cfg._ACTIVE_CPP_MODULE = None

CHECKPOINT_DIR = Path("checkpoints")
RECORD_DIR = Path("checkpoints/battle_records")
RECORD_DIR.mkdir(parents=True, exist_ok=True)
HOST = "127.0.0.1"
PORT = 8788

_VALID_POSITIONS: List[Tuple[int, int]] = [
    (r, c)
    for r, row in enumerate(BOARD_LAYOUT)
    for c, value in enumerate(row)
    if value != -1
]

# ── 模型載入 ─────────────────────────────────────────────────────────

def list_checkpoint_files() -> List[str]:
    """回傳 checkpoints/ 下可載入的 checkpoints 檔名（按修改時間排序）。"""
    files = sorted(
        [p.name for p in CHECKPOINT_DIR.glob("*.pt")],
        key=lambda name: (CHECKPOINT_DIR / name).stat().st_mtime,
        reverse=True,
    )
    return files


def load_model(path: str, device: torch.device) -> PolicyNetCNNMin17:
    """從 checkpoint 檔載入 PolicyNetCNNMin17 並切換到 eval 模式。"""
    ckpt_path = Path(path)
    if not ckpt_path.is_absolute():
        ckpt_path = CHECKPOINT_DIR / ckpt_path

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    arch = str(ckpt.get("architecture", "")).lower()
    if arch != "cnn_min17":
        raise ValueError(f"checkpoint {ckpt_path.name} 架構為 {arch}，非 cnn_min17")

    net = PolicyNetCNNMin17(
        global_feature_dim=int(ckpt.get("global_feature_dim", NETWORK_CFG.global_feature_dim)),
        dropout=float(ckpt.get("dropout", NETWORK_CFG.dropout)),
    ).to(device)
    net.load_state_dict(ckpt["policy_state"], strict=True)
    net.eval()
    return net


# ── 遊戲 / MCTS 輔助 ────────────────────────────────────────────────

def _build_phase_state_from_env(env: TzaarEnv) -> Any:
    """從 TzaarEnv 建立 MCTS 所需的 PhaseGameState（Python 後端）。"""
    from state.phase_state import PythonPhaseGameState, Stage
    from TzaarAI import TzaarAIInterface

    game = env.game
    if game is None:
        raise RuntimeError("Environment not reset")

    ai = TzaarAIInterface(game)
    # 依序檢查：無強制吃子 → 等待第二步 → 已結束 → 正常第一步。
    if ai.resolve_no_mandatory_capture_if_needed():
        stage = Stage.DONE
    elif game.is_waiting_second_step():
        stage = Stage.NEED_STEP2
    elif game.is_game_over():
        stage = Stage.DONE
    else:
        stage = Stage.NEED_STEP1
    return PythonPhaseGameState(game=game, stage=stage, ai=ai)


def _sample_action_from_visits(visits: torch.Tensor, legal_mask: torch.Tensor, temperature: float) -> int:
    """依 MCTS 訪問次數 + 溫度採樣動作。temperature<=1e-6 為貪婪。"""
    legal = legal_mask[: visits.shape[0]]
    legal_visits = visits.clone()
    legal_visits[~legal] = 0.0

    if legal_visits.sum().item() <= 0:
        return int(torch.multinomial(legal.float(), 1).item())
    if temperature <= 1e-6:
        return int(torch.argmax(legal_visits).item())

    adjusted = torch.pow(legal_visits, 1.0 / temperature)
    adjusted[~legal] = 0.0
    adjusted = adjusted / adjusted.sum().clamp_min(1e-8)
    return int(torch.multinomial(adjusted, 1).item())


_PIECE_TYPE_NAME = {1: "Tzaar", 2: "Tzarra", 3: "Tott"}


def extract_pieces(env: TzaarEnv) -> List[Dict[str, Any]]:
    """取出目前盤面所有棋子 [{r,c,p,t,h}]。

    使用 Python 後端 env.game.board：每格 (piece_code, height)，
    piece_code 1-3 White / 4-6 Black。
    """
    game = env.game
    board = game.board
    pieces: List[Dict[str, Any]] = []
    for (r, c) in board.valid_positions():
        cell = board.get_cell((r, c))
        if cell is None or cell[1] == 0:
            continue
        code, height = cell[0], cell[1]
        owner = "White" if code in (1, 2, 3) else "Black"
        ptype = _PIECE_TYPE_NAME[((code - 1) % 3) + 1]
        pieces.append({"r": int(r), "c": int(c), "p": owner, "t": ptype, "h": int(height)})
    return pieces


def _winner_from_env(env: TzaarEnv) -> Optional[str]:
    from core.constants import WHITE as _W, BLACK as _B
    # 優先使用 env 內建的結果；若尚未反映在 env 中，則直接讀取 game.result。
    res = env.last_game_result
    if res is None and env.game is not None and env.game.result is not None:
        res = env.game.result
    if res is None:
        return None
    winner = getattr(res, "winner", None)
    value = getattr(res, "value", res)
    if winner is not None:
        if winner == _W:
            return "White"
        if winner == _B:
            return "Black"
        return None
    if value == _W:
        return "White"
    if value == _B:
        return "Black"
    return None


# ── 對戰執行 ─────────────────────────────────────────────────────────

class BattleJob:
    """一個對戰任務。"""

    def __init__(self, job_id: str, config: Dict[str, Any]) -> None:
        self.job_id = job_id
        self.config = config
        self.status = "running"   # running / done / error
        self.progress = 0          # 已完成局數
        self.games_done = []       # 完成的 game_id 列表
        self.errors: List[str] = []
        self.result: Optional[Dict[str, Any]] = None
        self.created_at = time.time()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "total": self.config.get("games", 0),
            "games_done": self.games_done,
            "errors": self.errors[:5],
            "result": self.result,
        }

    def _run(self) -> None:
        try:
            self.result = run_battle(self)
            self.status = "done"
        except Exception as exc:
            self.status = "error"
            self.errors.append(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()


_GAME_RECORDS: Dict[str, Dict[str, Any]] = {}   # game_id -> record


def run_battle(job: BattleJob) -> Dict[str, Any]:
    cfg = job.config
    ckpt_a = cfg["ckptA"]
    ckpt_b = cfg["ckptB"]
    sims_a = int(cfg["simsA"])
    sims_b = int(cfg["simsB"])
    n_games = int(cfg["games"])
    temperature = float(cfg.get("temperature", 0.1))
    # 走法模式：greedy = 貪婪（取訪問次數最高），stochastic = 依訪問次數機率採樣
    mode_a = str(cfg.get("modeA", "stochastic")).strip().lower()
    mode_b = str(cfg.get("modeB", "stochastic")).strip().lower()
    limit_temp = 1e-6
    temp_a = limit_temp if mode_a == "greedy" else temperature
    temp_b = limit_temp if mode_b == "greedy" else temperature
    device = torch.device("cpu")
    env_cfg = EnvConfig()

    model_a = load_model(ckpt_a, device)
    model_b = load_model(ckpt_b, device)

    wins = {"A": 0, "B": 0, "draw": 0}
    completed = 0

    for game_idx in range(n_games):
        a_player = WHITE if game_idx % 2 == 0 else BLACK

        env = TzaarEnv(env_cfg)
        env.reset()
        game_id = f"battle_{job.job_id}_g{game_idx:03d}"
        record = {
            "schema_version": 3,
            "game_id": game_id,
            "job_id": job.job_id,
            "models": {"A": ckpt_a, "B": ckpt_b},
            "simulations": {"A": sims_a, "B": sims_b},
            "modes": {"A": mode_a, "B": mode_b},
            "a_is_white": bool(a_player == WHITE),
            "temperature": temperature,
            "plies": [],
            "winner": None,
        }

        plies = record["plies"]

        while env.game_in_progress:
            current_player = env.current_player
            policy = model_a if current_player == a_player else model_b

            root_state = _build_phase_state_from_env(env)
            # 建構時可能已偵測到終局（例如「無強制吃子」），直接結束本局。
            if root_state.is_done():
                env._game_in_progress = False
                break
            with torch.no_grad():
                _, _, legal_mask, visits, _, _ = run_mcts(
                    policy,
                    root_state,
                    device,
                    apply_dirichlet_noise=False,
                    simulations=(sims_a if current_player == a_player else sims_b),
                )
            player_temp = temp_a if current_player == a_player else temp_b
            action_idx = _sample_action_from_visits(visits, legal_mask, player_temp)

            plies.append({
                "player": "A" if current_player == a_player else "B",
                "move_player": "White" if current_player == WHITE else "Black",
                "action_idx": int(action_idx),
                "board": extract_pieces(env),
            })

            env.step(action_idx)

        winner = _winner_from_env(env)
        record["winner"] = winner
        record["final_board"] = extract_pieces(env)

        if winner is None:
            wins["draw"] += 1
        elif winner == ("White" if a_player == WHITE else "Black"):
            wins["A"] += 1
        else:
            wins["B"] += 1

        _GAME_RECORDS[game_id] = record
        job.games_done.append(game_id)
        completed += 1
        job.progress = completed

    total = max(1, sum(wins.values()))
    return {
        "games": total,
        "A": wins["A"],
        "B": wins["B"],
        "draw": wins["draw"],
        "A_win_rate": wins["A"] / total,
        "B_win_rate": wins["B"] / total,
        "models": {"A": ckpt_a, "B": ckpt_b},
        "simulations": {"A": sims_a, "B": sims_b},
        "modes": {"A": mode_a, "B": mode_b},
        "game_ids": job.games_done,
    }


# ── strategy-move（AI 代走目前回合）─────────────────────────────────

def _reconstruct_game_from_payload(payload: Dict[str, Any]) -> Any:
    """由前端 board + layer + turn/step 還原 TzaarGame。"""
    from Tzaar import TzaarGame

    board = payload["board"]
    layer = payload["layer"]
    turn = payload.get("turn", "White")
    step = payload.get("step", 1)

    game = TzaarGame()
    # 還原每個格子
    for r in range(9):
        for c in range(9):
            code = board[r][c]
            height = layer[r][c]
            if code == -1:
                continue
            game.board.set_cell((r, c), int(code), int(height))

    # 重建 counts
    game._counts = {WHITE: {1: 0, 2: 0, 3: 0}, BLACK: {1: 0, 2: 0, 3: 0}}
    for (r, c) in game.board.valid_positions():
        cell = game.board.get_cell((r, c))
        if cell is None or cell[1] == 0:
            continue
        from Tzaar import piece_owner, piece_type
        game._counts[piece_owner(cell[0])][piece_type(cell[0])] += 1

    game.current_player = WHITE if turn == "White" else BLACK
    game._waiting_second_step = (step == 2)
    return game


def _move_to_token(kind: Any, src: Tuple[int, int], dst: Tuple[int, int], color: str) -> str:
    return f"{color},{src[0]},{src[1]},{dst[0]},{dst[1]}"


def run_strategy_move(payload: Dict[str, Any]) -> Dict[str, Any]:
    """執行目前回合的一手（move1）/兩手（move2）策略，回傳字串 token。"""
    name = payload.get("model")
    sims = int(payload.get("simulations", 256))
    device = torch.device("cpu")

    if not name:
        raise ValueError("未指定模型 (model)")
    model = load_model(name, device)

    game = _reconstruct_game_from_payload(payload)
    from Tzaar import MoveKind, WinReason, GameResult
    from TzaarAI import TzaarAIInterface
    from state.phase_state import PythonPhaseGameState, Stage

    ai = TzaarAIInterface(game)
    is_first_turn = payload.get("turn_number", 1) == 1 and game.current_player == WHITE
    step = payload.get("step", 1)

    # 第一步：必須吃子
    stage = Stage.NEED_STEP1 if step == 1 else Stage.NEED_STEP2
    state = PythonPhaseGameState(game=game, stage=stage, ai=ai)

    color = "W" if game.current_player == WHITE else "B"

    moves: List[str] = []
    current_state = state.clone()

    def do_one_decision(awaiting_step2: bool) -> Optional[str]:
        with torch.no_grad():
            _, _, legal_mask, visits, _, _ = run_mcts(
                model, current_state, device,
                apply_dirichlet_noise=False, simulations=sims,
            )
        idx = _sample_action_from_visits(visits, legal_mask, 1e-6)
        if idx == PASS_ACTION_IDX:
            current_state.apply_action(idx)
            return "P"

        from core.action import decode_action_idx
        kind_name, _, src_idx, direction_idx = decode_action_idx(idx)
        src = _VALID_POSITIONS[src_idx]
        from Tzaar import DIRECTIONS
        dst = current_state.game.board.first_occupied_in_direction(src, DIRECTIONS[direction_idx])
        if dst is None:
            raise ValueError("strategy move 方向無目標")
        current_state.apply_action(idx)
        return _move_to_token(kind_name, src, dst, color)

    # move1
    m1 = do_one_decision(False)
    moves.append(m1)
    if m1 != "P" and current_state.phase() == "step2_action":
        m2 = do_one_decision(True)
        moves.append(m2 if m2 is not None else "P")

    return {
        "move1": moves[0],
        "move2": moves[1] if len(moves) > 1 else None,
    }


# ── 人類 vs 模型（單步模型對弈）────────────────────────────────────

def run_model_move(payload: Dict[str, Any]) -> Dict[str, Any]:
    """執行人類 vs 模型對弈中「模型該下的單一步」。

    與 run_strategy_move 不同：這裡只下「一步」。
    由前端控制：輪到模型時呼叫，回傳該步走法後由前端套用並更新畫面。

    參數（payload）
    ---------------
    board / layer / turn / turn_number / step / model / simulations / mode

    回傳
    ----
    {
        "move": "W,fromR,fromC,toR,toC" 或 "P"（第二步 pass），
        "step": 該步對應的步數（1 或 2），
    }
    """
    name = payload.get("model")
    sims = int(payload.get("simulations", 200))
    mode = str(payload.get("mode", "stochastic")).strip().lower()
    device = torch.device("cpu")

    if not name:
        raise ValueError("未指定模型 (model)")
    model = load_model(name, device)

    game = _reconstruct_game_from_payload(payload)
    from Tzaar import MoveKind, WinReason, GameResult
    from TzaarAI import TzaarAIInterface
    from state.phase_state import PythonPhaseGameState, Stage

    ai = TzaarAIInterface(game)
    step = payload.get("step", 1)

    # 依目前該下哪一步決定階段
    if step == 1:
        # 第一步：必吃。先檢查無強制吃子（此時該玩家直接判負）。
        if ai.resolve_no_mandatory_capture_if_needed():
            stage = Stage.DONE
        else:
            stage = Stage.NEED_STEP1
    else:
        stage = Stage.NEED_STEP2

    state = PythonPhaseGameState(game=game, stage=stage, ai=ai)
    color = "W" if game.current_player == WHITE else "B"

    if state.is_done():
        # 模型輪到無強制吃子：視為已結束，回傳特殊 PASS 訊號
        return {"move": "P", "step": step, "game_over": True}

    with torch.no_grad():
        _, _, legal_mask, visits, _, _ = run_mcts(
            model,
            state,
            device,
            apply_dirichlet_noise=False,
            simulations=sims,
        )

    idx = _sample_action_from_visits(visits, legal_mask, (1e-6 if mode == "greedy" else 0.1))
    if idx == PASS_ACTION_IDX:
        return {"move": "P", "step": step}

    from core.action import decode_action_idx
    kind_name, _, src_idx, direction_idx = decode_action_idx(idx)
    src = _VALID_POSITIONS[src_idx]
    from Tzaar import DIRECTIONS
    dst = state.game.board.first_occupied_in_direction(src, DIRECTIONS[direction_idx])
    if dst is None:
        raise ValueError("model move 方向無目標")
    move_token = _move_to_token(kind_name, src, dst, color)

    return {"move": move_token, "step": step}


# ── HTTP Handler ─────────────────────────────────────────────────────

_JOBS: Dict[str, BattleJob] = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = self.path
        try:
            if path == "/api/checkpoints":
                return self._send_json({"checkpoints": list_checkpoint_files()})

            if path == "/api/games":
                summary = []
                for gid in sorted(_GAME_RECORDS.keys()):
                    rec = _GAME_RECORDS[gid]
                    summary.append({
                        "game_id": gid,
                        "models": rec["models"],
                        "simulations": rec["simulations"],
                        "winner": rec["winner"],
                        "ply_count": len(rec["plies"]),
                    })
                return self._send_json({"games": summary})

            if path.startswith("/api/games/"):
                gid = path[len("/api/games/"):]
                rec = _GAME_RECORDS.get(gid)
                if rec is None:
                    return self._send_json({"error": "找不到對局"}, 404)
                return self._send_json(rec)

            if path.startswith("/api/battle/"):
                job_id = path[len("/api/battle/"):]
                job = _JOBS.get(job_id)
                if job is None:
                    return self._send_json({"error": "找不到對戰工作"}, 404)
                return self._send_json(job.to_dict())

            return self._send_json({"error": "未知端點"}, 404)
        except Exception as exc:
            traceback.print_exc()
            return self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:
        path = self.path
        try:
            if path == "/api/battle":
                payload = self._read_json()
                ckpt_a = payload.get("ckptA", "")
                ckpt_b = payload.get("ckptB", "")
                sims_a = int(payload.get("simsA", 256))
                sims_b = int(payload.get("simsB", 256))
                games = int(payload.get("games", 10))
                mode_a = str(payload.get("modeA", "stochastic")).strip().lower()
                mode_b = str(payload.get("modeB", "stochastic")).strip().lower()
                if mode_a not in ("greedy", "stochastic"):
                    mode_a = "stochastic"
                if mode_b not in ("greedy", "stochastic"):
                    mode_b = "stochastic"
                if not ckpt_a or not ckpt_b:
                    return self._send_json({"error": "需指定模型 A 與模型 B"}, 400)
                if games < 1:
                    return self._send_json({"error": "場數需 >= 1"}, 400)

                job_id = uuid.uuid4().hex[:12]
                job = BattleJob(job_id, {
                    "ckptA": ckpt_a, "ckptB": ckpt_b,
                    "simsA": sims_a, "simsB": sims_b,
                    "games": games,
                    "modeA": mode_a, "modeB": mode_b,
                    "temperature": payload.get("temperature", 0.1),
                })
                _JOBS[job_id] = job
                job.start()
                return self._send_json({"job_id": job_id, "status": "running"})

            if path == "/api/strategy-move":
                payload = self._read_json()
                result = run_strategy_move(payload)
                return self._send_json(result)

            if path == "/api/model-move":
                payload = self._read_json()
                result = run_model_move(payload)
                return self._send_json(result)

            return self._send_json({"error": "未知端點"}, 404)
        except Exception as exc:
            traceback.print_exc()
            return self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main() -> None:
    print(f"[battle_server] 啟動於 http://{HOST}:{PORT}")
    print(f"[battle_server] checkpoints: {[f for f in list_checkpoint_files()[:5]]}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[battle_server] 停止")
        server.shutdown()


if __name__ == "__main__":
    main()
