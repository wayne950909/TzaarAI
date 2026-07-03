from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module-dir",
        type=Path,
        default=Path("cpp/build/Release"),
        help="Directory containing tzaar_cpp*.pyd",
    )
    args = parser.parse_args()

    module_dir = args.module_dir.resolve()
    if not module_dir.exists():
        raise FileNotFoundError(f"module dir not found: {module_dir}")

    sys.path.insert(0, str(module_dir))

    import tzaar_cpp  # type: ignore

    print("module loaded:", tzaar_cpp.__doc__)
    print("N_ACTIONS:", tzaar_cpp.N_ACTIONS)
    print("PASS_ACTION_IDX:", tzaar_cpp.PASS_ACTION_IDX)

    capture_idx = tzaar_cpp.encode_action_idx("capture", 12)
    reinforce_idx = tzaar_cpp.encode_action_idx("reinforce", 12)
    pass_idx = tzaar_cpp.encode_action_idx("pass")

    print("encode capture edge 12 ->", capture_idx)
    print("encode reinforce edge 12 ->", reinforce_idx)
    print("encode pass ->", pass_idx)

    print("decode 12 ->", tzaar_cpp.decode_action_idx(12))
    print("decode 312 ->", tzaar_cpp.decode_action_idx(312))
    print("decode 600 ->", tzaar_cpp.decode_action_idx(600))
    print("edge table size ->", len(tzaar_cpp.action_edge_table()))

    state = tzaar_cpp.PhaseGameState()
    print("initial phase:", state.phase())
    print("initial turn:", state.turn_number())
    print("initial player:", state.current_player())
    print("initial piece counts:", state.piece_counts())
    print("initial occupied cells:", len(state.board_cells()))
    print("step1 legal count:", sum(1 for x in state.legal_mask() if x))

    state.apply_action(0)
    print("after step1 phase:", state.phase())
    print("step2 legal count:", sum(1 for x in state.legal_mask() if x))

    state.apply_action(tzaar_cpp.PASS_ACTION_IDX)
    print("after step2 phase:", state.phase())
    print("current player:", state.current_player())
    print("current turn:", state.turn_number())


if __name__ == "__main__":
    main()
