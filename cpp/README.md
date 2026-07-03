# tzaar_cpp

This folder contains the C++ rules-engine implementation for Tzaar state transitions.

## What is implemented now

- pybind11 extension module named `tzaar_cpp`
- Unified action index helpers:
  - `encode_action_idx(kind, edge_idx=None)`
  - `decode_action_idx(action_idx)`
  - `action_edge_table()`
- C++ `PhaseGameState` backed by a C++ `Board` + `TzaarGame` implementation:
  - stage machine: `NEED_STEP1 -> NEED_STEP2 -> NEED_STEP1`
  - terminal handling: extinction / no mandatory capture
  - legal mask generation over unified 601-action space
  - `apply_action(action_idx)` with legality checks
  - `clone()`

This is the core game logic migration (rules + mask + transitions), but Python training code is still defaulting to Python backend until parity testing is complete.

## Build (Windows, MSVC)

From workspace root:

```powershell
python -m pip install pybind11
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64 -Dpybind11_DIR="$((python -m pybind11 --cmakedir).Trim())"
cmake --build cpp/build --config Release
```

The built module will be at:

- `cpp/build/Release/tzaar_cpp*.pyd`

To import it from Python, add that directory to `sys.path`.

## Next implementation targets

1. Add parity tests against `Tzaar.py` and `TzaarAI.py` (bitwise mask and transition parity).
2. Wire self-play runtime to instantiate C++ state path when backend is `cpp`.
3. Optional next phase: migrate MCTS inner loop to C++.
