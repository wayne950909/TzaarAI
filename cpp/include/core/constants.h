#ifndef TZAAR_CORE_CONSTANTS_H_
#define TZAAR_CORE_CONSTANTS_H_

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace tzaar {

// ─── 棋盤尺寸 ─────────────────────────────────────────────
constexpr int kRows = 9;
constexpr int kCols = 9;
constexpr int kBoardChannels = 12;
constexpr int kGlobalFeatureDim = 12;
constexpr int kBoardFlatSize = kBoardChannels * kRows * kCols;  // 972

// ─── 動作空間 ─────────────────────────────────────────────
constexpr int kEdgeActionCount = 300;
constexpr int kCaptureOffset = 0;
constexpr int kReinforceOffset = 300;
constexpr int kPassActionIdx = 600;
constexpr int kActionCount = 601;

// ─── 玩家 ─────────────────────────────────────────────────
constexpr int kWhite = 1;
constexpr int kBlack = -1;

// ─── 方向 ─────────────────────────────────────────────────
constexpr int kDirectionCount = 6;

constexpr std::array<std::pair<int, int>, kDirectionCount> kDirections = {{
    {-1, 0},   // 0: Up
    {1, 0},    // 1: Down
    {0, -1},   // 2: Left
    {0, 1},    // 3: Right
    {-1, -1},  // 4: UpLeft
    {1, 1},    // 5: DnRight
}};

// ─── 棋盤佈局 (9x9, -1 = 洞位) ────────────────────────────
constexpr std::array<std::array<int, kCols>, kRows> kBoardLayout = {{
    {{3, 3, 3, 3, 6, -1, -1, -1, -1}},
    {{6, 2, 2, 2, 5, 6, -1, -1, -1}},
    {{6, 5, 1, 1, 4, 5, 6, -1, -1}},
    {{6, 5, 4, 3, 6, 4, 5, 6, -1}},
    {{6, 5, 4, 6, -1, 3, 1, 2, 3}},
    {{-1, 3, 2, 1, 3, 6, 1, 2, 3}},
    {{-1, -1, 3, 2, 1, 4, 4, 2, 3}},
    {{-1, -1, -1, 3, 2, 5, 5, 5, 3}},
    {{-1, -1, -1, -1, 3, 6, 6, 6, 6}},
}};

// ─── 列舉 ─────────────────────────────────────────────────
enum class MoveKind {
  Capture,
  Reinforce,
  Pass,
};

enum class WinReason {
  Extinction,
  NoMandatoryCapture,
};

enum class Stage {
  NeedStep1,
  NeedStep2,
  Done,
};

// ─── 座標 (必須在 Move 之前定義) ──────────────────────────
struct Pos {
  int row;
  int col;

  bool operator==(const Pos& other) const {
    return row == other.row && col == other.col;
  }
};

inline int pack_pos(const Pos& pos) {
  return pos.row * kCols + pos.col;
}

// ─── Move 結構 ────────────────────────────────────────────
struct Move {
  MoveKind kind;
  std::optional<Pos> src;
  std::optional<Pos> dst;
};

struct GameResult {
  int winner;
  WinReason reason;
};

// ─── 輔助函式 ─────────────────────────────────────────────
inline int player_to_index(int player) {
  if (player == kWhite) return 0;
  if (player == kBlack) return 1;
  throw std::invalid_argument("player must be WHITE(1) or BLACK(-1)");
}

inline int piece_owner(int piece) {
  if (piece >= 1 && piece <= 3) return kWhite;
  if (piece >= 4 && piece <= 6) return kBlack;
  throw std::invalid_argument("unknown piece code");
}

inline int piece_type(int piece) {
  if (piece == 1 || piece == 4) return 1;
  if (piece == 2 || piece == 5) return 2;
  if (piece == 3 || piece == 6) return 3;
  throw std::invalid_argument("unknown piece code");
}

inline std::string move_kind_to_string(MoveKind kind) {
  switch (kind) {
    case MoveKind::Capture:   return "capture";
    case MoveKind::Reinforce: return "reinforce";
    case MoveKind::Pass:      return "pass";
  }
  throw std::runtime_error("unreachable move kind");
}

inline float terminal_value_for_current_player(int winner, int current_player) {
  if (winner == 0) return 0.0f;
  return winner == current_player ? 1.0f : -1.0f;
}

}  // namespace tzaar

#endif  // TZAAR_CORE_CONSTANTS_H_

