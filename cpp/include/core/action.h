#ifndef TZAAR_CORE_ACTION_H_
#define TZAAR_CORE_ACTION_H_

#include "core/constants.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tzaar {

// Forward declarations (defined in constants.h or game.h)
struct Move;
class Board;

struct DecodedAction {
  MoveKind kind;
  int edge_idx;
  int src_idx;
  int direction_idx;
};

class ActionSpace {
 public:
  ActionSpace();

  const std::vector<std::pair<int, int>>& edge_table() const { return edge_table_; }
  std::optional<int> edge_to_idx(int src_idx, int direction_idx) const;

 private:
  std::vector<std::pair<int, int>> edge_table_;
  std::unordered_map<std::int64_t, int> edge_to_idx_;

  void build_from_geometry();
};

const ActionSpace& action_space();

// ─── 編解碼 ───────────────────────────────────────────────
DecodedAction decode_action_idx(int action_idx);
int encode_action_idx(const std::string& kind, std::optional<int> edge_idx);

// ─── 從解碼重建 Move ─────────────────────────────────────
// 需要 TzaarGame 來查詢 board 以找到實際 dst 位置
Move reconstruct_non_pass_move(const Board& board, MoveKind kind, int src_idx, int direction_idx);

}  // namespace tzaar

#endif  // TZAAR_CORE_ACTION_H_

