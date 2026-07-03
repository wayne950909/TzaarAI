#include "core/action.h"
#include "core/board.h"

#include <stdexcept>

namespace tzaar {

ActionSpace::ActionSpace() {
  build_from_geometry();
}

std::optional<int> ActionSpace::edge_to_idx(int src_idx, int direction_idx) const {
  const std::int64_t key = (static_cast<std::int64_t>(src_idx) << 8) + direction_idx;
  auto it = edge_to_idx_.find(key);
  if (it == edge_to_idx_.end()) return std::nullopt;
  return it->second;
}

void ActionSpace::build_from_geometry() {
  Board board;
  const auto& positions = board.valid_positions();
  for (int src_idx = 0; src_idx < static_cast<int>(positions.size()); ++src_idx) {
    const Pos& src = positions[src_idx];
    for (int d = 0; d < kDirectionCount; ++d) {
      auto dst = board.first_occupied_in_direction(src, d);
      if (!dst.has_value()) continue;

      const int edge_idx = static_cast<int>(edge_table_.size());
      edge_table_.push_back({src_idx, d});
      const std::int64_t key = (static_cast<std::int64_t>(src_idx) << 8) + d;
      edge_to_idx_[key] = edge_idx;
    }
  }

  if (static_cast<int>(edge_table_.size()) != kEdgeActionCount) {
    throw std::runtime_error("Generated ACTION_EDGE_TABLE size mismatch");
  }
}

const ActionSpace& action_space() {
  static ActionSpace space;
  return space;
}

// ─── 編解碼 ───────────────────────────────────────────────

DecodedAction decode_action_idx(int action_idx) {
  if (action_idx == kPassActionIdx) {
    return DecodedAction{MoveKind::Pass, -1, -1, -1};
  }
  if (action_idx < 0 || action_idx >= kPassActionIdx) {
    throw std::out_of_range("action_idx out of range");
  }

  MoveKind kind = MoveKind::Capture;
  int edge_idx = action_idx;
  if (action_idx >= kReinforceOffset) {
    kind = MoveKind::Reinforce;
    edge_idx = action_idx - kReinforceOffset;
  }
  const auto& edge = action_space().edge_table().at(static_cast<std::size_t>(edge_idx));
  return DecodedAction{kind, edge_idx, edge.first, edge.second};
}

int encode_action_idx(const std::string& kind, std::optional<int> edge_idx) {
  if (kind == "pass") return kPassActionIdx;
  if (!edge_idx.has_value()) throw std::invalid_argument("edge_idx required for non-pass actions");
  const int idx = edge_idx.value();
  if (idx < 0 || idx >= kEdgeActionCount) throw std::out_of_range("edge_idx out of range");
  if (kind == "capture")  return kCaptureOffset + idx;
  if (kind == "reinforce") return kReinforceOffset + idx;
  throw std::invalid_argument("kind must be capture, reinforce, or pass");
}

Move reconstruct_non_pass_move(const Board& board, MoveKind kind, int src_idx, int direction_idx) {
  const auto& positions = board.valid_positions();
  if (src_idx < 0 || src_idx >= static_cast<int>(positions.size()))
    throw std::out_of_range("src_idx out of range");

  const Pos src = positions[src_idx];
  auto dst = board.first_occupied_in_direction(src, direction_idx);
  if (!dst.has_value())
    throw std::invalid_argument("No occupied cell found in chosen direction");

  Move m;
  m.kind = kind;
  m.src = src;
  m.dst = *dst;
  return m;
}

}  // namespace tzaar
