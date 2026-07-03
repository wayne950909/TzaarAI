#include "state/phase_state.h"
#include "core/action.h"

#include <cstring>
#include <stdexcept>

namespace tzaar {

PhaseGameState::PhaseGameState() : game_(), stage_(Stage::NeedStep1) {}

bool PhaseGameState::is_done() const {
  return stage_ == Stage::Done || game_.is_game_over();
}

int PhaseGameState::current_player() const {
  return game_.current_player();
}

int PhaseGameState::winner() const {
  const auto w = game_.winner();
  return w.has_value() ? *w : 0;
}

std::string PhaseGameState::phase() const {
  if (is_done()) return "";
  if (stage_ == Stage::NeedStep1) return "step1_capture";
  if (stage_ == Stage::NeedStep2) return "step2_action";
  return "";
}

int PhaseGameState::turn_number() const {
  return game_.turn_number();
}

// ─── 合法動作遮罩 ─────────────────────────────────────────

std::vector<bool> PhaseGameState::legal_mask() {
  if (is_done()) return std::vector<bool>(kActionCount, false);
  if (cache_valid_) return cached_mask_;

  std::vector<bool> mask(kActionCount, false);
  if (stage_ == Stage::NeedStep1) {
    if (game_.resolve_no_mandatory_capture_if_needed()) {
      stage_ = Stage::Done;
      invalidate_cache();
      return std::vector<bool>(kActionCount, false);
    }
    fill_step1_mask(mask);
  } else if (stage_ == Stage::NeedStep2) {
    fill_step2_mask(mask);
  } else {
    throw std::runtime_error("Unsupported stage");
  }

  cached_mask_ = mask;
  cache_valid_ = true;
  return cached_mask_;
}

// ─── 動作執行 ─────────────────────────────────────────────

void PhaseGameState::apply_action(int action_idx) {
  if (is_done()) return;
  if (action_idx < 0 || action_idx >= kActionCount)
    throw std::out_of_range("action_idx out of range");

  std::vector<bool> mask = legal_mask();
  if (!mask[static_cast<std::size_t>(action_idx)])
    throw std::invalid_argument("illegal action");
  invalidate_cache();

  if (stage_ == Stage::NeedStep1) {
    Move move = reconstruct_unified_action(action_idx, true);
    game_.play_first_step(move);
    if (game_.is_game_over()) {
      stage_ = Stage::Done;
    } else if (game_.is_waiting_second_step()) {
      stage_ = Stage::NeedStep2;
    } else {
      stage_ = Stage::NeedStep1;
      advance_from_first_src();
    }
    return;
  }

  if (stage_ == Stage::NeedStep2) {
    if (action_idx == kPassActionIdx) {
      game_.play_second_step(std::nullopt);
    } else {
      Move move = reconstruct_unified_action(action_idx, false);
      game_.play_second_step(move);
    }
    stage_ = game_.is_game_over() ? Stage::Done : Stage::NeedStep1;
    advance_from_first_src();
    return;
  }

  throw std::runtime_error("Unsupported stage for apply_action");
}

void PhaseGameState::apply_action_trusted(int action_idx) {
  if (is_done()) return;
  if (action_idx < 0 || action_idx >= kActionCount)
    throw std::out_of_range("action_idx out of range");

  invalidate_cache();

  if (stage_ == Stage::NeedStep1) {
    Move move = reconstruct_unified_action(action_idx, true);
    game_.play_first_step(move);
    if (game_.is_game_over()) {
      stage_ = Stage::Done;
    } else if (game_.is_waiting_second_step()) {
      stage_ = Stage::NeedStep2;
    } else {
      stage_ = Stage::NeedStep1;
      advance_from_first_src();
    }
    return;
  }

  if (stage_ == Stage::NeedStep2) {
    if (action_idx == kPassActionIdx) {
      game_.play_second_step(std::nullopt);
    } else {
      Move move = reconstruct_unified_action(action_idx, false);
      game_.play_second_step(move);
    }
    stage_ = game_.is_game_over() ? Stage::Done : Stage::NeedStep1;
    advance_from_first_src();
    return;
  }

  throw std::runtime_error("Unsupported stage for apply_action_trusted");
}

// ─── 複製 ─────────────────────────────────────────────────

PhaseGameState PhaseGameState::clone() const {
  PhaseGameState copied;
  copied.game_ = game_;  // TzaarGame copy
  copied.stage_ = stage_;
  copied.cache_valid_ = false;  // cache invalidated on clone
  // cached_mask_ intentionally not copied
  return copied;
}

// ─── 快照（葉節點用） ─────────────────────────────────────

void PhaseGameState::fill_snapshot_metadata(LeafSnapshot& snapshot, int node_id) {
  snapshot.node_id = node_id;
  snapshot.current_player = current_player();
  snapshot.turn_number = turn_number();
  snapshot.winner = winner();
  snapshot.is_done = is_done();
  snapshot.phase = phase();

  const auto counts = game_.piece_counts();
  for (int i = 0; i < 3; ++i) {
    snapshot.white_counts[static_cast<std::size_t>(i)] = counts[0][i + 1];
    snapshot.black_counts[static_cast<std::size_t>(i)] = counts[1][i + 1];
  }

  const std::vector<bool> mask = legal_mask();
  snapshot.legal_mask.resize(mask.size());
  for (std::size_t i = 0; i < mask.size(); ++i) {
    snapshot.legal_mask[i] = mask[i] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
  }
}

LeafSnapshot PhaseGameState::leaf_snapshot(int node_id) {
  LeafSnapshot snapshot;
  fill_snapshot_metadata(snapshot, node_id);
  build_cnn_features(
      game_.board(), current_player(), turn_number(), phase(),
      game_.piece_counts(),
      snapshot.board_state_flat, snapshot.global_features);
  return snapshot;
}

// ─── 私有方法 ─────────────────────────────────────────────

void PhaseGameState::invalidate_cache() {
  cache_valid_ = false;
  cached_mask_.clear();
}

void PhaseGameState::advance_from_first_src() {
  if (stage_ == Stage::NeedStep1) {
    if (game_.resolve_no_mandatory_capture_if_needed()) {
      stage_ = Stage::Done;
      invalidate_cache();
    }
  }
}

void PhaseGameState::fill_step1_mask(std::vector<bool>& mask) const {
  const auto& board = game_.board();
  const auto& pos_to_idx = board.pos_to_idx_map();
  const int player = game_.current_player();

  for (const Pos& src : board.valid_positions()) {
    if (board.is_empty(src)) continue;
    if (piece_owner(board.top_piece(src)) != player) continue;

    const int src_height = board.height(src);
    const int src_idx = pos_to_idx.at(pack_pos(src));

    for (int d = 0; d < kDirectionCount; ++d) {
      auto dst = board.first_occupied_in_direction(src, d);
      if (!dst.has_value()) continue;
      if (piece_owner(board.top_piece(*dst)) == player) continue;
      if (board.height(*dst) > src_height) continue;

      auto edge_idx = action_space().edge_to_idx(src_idx, d);
      if (edge_idx.has_value()) {
        mask[static_cast<std::size_t>(kCaptureOffset + *edge_idx)] = true;
      }
    }
  }
}

void PhaseGameState::fill_step2_mask(std::vector<bool>& mask) const {
  const auto& board = game_.board();
  const auto& pos_to_idx = board.pos_to_idx_map();
  const int player = game_.current_player();

  for (const Pos& src : board.valid_positions()) {
    if (board.is_empty(src)) continue;
    if (piece_owner(board.top_piece(src)) != player) continue;

    const int src_height = board.height(src);
    const int src_idx = pos_to_idx.at(pack_pos(src));

    for (int d = 0; d < kDirectionCount; ++d) {
      auto dst = board.first_occupied_in_direction(src, d);
      if (!dst.has_value()) continue;

      auto edge_idx = action_space().edge_to_idx(src_idx, d);
      if (!edge_idx.has_value()) continue;

      const bool dst_is_self = piece_owner(board.top_piece(*dst)) == player;
      if (dst_is_self) {
        mask[static_cast<std::size_t>(kReinforceOffset + *edge_idx)] = true;
      } else if (board.height(*dst) <= src_height) {
        mask[static_cast<std::size_t>(kCaptureOffset + *edge_idx)] = true;
      }
    }
  }

  if (game_.can_second_step_pass()) {
    mask[static_cast<std::size_t>(kPassActionIdx)] = true;
  }
}

Move PhaseGameState::reconstruct_unified_action(int action_idx, bool is_step1) const {
  DecodedAction decoded = decode_action_idx(action_idx);
  if (decoded.kind == MoveKind::Pass) {
    if (is_step1) throw std::invalid_argument("step1 only allows capture actions");
    Move m;
    m.kind = MoveKind::Pass;
    m.src = std::nullopt;
    m.dst = std::nullopt;
    return m;
  }
  if (is_step1 && decoded.kind != MoveKind::Capture)
    throw std::invalid_argument("step1 only allows capture actions");

  return reconstruct_non_pass_move(game_.board(), decoded.kind, decoded.src_idx, decoded.direction_idx);
}

}  // namespace tzaar
