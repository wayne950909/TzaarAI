#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kActionCount = 601;
constexpr int kPassActionIdx = 600;
constexpr int kCaptureOffset = 0;
constexpr int kReinforceOffset = 300;
constexpr int kEdgeActionCount = 300;
constexpr int kWhite = 1;
constexpr int kBlack = -1;
constexpr int kRows = 9;
constexpr int kCols = 9;
constexpr int kDirectionCount = 6;
constexpr int kBoardChannels = 12;
constexpr int kGlobalFeatureDim = 12;
constexpr int kBoardFlatSize = kBoardChannels * kRows * kCols;

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

constexpr std::array<std::pair<int, int>, kDirectionCount> kDirections = {{
    {-1, 0},
    {1, 0},
    {0, -1},
    {0, 1},
    {-1, -1},
    {1, 1},
}};

enum class Stage {
  NeedStep1,
  NeedStep2,
  Done,
};

enum class WinReason {
  Extinction,
  NoMandatoryCapture,
};

enum class MoveKind {
  Capture,
  Reinforce,
  Pass,
};

struct Pos {
  int row;
  int col;

  // function: function implementation.
  bool operator==(const Pos& other) const {
    return row == other.row && col == other.col;
  }
};

struct Move {
  MoveKind kind;
  std::optional<Pos> src;
  std::optional<Pos> dst;
};

struct GameResult {
  int winner;
  WinReason reason;
};

struct Cell {
  bool valid = false;
  int top_piece = 0;
  int height = 0;
};

// pack_pos: function implementation.
int pack_pos(const Pos& pos) {
  return pos.row * kCols + pos.col;
}

// player_to_index: function implementation.
int player_to_index(int player) {
  if (player == kWhite) {
    return 0;
  }
  if (player == kBlack) {
    return 1;
  }
  throw std::invalid_argument("player must be WHITE(1) or BLACK(-1)");
}

// piece_owner: function implementation.
int piece_owner(int piece) {
  if (piece >= 1 && piece <= 3) {
    return kWhite;
  }
  if (piece >= 4 && piece <= 6) {
    return kBlack;
  }
  throw std::invalid_argument("unknown piece code");
}

// piece_type: function implementation.
int piece_type(int piece) {
  if (piece == 1 || piece == 4) {
    return 1;
  }
  if (piece == 2 || piece == 5) {
    return 2;
  }
  if (piece == 3 || piece == 6) {
    return 3;
  }
  throw std::invalid_argument("unknown piece code");
}

// move_kind_to_string: function implementation.
std::string move_kind_to_string(MoveKind kind) {
  switch (kind) {
    case MoveKind::Capture:
      return "capture";
    case MoveKind::Reinforce:
      return "reinforce";
    case MoveKind::Pass:
      return "pass";
  }
  throw std::runtime_error("unreachable move kind");
}

class Board {
 public:
  // Board: function implementation.
  Board() {
    init_from_layout();
  }

  // is_valid_cell: function implementation.
  bool is_valid_cell(const Pos& pos) const {
    return is_inside(pos) && cells_[pos.row][pos.col].valid;
  }

  // is_empty: function implementation.
  bool is_empty(const Pos& pos) const {
    const Cell& cell = get_cell(pos);
    return cell.height == 0;
  }

  // top_piece: function implementation.
  int top_piece(const Pos& pos) const {
    const Cell& cell = get_cell(pos);
    if (cell.height == 0) {
      throw std::invalid_argument("No top piece on empty cell");
    }
    return cell.top_piece;
  }

  // height: function implementation.
  int height(const Pos& pos) const {
    return get_cell(pos).height;
  }

  // set_cell: function implementation.
  void set_cell(const Pos& pos, int piece_code, int height) {
    Cell& cell = get_mut_cell(pos);
    cell.top_piece = piece_code;
    cell.height = height;
  }

  // valid_positions: function implementation.
  const std::vector<Pos>& valid_positions() const {
    return valid_positions_;
  }

  // pos_to_idx_map: function implementation.
  const std::unordered_map<int, int>& pos_to_idx_map() const {
    return pos_to_idx_;
  }

  // first_occupied_in_direction: function implementation.
  std::optional<Pos> first_occupied_in_direction(const Pos& start, int direction_idx) const {
    auto ray_it = rays_.find(pack_pos(start));
    if (ray_it == rays_.end()) {
      return std::nullopt;
    }
    if (direction_idx < 0 || direction_idx >= kDirectionCount) {
      throw std::out_of_range("direction_idx out of range");
    }
    const std::vector<Pos>& ray = ray_it->second[direction_idx];
    for (const Pos& pos : ray) {
      if (!is_empty(pos)) {
        return pos;
      }
    }
    return std::nullopt;
  }

 private:
  std::array<std::array<Cell, kCols>, kRows> cells_{};
  std::vector<Pos> valid_positions_;
  std::unordered_map<int, int> pos_to_idx_;
  std::unordered_map<int, std::array<std::vector<Pos>, kDirectionCount>> rays_;

  // is_inside: function implementation.
  bool is_inside(const Pos& pos) const {
    return pos.row >= 0 && pos.row < kRows && pos.col >= 0 && pos.col < kCols;
  }

  // get_cell: function implementation.
  const Cell& get_cell(const Pos& pos) const {
    if (!is_inside(pos)) {
      throw std::out_of_range("position out of bounds");
    }
    const Cell& cell = cells_[pos.row][pos.col];
    if (!cell.valid) {
      throw std::invalid_argument("position is invalid cell");
    }
    return cell;
  }

  // get_mut_cell: function implementation.
  Cell& get_mut_cell(const Pos& pos) {
    if (!is_inside(pos)) {
      throw std::out_of_range("position out of bounds");
    }
    Cell& cell = cells_[pos.row][pos.col];
    if (!cell.valid) {
      throw std::invalid_argument("position is invalid cell");
    }
    return cell;
  }

  // init_from_layout: function implementation.
  void init_from_layout() {
    valid_positions_.clear();
    pos_to_idx_.clear();
    rays_.clear();

    for (int r = 0; r < kRows; ++r) {
      for (int c = 0; c < kCols; ++c) {
        const int v = kBoardLayout[r][c];
        Cell cell;
        if (v == -1) {
          cell.valid = false;
          cell.top_piece = 0;
          cell.height = 0;
        // if: function implementation.
        } else if (v == 0) {
          cell.valid = true;
          cell.top_piece = 0;
          cell.height = 0;
        } else {
          cell.valid = true;
          cell.top_piece = v;
          cell.height = 1;
        }
        cells_[r][c] = cell;
        if (cell.valid) {
          valid_positions_.push_back(Pos{r, c});
        }
      }
    }

    for (int i = 0; i < static_cast<int>(valid_positions_.size()); ++i) {
      pos_to_idx_[pack_pos(valid_positions_[i])] = i;
    }

    for (const Pos& pos : valid_positions_) {
      std::array<std::vector<Pos>, kDirectionCount> ray_pack;
      for (int d = 0; d < kDirectionCount; ++d) {
        const auto [dr, dc] = kDirections[d];
        int rr = pos.row + dr;
        int cc = pos.col + dc;
        while (rr >= 0 && rr < kRows && cc >= 0 && cc < kCols && cells_[rr][cc].valid) {
          ray_pack[d].push_back(Pos{rr, cc});
          rr += dr;
          cc += dc;
        }
      }
      rays_[pack_pos(pos)] = ray_pack;
    }
  }
};

class TzaarGame {
 public:
  // TzaarGame: function implementation.
  TzaarGame() : board_(), current_player_(kWhite), turn_number_(1), waiting_second_step_(false) {
    counts_[0] = {0, 0, 0, 0};
    counts_[1] = {0, 0, 0, 0};
    for (const Pos& pos : board_.valid_positions()) {
      if (board_.is_empty(pos)) {
        continue;
      }
      const int top = board_.top_piece(pos);
      counts_[player_to_index(piece_owner(top))][piece_type(top)] += 1;
    }
  }

  // board: function implementation.
  const Board& board() const {
    return board_;
  }

  // is_waiting_second_step: function implementation.
  bool is_waiting_second_step() const {
    return waiting_second_step_;
  }

  // is_game_over: function implementation.
  bool is_game_over() const {
    return result_.has_value();
  }

  // current_player: function implementation.
  int current_player() const {
    return current_player_;
  }

  // turn_number: function implementation.
  int turn_number() const {
    return turn_number_;
  }

  // winner: function implementation.
  std::optional<int> winner() const {
    if (!result_.has_value()) {
      return std::nullopt;
    }
    return result_->winner;
  }

  // piece_counts: function implementation.
  std::array<std::array<int, 4>, 2> piece_counts() const {
    return counts_;
  }

  // can_second_step_pass: function implementation.
  bool can_second_step_pass() const {
    return !is_game_over() && waiting_second_step_;
  }

  // has_capture_for_player: function implementation.
  bool has_capture_for_player(int player) const {
    for (const Pos& src : board_.valid_positions()) {
      if (board_.is_empty(src)) {
        continue;
      }
      if (piece_owner(board_.top_piece(src)) != player) {
        continue;
      }
      const int src_height = board_.height(src);
      for (int d = 0; d < kDirectionCount; ++d) {
        std::optional<Pos> dst = board_.first_occupied_in_direction(src, d);
        if (!dst.has_value()) {
          continue;
        }
        if (piece_owner(board_.top_piece(*dst)) == player) {
          continue;
        }
        if (board_.height(*dst) <= src_height) {
          return true;
        }
      }
    }
    return false;
  }

  // has_reinforce_for_player: function implementation.
  bool has_reinforce_for_player(int player) const {
    for (const Pos& src : board_.valid_positions()) {
      if (board_.is_empty(src)) {
        continue;
      }
      if (piece_owner(board_.top_piece(src)) != player) {
        continue;
      }
      for (int d = 0; d < kDirectionCount; ++d) {
        std::optional<Pos> dst = board_.first_occupied_in_direction(src, d);
        if (!dst.has_value()) {
          continue;
        }
        if (piece_owner(board_.top_piece(*dst)) == player) {
          return true;
        }
      }
    }
    return false;
  }

  // resolve_no_mandatory_capture_if_needed: function implementation.
  bool resolve_no_mandatory_capture_if_needed() {
    if (is_game_over() || waiting_second_step_) {
      return false;
    }
    if (has_capture_for_player(current_player_)) {
      return false;
    }
    result_ = GameResult{-current_player_, WinReason::NoMandatoryCapture};
    return true;
  }

  // play_first_step: function implementation.
  void play_first_step(const Move& move) {
    if (is_game_over()) {
      throw std::invalid_argument("Game is already over");
    }
    if (waiting_second_step_) {
      throw std::invalid_argument("Cannot play first step while waiting second step");
    }
    if (resolve_no_mandatory_capture_if_needed()) {
      return;
    }

    const bool white_first_turn = (turn_number_ == 1 && current_player_ == kWhite);
    apply_capture(move, current_player_);
    if (check_immediate_extinction()) {
      return;
    }

    if (white_first_turn) {
      waiting_second_step_ = false;
      end_turn();
      return;
    }
    waiting_second_step_ = true;
  }

  // play_second_step: function implementation.
  void play_second_step(const std::optional<Move>& move_opt) {
    if (is_game_over()) {
      throw std::invalid_argument("Game is already over");
    }
    if (!waiting_second_step_) {
      throw std::invalid_argument("Cannot play second step before first step");
    }

    Move move = move_opt.value_or(Move{MoveKind::Pass, std::nullopt, std::nullopt});
    const int player = current_player_;

    if (move.kind == MoveKind::Capture) {
      apply_capture(move, player);
      if (check_immediate_extinction()) {
        waiting_second_step_ = false;
        return;
      }
    // if: function implementation.
    } else if (move.kind == MoveKind::Reinforce) {
      apply_reinforce(move, player);
      if (check_immediate_extinction()) {
        waiting_second_step_ = false;
        return;
      }
    } else {
      if (move.kind != MoveKind::Pass || !can_second_step_pass()) {
        throw std::invalid_argument("Second move is not legal");
      }
    }

    waiting_second_step_ = false;
    end_turn();
  }

 private:
  Board board_;
  int current_player_;
  int turn_number_;
  bool waiting_second_step_;
  std::optional<GameResult> result_;
  std::array<std::array<int, 4>, 2> counts_{};

  // extinction_winner: function implementation.
  std::optional<int> extinction_winner() const {
    for (int player : {kWhite, kBlack}) {
      const int pi = player_to_index(player);
      if (counts_[pi][1] == 0 || counts_[pi][2] == 0 || counts_[pi][3] == 0) {
        return -player;
      }
    }
    return std::nullopt;
  }

  // apply_capture: function implementation.
  void apply_capture(const Move& move, int player) {
    if (move.kind != MoveKind::Capture || !move.src.has_value() || !move.dst.has_value()) {
      throw std::invalid_argument("Invalid capture move");
    }
    const Pos src = *move.src;
    const Pos dst = *move.dst;
    if (board_.is_empty(src) || board_.is_empty(dst)) {
      throw std::invalid_argument("Capture move on empty cell");
    }
    const int src_top = board_.top_piece(src);
    const int dst_top = board_.top_piece(dst);
    const int src_height = board_.height(src);
    const int dst_height = board_.height(dst);
    if (piece_owner(src_top) != player || piece_owner(dst_top) == player) {
      throw std::invalid_argument("Capture ownership constraints violated");
    }
    if (dst_height > src_height) {
      throw std::invalid_argument("Cannot capture stronger stack");
    }

    counts_[player_to_index(piece_owner(dst_top))][piece_type(dst_top)] -= 1;
    board_.set_cell(src, 0, 0);
    board_.set_cell(dst, src_top, src_height);
  }

  // apply_reinforce: function implementation.
  void apply_reinforce(const Move& move, int player) {
    if (move.kind != MoveKind::Reinforce || !move.src.has_value() || !move.dst.has_value()) {
      throw std::invalid_argument("Invalid reinforce move");
    }
    const Pos src = *move.src;
    const Pos dst = *move.dst;
    if (board_.is_empty(src) || board_.is_empty(dst)) {
      throw std::invalid_argument("Reinforce move on empty cell");
    }
    const int src_top = board_.top_piece(src);
    const int dst_top = board_.top_piece(dst);
    const int src_height = board_.height(src);
    const int dst_height = board_.height(dst);

    if (piece_owner(src_top) != player || piece_owner(dst_top) != player) {
      throw std::invalid_argument("Reinforce ownership constraints violated");
    }

    counts_[player_to_index(piece_owner(dst_top))][piece_type(dst_top)] -= 1;
    board_.set_cell(src, 0, 0);
    board_.set_cell(dst, src_top, src_height + dst_height);
  }

  // check_immediate_extinction: function implementation.
  bool check_immediate_extinction() {
    const std::optional<int> winner = extinction_winner();
    if (!winner.has_value()) {
      return false;
    }
    result_ = GameResult{*winner, WinReason::Extinction};
    return true;
  }

  // end_turn: function implementation.
  void end_turn() {
    current_player_ *= -1;
    turn_number_ += 1;
  }
};

class ActionSpace {
 public:
  // ActionSpace: function implementation.
  ActionSpace() {
    build_from_geometry();
  }

  // edge_table: function implementation.
  const std::vector<std::pair<int, int>>& edge_table() const {
    return edge_table_;
  }

  // edge_to_idx: function implementation.
  std::optional<int> edge_to_idx(int src_idx, int direction_idx) const {
    const std::int64_t key = (static_cast<std::int64_t>(src_idx) << 8) + direction_idx;
    auto it = edge_to_idx_.find(key);
    if (it == edge_to_idx_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

 private:
  std::vector<std::pair<int, int>> edge_table_;
  std::unordered_map<std::int64_t, int> edge_to_idx_;

  // build_from_geometry: function implementation.
  void build_from_geometry() {
    Board board;
    const auto& positions = board.valid_positions();
    for (int src_idx = 0; src_idx < static_cast<int>(positions.size()); ++src_idx) {
      const Pos& src = positions[src_idx];
      for (int d = 0; d < kDirectionCount; ++d) {
        std::optional<Pos> dst = board.first_occupied_in_direction(src, d);
        if (!dst.has_value()) {
          continue;
        }
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
};

// action_space: function implementation.
const ActionSpace& action_space() {
  static ActionSpace space;
  return space;
}

struct DecodedAction {
  MoveKind kind;
  int edge_idx;
  int src_idx;
  int direction_idx;
};

struct SnapshotCell {
  int row = 0;
  int col = 0;
  int top_piece = 0;
  int height = 0;
};

struct LeafSnapshot {
  int node_id = 0;
  int current_player = 0;
  int turn_number = 0;
  int winner = 0;
  bool is_done = false;
  std::string phase;
  std::array<int, 3> white_counts = {{0, 0, 0}};
  std::array<int, 3> black_counts = {{0, 0, 0}};
  std::vector<float> board_state_flat;  // 12*9*9 = 972 floats
  std::vector<float> global_features;   // 12 floats
  std::vector<std::uint8_t> legal_mask;
};

struct SearchConfig {
  int simulations = 0;
  int leaf_batch_size = 1;
  float puct_c = 1.5f;
  bool add_root_dirichlet_noise = false;
  float root_dirichlet_eps = 0.25f;
  float root_dirichlet_alpha = 0.05f;
};

struct SearchResult {
  int root_node_id = 0;
  int root_player = 0;
  int winner = 0;
  bool is_done = false;
  bool is_complete = false;
  bool needs_root_eval = false;
  int simulations_requested = 0;
  int simulations_processed = 0;
  int pending_leaf_count = 0;
  float root_value = 0.0f;
  std::vector<std::uint8_t> legal_mask;
  std::vector<float> root_policy;
  std::vector<float> root_visits;
};

// terminal_value_for_current_player: function implementation.
float terminal_value_for_current_player(int winner, int current_player) {
  if (winner == 0) {
    return 0.0f;
  }
  return winner == current_player ? 1.0f : -1.0f;
}

// decode_action_idx: function implementation.
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

// encode_action_idx: function implementation.
int encode_action_idx(const std::string& kind, std::optional<int> edge_idx) {
  if (kind == "pass") {
    return kPassActionIdx;
  }
  if (!edge_idx.has_value()) {
    throw std::invalid_argument("edge_idx is required for non-pass actions");
  }
  const int idx = edge_idx.value();
  if (idx < 0 || idx >= kEdgeActionCount) {
    throw std::out_of_range("edge_idx out of range");
  }
  if (kind == "capture") {
    return kCaptureOffset + idx;
  }
  if (kind == "reinforce") {
    return kReinforceOffset + idx;
  }
  throw std::invalid_argument("kind must be one of: capture, reinforce, pass");
}

// reconstruct_non_pass_move: function implementation.
Move reconstruct_non_pass_move(const TzaarGame& game, MoveKind kind, int src_idx, int direction_idx) {
  const auto& positions = game.board().valid_positions();
  if (src_idx < 0 || src_idx >= static_cast<int>(positions.size())) {
    throw std::out_of_range("src_idx out of range");
  }
  const Pos src = positions[src_idx];
  std::optional<Pos> dst = game.board().first_occupied_in_direction(src, direction_idx);
  if (!dst.has_value()) {
    throw std::invalid_argument("No occupied cell found in chosen direction");
  }
  return Move{kind, src, *dst};
}

class PhaseGameState {
 public:
  PhaseGameState() : game_(), stage_(Stage::NeedStep1) {}

  // is_done: function implementation.
  bool is_done() const {
    return stage_ == Stage::Done || game_.is_game_over();
  }

  // current_player: function implementation.
  int current_player() const {
    return game_.current_player();
  }

  // winner: function implementation.
  int winner() const {
    const std::optional<int> w = game_.winner();
    return w.has_value() ? *w : 0;
  }

  // phase: function implementation.
  std::string phase() const {
    if (is_done()) {
      return "";
    }
    if (stage_ == Stage::NeedStep1) {
      return "step1_capture";
    }
    if (stage_ == Stage::NeedStep2) {
      return "step2_action";
    }
    return "";
  }

  // legal_mask: function implementation.
  std::vector<bool> legal_mask() {
    if (is_done()) {
      return std::vector<bool>(kActionCount, false);
    }
    if (cache_valid_) {
      return cached_mask_;
    }

    std::vector<bool> mask(kActionCount, false);
    if (stage_ == Stage::NeedStep1) {
      if (game_.resolve_no_mandatory_capture_if_needed()) {
        stage_ = Stage::Done;
        invalidate_cache();
        return std::vector<bool>(kActionCount, false);
      }
      fill_step1_mask(mask);
    // if: function implementation.
    } else if (stage_ == Stage::NeedStep2) {
      fill_step2_mask(mask);
    } else {
      throw std::runtime_error("Unsupported stage");
    }

    cached_mask_ = mask;
    cache_valid_ = true;
    return cached_mask_;
  }

  // apply_action: function implementation.
  void apply_action(int action_idx) {
    if (is_done()) {
      return;
    }
    if (action_idx < 0 || action_idx >= kActionCount) {
      throw std::out_of_range("action_idx out of range");
    }

    std::vector<bool> mask = legal_mask();
    if (!mask[static_cast<std::size_t>(action_idx)]) {
      throw std::invalid_argument("illegal action");
    }
    invalidate_cache();

    if (stage_ == Stage::NeedStep1) {
      Move move = reconstruct_unified_action(action_idx, true);
      game_.play_first_step(move);
      if (game_.is_game_over()) {
        stage_ = Stage::Done;
      // if: function implementation.
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

  // apply_action_trusted: internal fast path for replaying known-legal actions.
  // Used by MCTS state reconstruction to avoid recomputing legal_mask on each step.
  void apply_action_trusted(int action_idx) {
    if (is_done()) {
      return;
    }
    if (action_idx < 0 || action_idx >= kActionCount) {
      throw std::out_of_range("action_idx out of range");
    }

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

  // clone: function implementation.
  PhaseGameState clone() const {
    PhaseGameState copied;
    copied.game_ = game_;
    copied.stage_ = stage_;
    copied.cache_valid_ = cache_valid_;
    copied.cached_mask_ = cached_mask_;
    return copied;
  }

  // turn_number: function implementation.
  int turn_number() const {
    return game_.turn_number();
  }

  // piece_counts: function implementation.
  py::dict piece_counts() const {
    const auto counts = game_.piece_counts();
    py::dict out;
    py::dict white;
    py::dict black;
    white[py::int_(1)] = py::int_(counts[0][1]);
    white[py::int_(2)] = py::int_(counts[0][2]);
    white[py::int_(3)] = py::int_(counts[0][3]);
    black[py::int_(1)] = py::int_(counts[1][1]);
    black[py::int_(2)] = py::int_(counts[1][2]);
    black[py::int_(3)] = py::int_(counts[1][3]);
    out[py::int_(kWhite)] = std::move(white);
    out[py::int_(kBlack)] = std::move(black);
    return out;
  }

  // board_cells: function implementation.
  py::list board_cells() const {
    py::list out;
    const auto& board = game_.board();
    for (const Pos& pos : board.valid_positions()) {
      if (board.is_empty(pos)) {
        continue;
      }
      py::dict cell;
      cell["row"] = pos.row;
      cell["col"] = pos.col;
      cell["top_piece"] = board.top_piece(pos);
      cell["height"] = board.height(pos);
      out.append(cell);
    }
    return out;
  }

  // fill_snapshot_metadata: function implementation.
  void fill_snapshot_metadata(LeafSnapshot& snapshot, int node_id) {
    snapshot.node_id = node_id;
    snapshot.current_player = current_player();
    snapshot.turn_number = turn_number();
    snapshot.winner = winner();
    snapshot.is_done = is_done();
    snapshot.phase = phase();

    const auto counts = game_.piece_counts();
    for (int piece_idx = 1; piece_idx <= 3; ++piece_idx) {
      snapshot.white_counts[static_cast<std::size_t>(piece_idx - 1)] = counts[0][piece_idx];
      snapshot.black_counts[static_cast<std::size_t>(piece_idx - 1)] = counts[1][piece_idx];
    }

    const std::vector<bool> mask = legal_mask();
    snapshot.legal_mask.reserve(mask.size());
    for (const bool is_legal : mask) {
      snapshot.legal_mask.push_back(is_legal ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0));
    }
  }

  // build_cnn_features: function implementation.
  void build_cnn_features(std::vector<float>& board_state_flat, std::vector<float>& global_features) const {
    const auto counts = game_.piece_counts();

    // Build 12x9x9 board state as flat array (972 floats)
    const float max_height_norm = 8.0f;
    board_state_flat.assign(kBoardFlatSize, 0.0f);

    const int player = current_player();
    const auto& board = game_.board();
    for (const Pos& pos : board.valid_positions()) {
      if (board.is_empty(pos)) {
        continue;
      }
      const int top = board.top_piece(pos);
      const int owner = piece_owner(top);
      int type_idx = 0;
      if (top == 1 || top == 4) {
        type_idx = 0;
      // if: function implementation.
      } else if (top == 2 || top == 5) {
        type_idx = 1;
      // if: function implementation.
      } else if (top == 3 || top == 6) {
        type_idx = 2;
      }

      const float h_norm = std::min(static_cast<float>(board.height(pos)), max_height_norm) / max_height_norm;
      int occ_ch = 0;
      int h_ch = 0;
      if (owner == player) {
        occ_ch = type_idx;
        h_ch = 6 + type_idx;
      } else {
        occ_ch = 3 + type_idx;
        h_ch = 9 + type_idx;
      }

      board_state_flat[occ_ch * 81 + pos.row * 9 + pos.col] = 1.0f;
      board_state_flat[h_ch * 81 + pos.row * 9 + pos.col] = h_norm;
    }

    // Build 12-dim global features
    global_features.assign(kGlobalFeatureDim, 0.0f);
    const std::string phase_str = phase();
    int phase_idx = 0;
    if (phase_str == "step2_action") {
      phase_idx = 1;
    }

    const int own_idx = (player == kWhite) ? 0 : 1;
    const int opp_idx = (player == kWhite) ? 1 : 0;
    const int own_t1 = counts[own_idx][1];
    const int own_t2 = counts[own_idx][2];
    const int own_t3 = counts[own_idx][3];
    const int opp_t1 = counts[opp_idx][1];
    const int opp_t2 = counts[opp_idx][2];
    const int opp_t3 = counts[opp_idx][3];

    const float turn_norm = std::min(static_cast<float>(turn_number()), 200.0f) / 200.0f;
    const float player_sign = (player == kWhite) ? 1.0f : -1.0f;

    global_features[0] = turn_norm;
    global_features[1] = player_sign;
    global_features[2] = static_cast<float>(own_t1) / 15.0f;
    global_features[3] = static_cast<float>(own_t2) / 15.0f;
    global_features[4] = static_cast<float>(own_t3) / 15.0f;
    global_features[5] = static_cast<float>(opp_t1) / 15.0f;
    global_features[6] = static_cast<float>(opp_t2) / 15.0f;
    global_features[7] = static_cast<float>(opp_t3) / 15.0f;
    global_features[8] = static_cast<float>(own_t1 + own_t2 + own_t3) / 45.0f;
    global_features[9] = static_cast<float>(opp_t1 + opp_t2 + opp_t3) / 45.0f;
    global_features[10] = (phase_idx == 0) ? 1.0f : 0.0f;
    global_features[11] = (phase_idx == 1) ? 1.0f : 0.0f;
  }

  // build_cnn_features_into: write features directly into pre-allocated buffers (no intermediate allocation).
  void build_cnn_features_into(float* board_out, float* global_out) const {
    const auto counts = game_.piece_counts();
    const float max_height_norm = 8.0f;
    std::memset(board_out, 0, static_cast<std::size_t>(kBoardFlatSize) * sizeof(float));

    const int player = current_player();
    const auto& board = game_.board();
    for (const Pos& pos : board.valid_positions()) {
      if (board.is_empty(pos)) continue;
      const int top = board.top_piece(pos);
      const int owner = piece_owner(top);
      int type_idx = 0;
      if (top == 1 || top == 4) { type_idx = 0; }
      else if (top == 2 || top == 5) { type_idx = 1; }
      else if (top == 3 || top == 6) { type_idx = 2; }
      const float h_norm = std::min(static_cast<float>(board.height(pos)), max_height_norm) / max_height_norm;
      int occ_ch = 0, h_ch = 0;
      if (owner == player) { occ_ch = type_idx; h_ch = 6 + type_idx; }
      else { occ_ch = 3 + type_idx; h_ch = 9 + type_idx; }
      board_out[occ_ch * 81 + pos.row * 9 + pos.col] = 1.0f;
      board_out[h_ch * 81 + pos.row * 9 + pos.col] = h_norm;
    }

    const std::string phase_str = phase();
    const int phase_idx = (phase_str == "step2_action") ? 1 : 0;
    const int own_idx = (player == kWhite) ? 0 : 1;
    const int opp_idx = (player == kWhite) ? 1 : 0;
    const int own_t1 = counts[own_idx][1], own_t2 = counts[own_idx][2], own_t3 = counts[own_idx][3];
    const int opp_t1 = counts[opp_idx][1], opp_t2 = counts[opp_idx][2], opp_t3 = counts[opp_idx][3];
    const float turn_norm = std::min(static_cast<float>(turn_number()), 200.0f) / 200.0f;
    const float player_sign = (player == kWhite) ? 1.0f : -1.0f;
    global_out[0]  = turn_norm;
    global_out[1]  = player_sign;
    global_out[2]  = static_cast<float>(own_t1) / 15.0f;
    global_out[3]  = static_cast<float>(own_t2) / 15.0f;
    global_out[4]  = static_cast<float>(own_t3) / 15.0f;
    global_out[5]  = static_cast<float>(opp_t1) / 15.0f;
    global_out[6]  = static_cast<float>(opp_t2) / 15.0f;
    global_out[7]  = static_cast<float>(opp_t3) / 15.0f;
    global_out[8]  = static_cast<float>(own_t1 + own_t2 + own_t3) / 45.0f;
    global_out[9]  = static_cast<float>(opp_t1 + opp_t2 + opp_t3) / 45.0f;
    global_out[10] = (phase_idx == 0) ? 1.0f : 0.0f;
    global_out[11] = (phase_idx == 1) ? 1.0f : 0.0f;
  }

  // leaf_snapshot: function implementation.
  LeafSnapshot leaf_snapshot(int node_id) {
    LeafSnapshot snapshot;
    fill_snapshot_metadata(snapshot, node_id);
    build_cnn_features(snapshot.board_state_flat, snapshot.global_features);
    return snapshot;
  }

 private:
  TzaarGame game_;
  Stage stage_;
  bool cache_valid_ = false;
  std::vector<bool> cached_mask_;

  // invalidate_cache: function implementation.
  void invalidate_cache() {
    cache_valid_ = false;
    cached_mask_.clear();
  }

  // advance_from_first_src: function implementation.
  void advance_from_first_src() {
    if (stage_ == Stage::NeedStep1) {
      if (game_.resolve_no_mandatory_capture_if_needed()) {
        stage_ = Stage::Done;
        invalidate_cache();
      }
    }
  }

  // fill_step1_mask: function implementation.
  void fill_step1_mask(std::vector<bool>& mask) const {
    const auto& board = game_.board();
    const auto& pos_to_idx = board.pos_to_idx_map();
    const int player = game_.current_player();

    for (const Pos& src : board.valid_positions()) {
      if (board.is_empty(src)) {
        continue;
      }
      if (piece_owner(board.top_piece(src)) != player) {
        continue;
      }
      const int src_height = board.height(src);
      const int src_idx = pos_to_idx.at(pack_pos(src));
      for (int d = 0; d < kDirectionCount; ++d) {
        std::optional<Pos> dst = board.first_occupied_in_direction(src, d);
        if (!dst.has_value()) {
          continue;
        }
        if (piece_owner(board.top_piece(*dst)) == player) {
          continue;
        }
        if (board.height(*dst) > src_height) {
          continue;
        }
        std::optional<int> edge_idx = action_space().edge_to_idx(src_idx, d);
        if (edge_idx.has_value()) {
          mask[static_cast<std::size_t>(kCaptureOffset + *edge_idx)] = true;
        }
      }
    }
  }

  // fill_step2_mask: function implementation.
  void fill_step2_mask(std::vector<bool>& mask) const {
    const auto& board = game_.board();
    const auto& pos_to_idx = board.pos_to_idx_map();
    const int player = game_.current_player();

    for (const Pos& src : board.valid_positions()) {
      if (board.is_empty(src)) {
        continue;
      }
      if (piece_owner(board.top_piece(src)) != player) {
        continue;
      }
      const int src_height = board.height(src);
      const int src_idx = pos_to_idx.at(pack_pos(src));
      for (int d = 0; d < kDirectionCount; ++d) {
        std::optional<Pos> dst = board.first_occupied_in_direction(src, d);
        if (!dst.has_value()) {
          continue;
        }
        std::optional<int> edge_idx = action_space().edge_to_idx(src_idx, d);
        if (!edge_idx.has_value()) {
          continue;
        }
        const bool dst_is_self = piece_owner(board.top_piece(*dst)) == player;
        if (dst_is_self) {
          mask[static_cast<std::size_t>(kReinforceOffset + *edge_idx)] = true;
        // if: function implementation.
        } else if (board.height(*dst) <= src_height) {
          mask[static_cast<std::size_t>(kCaptureOffset + *edge_idx)] = true;
        }
      }
    }

    if (game_.can_second_step_pass()) {
      mask[static_cast<std::size_t>(kPassActionIdx)] = true;
    }
  }

  // reconstruct_unified_action: function implementation.
  Move reconstruct_unified_action(int action_idx, bool is_step1) const {
    DecodedAction decoded = decode_action_idx(action_idx);
    if (decoded.kind == MoveKind::Pass) {
      if (is_step1) {
        throw std::invalid_argument("step1 only allows capture actions");
      }
      return Move{MoveKind::Pass, std::nullopt, std::nullopt};
    }

    if (is_step1 && decoded.kind != MoveKind::Capture) {
      throw std::invalid_argument("step1 only allows capture actions");
    }

    return reconstruct_non_pass_move(game_, decoded.kind, decoded.src_idx, decoded.direction_idx);
  }
};

class SearchSession {
 public:
  SearchSession(const PhaseGameState& root_state, SearchConfig config)
      // config_: function implementation.
      : config_(std::move(config)), rng_(std::random_device{}()) {
    if (config_.simulations < 0) {
      throw std::invalid_argument("simulations must be >= 0");
    }
    if (config_.leaf_batch_size <= 0) {
      throw std::invalid_argument("leaf_batch_size must be >= 1");
    }
    if (config_.puct_c <= 0.0f) {
      throw std::invalid_argument("puct_c must be > 0");
    }

    root_state_ = root_state.clone();
    MctsNode root;
    root.prior = 1.0f;
    root.parent_idx = -1;
    root.action_from_parent = -1;
    root.to_play = root_state_.current_player();
    root.is_terminal = root_state_.is_done();
    root.winner = root_state_.winner();
    root.has_player = !root.is_terminal;
    if (root.is_terminal) {
      root.legal_mask_ready = true;
      root.cached_legal_mask.assign(static_cast<std::size_t>(kActionCount), static_cast<std::uint8_t>(0));
    }

    nodes_.push_back(std::move(root));

    // Cache root CNN features once (spec: "根節點的state全部都做一份複製，並放進快取")
    root_board_flat_.resize(static_cast<std::size_t>(kBoardFlatSize), 0.0f);
    root_global_feat_.resize(static_cast<std::size_t>(kGlobalFeatureDim), 0.0f);
    if (!root_state_.is_done()) {
      root_state_.build_cnn_features_into(root_board_flat_.data(), root_global_feat_.data());
    }
  }

  // config: function implementation.
  SearchConfig config() const {
    return config_;
  }

  // has_pending_leaves: function implementation.
  bool has_pending_leaves() const {
    return !pending_node_order_.empty();
  }

  // collect_pending_leaves: function implementation.
  std::vector<LeafSnapshot> collect_pending_leaves(int max_batch) {
    bool has_pending = false;
    {
      py::gil_scoped_release release;
      has_pending = prepare_pending_leaves(max_batch);
      if (has_pending) {
        return pending_leaf_snapshots();
      }
    }
    if (!has_pending) {
      return {};
    }
    return {};
  }

  // collect_pending_leaves_packed: function implementation.
  // Zero-copy: wrap C++ batch buffers directly as numpy views — no memcpy.
  // Safety: buffers are valid until the next simulate_chunk call. Python callers that
  // hand results across threads must copy arrays immediately after collect.
  py::dict collect_pending_leaves_packed(int max_batch) {
    bool has_pending = false;
    {
      py::gil_scoped_release release;
      has_pending = prepare_pending_leaves(max_batch);
    }

    if (!has_pending) {
      py::dict empty;
      empty["node_ids"]         = py::array_t<int32_t>({0});
      empty["legal_masks"]      = py::array_t<uint8_t>({0, kActionCount});
      empty["board_state_flat"] = py::array_t<float>({0, kBoardFlatSize});
      empty["global_features"]  = py::array_t<float>({0, kGlobalFeatureDim});
      return empty;
    }

    const py::ssize_t bsz = static_cast<py::ssize_t>(pending_node_order_.size());

    // Capsules with no-op destructors: the SearchSession vectors own the memory.
    py::capsule ids_cap  (batch_node_ids_.data(),   [](void*) {});
    py::capsule mask_cap (batch_legal_mask_.data(),  [](void*) {});
    py::capsule board_cap(batch_board_flat_.data(),  [](void*) {});
    py::capsule glob_cap (batch_global_feat_.data(), [](void*) {});

    py::dict out;
    out["node_ids"]         = py::array_t<int32_t>(
        {bsz}, batch_node_ids_.data(), ids_cap);
    out["legal_masks"]      = py::array_t<uint8_t>(
        {bsz, static_cast<py::ssize_t>(kActionCount)}, batch_legal_mask_.data(), mask_cap);
    out["board_state_flat"] = py::array_t<float>(
        {bsz, static_cast<py::ssize_t>(kBoardFlatSize)}, batch_board_flat_.data(), board_cap);
    out["global_features"]  = py::array_t<float>(
        {bsz, static_cast<py::ssize_t>(kGlobalFeatureDim)}, batch_global_feat_.data(), glob_cap);
    return out;
  }

  // submit_leaf_eval: function implementation.
  void submit_leaf_eval(int node_id, const std::vector<float>& priors, float value) {
    const int node_idx = node_id - 1;
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size())) {
      throw std::invalid_argument("unknown node_id for current search session");
    }
    if (!has_pending_leaves()) {
      throw std::invalid_argument("no pending leaves to evaluate");
    }
    if (pending_paths_.find(node_idx) == pending_paths_.end()) {
      throw std::invalid_argument("node_id is not in pending leaves");
    }
    if (pending_eval_map_.find(node_idx) != pending_eval_map_.end()) {
      throw std::invalid_argument("leaf evaluation already submitted for this node");
    }
    if (priors.size() != static_cast<std::size_t>(kActionCount)) {
      throw std::invalid_argument("priors must have length N_ACTIONS");
    }

    pending_eval_map_.emplace(node_idx, PendingEval{priors, value});

    if (pending_eval_map_.size() == pending_node_order_.size()) {
      py::gil_scoped_release release;
      process_pending_evals();
    }
  }

  void submit_leaf_eval_batch(
      py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> node_ids,
      py::array_t<float, py::array::c_style | py::array::forcecast> priors,
      py::array_t<float, py::array::c_style | py::array::forcecast> values) {
    if (node_ids.ndim() != 1) {
      throw std::invalid_argument("node_ids must be a 1D array");
    }
    if (priors.ndim() != 2) {
      throw std::invalid_argument("priors must be a 2D array [B, N_ACTIONS]");
    }
    if (values.ndim() != 1) {
      throw std::invalid_argument("values must be a 1D array [B]");
    }

    const py::ssize_t bsz = node_ids.shape(0);
    if (priors.shape(0) != bsz || values.shape(0) != bsz) {
      throw std::invalid_argument("batch size mismatch among node_ids/priors/values");
    }
    if (priors.shape(1) != static_cast<py::ssize_t>(kActionCount)) {
      throw std::invalid_argument("priors second dim must equal N_ACTIONS");
    }

    if (!has_pending_leaves()) {
      throw std::invalid_argument("no pending leaves to evaluate");
    }

    auto ids = node_ids.unchecked<1>();
    auto vs = values.unchecked<1>();

    const auto priors_buf = priors.request();
    const float* priors_ptr = static_cast<const float*>(priors_buf.ptr);

    std::vector<float> prior_row(static_cast<std::size_t>(kActionCount), 0.0f);
    pending_eval_map_.reserve(pending_eval_map_.size() + static_cast<std::size_t>(bsz));
    for (py::ssize_t i = 0; i < bsz; ++i) {
      const int node_id = static_cast<int>(ids(i));
      const int node_idx = node_id - 1;
      if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size())) {
        throw std::invalid_argument("unknown node_id for current search session");
      }
      if (pending_paths_.find(node_idx) == pending_paths_.end()) {
        throw std::invalid_argument("node_id is not in pending leaves");
      }
      if (pending_eval_map_.find(node_idx) != pending_eval_map_.end()) {
        throw std::invalid_argument("leaf evaluation already submitted for this node");
      }

      const float* row_ptr = priors_ptr + (i * static_cast<py::ssize_t>(kActionCount));
      std::memcpy(prior_row.data(), row_ptr, static_cast<std::size_t>(kActionCount) * sizeof(float));
      pending_eval_map_.emplace(node_idx, PendingEval{prior_row, static_cast<float>(vs(i))});
    }

    if (pending_eval_map_.size() == pending_node_order_.size()) {
      py::gil_scoped_release release;
      process_pending_evals();
    }
  }

  // finish: function implementation.
  SearchResult finish() {
    SearchResult result;
    result.root_node_id = root_node_id_;
    result.root_player = nodes_[root_node_index_].to_play;
    result.winner = nodes_[root_node_index_].winner;
    result.is_done = nodes_[root_node_index_].is_terminal;
    result.is_complete = is_complete();
    result.needs_root_eval = has_pending_leaves();
    result.simulations_requested = config_.simulations;
    result.simulations_processed = simulations_processed_;
    result.pending_leaf_count = static_cast<int>(pending_node_order_.size());
    result.root_value = nodes_[root_node_index_].mean_value();
    // Root state is always available as root_state_ — no reconstruction needed
    MctsNode& root_node = nodes_[root_node_index_];
    if (!root_node.legal_mask_ready) {
      // Root was never visited as a leaf (e.g. simulations=0), compute mask from root_state_
      PhaseGameState tmp = root_state_.clone();
      get_legal_mask_for_node(root_node_index_, tmp);
    }
    const std::vector<bool> mask = [&]() {
      std::vector<bool> m(static_cast<std::size_t>(kActionCount), false);
      const std::size_t n = std::min(root_node.cached_legal_mask.size(),
                                     static_cast<std::size_t>(kActionCount));
      for (std::size_t i = 0; i < n; ++i) m[i] = root_node.cached_legal_mask[i] != 0;
      return m;
    }();
    result.legal_mask.reserve(mask.size());
    for (const bool is_legal : mask) {
      result.legal_mask.push_back(is_legal ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0));
    }

    result.root_policy.assign(static_cast<std::size_t>(kActionCount), 0.0f);
    result.root_visits.assign(static_cast<std::size_t>(kActionCount), 0.0f);

    const MctsNode& root = nodes_[root_node_index_];
    for (const auto& pair : root.children) {
      const int action = pair.first;
      const int child_idx = pair.second;
      if (action < 0 || action >= kActionCount) {
        continue;
      }
      const MctsNode& child = nodes_[child_idx];
      result.root_policy[static_cast<std::size_t>(action)] = child.prior;
      result.root_visits[static_cast<std::size_t>(action)] = static_cast<float>(child.visit_count);
    }

    return result;
  }

  // root_snapshot: function implementation.
  LeafSnapshot root_snapshot() {
    LeafSnapshot snapshot;
    root_state_.fill_snapshot_metadata(snapshot, root_node_id_);
    snapshot.board_state_flat = root_board_flat_;
    snapshot.global_features = root_global_feat_;
    return snapshot;
  }

 private:
  struct PendingEval {
    std::vector<float> priors;
    float value = 0.0f;
  };

  struct MctsNode {
    float prior = 0.0f;
    int parent_idx = -1;
    int action_from_parent = -1;
    int to_play = 0;
    bool has_player = false;
    bool is_terminal = false;
    int winner = 0;
    int visit_count = 0;
    float value_sum = 0.0f;
    bool expanded = false;
    bool legal_mask_ready = false;
    std::vector<std::uint8_t> cached_legal_mask;
    std::unordered_map<int, int> children;

    // mean_value: function implementation.
    float mean_value() const {
      if (visit_count <= 0) {
        return 0.0f;
      }
      return value_sum / static_cast<float>(visit_count);
    }
  };

  static constexpr int root_node_index_ = 0;
  SearchConfig config_;
  int root_node_id_ = 1;
  int simulations_processed_ = 0;
  bool root_noise_applied_ = false;

  PhaseGameState root_state_;
  std::vector<MctsNode> nodes_;

  std::vector<int> pending_node_order_;
  std::unordered_map<int, std::vector<std::vector<int>>> pending_paths_;
  std::unordered_map<int, PendingEval> pending_eval_map_;

  // Contiguous batch buffers filled during simulate_chunk (spec: O(1) tensor creation)
  std::vector<float>    batch_board_flat_;   // [N x kBoardFlatSize]
  std::vector<float>    batch_global_feat_;  // [N x kGlobalFeatureDim]
  std::vector<uint8_t>  batch_legal_mask_;   // [N x kActionCount]
  std::vector<int32_t>  batch_node_ids_;     // [N]

  // Root CNN features cached once in constructor
  std::vector<float> root_board_flat_;
  std::vector<float> root_global_feat_;

  std::mt19937 rng_;

  // collect_action_path_to_node: function implementation.
  std::vector<int> collect_action_path_to_node(int node_idx) const {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size())) {
      throw std::out_of_range("node_idx out of range while collecting action path");
    }

    std::vector<int> reversed;
    int current = node_idx;
    while (current != root_node_index_) {
      if (current < 0 || current >= static_cast<int>(nodes_.size())) {
        throw std::runtime_error("invalid parent chain while collecting action path");
      }
      const MctsNode& node = nodes_[current];
      if (node.parent_idx < 0 || node.action_from_parent < 0) {
        throw std::runtime_error("broken parent/action link in search tree");
      }
      reversed.push_back(node.action_from_parent);
      current = node.parent_idx;
    }

    std::reverse(reversed.begin(), reversed.end());
    return reversed;
  }

  // reconstruct_state_for_node: function implementation.
  PhaseGameState reconstruct_state_for_node(int node_idx) const {
    PhaseGameState state = root_state_.clone();
    const std::vector<int> actions = collect_action_path_to_node(node_idx);
    for (const int action : actions) {
      state.apply_action_trusted(action);
    }
    return state;
  }

  // update_node_metadata_from_state: function implementation.
  void update_node_metadata_from_state(int node_idx, const PhaseGameState& state) {
    MctsNode& node = nodes_[node_idx];
    node.to_play = state.current_player();
    node.is_terminal = state.is_done();
    node.winner = state.winner();
    node.has_player = !node.is_terminal;
    if (node.is_terminal) {
      node.legal_mask_ready = true;
      node.cached_legal_mask.assign(static_cast<std::size_t>(kActionCount), static_cast<std::uint8_t>(0));
    }
  }

  // get_legal_mask_for_node: function implementation.
  std::vector<bool> get_legal_mask_for_node(int node_idx, PhaseGameState& state) {
    MctsNode& node = nodes_[node_idx];
    if (node.legal_mask_ready) {
      std::vector<bool> cached(static_cast<std::size_t>(kActionCount), false);
      const std::size_t n = std::min(cached.size(), node.cached_legal_mask.size());
      for (std::size_t i = 0; i < n; ++i) {
        cached[i] = node.cached_legal_mask[i] != 0;
      }
      return cached;
    }

    std::vector<bool> mask = state.legal_mask();
    node.cached_legal_mask.assign(mask.size(), static_cast<std::uint8_t>(0));
    for (std::size_t i = 0; i < mask.size(); ++i) {
      node.cached_legal_mask[i] = mask[i] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    }
    node.legal_mask_ready = true;
    return mask;
  }

  // prepare_pending_leaves: function implementation.
  bool prepare_pending_leaves(int max_batch) {
    if (max_batch <= 0) {
      throw std::invalid_argument("max_batch must be >= 1");
    }

    if (is_complete()) {
      return false;
    }

    if (has_pending_leaves()) {
      if (static_cast<int>(pending_node_order_.size()) > max_batch) {
        throw std::invalid_argument("max_batch is smaller than current pending leaf count");
      }
      return true;
    }

    while (!is_complete() && !has_pending_leaves()) {
      const int remaining = config_.simulations - simulations_processed_;
      if (remaining <= 0) {
        break;
      }
      const int chunk = std::min(config_.leaf_batch_size, remaining);
      simulate_chunk(chunk);
    }

    if (!has_pending_leaves()) {
      return false;
    }
    if (static_cast<int>(pending_node_order_.size()) > max_batch) {
      throw std::invalid_argument("max_batch is smaller than current pending leaf count");
    }
    return true;
  }

  // is_complete: function implementation.
  bool is_complete() const {
    return simulations_processed_ >= config_.simulations && !has_pending_leaves();
  }

  // pending_leaf_snapshots: function implementation.
  std::vector<LeafSnapshot> pending_leaf_snapshots() {
    std::vector<LeafSnapshot> leaves;
    leaves.reserve(pending_node_order_.size());
    for (const int node_idx : pending_node_order_) {
      leaves.push_back(build_leaf_snapshot(node_idx));
    }
    return leaves;
  }

  // build_leaf_snapshot: function implementation.
  LeafSnapshot build_leaf_snapshot(int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size())) {
      throw std::out_of_range("node_idx out of range when building snapshot");
    }

    MctsNode& node = nodes_[node_idx];
    PhaseGameState state = reconstruct_state_for_node(node_idx);
    update_node_metadata_from_state(node_idx, state);

    LeafSnapshot snapshot;
    state.fill_snapshot_metadata(snapshot, node_idx + 1);
    if (!node.is_terminal) {
      state.build_cnn_features(snapshot.board_state_flat, snapshot.global_features);
    }
    return snapshot;
  }

  // simulate_chunk: function implementation.
  // Spec: on first visit to unexpanded leaf, reconstruct state ONCE, build CNN features into
  // contiguous batch buffers, create child stubs (action only, no child state clone).
  void simulate_chunk(int chunk) {
    pending_node_order_.clear();
    pending_paths_.clear();
    pending_eval_map_.clear();
    batch_board_flat_.clear();
    batch_global_feat_.clear();
    batch_legal_mask_.clear();
    batch_node_ids_.clear();

    for (int sim = 0; sim < chunk; ++sim) {
      simulations_processed_ += 1;

      int node_idx = root_node_index_;
      std::vector<int> path;
      path.reserve(128);
      path.push_back(node_idx);

      while (true) {
        MctsNode& node = nodes_[node_idx];

        if (node.is_terminal) {
          // Bottom node: value known, no NN eval needed
          const int leaf_to_play = node.to_play;
          const float leaf_value = terminal_value_for_current_player(node.winner, leaf_to_play);
          backup(path, leaf_value, leaf_to_play);
          break;
        }

        if (node.expanded) {
          // Already expanded: keep descending
          if (node.children.empty()) {
            backup(path, 0.0f, node.to_play);
            break;
          }
          const int action = select_child_action(node_idx);
          node_idx = node.children.at(action);
          path.push_back(node_idx);
          continue;
        }

        // Unexpanded leaf
        if (pending_paths_.count(node_idx) > 0) {
          // Already queued this chunk: just add another path for backup
          // Apply virtual loss so PUCT scores drop while batch is pending
          for (const int idx : path) {
            nodes_[idx].visit_count += 1;
            nodes_[idx].value_sum -= 1.0f;
          }
          pending_paths_[node_idx].push_back(path);
          break;
        }

        // First visit: reconstruct state exactly once (spec)
        PhaseGameState state = reconstruct_state_for_node(node_idx);

        // Update node metadata from reconstructed state
        node.to_play = state.current_player();
        node.is_terminal = state.is_done();
        node.winner = state.winner();
        node.has_player = !node.is_terminal;

        if (node.is_terminal) {
          // Terminal discovered on first visit: backup immediately, no NN eval
          node.expanded = true;
          node.legal_mask_ready = true;
          node.cached_legal_mask.assign(static_cast<std::size_t>(kActionCount),
                                        static_cast<std::uint8_t>(0));
          const float leaf_value = terminal_value_for_current_player(node.winner, node.to_play);
          backup(path, leaf_value, node.to_play);
          break;
        }

        // Build CNN features directly into contiguous batch buffers (spec: no intermediate allocation)
        {
          const std::size_t offset_board = batch_board_flat_.size();
          const std::size_t offset_global = batch_global_feat_.size();
          batch_board_flat_.resize(offset_board + static_cast<std::size_t>(kBoardFlatSize));
          batch_global_feat_.resize(offset_global + static_cast<std::size_t>(kGlobalFeatureDim));
          state.build_cnn_features_into(
              batch_board_flat_.data() + offset_board,
              batch_global_feat_.data() + offset_global);
        }

        // Cache legal mask on node AND fill batch_legal_mask_ (zero-copy path)
        const std::vector<bool> legal = state.legal_mask();
        node.cached_legal_mask.resize(static_cast<std::size_t>(kActionCount), 0);
        const std::size_t offset_mask = batch_legal_mask_.size();
        batch_legal_mask_.resize(offset_mask + static_cast<std::size_t>(kActionCount), 0);
        for (std::size_t j = 0; j < legal.size() && j < static_cast<std::size_t>(kActionCount); ++j) {
          const uint8_t v = legal[j] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
          node.cached_legal_mask[j] = v;
          batch_legal_mask_[offset_mask + j] = v;
        }
        node.legal_mask_ready = true;

        // Create child stubs: only record parent+action, NO child state clone (spec)
        for (int action = 0; action < kActionCount; ++action) {
          if (!legal[static_cast<std::size_t>(action)]) continue;
          const int new_idx = static_cast<int>(nodes_.size());
          MctsNode child;
          child.prior = 0.0f;  // set in expand_node after NN eval
          child.parent_idx = node_idx;
          child.action_from_parent = action;
          // to_play / is_terminal / winner filled on first visit to child
          nodes_.push_back(std::move(child));
          nodes_[node_idx].children[action] = new_idx;
        }

        // Queue for NN batch — apply virtual loss so PUCT scores drop while batch is pending
        for (const int idx : path) {
          nodes_[idx].visit_count += 1;
          nodes_[idx].value_sum -= 1.0f;
        }
        pending_node_order_.push_back(node_idx);
        batch_node_ids_.push_back(static_cast<int32_t>(node_idx + 1));
        pending_paths_[node_idx].push_back(path);
        break;
      }
    }
  }

  // select_child_action: function implementation.
  int select_child_action(int node_idx) {
    MctsNode& node = nodes_[node_idx];
    if (node.children.empty()) {
      throw std::runtime_error("select_child_action called on node without children");
    }

    int total_visits = 0;
    for (const auto& pair : node.children) {
      total_visits += nodes_[pair.second].visit_count;
    }

    const float sqrt_total = std::sqrt(static_cast<float>(total_visits + 1));
    float best_score = -std::numeric_limits<float>::infinity();
    std::vector<int> best_actions;

    for (const auto& pair : node.children) {
      const int action = pair.first;
      const MctsNode& child = nodes_[pair.second];
      const float q_child = child.mean_value();

      float q = q_child;
      if (node.has_player && child.has_player && child.to_play != node.to_play) {
        q = -q_child;
      }

      const float u = config_.puct_c * child.prior * sqrt_total / static_cast<float>(1 + child.visit_count);
      const float score = q + u;

      if (score > best_score + 1e-12f) {
        best_score = score;
        best_actions.clear();
        best_actions.push_back(action);
      // if: function implementation.
      } else if (std::fabs(score - best_score) <= 1e-12f) {
        best_actions.push_back(action);
      }
    }

    if (best_actions.empty()) {
      throw std::runtime_error("no best action found during child selection");
    }

    std::uniform_int_distribution<int> pick(0, static_cast<int>(best_actions.size()) - 1);
    return best_actions[static_cast<std::size_t>(pick(rng_))];
  }

  // backup: function implementation.
  void backup(const std::vector<int>& path, float leaf_value, int leaf_to_play) {
    for (const int idx : path) {
      MctsNode& node = nodes_[idx];
      const int node_player = node.has_player ? node.to_play : leaf_to_play;
      const float value_for_node = (node_player == leaf_to_play) ? leaf_value : -leaf_value;
      node.value_sum += value_for_node;
      node.visit_count += 1;
    }
  }

  // process_pending_evals: function implementation.
  void process_pending_evals() {
    for (const int node_idx : pending_node_order_) {
      auto eval_it = pending_eval_map_.find(node_idx);
      if (eval_it == pending_eval_map_.end()) {
        throw std::runtime_error("missing pending eval while processing leaf batch");
      }
      expand_node(node_idx, eval_it->second.priors);

      const int leaf_to_play = nodes_[node_idx].to_play;
      const float leaf_value = eval_it->second.value;

      const auto path_it = pending_paths_.find(node_idx);
      if (path_it == pending_paths_.end()) {
        throw std::runtime_error("missing pending paths while processing leaf batch");
      }
      for (const auto& path : path_it->second) {
        // Revert virtual loss applied during selection before real backup
        for (const int idx : path) {
          nodes_[idx].visit_count -= 1;
          nodes_[idx].value_sum += 1.0f;
        }
        backup(path, leaf_value, leaf_to_play);
      }
    }

    pending_node_order_.clear();
    pending_paths_.clear();
    pending_eval_map_.clear();
  }

  // expand_node: function implementation.
  // Spec: child stubs already created in simulate_chunk. This function only assigns normalized
  // priors from NN output and marks the node as expanded. No state reconstruction, no cloning.
  void expand_node(int node_idx, const std::vector<float>& priors) {
    if (nodes_[node_idx].expanded) {
      return;
    }

    MctsNode& parent = nodes_[node_idx];
    if (parent.children.empty()) {
      parent.expanded = true;
      return;
    }

    // Collect legal priors for existing children (created in simulate_chunk)
    float sum_legal = 0.0f;
    for (const auto& kv : parent.children) {
      const int action = kv.first;
      const float p = (action >= 0 && action < static_cast<int>(priors.size()))
                          ? priors[static_cast<std::size_t>(action)]
                          : 0.0f;
      const float safe = std::isfinite(p) && p > 0.0f ? p : 0.0f;
      nodes_[kv.second].prior = safe;
      sum_legal += safe;
    }

    // Normalize
    if (!(sum_legal > 0.0f)) {
      const float uniform = 1.0f / static_cast<float>(parent.children.size());
      for (const auto& kv : parent.children) {
        nodes_[kv.second].prior = uniform;
      }
    } else {
      for (const auto& kv : parent.children) {
        nodes_[kv.second].prior /= sum_legal;
      }
    }

    parent.expanded = true;

    if (node_idx == root_node_index_ && config_.add_root_dirichlet_noise && !root_noise_applied_) {
      apply_root_dirichlet_noise();
      root_noise_applied_ = true;
    }
  }

  // apply_root_dirichlet_noise: function implementation.
  void apply_root_dirichlet_noise() {
    MctsNode& root = nodes_[root_node_index_];
    if (root.children.empty()) {
      return;
    }
    const float alpha = config_.root_dirichlet_alpha;
    if (!(alpha > 0.0f)) {
      return;
    }

    std::vector<int> actions;
    actions.reserve(root.children.size());
    for (const auto& pair : root.children) {
      actions.push_back(pair.first);
    }

    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> noise(actions.size(), 0.0f);
    float noise_sum = 0.0f;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      const float sample = gamma(rng_);
      noise[i] = sample;
      noise_sum += sample;
    }
    if (!(noise_sum > 0.0f)) {
      return;
    }

    const float eps = std::clamp(config_.root_dirichlet_eps, 0.0f, 1.0f);
    for (std::size_t i = 0; i < actions.size(); ++i) {
      const int child_idx = root.children[actions[i]];
      MctsNode& child = nodes_[child_idx];
      const float dir = noise[i] / noise_sum;
      child.prior = (1.0f - eps) * child.prior + eps * dir;
    }
  }
};

}  // namespace

// PYBIND11_MODULE: function implementation.
PYBIND11_MODULE(tzaar_cpp, m) {
  m.doc() = "Tzaar C++ rules engine bridge module";

  m.attr("N_ACTIONS") = py::int_(kActionCount);
  m.attr("PASS_ACTION_IDX") = py::int_(kPassActionIdx);
  m.attr("CAPTURE_OFFSET") = py::int_(kCaptureOffset);
  m.attr("REINFORCE_OFFSET") = py::int_(kReinforceOffset);
  m.attr("N_EDGE_ACTIONS") = py::int_(kEdgeActionCount);

  m.def("encode_action_idx", &encode_action_idx, py::arg("kind"), py::arg("edge_idx") = py::none(),
        "Encode (kind, edge_idx) into unified action index.");

  m.def(
      "decode_action_idx",
      [](int action_idx) {
        const DecodedAction decoded = decode_action_idx(action_idx);
        py::dict out;
        out["kind"] = py::str(move_kind_to_string(decoded.kind));
        out["edge_idx"] = py::int_(decoded.edge_idx);
        out["src_idx"] = py::int_(decoded.src_idx);
        out["direction_idx"] = py::int_(decoded.direction_idx);
        return out;
      },
      py::arg("action_idx"),
      "Decode unified action index into kind/edge/src/direction.");

  m.def(
      "action_edge_table",
      []() {
        py::list out;
        for (const auto& edge : action_space().edge_table()) {
          out.append(py::make_tuple(edge.first, edge.second));
        }
        return out;
      },
      "Return generated fixed edge table as (src_idx, direction_idx) pairs.");

      py::class_<SnapshotCell>(m, "SnapshotCell")
        .def(py::init<>())
        .def_readwrite("row", &SnapshotCell::row)
        .def_readwrite("col", &SnapshotCell::col)
        .def_readwrite("top_piece", &SnapshotCell::top_piece)
        .def_readwrite("height", &SnapshotCell::height);

      py::class_<LeafSnapshot>(m, "LeafSnapshot")
        .def(py::init<>())
        .def_readwrite("node_id", &LeafSnapshot::node_id)
        .def_readwrite("current_player", &LeafSnapshot::current_player)
        .def_readwrite("turn_number", &LeafSnapshot::turn_number)
        .def_readwrite("winner", &LeafSnapshot::winner)
        .def_readwrite("is_done", &LeafSnapshot::is_done)
        .def_readwrite("phase", &LeafSnapshot::phase)
        .def_readwrite("white_counts", &LeafSnapshot::white_counts)
        .def_readwrite("black_counts", &LeafSnapshot::black_counts)
        .def_readwrite("board_state_flat", &LeafSnapshot::board_state_flat)
        .def_readwrite("global_features", &LeafSnapshot::global_features)
        .def_readwrite("legal_mask", &LeafSnapshot::legal_mask);

      py::class_<SearchConfig>(m, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("simulations", &SearchConfig::simulations)
        .def_readwrite("leaf_batch_size", &SearchConfig::leaf_batch_size)
        .def_readwrite("puct_c", &SearchConfig::puct_c)
        .def_readwrite("add_root_dirichlet_noise", &SearchConfig::add_root_dirichlet_noise)
        .def_readwrite("root_dirichlet_eps", &SearchConfig::root_dirichlet_eps)
        .def_readwrite("root_dirichlet_alpha", &SearchConfig::root_dirichlet_alpha);

      py::class_<SearchResult>(m, "SearchResult")
        .def(py::init<>())
        .def_readwrite("root_node_id", &SearchResult::root_node_id)
        .def_readwrite("root_player", &SearchResult::root_player)
        .def_readwrite("winner", &SearchResult::winner)
        .def_readwrite("is_done", &SearchResult::is_done)
          .def_readwrite("is_complete", &SearchResult::is_complete)
        .def_readwrite("needs_root_eval", &SearchResult::needs_root_eval)
          .def_readwrite("simulations_requested", &SearchResult::simulations_requested)
          .def_readwrite("simulations_processed", &SearchResult::simulations_processed)
          .def_readwrite("pending_leaf_count", &SearchResult::pending_leaf_count)
        .def_readwrite("root_value", &SearchResult::root_value)
        .def_readwrite("legal_mask", &SearchResult::legal_mask)
          .def_readwrite("root_policy", &SearchResult::root_policy)
          .def_readwrite("root_visits", &SearchResult::root_visits);

  py::class_<PhaseGameState>(m, "PhaseGameState")
      .def(py::init<>())
      .def("is_done", &PhaseGameState::is_done)
      .def("current_player", &PhaseGameState::current_player)
      .def("turn_number", &PhaseGameState::turn_number)
      .def("winner", &PhaseGameState::winner)
      .def("phase", &PhaseGameState::phase)
      .def("piece_counts", &PhaseGameState::piece_counts)
      .def("board_cells", &PhaseGameState::board_cells)
        .def("leaf_snapshot", &PhaseGameState::leaf_snapshot, py::arg("node_id") = 0)
      .def("legal_mask", &PhaseGameState::legal_mask)
      .def("apply_action", &PhaseGameState::apply_action, py::arg("action_idx"))
      .def("clone", &PhaseGameState::clone);

      py::class_<SearchSession>(m, "SearchSession")
        .def(py::init<const PhaseGameState&, SearchConfig>(), py::arg("root_state"), py::arg("config"))
        .def("config", &SearchSession::config)
        .def("has_pending_leaves", &SearchSession::has_pending_leaves)
        .def("collect_pending_leaves", &SearchSession::collect_pending_leaves, py::arg("max_batch"))
        .def("collect_pending_leaves_packed", &SearchSession::collect_pending_leaves_packed, py::arg("max_batch"))
        .def("submit_leaf_eval", &SearchSession::submit_leaf_eval, py::arg("node_id"), py::arg("priors"),
           py::arg("value"))
        .def("submit_leaf_eval_batch", &SearchSession::submit_leaf_eval_batch, py::arg("node_ids"), py::arg("priors"),
          py::arg("values"))
        .def("finish", &SearchSession::finish, py::call_guard<py::gil_scoped_release>())
        .def("root_snapshot", &SearchSession::root_snapshot);
}
