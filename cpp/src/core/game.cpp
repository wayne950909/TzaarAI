#include "core/game.h"

#include <stdexcept>

namespace tzaar {

TzaarGame::TzaarGame()
  : current_player_(kWhite), turn_number_(1), waiting_second_step_(false) {
  counts_[0] = {0, 0, 0, 0};
  counts_[1] = {0, 0, 0, 0};
  for (const Pos& pos : board_.valid_positions()) {
    if (board_.is_empty(pos)) continue;
    const int top = board_.top_piece(pos);
    counts_[player_to_index(piece_owner(top))][piece_type(top)] += 1;
  }
}

std::optional<int> TzaarGame::winner() const {
  if (!result_.has_value()) return std::nullopt;
  return result_->winner;
}

bool TzaarGame::can_second_step_pass() const {
  return !is_game_over() && waiting_second_step_;
}

bool TzaarGame::resolve_no_mandatory_capture_if_needed() {
  if (is_game_over() || waiting_second_step_) return false;
  if (has_capture_for_player(current_player_)) return false;
  result_ = GameResult{-current_player_, WinReason::NoMandatoryCapture};
  return true;
}

// ─── 私有方法 ─────────────────────────────────────────────

std::optional<int> TzaarGame::extinction_winner() const {
  for (int player : {kWhite, kBlack}) {
    const int pi = player_to_index(player);
    if (counts_[pi][1] == 0 || counts_[pi][2] == 0 || counts_[pi][3] == 0) {
      return -player;
    }
  }
  return std::nullopt;
}

void TzaarGame::apply_capture(const Move& move, int player) {
  if (move.kind != MoveKind::Capture || !move.src.has_value() || !move.dst.has_value())
    throw std::invalid_argument("Invalid capture move");

  const Pos src = *move.src;
  const Pos dst = *move.dst;
  if (board_.is_empty(src) || board_.is_empty(dst))
    throw std::invalid_argument("Capture move on empty cell");

  const int src_top = board_.top_piece(src);
  const int dst_top = board_.top_piece(dst);
  const int src_height = board_.height(src);
  const int dst_height = board_.height(dst);

  if (piece_owner(src_top) != player || piece_owner(dst_top) == player)
    throw std::invalid_argument("Capture ownership constraints violated");
  if (dst_height > src_height)
    throw std::invalid_argument("Cannot capture stronger stack");

  counts_[player_to_index(piece_owner(dst_top))][piece_type(dst_top)] -= 1;
  board_.set_cell(src, 0, 0);
  board_.set_cell(dst, src_top, src_height);
}

void TzaarGame::apply_reinforce(const Move& move, int player) {
  if (move.kind != MoveKind::Reinforce || !move.src.has_value() || !move.dst.has_value())
    throw std::invalid_argument("Invalid reinforce move");

  const Pos src = *move.src;
  const Pos dst = *move.dst;
  if (board_.is_empty(src) || board_.is_empty(dst))
    throw std::invalid_argument("Reinforce move on empty cell");

  const int src_top = board_.top_piece(src);
  const int dst_top = board_.top_piece(dst);
  const int src_height = board_.height(src);
  const int dst_height = board_.height(dst);

  if (piece_owner(src_top) != player || piece_owner(dst_top) != player)
    throw std::invalid_argument("Reinforce ownership constraints violated");

  counts_[player_to_index(piece_owner(dst_top))][piece_type(dst_top)] -= 1;
  board_.set_cell(src, 0, 0);
  board_.set_cell(dst, src_top, src_height + dst_height);
}

bool TzaarGame::check_immediate_extinction() {
  const auto winner = extinction_winner();
  if (!winner.has_value()) return false;
  result_ = GameResult{*winner, WinReason::Extinction};
  return true;
}

void TzaarGame::end_turn() {
  current_player_ *= -1;
  turn_number_ += 1;
}

// ─── 公開遊戲流程 ─────────────────────────────────────────

void TzaarGame::play_first_step(const Move& move) {
  if (is_game_over()) throw std::invalid_argument("Game is already over");
  if (waiting_second_step_) throw std::invalid_argument("Cannot play first step while waiting second step");
  if (resolve_no_mandatory_capture_if_needed()) return;

  const bool white_first_turn = (turn_number_ == 1 && current_player_ == kWhite);
  apply_capture(move, current_player_);
  if (check_immediate_extinction()) return;

  if (white_first_turn) {
    waiting_second_step_ = false;
    end_turn();
    return;
  }
  waiting_second_step_ = true;
}

void TzaarGame::play_second_step(const std::optional<Move>& move_opt) {
  if (is_game_over()) throw std::invalid_argument("Game is already over");
  if (!waiting_second_step_) throw std::invalid_argument("Cannot play second step before first step");

  Move move = move_opt.value_or(Move{MoveKind::Pass, std::nullopt, std::nullopt});
  const int player = current_player_;

  if (move.kind == MoveKind::Capture) {
    apply_capture(move, player);
    if (check_immediate_extinction()) { waiting_second_step_ = false; return; }
  } else if (move.kind == MoveKind::Reinforce) {
    apply_reinforce(move, player);
    if (check_immediate_extinction()) { waiting_second_step_ = false; return; }
  } else {
    if (move.kind != MoveKind::Pass || !can_second_step_pass())
      throw std::invalid_argument("Second move is not legal");
  }

  waiting_second_step_ = false;
  end_turn();
}

}  // namespace tzaar
