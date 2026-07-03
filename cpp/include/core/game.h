#ifndef TZAAR_CORE_GAME_H_
#define TZAAR_CORE_GAME_H_

#include "core/constants.h"
#include "core/board.h"

#include <array>
#include <optional>

namespace tzaar {

class TzaarGame {
 public:
  TzaarGame();

  // ─── 唯讀查詢 ─────────────────────────────────────────
  const Board& board() const { return board_; }
  bool is_waiting_second_step() const { return waiting_second_step_; }
  bool is_game_over() const { return result_.has_value(); }
  int current_player() const { return current_player_; }
  int turn_number() const { return turn_number_; }
  std::optional<int> winner() const;
  std::array<std::array<int, 4>, 2> piece_counts() const { return counts_; }
  bool can_second_step_pass() const;

  // ─── 規則查詢（用於 PhaseGameState） ──────────────────
  bool has_capture_for_player(int player) const { return board_.has_capture_for_player(player); }
  bool has_reinforce_for_player(int player) const { return board_.has_reinforce_for_player(player); }

  // ─── 遊戲流程 ─────────────────────────────────────────
  bool resolve_no_mandatory_capture_if_needed();
  void play_first_step(const Move& move);
  void play_second_step(const std::optional<Move>& move_opt);

 private:
  Board board_;
  int current_player_;
  int turn_number_;
  bool waiting_second_step_;
  std::optional<GameResult> result_;
  std::array<std::array<int, 4>, 2> counts_{};

  std::optional<int> extinction_winner() const;
  void apply_capture(const Move& move, int player);
  void apply_reinforce(const Move& move, int player);
  bool check_immediate_extinction();
  void end_turn();
};

}  // namespace tzaar

#endif  // TZAAR_CORE_GAME_H_
