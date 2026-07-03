#ifndef TZAAR_CORE_BOARD_H_
#define TZAAR_CORE_BOARD_H_

#include "core/constants.h"

#include <array>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tzaar {

struct Cell {
  bool valid = false;
  int top_piece = 0;
  int height = 0;
};

class Board {
 public:
  Board();

  // ─── 查詢 ─────────────────────────────────────────────
  bool is_valid_cell(const Pos& pos) const;
  bool is_empty(const Pos& pos) const;
  int top_piece(const Pos& pos) const;
  int height(const Pos& pos) const;

  // ─── 變異 ─────────────────────────────────────────────
  void set_cell(const Pos& pos, int piece_code, int height);

  // ─── 迭代 ─────────────────────────────────────────────
  const std::vector<Pos>& valid_positions() const { return valid_positions_; }
  const std::unordered_map<int, int>& pos_to_idx_map() const { return pos_to_idx_; }

  // ─── 射線查詢 ─────────────────────────────────────────
  std::optional<Pos> first_occupied_in_direction(const Pos& start, int direction_idx) const;

  // quick checks for phase state
  bool has_capture_for_player(int player) const;
  bool has_reinforce_for_player(int player) const;

 private:
  std::array<std::array<Cell, kCols>, kRows> cells_{};
  std::vector<Pos> valid_positions_;
  std::unordered_map<int, int> pos_to_idx_;
  std::unordered_map<int, std::array<std::vector<Pos>, kDirectionCount>> rays_;

  bool is_inside(const Pos& pos) const;
  const Cell& get_cell(const Pos& pos) const;
  Cell& get_mut_cell(const Pos& pos);
  void init_from_layout();
};

}  // namespace tzaar

#endif  // TZAAR_CORE_BOARD_H_
