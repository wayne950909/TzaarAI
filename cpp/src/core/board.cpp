#include "core/board.h"

#include <stdexcept>

namespace tzaar {

Board::Board() {
  init_from_layout();
}

bool Board::is_inside(const Pos& pos) const {
  return pos.row >= 0 && pos.row < kRows && pos.col >= 0 && pos.col < kCols;
}

const Cell& Board::get_cell(const Pos& pos) const {
  if (!is_inside(pos)) throw std::out_of_range("position out of bounds");
  const Cell& cell = cells_[pos.row][pos.col];
  if (!cell.valid) throw std::invalid_argument("position is invalid cell");
  return cell;
}

Cell& Board::get_mut_cell(const Pos& pos) {
  if (!is_inside(pos)) throw std::out_of_range("position out of bounds");
  Cell& cell = cells_[pos.row][pos.col];
  if (!cell.valid) throw std::invalid_argument("position is invalid cell");
  return cell;
}

bool Board::is_valid_cell(const Pos& pos) const {
  return is_inside(pos) && cells_[pos.row][pos.col].valid;
}

bool Board::is_empty(const Pos& pos) const {
  return get_cell(pos).height == 0;
}

int Board::top_piece(const Pos& pos) const {
  const Cell& cell = get_cell(pos);
  if (cell.height == 0) throw std::invalid_argument("No top piece on empty cell");
  return cell.top_piece;
}

int Board::height(const Pos& pos) const {
  return get_cell(pos).height;
}

void Board::set_cell(const Pos& pos, int piece_code, int height) {
  Cell& cell = get_mut_cell(pos);
  cell.top_piece = piece_code;
  cell.height = height;
}

std::optional<Pos> Board::first_occupied_in_direction(const Pos& start, int direction_idx) const {
  auto ray_it = rays_.find(pack_pos(start));
  if (ray_it == rays_.end()) return std::nullopt;
  if (direction_idx < 0 || direction_idx >= kDirectionCount)
    throw std::out_of_range("direction_idx out of range");
  for (const Pos& pos : ray_it->second[direction_idx]) {
    if (!is_empty(pos)) return pos;
  }
  return std::nullopt;
}

bool Board::has_capture_for_player(int player) const {
  for (const Pos& src : valid_positions_) {
    if (is_empty(src)) continue;
    if (piece_owner(top_piece(src)) != player) continue;
    const int src_height = height(src);
    for (int d = 0; d < kDirectionCount; ++d) {
      auto dst = first_occupied_in_direction(src, d);
      if (!dst.has_value()) continue;
      if (piece_owner(top_piece(*dst)) == player) continue;
      if (height(*dst) <= src_height) return true;
    }
  }
  return false;
}

bool Board::has_reinforce_for_player(int player) const {
  for (const Pos& src : valid_positions_) {
    if (is_empty(src)) continue;
    if (piece_owner(top_piece(src)) != player) continue;
    for (int d = 0; d < kDirectionCount; ++d) {
      auto dst = first_occupied_in_direction(src, d);
      if (!dst.has_value()) continue;
      if (piece_owner(top_piece(*dst)) == player) return true;
    }
  }
  return false;
}

void Board::init_from_layout() {
  valid_positions_.clear();
  pos_to_idx_.clear();
  rays_.clear();

  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const int v = kBoardLayout[r][c];
      Cell cell;
      if (v == -1) {
        cell.valid = false; cell.top_piece = 0; cell.height = 0;
      } else if (v == 0) {
        cell.valid = true; cell.top_piece = 0; cell.height = 0;
      } else {
        cell.valid = true; cell.top_piece = v; cell.height = 1;
      }
      cells_[r][c] = cell;
      if (cell.valid) valid_positions_.push_back(Pos{r, c});
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

}  // namespace tzaar
