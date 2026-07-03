#include "features/cnn_builder.h"
#include "core/board.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tzaar {

void build_cnn_features(
    const Board& board,
    int current_player,
    int turn_number,
    const std::string& phase,
    const std::array<std::array<int, 4>, 2>& counts,
    std::vector<float>& board_state_flat,
    std::vector<float>& global_features) {

  board_state_flat.assign(kBoardFlatSize, 0.0f);
  global_features.assign(kGlobalFeatureDim, 0.0f);
  build_cnn_features_into(board, current_player, turn_number, phase, counts,
                           board_state_flat.data(), global_features.data());
}

void build_cnn_features_into(
    const Board& board,
    int current_player,
    int turn_number,
    const std::string& phase,
    const std::array<std::array<int, 4>, 2>& counts,
    float* board_out,
    float* global_out) {

  constexpr float max_height_norm = 8.0f;
  std::memset(board_out, 0, static_cast<std::size_t>(kBoardFlatSize) * sizeof(float));

  // ─── 棋盤特徵 12ch × 9 × 9 ─────────────────────────
  for (const Pos& pos : board.valid_positions()) {
    if (board.is_empty(pos)) continue;

    const int top = board.top_piece(pos);
    const int owner = piece_owner(top);
    int type_idx = 0;
    if (top == 1 || top == 4)       type_idx = 0;
    else if (top == 2 || top == 5)  type_idx = 1;
    else if (top == 3 || top == 6)  type_idx = 2;

    const float h_norm = std::min(static_cast<float>(board.height(pos)), max_height_norm) / max_height_norm;
    int occ_ch = 0, h_ch = 0;
    if (owner == current_player) {
      occ_ch = type_idx;
      h_ch = 6 + type_idx;
    } else {
      occ_ch = 3 + type_idx;
      h_ch = 9 + type_idx;
    }

    board_out[occ_ch * 81 + pos.row * 9 + pos.col] = 1.0f;
    board_out[h_ch * 81 + pos.row * 9 + pos.col] = h_norm;
  }

  // ─── 全域特徵 12 維 ─────────────────────────────────
  const int phase_idx = (phase == "step2_action") ? 1 : 0;
  const int own_idx = (current_player == kWhite) ? 0 : 1;
  const int opp_idx = (current_player == kWhite) ? 1 : 0;

  const int own_t1 = counts[own_idx][1];
  const int own_t2 = counts[own_idx][2];
  const int own_t3 = counts[own_idx][3];
  const int opp_t1 = counts[opp_idx][1];
  const int opp_t2 = counts[opp_idx][2];
  const int opp_t3 = counts[opp_idx][3];

  const float turn_norm = std::min(static_cast<float>(turn_number), 200.0f) / 200.0f;
  const float player_sign = (current_player == kWhite) ? 1.0f : -1.0f;

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

}  // namespace tzaar
