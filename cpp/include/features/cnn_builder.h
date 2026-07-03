#ifndef TZAAR_FEATURES_CNN_BUILDER_H_
#define TZAAR_FEATURES_CNN_BUILDER_H_

#include "core/constants.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tzaar {

// Forward 宣告
class Board;
struct Cell;

// ─── LeafSnapshot（葉節點快照資料結構） ─────────────────
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
  std::vector<std::uint8_t> legal_mask; // kActionCount bytes
};

// ─── 特徵建構 ─────────────────────────────────────────────
void build_cnn_features(
    const Board& board,
    int current_player,
    int turn_number,
    const std::string& phase,
    const std::array<std::array<int, 4>, 2>& counts,
    std::vector<float>& board_state_flat,
    std::vector<float>& global_features);

// 零分配版本：寫入預先配置的 buffer
void build_cnn_features_into(
    const Board& board,
    int current_player,
    int turn_number,
    const std::string& phase,
    const std::array<std::array<int, 4>, 2>& counts,
    float* board_out,
    float* global_out);

}  // namespace tzaar

#endif  // TZAAR_FEATURES_CNN_BUILDER_H_
