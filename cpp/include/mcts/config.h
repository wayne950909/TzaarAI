#ifndef TZAAR_MCTS_CONFIG_H_
#define TZAAR_MCTS_CONFIG_H_

#include "core/constants.h"

#include <cstdint>
#include <vector>

namespace tzaar {

struct SearchConfig {
  int simulations = 0;
  int leaf_batch_size = 1;
  float puct_c = 1.5f;
  bool add_root_dirichlet_noise = false;
  float root_dirichlet_eps = 0.25f;
  float root_dirichlet_alpha = 0.05f;
  int min_batch_for_swap = 0;    // 最小累積 leaf 數才送 GPU (0=用 max_batch)
  int flush_timeout_ms = 0;      // 強制送 GPU 的 timeout (0=不使用)

  // Worker-Local Double Buffer 參數（對應 adjust.md）
  // buffer 最大容量 = tree_count * buffer_capacity_per_tree（填不滿）
  int buffer_capacity_per_tree = 32;
  // 單側尚未滿載前，「到達一定資料量」即觸發 is_ready 的 leaf 數（≠容量）
  int ready_flush_leaves = 32;
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

}  // namespace tzaar

#endif  // TZAAR_MCTS_CONFIG_H_


