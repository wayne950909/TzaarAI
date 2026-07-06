#ifndef TZAAR_MCTS_BENCH_H_
#define TZAAR_MCTS_BENCH_H_

#include "mcts/config.h"
#include "mcts/search.h"
#include "state/phase_state.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace tzaar {

// ─── CPU 多執行緒 MCTS 模擬效能測試 ──────────────────────
// 完全獨立於 SearchManager，不走 GPU batch 流程。
//
// 流程：
//   1. 建立 num_trees 棵搜尋樹（SearchSession）
//   2. 啟動 num_threads 個 worker thread
//   3. 每個 worker 用 CAS 鎖定一棵未完成的樹
//   4. 對該樹執行 simulate_into_buffers(1, ...) → mock priors → submit_single_eval
//   5. 重複直到該樹完成 simulations 次模擬
//   6. 解鎖，繼續搶下一棵樹
//   7. 所有樹完成後，回傳耗時與統計資料
// ──────────────────────────────────────────────────────────

struct CpuBenchResult {
  double elapsed_seconds = 0.0;     // 總耗時（秒）
  int num_trees = 0;                 // 樹數量
  int num_threads = 0;               // 執行緒數量
  int simulations_per_tree = 0;      // 每棵樹模擬次數
  int total_simulations = 0;         // 總模擬次數
  int total_simulations_done = 0;    // 實際完成的模擬次數
  int total_nodes_created = 0;       // 所有樹的總節點數
  double sims_per_second = 0.0;      // 每秒模擬次數

  // 每棵樹的詳細資料
  std::vector<int> tree_simulations_done;   // 各樹完成的模擬數
  std::vector<int> tree_node_counts;        // 各樹的節點數
};

class CpuBench {
 public:
  // 執行單一測試（阻塞，直到完成）
  static CpuBenchResult run(int num_trees, int num_threads, int simulations,
                            int leaf_batch_size = 8);

 private:
  // Worker thread 主迴圈
  static void worker_loop(
      int thread_id,
      int num_trees,
      std::vector<std::unique_ptr<SearchSession>>& sessions,
      std::vector<std::atomic<bool>>& tree_locks,
      std::vector<std::atomic<bool>>& tree_completed,
      int simulations,
      int leaf_batch_size,
      std::atomic<int>& next_tree_index);

  // 產生隨機 priors（Dirichlet-like 均勻分佈）
  static void fill_random_priors(float* priors, int num_actions,
                                  std::mt19937& rng);
};

}  // namespace tzaar

#endif  // TZAAR_MCTS_BENCH_H_
