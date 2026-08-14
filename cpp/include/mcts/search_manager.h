#ifndef TZAAR_MCTS_SEARCH_MANAGER_H_
#define TZAAR_MCTS_SEARCH_MANAGER_H_

#include "core/constants.h"
#include "state/phase_state.h"
#include "mcts/config.h"
#include "mcts/search.h"
#include "features/cnn_builder.h"
#include "mcts/concurrent_queue.h"
#include "mcts/debug_log.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace tzaar {

// Worker-local double-buffer multi-tree search manager.
// See mcts_optimization_plan_zh-TW.md for the design.

class SearchManager {
 public:
  SearchManager(SearchConfig config,
                int num_threads = 8,
                int max_batch = 480);

  SearchManager(const SearchManager&) = delete;
  SearchManager& operator=(const SearchManager&) = delete;

  void reset(const std::vector<PhaseGameState>& root_states,
             SearchConfig config);

  void shutdown();
  void join_workers();

  bool has_ready_batch() const;

  struct PackedBatch {
    int batch_size = 0;
    const int32_t* node_ids = nullptr;
    const float* board_state_flat = nullptr;
    const float* global_features = nullptr;
    const uint8_t* legal_masks = nullptr;
    const int32_t* tree_ids = nullptr;
  };
  PackedBatch get_ready_batch();

  void submit_eval_batch(const int32_t* node_ids,
                         const float* priors,
                         const float* values,
                         int batch_size);
  bool is_complete() const;
  int completed_tree_count() const;
  int total_remaining_simulations() const;
  std::string last_swap_reason() const;
  std::vector<SearchResult> finish_all();

 private:
  struct LocalEvalBuffer {
    std::vector<float>   board_flat;
    std::vector<float>   global_feat;
    std::vector<uint8_t> legal_mask;
    std::vector<int32_t> node_ids;
    std::vector<int32_t> tree_ids;
    int count = 0;
    std::atomic<bool> is_ready{false};
  };

  struct WorkerBuffers {
    LocalEvalBuffer bufs[2];
    int active_idx = 0;
  };

  struct AggregateBuffer {
    std::vector<float>   board_flat;
    std::vector<float>   global_feat;
    std::vector<uint8_t> legal_mask;
    std::vector<int32_t> node_ids;
    std::vector<int32_t> tree_ids;
    int count = 0;
  };

  struct PendingEvalResult {
    int batch_size;
    std::vector<int32_t> node_ids;
    std::vector<int32_t> tree_ids;
    std::vector<float> priors;
    std::vector<float> values;
  };

  SearchConfig config_;
  int num_threads_;
  int max_batch_;
  int tree_count_;
  std::atomic<bool> stop_{false};
  std::atomic<int> completed_count_{0};

  struct SearchTree {
    PhaseGameState root_state;
    std::unique_ptr<SearchSession> session;
    bool completed = false;
    int tree_id = -1;
  };
  std::vector<std::unique_ptr<SearchTree>> trees_;

  ConcurrentQueue<int> tree_queue_;

  std::vector<std::unique_ptr<WorkerBuffers>> worker_buffers_;
  int local_capacity_ = 0;      // 每側邊緩衝區容量 = tree_count * buffer_capacity_per_tree
  int ready_flush_leaves_ = 0;  // 單側到達此 leaf 數即觸發 ready（adjust.md 的「一定資料量」）
  AggregateBuffer agg_buf_;

  std::vector<std::thread> pool_;
  std::thread result_handler_;
  std::atomic<bool> result_handler_stop_{false};

  mutable std::mutex swap_reason_mtx_;
  std::string last_swap_reason_;

  mutable std::mutex cv_mtx_;
  std::condition_variable batch_ready_cv_;
  bool batch_ready_ = false;

  mutable std::mutex results_mtx_;
  std::condition_variable results_cv_;
  std::vector<PendingEvalResult> pending_results_;

  void worker_loop(int thread_id);
  void result_handler_loop();
  int wait_for_simulable_tree(WorkerBuffers& wb, int thread_id); // 等待並取得一棵「可模擬」樹 id；stop 時回傳 -1
  bool classify_popped_tree(int tree_id, int thread_id); // 取出樹後判定：true=可模擬（回傳模擬）；false=已處理（等 GPU / 已完成）
  int simulate_tree_into_local(int tree_id, WorkerBuffers& wb);
  bool seal_ready(WorkerBuffers& wb, int thread_id, bool force = false); // 依 adjust.md：設 active ready 並 flip；回傳是否成功封存
  void wait_for_other_collected(WorkerBuffers& wb, int which); // 等待指定側 buffer 被主執行緒收走（is_ready→false）
  void init_buffers();
  void resize_buffer(LocalEvalBuffer& buf, int cap);  // 依指定容量重設單一 buffer
  void resize_buffers(int tree_count);      // 依樹數量調整每側容量
  void reenqueue_tree_if_needed(int tree_id);
  void complete_tree(SearchTree& tree);
  static int64_t now_ms();
  void aggregate_ready();
  void process_pending_results();
};

}  // namespace tzaar

#endif  // TZAAR_MCTS_SEARCH_MANAGER_H_

