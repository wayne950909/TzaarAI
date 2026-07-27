#ifndef TZAAR_MCTS_SEARCH_MANAGER_H_
#define TZAAR_MCTS_SEARCH_MANAGER_H_

#include "core/constants.h"
#include "state/phase_state.h"
#include "mcts/config.h"
#include "mcts/search.h"
#include "features/cnn_builder.h"
#include "external/concurrentqueue.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace tzaar {

// ─── 多樹搜尋管理器（新架構 v3） ──────────────────────────
// 管理多棵 MCTS 搜尋樹，使用多執行緒讓 CPU 和 GPU 可以同時工作。
//
// 實作 multi_thread_logic.md 的新規劃：
//   - 固定分配樹給每個 CPU worker，每個 worker 持有內部活躍樹 ID 清單
//   - 每輪最多 16 次模擬（kBatchIncrement），收集一批葉節點指標送 ConcurrentQueue
//   - 指標型 ConcurrentQueue 傳遞，零拷貝
//   - 1:1 配對：每個 CPU worker 伴隨一個專屬 Result Handler
//   - 防空轉睡眠機制：worker 無事可做時透過 condition_variable 睡眠，
//     由對應的 Result Handler 喚醒
//   - 樹完成後立即從 worker 內部清單中 erase
//
// worker 的存活
//   - 所有 thread 在建構時就被建立
//   - 每次 reset() 重新分配樹並喚醒 workers
//   - 所有樹完成後，workers 進入睡眠等待下一次 reset
//   - shutdown() + join_workers() 時才真正 join threads
// ──────────────────────────────────────────────────────────

class SearchManager {
 public:
  // ─── 建構子 ─────────────────────────────────────────
  // 不傳入 root_states，只設定固定參數，直接建立常駐 threads
  // threads 建立後立即進入 pause wait 狀態
  // config      : 預設搜尋設定（reset 時可重新設定）
  // num_threads : CPU worker 執行緒數量（預設 8）
  //              同時也是 Result Handler 的數量（1:1 配對）
  // max_batch   : 單次 GPU 推論的最大葉節點數（預設 480）
  SearchManager(SearchConfig config,
                int num_threads = 8,
                int max_batch = 480);

  SearchManager(const SearchManager&) = delete;
  SearchManager& operator=(const SearchManager&) = delete;

  // ─── 生命週期 ───────────────────────────────────────
  // 重置樹和狀態，將樹固定分配給各 worker 並喚醒所有 threads
  void reset(const std::vector<PhaseGameState>& root_states,
             SearchConfig config);

  // 停止 workers（設 stop flag + notify_all，不 join）
  void shutdown();

  // 等待所有 workers 結束（真正 join threads）
  void join_workers();

  // ─── Python 端專用介面 ────────────────────────────────

    // 從 ConcurrentQueue 取出一個批次（累積多個 EvalBatch）
  // 從 queue 中儘可能取出所有 EvalBatch，合併成一個大 PackedBatch。
  // max_batch   : 單次回傳的最大葉節點數（預設 480）
  // timeout_ms  : 等待時間（預設 100ms，0 = non-blocking）
  // 回傳 empty batch（batch_size=0）表示沒有資料
  struct PackedBatch {
    int buffer_id = -1;  // 保留相容性
    int batch_size = 0;
    const int32_t* node_ids = nullptr;
    const float* board_state_flat = nullptr;
    const float* global_features = nullptr;
    const uint8_t* legal_masks = nullptr;
    const int32_t* tree_ids = nullptr;
    const int32_t* worker_ids = nullptr;  // 每個 leaf 對應的 worker_id
  };
  
  PackedBatch dequeue_batch(int max_batch = 480, int timeout_ms = 100);

      // 提交 GPU 推論結果
  // worker_ids 陣列，每個 leaf 對應所屬的 worker_id，讓對應的 Result Handler 處理
  void submit_leaf_evals(
      const int32_t* worker_ids,
      const int32_t* node_ids,
      const int32_t* tree_ids,
      const float* priors,
      const float* values,
      int batch_size);

  // ─── 結果取得 ───────────────────────────────────────
  bool is_complete() const;
  int completed_tree_count() const;
  int total_remaining_simulations() const;
  std::vector<SearchResult> finish_all();

  // 內部批次大小常數（每 16 次模擬為一個單位）
  static constexpr int kBatchIncrement = 16;

 private:
  // ─── 內部資料結構 ──────────────────────────────────

  // 從 SearchSession 模擬產出的葉節點資料（指標傳遞）
  // simulate_single_step 會回傳一個 LeafData*（由 caller 負責 delete）
  struct LeafData {
    int node_id;
    int tree_id;
    float board_flat[kBoardFlatSize];
    float global_feat[kGlobalFeatureDim];
    uint8_t legal_mask[kActionCount];
  };

  // ConcurrentQueue 中的批次單位
  struct EvalBatch {
    int worker_id;
    std::vector<LeafData*> leaves;  // 指標向量
  };

  // GPU 送回但還沒寫回樹的結果
  struct PendingEvalResult {
    int buffer_id;  // 保留相容性
    int batch_size;
    std::vector<int32_t> node_ids;
    std::vector<int32_t> tree_ids;
    std::vector<float> priors;
    std::vector<float> values;
  };

  // ─── 成員變數 ──────────────────────────────────────
  SearchConfig config_;
  int num_threads_;
  int max_batch_;
  int tree_count_;
  std::atomic<bool> stop_{false};
  std::atomic<int> completed_count_{0};

  // 樹的管理
    struct SearchTree {
    PhaseGameState root_state;
    std::unique_ptr<SearchSession> session;
    bool completed = false;
    int tree_id = -1;
  };
  std::vector<std::unique_ptr<SearchTree>> trees_;

  // ─── 固定分配：每個 worker 持有自己的活躍樹 ID 清單 ──
  std::vector<std::vector<int>> assigned_trees_;  // [worker_id] -> list of tree IDs

  // ─── ConcurrentQueue（指標傳遞，零拷貝） ─────────
  moodycamel::ConcurrentQueue<EvalBatch> eval_queue_;

    // ─── 每個 worker 的睡眠/喚醒機制（用 unique_ptr 繞過不可複製限制） ──
  std::vector<std::unique_ptr<std::mutex>> worker_cv_mtx_;
  std::vector<std::unique_ptr<std::condition_variable>> worker_cv_;

  // ─── 執行緒池 ─────────────────────────────────────
  std::vector<std::thread> pool_;               // CPU workers
  std::vector<std::thread> result_handlers_;     // 1:1 Result Handlers
  std::atomic<bool> result_handler_stop_{false};

  // ─── Python 搜尋完成通知 ──────────────────────────
  mutable std::mutex search_done_mtx_;
  std::condition_variable search_done_cv_;
  bool search_done_ = false;

        // ─── 每個 Result Handler 專屬的 pending queue ──────
  //  submit_leaf_evals 根據 worker_id 直接放進對應的 queue
  //  每個 Result Handler 只等自己的 queue，不需競爭
  //  使用 unique_ptr 繞過 mutex/condition_variable 不可複製限制
  std::vector<std::unique_ptr<std::mutex>> per_worker_pending_mtx_;
  std::vector<std::vector<PendingEvalResult>> per_worker_pending_results_;
  std::vector<std::unique_ptr<std::condition_variable>> per_worker_pending_cv_;

    // ─── 累積葉節點數閥值（達到此值通知 Python） ────
  std::atomic<int> accumulated_leaves_{0};

        // ─── dequeue_batch 暫存區 ──
  std::vector<int32_t> dequeued_node_ids_;
  std::vector<int32_t> dequeued_tree_ids_;
  std::vector<int32_t> dequeued_worker_ids_;
  std::vector<float> dequeued_board_flat_;
  std::vector<float> dequeued_global_feat_;
  std::vector<uint8_t> dequeued_legal_mask_;

  // ─── 內部方法 ──────────────────────────────────────

  // CPU worker 主迴圈（固定分配樹 + 16次模擬 + 睡眠機制）
  void worker_loop(int worker_id);

  // 1:1 配對的 Result Handler 主迴圈
  void result_handler_loop(int worker_id);

  // 對一棵樹執行一次模擬步驟，回傳 LeafData*（若無則 nullptr）
  // 呼叫方負責 delete
  LeafData* simulate_single_step(int tree_id);

  // 檢查是否有「處在就緒狀態」的樹（未完成且無 pending leaves）
  bool HasAnyReadyTree(const std::vector<int>& tree_ids) const;

    // 處理 GPU 推論結果（寫回樹、解除 pending、喚醒 worker）
  void process_pending_eval_result(int tree_id, int node_id,
                                   const float* priors, float value);

  // 將一棵樹標記為完成
  void complete_tree(SearchTree& tree);

  // 檢查是否所有樹已完成
  bool all_trees_completed() const;
};

}  // namespace tzaar

#endif  // TZAAR_MCTS_SEARCH_MANAGER_H_
