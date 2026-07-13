#ifndef TZAAR_MCTS_SEARCH_MANAGER_H_
#define TZAAR_MCTS_SEARCH_MANAGER_H_

#include "core/constants.h"
#include "state/phase_state.h"
#include "mcts/config.h"
#include "mcts/search.h"
#include "features/cnn_builder.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace tzaar {

// ─── 多樹搜尋管理器 ───────────────────────────────────────
// 管理多棵 MCTS 搜尋樹，使用多執行緒 + 雙 buffer 架構
// 讓 CPU 和 GPU 可以同時工作。
//
// 這個類別實作 multi_thread_logic.md 的規劃：
//   - 固定數量 CPU worker threads
//   - 一個佇列存放尚需模擬的樹 ID，worker 從中取出，無樹時 wait
//   - 一棵樹只會由一個執行緒模擬（per-tree mutex）
//   - 每個 worker 的 thread-local leaf buffer（leaf_batch_size 大小）
//   - shared double buffer（write_index / active_writers 原子變數）
//   - 原子交換鎖避免多 worker 同時 swap_buffer
//   - result_handler 負責回寫 GPU 結果，並將未完成的樹重新入隊
// ──────────────────────────────────────────────────────────

class SearchManager {
 public:
  // ─── 建構子 ─────────────────────────────────────────
  // root_states : 每棵樹的根狀態（每個元素 clone 後獨立使用）
  // config      : 共用搜尋設定（每棵樹使用相同設定）
  // num_threads : CPU worker 執行緒數量（預設 10）
  // max_batch   : 單個 buffer 的最大葉節點數（預設 480 = 30 棵樹 * 16 leaf_batch_size）
  SearchManager(const std::vector<PhaseGameState>& root_states,
                SearchConfig config,
                int num_threads = 10,
                int max_batch = 480);

  SearchManager(const SearchManager&) = delete;
  SearchManager& operator=(const SearchManager&) = delete;

  // ─── 生命週期 ───────────────────────────────────────
  void start_workers();
  void wait_for_completion();
  void run();
  void shutdown();

  // ─── Python 端專用介面 ────────────────────────────────
  bool has_ready_batch() const;

  struct PackedBatch {
    int buffer_id = -1;
    int batch_size = 0;
    const int32_t* node_ids = nullptr;
    const float* board_state_flat = nullptr;
    const float* global_features = nullptr;
    const uint8_t* legal_masks = nullptr;
    const int32_t* tree_ids = nullptr;
  };
  PackedBatch get_ready_batch();

  void submit_eval_batch(int buffer_id,
                         const int32_t* node_ids,
                         const float* priors,
                         const float* values,
                         int batch_size);

  // ─── 結果取得 ───────────────────────────────────────
  bool is_complete() const;
  int total_remaining_simulations() const;
  std::vector<SearchResult> finish_all();

 private:
  // ─── 內部資料結構 ──────────────────────────────────

  // 雙 buffer（依 multi_thread_logic.md 規劃）：
  // 每個 buffer 有 write_index 和 active_writers 原子變數，
  // 置於不同 cache line 避免 false sharing。
  struct EvalBuffer {
    // 預先分配的連續記憶體
    std::vector<float> board_flat;       // [max_batch * kBoardFlatSize]
    std::vector<float> global_feat;      // [max_batch * kGlobalFeatureDim]
    std::vector<uint8_t> legal_mask;     // [max_batch * kActionCount]
    std::vector<int32_t> node_ids;       // [max_batch]
    std::vector<int32_t> tree_ids;       // [max_batch]

    int capacity = 0;

    // ── 原子變數（避免 false sharing） ──────────────
    // write_index   ：下一個寫入位置（worker fetch_add 取得）
    // active_writers：目前正在寫入的 worker 數量
    // pending_count ：目前已累積的葉節點總數
    alignas(64) std::atomic<int> write_index{0};
    alignas(64) std::atomic<int> active_writers{0};
    alignas(64) std::atomic<int> pending_count{0};

    bool eval_done = true;  // GPU 是否已處理完這個 buffer
  };

  // GPU 送回但還沒寫回樹的結果。
  struct PendingEvalResult {
    int buffer_id;
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

  // 樹的管理
  struct SearchTree {
    PhaseGameState root_state;
    std::unique_ptr<SearchSession> session;
    std::mutex mtx;                      // per-tree mutex（取代 atomic<bool> locked）
    bool completed = false;
    bool flushed_to_buffer = false;      // 最近一批 leaf 是否已 flush 進 buffer
    int tree_id = -1;
  };
  std::vector<std::unique_ptr<SearchTree>> trees_;

  // ─── 樹 ID 佇列（依 md 規劃） ────────────────────
  // 佇列內放著尚需模擬的樹 ID。
  // worker 從中取出，若佇列為空則 wait。
  // result_handler 處理完 GPU 結果後，若樹尚未完成則重新入隊。
  std::queue<int> tree_queue_;
  mutable std::mutex queue_mtx_;
  std::condition_variable queue_cv_;

  // ─── 雙 buffer ─────────────────────────────────────
  EvalBuffer buffers_[2];
  std::atomic<int> active_buffer_{0};     // CPU 正在填入的 buffer 索引
  bool fillable_[2] = {true, false};      // buffer 是否可以填入

  // 原子交換鎖：避免多個 worker 同時 swap_buffer
  std::atomic<bool> swapping_{false};

  // 強制 swap 的時序控制
  std::atomic<int64_t> last_swap_time_ms_{0};

  // 執行緒池
  std::vector<std::thread> pool_;
  std::thread result_handler_;
  std::atomic<bool> result_handler_stop_{false};

  // 同步（Python 通知用）
  mutable std::mutex cv_mtx_;
  std::condition_variable batch_ready_cv_;
  bool batch_ready_ = false;

  // 同步（結果處理用）
  mutable std::mutex results_mtx_;
  std::condition_variable results_cv_;
  std::vector<PendingEvalResult> pending_results_;

  // ─── 內部方法 ──────────────────────────────────────

  // CPU worker 主迴圈
  void worker_loop(int thread_id);

  // 專用 GPU 結果處理執行緒主迴圈
  void result_handler_loop();

  // Thread-local buffer
  struct ThreadLocalSlot {
    float board_flat[kBoardFlatSize];
    float global_feat[kGlobalFeatureDim];
    uint8_t legal_mask[kActionCount];
    int32_t node_id;
    int32_t tree_id;
  };
  struct ThreadLocalBuffer {
    std::vector<ThreadLocalSlot> slots;
    int count = 0;
  };

  // 從佇列取得一棵樹（佇列空則 wait）
  // 回傳樹 ID，若收到停止訊號則回傳 -1
  int acquire_tree_from_queue();

  // 在一棵樹上模擬 leaf_batch_size 次，填入 local buffer
  void simulate_tree_into_local(int tree_id, ThreadLocalBuffer& local);

  // 將 local buffer 的內容 flush 到 shared buffer
  // 使用 write_index / active_writers 原子變數（不用 mutex）
  void flush_local_to_shared(int thread_id, ThreadLocalBuffer& local, int tree_id);

  // 嘗試 swap_buffer（原子交換鎖保護）
  // 回傳 true 表示成功 swap，false 表示條件不滿足或已被其他 thread swap
  bool try_swap_buffer();

  // 處理 GPU 回傳的評估結果（寫回對應的樹）
  void process_pending_results();

  // 初始化 buffer 記憶體
  void init_buffers();

  // 將所有未完成的樹放入佇列（start_workers 時呼叫）
  void enqueue_all_trees();

  // 將一棵樹重新入隊（若未完成且無 pending leaves）
  void reenqueue_tree_if_needed(int tree_id);

  // 輔助：取得目前毫秒時間
  static int64_t now_ms();

  // 檢查是否所有未完成的樹都已 flush 完在等 GPU
  bool trees_all_stalled() const;
};

}  // namespace tzaar

#endif  // TZAAR_MCTS_SEARCH_MANAGER_H_
