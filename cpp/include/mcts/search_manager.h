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
#include <thread>
#include <vector>

namespace tzaar {

// ─── 多樹搜尋管理器 ───────────────────────────────────────
// 管理多棵 MCTS 搜尋樹，使用多執行緒 + 雙 buffer 架構
// 讓 CPU 和 GPU 可以同時工作。
//
// 這個類別是文件中「最終目標架構」的核心實作：
//   - 固定數量 CPU worker threads
//   - 每個 worker 的 thread-local leaf buffer
//   - shared double buffer 給 Python / GPU 讀取
//   - GPU 結果回來後再分發回對應搜尋樹
//
// 目前程式庫已經有這個底層，但正式訓練主流程還沒有切到這裡。
// ──────────────────────────────────────────────────────────

class SearchManager {
 public:
  // ─── 建構子 ─────────────────────────────────────────
  // root_states : 每棵樹的根狀態（每個元素 clone 後獨立使用）
  // config      : 共用搜尋設定（每棵樹使用相同設定）
  // num_threads : CPU worker 執行緒數量（預設 10）
  // max_batch   : 單個 buffer 的最大葉節點數（預設 480 = 20 棵樹 * 16 leaf_batch_size）
  SearchManager(const std::vector<PhaseGameState>& root_states,
                SearchConfig config,
                int num_threads = 10,
                int max_batch = 480);

  // 禁止複製與移動
  SearchManager(const SearchManager&) = delete;
  SearchManager& operator=(const SearchManager&) = delete;

  // ─── 生命週期 ───────────────────────────────────────
  // 這組 API 預期對應的是「長生命週期 manager」，
  // 而不是像目前 Python async_worker 那樣每次搜尋都重建 thread。
  
  // 啟動所有 worker thread（非阻塞）
  void start_workers();

  // 等待所有 worker thread 完成（阻塞）
  void wait_for_completion();

  // 啟動並等待完成（阻塞，等於 start_workers() + wait_for_completion()）
  void run();

  // 要求所有 thread 停止（安全停止）
  void shutdown();

  // ─── CPU Worker 專用介面（給 Python 端） ────────────
  // 檢查是否有待推論的 batch
  bool has_ready_batch() const;

  // 取得已準備好的 batch（需在呼叫 submit_eval_batch 後才可再次呼叫）
  // 如果沒有 ready batch，會阻塞等待
  // 回傳值：batch 資料（零拷貝指標指向內部 buffer）
  struct PackedBatch {
    int buffer_id = -1;
    int batch_size = 0;
    const int32_t* node_ids = nullptr;
    const float* board_state_flat = nullptr;
    const float* global_features = nullptr;
    const uint8_t* legal_masks = nullptr;
    const int32_t* tree_ids = nullptr;  // 每個 node 來自哪棵樹
  };
  PackedBatch get_ready_batch();

  // 提交 NN 評估結果（由 Python 端呼叫）
  // buffer_id : 對應 get_ready_batch 回傳的 buffer_id
  void submit_eval_batch(int buffer_id,
                         const int32_t* node_ids,
                         const float* priors,
                         const float* values,
                         int batch_size);

  // ─── 結果取得 ───────────────────────────────────────
  bool is_complete() const;
  std::vector<SearchResult> finish_all();

 private:
  // ─── 內部資料結構 ──────────────────────────────────

  // 雙 buffer：一個讓 CPU 填，一個讓 GPU 讀。
  // 設計目的是讓 CPU 不必等待 GPU 完成後才繼續下一批模擬。
  struct EvalBuffer {
    // 預先分配的連續記憶體
    std::vector<float> board_flat;       // [max_batch * kBoardFlatSize]
    std::vector<float> global_feat;      // [max_batch * kGlobalFeatureDim]
    std::vector<uint8_t> legal_mask;     // [max_batch * kActionCount]
    std::vector<int32_t> node_ids;       // [max_batch]
    std::vector<int32_t> tree_ids;       // [max_batch]  ← 每個 node 的來源樹

    int capacity = 0;
    std::atomic<int> pending_count{0};   // 目前累積了多少葉節點
    bool eval_done = true;               // GPU 是否已處理完這個 buffer
  };

  // GPU 送回但還沒寫回樹的結果。
  // 這讓「GPU forward 完成」和「回寫 session」可以解耦。
  struct PendingEvalResult {
    int buffer_id;
    int batch_size;
    std::vector<int32_t> node_ids;
    std::vector<int32_t> tree_ids;
    std::vector<float> priors;   // flat: [batch_size * kActionCount]
    std::vector<float> values;   // [batch_size]
  };

  // ─── 成員變數 ──────────────────────────────────────
  SearchConfig config_;
  int num_threads_;
  int max_batch_;
  int tree_count_;
  std::atomic<bool> stop_{false};

  // 樹的管理 — 用 unique_ptr 避免 atomic+unique_ptr 的拷貝問題
  struct SearchTree {
    PhaseGameState root_state;
    std::unique_ptr<SearchSession> session;
    std::atomic<bool> locked{false};
    bool completed = false;
    int tree_id = -1;
  };
  std::vector<std::unique_ptr<SearchTree>> trees_;
  std::atomic<int> next_tree_index_{0};  // round-robin 鎖定用

  // 雙 buffer
  EvalBuffer buffers_[2];
  std::atomic<int> active_buffer_{0};  // CPU 正在填入的 buffer 索引

  // 執行緒池
  std::vector<std::thread> pool_;

  // 同步
  mutable std::mutex cv_mtx_;
  std::condition_variable batch_ready_cv_;   // 通知 Python："buffer 滿了"
  std::condition_variable eval_done_cv_;     // 通知 CPU thread："buffer 可用"
  bool batch_ready_ = false;                 // predicate for batch_ready_cv_

  mutable std::mutex results_mtx_;
  std::condition_variable results_cv_;
  std::vector<PendingEvalResult> pending_results_;

  // ─── 內部方法 ──────────────────────────────────────

  // CPU worker 主迴圈
  void worker_loop(int thread_id);

  // Thread-local buffer：每個 thread 有自己的小 buffer（leaf_batch_size 大小）
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

  // 嘗試鎖定一棵樹（CAS）
  int try_lock_tree();

  // 在一棵樹上模擬一輪 (leaf_batch_size 次)，填入 local buffer
  void simulate_tree_into_local(int tree_id, ThreadLocalBuffer& local);

  // 將 local buffer 的內容 flush 到 shared buffer
  void flush_local_to_shared(int thread_id, ThreadLocalBuffer& local);

  // 交換 buffer（active_buffer 切換）
  void swap_buffer(int old_buffer_id);

  // 處理 GPU 回傳的評估結果（寫回對應的樹）
  void process_pending_results();

  // 初始化 buffer 記憶體
  void init_buffers();
};

}  // namespace tzaar

#endif  // TZAAR_MCTS_SEARCH_MANAGER_H_
