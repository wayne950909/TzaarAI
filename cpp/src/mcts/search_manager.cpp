// search_manager.cpp
// 實作 SearchManager：多樹 MCTS 搜尋的管理器。
// 使用固定數量 worker thread + 樹 ID 佇列 + 每個執行緒專屬雙緩衝區。
//
// 本檔案對應 multi_thread_logic.md + mcts_optimization_plan_zh-TW.md 的設計：
//   - Worker 從 ConcurrentQueue 取出樹 ID，無樹時 wait
//   - 一棵樹由 ConcurrentQueue 確保唯一持有，無需 per-tree mutex
//   - 每個 Worker 有「執行緒專屬雙緩衝區（Worker-Local Double Buffer）」
//     （無鎖、單一寫入者，不需 write_index/active_writers 原子變數）
//   - 緩衝區填滿容量或全體樹 Stalled 時封存為 is_ready（release 語意）
//   - 主執行緒 get_ready_batch() 掃描各 worker 就緒 buffer 並 memcpy 彙整
//   - result_handler 處理 GPU 結果並將未完成樹重新入隊

#include "mcts/search_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <mutex>

// ─── NVTX Profiling ────────────────────────────────────
#ifdef TZAAR_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#else
#define nvtxRangePushA(x)
#define nvtxRangePop()
#endif

namespace tzaar {

// ══════════════════════════════════════════════════════════════════════
// 建構子
// ══════════════════════════════════════════════════════════════════════

SearchManager::SearchManager(SearchConfig config,
                             int num_threads,
                             int max_batch)
    : config_(std::move(config)),
      num_threads_(num_threads),
      max_batch_(max_batch),
      tree_count_(0) {

  if (num_threads_ <= 0)
    throw std::invalid_argument("num_threads must be >= 1");
  if (max_batch_ <= 0)
    throw std::invalid_argument("max_batch must be >= 1");

  ready_flush_leaves_ = std::max(1, config_.ready_flush_leaves);

  // 初始化 worker-local buffers + 彙整暫存區。
  // 必須在建立 threads「之前」，否則 worker 一啟動就會解引用空向量而出錯。
  init_buffers();

  // 建立常駐 threads（佇列空的，workers 直接在 tree_queue_.wait_and_pop 中 wait）
  stop_ = false;
  result_handler_stop_ = false;

  result_handler_ = std::thread(&SearchManager::result_handler_loop, this);
  pool_.reserve(static_cast<std::size_t>(num_threads_));
  for (int i = 0; i < num_threads_; ++i) {
    pool_.emplace_back(&SearchManager::worker_loop, this, i);
  }
}

// ══════════════════════════════════════════════════════════════════════
// Worker 專屬雙緩衝區初始化
// ══════════════════════════════════════════════════════════════════════

void SearchManager::init_buffers() {
  // 建構時尚無樹數量，先以預設單側容量建立 worker-buffers。
  // 真正的容量會在建構後第一次 reset() 依樹數量重設（resize_buffers）。
  if (local_capacity_ <= 0)
    local_capacity_ = std::max(1, config_.buffer_capacity_per_tree);

    worker_buffers_.clear();
  worker_buffers_.reserve(static_cast<std::size_t>(num_threads_));
  for (int w = 0; w < num_threads_; ++w) {
    auto wb_ptr = std::make_unique<WorkerBuffers>();
    WorkerBuffers& wb = *wb_ptr;
    wb.active_idx = 0;
    for (int b = 0; b < 2; ++b) {
      resize_buffer(wb.bufs[b], local_capacity_);
    }
    worker_buffers_.push_back(std::move(wb_ptr));
  }

  // 彙整暫存區（容量 = max_batch_）
  agg_buf_.board_flat.assign(
      static_cast<std::size_t>(max_batch_) *
          static_cast<std::size_t>(kBoardFlatSize),
      0.0f);
  agg_buf_.global_feat.assign(
      static_cast<std::size_t>(max_batch_) *
          static_cast<std::size_t>(kGlobalFeatureDim),
      0.0f);
  agg_buf_.legal_mask.assign(
      static_cast<std::size_t>(max_batch_) *
          static_cast<std::size_t>(kActionCount),
      0);
  agg_buf_.node_ids.assign(static_cast<std::size_t>(max_batch_), 0);
  agg_buf_.tree_ids.assign(static_cast<std::size_t>(max_batch_), 0);
  agg_buf_.count = 0;
}

// 依指定容量重設單一側邊 buffer（不移動 WorkerBuffers 物件，執行緒參考安全）。
void SearchManager::resize_buffer(LocalEvalBuffer& buf, int cap) {
  cap = std::max(cap, 1);
  buf.board_flat.assign(static_cast<std::size_t>(cap) *
                            static_cast<std::size_t>(kBoardFlatSize),
                        0.0f);
  buf.global_feat.assign(static_cast<std::size_t>(cap) *
                             static_cast<std::size_t>(kGlobalFeatureDim),
                         0.0f);
  buf.legal_mask.assign(static_cast<std::size_t>(cap) *
                            static_cast<std::size_t>(kActionCount),
                        0);
  buf.node_ids.assign(static_cast<std::size_t>(cap), 0);
  buf.tree_ids.assign(static_cast<std::size_t>(cap), 0);
  buf.count = 0;
  buf.is_ready.store(false, std::memory_order_relaxed);
}

// 依樹數量調整每側緩衝區容量：容量 = tree_count * buffer_capacity_per_tree。
// 執行緒中快取的 WorkerBuffers& 參考指向 vector<unique_ptr> 的元素，
// 此處僅 resize 內部 LocalEvalBuffer 的 vector，不回重建 worker_buffers_，
// 因此執行緒持有的參考不會失效。
void SearchManager::resize_buffers(int tree_count) {
  local_capacity_ = std::max(1, tree_count) *
                    std::max(1, config_.buffer_capacity_per_tree);
  for (auto& wb_ptr : worker_buffers_) {
    WorkerBuffers& wb = *wb_ptr;
    for (int b = 0; b < 2; ++b) {
      resize_buffer(wb.bufs[b], local_capacity_);
    }
  }
}

// ══════════════════════════════════════════════════════════════════════
// 生命週期
// ══════════════════════════════════════════════════════════════════════

void SearchManager::reset(const std::vector<PhaseGameState>& root_states,
                           SearchConfig config) {
  // 不用暫停 workers，直接重置
  config_ = std::move(config);
  tree_count_ = static_cast<int>(root_states.size());

  if (tree_count_ <= 0)
    throw std::invalid_argument("root_states must not be empty");

  ready_flush_leaves_ = std::max(1, config_.ready_flush_leaves);

  // 依樹數量調整每側緩衝區容量（adjust.md：容量 = 樹數量 * buffer_capacity_per_tree）
  resize_buffers(tree_count_);

  // 重置 trees
  trees_.clear();
  trees_.reserve(static_cast<std::size_t>(tree_count_));
  for (int i = 0; i < tree_count_; ++i) {
    auto tree = std::make_unique<SearchTree>();
    tree->root_state = root_states[static_cast<std::size_t>(i)].clone();
    tree->session = std::make_unique<SearchSession>(tree->root_state, config_);
    tree->tree_id = i;
    trees_.push_back(std::move(tree));
  }

  // 重置 worker-local buffers + 彙整暫存區的「狀態」。
  // 注意：不能呼叫 init_buffers()（那會 clear + 重建 worker_buffers_，
  // 使執行緒中快取的 WorkerBuffers& 參考失效）。此處只在原地清空計數/旗標。
  for (auto& wb_ptr : worker_buffers_) {
    WorkerBuffers& wb = *wb_ptr;
    wb.active_idx = 0;
    for (int b = 0; b < 2; ++b) {
      LocalEvalBuffer& buf = wb.bufs[b];
      buf.count = 0;
      buf.is_ready.store(false, std::memory_order_relaxed);
    }
  }
  agg_buf_.count = 0;
  last_swap_reason_.clear();

  // 重置所有計數器和狀態
  completed_count_ = 0;
  stop_ = false;
  result_handler_stop_ = false;
  pending_results_.clear();

  // 清空佇列並重新入隊
  tree_queue_.clear();
  tree_queue_.reset_stop();
  for (int i = 0; i < tree_count_; ++i) {
    tree_queue_.push(i);
  }
}

void SearchManager::shutdown() {
  stop_ = true;
  result_handler_stop_ = true;

  tree_queue_.notify_all();
  batch_ready_cv_.notify_all();
  results_cv_.notify_all();
}

void SearchManager::join_workers() {
  // 確保 thread 已經被通知停止
  shutdown();

  for (auto& t : pool_) {
    if (t.joinable()) {
      t.join();
    }
  }
  pool_.clear();

  if (result_handler_.joinable()) {
    result_handler_.join();
  }

  // 處理殘留結果
  process_pending_results();
}

// ══════════════════════════════════════════════════════════════════════
// Worker Thread 主迴圈
// ══════════════════════════════════════════════════════════════════════

// Worker 的節奏（依 adjust.md 實作 worker-local double buffer 狀態機）：
// 1. 嘗試從佇列取得一棵「可模擬」的樹；拿不到（全體樹 stalled / 都在等 GPU）
//    → 依 adjust.md「無法從佇列獲得樹時將正在寫入的 buffer 設為 ready」→ seal + 睡眠。
// 2. 在同一棵樹上模擬，直接寫入本 worker 的 active buffer。
// 3. 依 adjust.md 條件檢查是否把 active buffer 設為 ready（須兩側都 false 且達資料量）。
// 4. 若樹已完成則標記；全完成則將最後的 buffer 送出並結束。
void SearchManager::worker_loop(int thread_id) {
  WorkerBuffers& wb = *worker_buffers_[static_cast<std::size_t>(thread_id)];
  while (!stop_) {
    // ── 嘗試取得一棵可模擬樹 ──────────────────────────────
    int tree_id = -1;
    bool got_simulable = false;
    while (tree_queue_.try_pop(tree_id)) {
      SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];

      // 還在等 GPU：無法模擬。此樹由 result_handler 處理完後 reenqueue。
      if (tree.session->has_pending_leaves()) continue;

      // 已完整模擬：標記完成後試下一棵。
      if (tree.session->is_complete()) {
        complete_tree(tree);
        if (is_complete()) {
          // 全部完成 → 把殘留 active buffer 設 ready 送出，結束。
          seal_ready(wb, true);
          return;
        }
        continue;
      }

      got_simulable = true;
      break;
    }

    if (!got_simulable) {
      // 無法從佇列獲得可模擬樹（所有未完成樹都在等 GPU / 已完成）。
      // adjust.md：執行緒無法從佇列獲得樹時，將正在寫入的 buffer 設為 ready。
      seal_ready(wb, true);
      if (is_complete()) return;

      // 睡眠：等 result_handler 處理完一批後 reenqueue 喚醒。
      int next;
      if (tree_queue_.wait_and_pop(next)) {
        continue;  // 有新樹可模擬（或該樹由下一輪處理）
      }
      return;  // stop
    }

    // ── 模擬並寫入 active buffer ──────────────────────────
    simulate_tree_into_local(tree_id, wb);

    // ── 依 adjust.md 檢查是否設 active buffer 為 ready ──
    // 到達「一定資料量」且另一側不是 ready 時才會觸發（seal_ready 內部檢查）。
    bool sealed = seal_ready(wb, false);

    // adjust.md 狀況 1：若另一側 ready 且本側已達資料量，則不能設 ready，
    // 但也不能無止盡填下去。此時等待另一側被主執行緒收走（→狀況 2），
    // 再重試封存。（此處尚未 flip，要等的是另一側 1-active_idx）
    if (!sealed &&
        wb.bufs[wb.active_idx].count >= ready_flush_leaves_ &&
        wb.bufs[1 - wb.active_idx].is_ready.load(std::memory_order_acquire)) {
      wait_for_other_collected(wb, 1 - wb.active_idx);  // 等另一側被收走（true→false）
      seal_ready(wb, false);
    }

    // ── 處理樹的完成狀態 ─────────────────────────────────
    SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
    if (tree.session->is_complete()) {
      complete_tree(tree);
      if (is_complete()) {
        seal_ready(wb, true);
        return;
      }
    }
  }
}

// 若樹未完成且無 pending leaves，重新放入佇列
void SearchManager::reenqueue_tree_if_needed(int tree_id) {
  SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
  if (tree.completed) return;
  if (tree.session->has_pending_leaves()) return;  // 還在等 GPU
  tree_queue_.push(tree_id);
}

void SearchManager::complete_tree(SearchTree& tree) {
  if (tree.completed) return;  // 防止重複標記

  tree.completed = true;
  completed_count_.fetch_add(1, std::memory_order_release);

  if (is_complete()) {
    {
      std::lock_guard<std::mutex> cv_lock(cv_mtx_);
      batch_ready_ = true;
    }
    batch_ready_cv_.notify_all();
  }
}

// ══════════════════════════════════════════════════════════════════════
// 模擬並寫入 worker 專屬 active buffer
// ══════════════════════════════════════════════════════════════════════

void SearchManager::simulate_tree_into_local(int tree_id, WorkerBuffers& wb) {
  SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
  auto& session = tree.session;

  LocalEvalBuffer& buf = wb.bufs[wb.active_idx];

  // adjust.md：buffer 最大容量是「硬上限」（填不滿），但 is_ready 的觸發是
  // 「到達一定資料量」。此處把單次寫入目標設為 ready_flush_leaves_，
  // 讓 worker 一到資料量就交由 worker_loop 觸發 seal_ready（不填滿容量）。
  const int fill_target = std::min(local_capacity_, ready_flush_leaves_);

  // 在同一棵樹上持續模擬，直到：
  //   - active buffer 到達資料量（count >= fill_target）
  //   - 樹的模擬次數已耗盡（is_complete）
  //   - 樹 stalled（simulate_into_buffers 回傳 0 且有 pending leaves）
  while (buf.count < fill_target) {
    if (session->is_complete()) break;
    if (session->has_pending_leaves()) break;

    const int remaining_cap = fill_target - buf.count;
    const int chunk = std::min(config_.leaf_batch_size, remaining_cap);
    if (chunk <= 0) break;

    // 計算要寫入 active buffer 的位置偏移
    float* board_ptr = buf.board_flat.data() +
        static_cast<std::size_t>(buf.count) *
            static_cast<std::size_t>(kBoardFlatSize);
    float* global_ptr = buf.global_feat.data() +
        static_cast<std::size_t>(buf.count) *
            static_cast<std::size_t>(kGlobalFeatureDim);
    uint8_t* mask_ptr = buf.legal_mask.data() +
        static_cast<std::size_t>(buf.count) *
            static_cast<std::size_t>(kActionCount);
    int32_t* node_ptr = buf.node_ids.data() + static_cast<std::size_t>(buf.count);
    int32_t* tree_ptr = buf.tree_ids.data() + static_cast<std::size_t>(buf.count);

    int n = session->simulate_into_buffers(
        chunk,
        board_ptr, global_ptr, mask_ptr,
        node_ptr, tree_ptr,
        tree_id);

    if (n > 0) {
      buf.count += n;
    }

    // simulate_into_buffers 回傳 0 表示 stalled 或樹已 complete。
    // 跳出 while 迴圈，不要 busy loop。
    if (n == 0) {
      break;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════
// 封存 active buffer 並翻轉
// ══════════════════════════════════════════════════════════════════════

// ── 依 adjust.md 實作的 is_ready→ flip 狀態機 ──────────────────────────
//
// adjust.md「is_ready 變為 true 的條件」：
//   前提：兩個 buffer 都要為 false。
//   條件 a：到達一定的資料量（ready_flush_leaves_），如果沒有其他 ready 的 buffer。
//   條件 b（強制 force）：所有的樹都 stalled → 執行緒無法從佇列獲得樹時，
//           將自己正在寫入的 buffer 設為 ready。
// 註：ready 從 false→true 時以 release 發布，執行緒會 flip 到另一側續寫。
//
// 回傳 true 表示成功封存並 flip；false 表示不符合條件（未封存）。

bool SearchManager::seal_ready(WorkerBuffers& wb, bool force) {
  LocalEvalBuffer& active = wb.bufs[wb.active_idx];
  if (active.count <= 0) return false;   // 無資料可送出
  if (active.is_ready.load(std::memory_order_acquire)) return false;

  LocalEvalBuffer& other = wb.bufs[1 - wb.active_idx];
  const bool other_ready = other.is_ready.load(std::memory_order_acquire);

  if (!force) {
    // adjust.md 前提：兩側都必須 false，且「沒有其他 ready 的 buffer」。
    if (other_ready) return false;                          // 狀況 1：繼續填
    if (active.count < ready_flush_leaves_) return false;   // 未達一定資料量
  }
  // force：全體樹 stalled / 無法取得樹。即使另一側也 ready 仍封存，
  // 主執行緒會收走兩側。

  // 設 ready（false→true，release 語意，確保先前寫入可見）
  active.is_ready.store(true, std::memory_order_release);

  // flip 到另一側
  wb.active_idx = 1 - wb.active_idx;
  LocalEvalBuffer& next = wb.bufs[wb.active_idx];

  // 等 flip 後的新 active（= 另一側）被主執行緒收走清空後才可重寫。
  wait_for_other_collected(wb, wb.active_idx);
  next.count = 0;

  // 通知主執行緒有就緒資料
  {
    std::lock_guard<std::mutex> lock(cv_mtx_);
    batch_ready_ = true;
  }
  batch_ready_cv_.notify_one();
  return true;
}

// 等待指定側 buffer 被主執行緒收走清空（is_ready true→false）。
void SearchManager::wait_for_other_collected(WorkerBuffers& wb, int which) {
  LocalEvalBuffer& tgt = wb.bufs[which];
  int spin = 0;
  while (tgt.is_ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
    if (++spin > 10000) {
      // 長時間等待仍未被收走，交還 CPU 避免吃滿核心
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      spin = 0;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════
// Python 端專用介面
// ══════════════════════════════════════════════════════════════════════

bool SearchManager::has_ready_batch() const {
  for (const auto& wb_ptr : worker_buffers_) {
    const WorkerBuffers& wb = *wb_ptr;
    for (int b = 0; b < 2; ++b) {
      if (wb.bufs[b].is_ready.load(std::memory_order_acquire)) {
        return true;
      }
    }
  }
  return false;
}

// 依 mcts_optimization_plan_zh-TW.md 三：
// 1. 條件等待，直到就緒總量達批次閥值、全體 Stalled、或搜尋完成
// 2. 掃描所有 worker 的 bufs[0]/bufs[1]，對 is_ready 者 memcpy 彙整至 agg_buf_
// 3. 重置被彙整的 buffer（count=0, is_ready=false）歸還給 worker
SearchManager::PackedBatch SearchManager::get_ready_batch() {
  {
    std::unique_lock<std::mutex> lock(cv_mtx_);
    // 依 adjust.md：主執行緒偵測到 worker 設 ready 的 buffer 後即可收走彙整。
    // 等到「任一 worker buffer 就緒」或「搜尋完成」才喚醒。
    batch_ready_cv_.wait(lock, [this]() {
      return has_ready_batch() ||
             (completed_count_.load(std::memory_order_acquire) >= tree_count_ &&
              batch_ready_);
    });
    if (has_ready_batch()) {
      batch_ready_ = false;
    }
  }

  // 彙整各就緒 worker buffer 到 agg_buf_
  aggregate_ready();

  PackedBatch result;
  result.batch_size = agg_buf_.count;
  if (agg_buf_.count > 0) {
    result.node_ids = agg_buf_.node_ids.data();
    result.board_state_flat = agg_buf_.board_flat.data();
    result.global_features = agg_buf_.global_feat.data();
    result.legal_masks = agg_buf_.legal_mask.data();
    result.tree_ids = agg_buf_.tree_ids.data();
  }
  // 清空（aggregate_ready 已處理索引；若無資料 batch_size=0，Python 端會 sleep）
  return result;
}

void SearchManager::submit_eval_batch(const int32_t* /*node_ids*/,
                                       const float* priors,
                                       const float* values,
                                       int batch_size) {
  if (batch_size <= 0)
    throw std::invalid_argument("batch_size must be >= 1");

  // 將 GPU 結果放入 pending_results_（回填 tree 的邏輯以 node/tree id 為準）
  {
    std::lock_guard<std::mutex> lock(results_mtx_);

    PendingEvalResult res;
    res.batch_size = batch_size;
    res.node_ids.assign(agg_buf_.node_ids.begin(),
                        agg_buf_.node_ids.begin() +
                            static_cast<std::size_t>(batch_size));
    res.tree_ids.assign(agg_buf_.tree_ids.begin(),
                        agg_buf_.tree_ids.begin() +
                            static_cast<std::size_t>(batch_size));
    res.priors.resize(static_cast<std::size_t>(batch_size) *
                      static_cast<std::size_t>(kActionCount));
    res.values.resize(static_cast<std::size_t>(batch_size));

    std::memcpy(res.priors.data(), priors,
                static_cast<std::size_t>(batch_size) *
                    static_cast<std::size_t>(kActionCount) * sizeof(float));
    std::memcpy(res.values.data(), values,
                static_cast<std::size_t>(batch_size) * sizeof(float));

    pending_results_.push_back(std::move(res));
  }
  results_cv_.notify_all();
}

// ══════════════════════════════════════════════════════════════════════
// 專用 GPU 結果處理執行緒
// ══════════════════════════════════════════════════════════════════════

void SearchManager::result_handler_loop() {
  while (!result_handler_stop_) {
    std::unique_lock<std::mutex> lock(results_mtx_);

    results_cv_.wait(lock, [this]() {
      return !pending_results_.empty() || result_handler_stop_;
    });

    if (result_handler_stop_ && pending_results_.empty()) {
      break;
    }

    auto results_to_process = std::move(pending_results_);
    pending_results_.clear();
    lock.unlock();

    for (auto& res : results_to_process) {
      for (int i = 0; i < res.batch_size; ++i) {
        const int tree_id = res.tree_ids[static_cast<std::size_t>(i)];
        const int node_id = res.node_ids[static_cast<std::size_t>(i)];

        if (tree_id < 0 || tree_id >= tree_count_) {
          continue;
        }
        SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
        if (tree.completed) {
          continue;
        }

        const float* priors_row =
            res.priors.data() + (static_cast<std::size_t>(i) *
                                 static_cast<std::size_t>(kActionCount));
        const float value = res.values[static_cast<std::size_t>(i)];

        // 將 NN 評估結果寫入樹（復原 virtual loss + expand + backup）
        // Worker 操作的是尚未送出的 pending leaves，
        // result_handler 操作的是已送回來的 GPU 結果，兩者不重疊。
        tree.session->submit_single_eval(node_id, priors_row, value);

        if (!tree.session->is_complete() && !tree.session->has_pending_leaves()) {
          reenqueue_tree_if_needed(tree_id);
        } else if (tree.session->is_complete()) {
          complete_tree(tree);
        }
      }
    }

    // 記憶體屏障：確保寫入對 worker 可見
    std::atomic_thread_fence(std::memory_order_release);
  }
}

// ══════════════════════════════════════════════════════════════════════
// 處理 GPU 回傳的結果（保留給 join_workers 的最終清理用）
// ══════════════════════════════════════════════════════════════════════

void SearchManager::process_pending_results() {
  std::vector<PendingEvalResult> results_to_process;
  {
    std::lock_guard<std::mutex> lock(results_mtx_);
    results_to_process.swap(pending_results_);
  }

  for (auto& res : results_to_process) {
    for (int i = 0; i < res.batch_size; ++i) {
      const int tree_id = res.tree_ids[static_cast<std::size_t>(i)];
      const int node_id = res.node_ids[static_cast<std::size_t>(i)];

      if (tree_id < 0 || tree_id >= tree_count_) {
        continue;
      }
      SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
      if (tree.completed) {
        continue;
      }

      const float* priors_row =
          res.priors.data() + (static_cast<std::size_t>(i) *
                               static_cast<std::size_t>(kActionCount));
      const float value = res.values[static_cast<std::size_t>(i)];

      tree.session->submit_single_eval(node_id, priors_row, value);
    }
  }

  std::atomic_thread_fence(std::memory_order_release);
}

// ══════════════════════════════════════════════════════════════════════
// 彙整機制
// ══════════════════════════════════════════════════════════════════════

void SearchManager::aggregate_ready() {
  agg_buf_.count = 0;

  for (auto& wb_ptr : worker_buffers_) {
    WorkerBuffers& wb = *wb_ptr;
    for (int b = 0; b < 2; ++b) {
      LocalEvalBuffer& buf = wb.bufs[b];
      if (!buf.is_ready.load(std::memory_order_acquire)) {
        continue;
      }
      const int n = buf.count;
      if (n <= 0) {
        buf.is_ready.store(false, std::memory_order_release);
        continue;
      }
      // 不超過 agg_buf_ 上限 max_batch_
      if (agg_buf_.count + n > max_batch_) {
        continue;  // 本 buffer 不彙整（保留就緒供下次）
      }
      const int dst = agg_buf_.count;
      std::memcpy(
          agg_buf_.board_flat.data() +
              static_cast<std::size_t>(dst) * static_cast<std::size_t>(kBoardFlatSize),
          buf.board_flat.data(),
          static_cast<std::size_t>(n) *
              static_cast<std::size_t>(kBoardFlatSize) * sizeof(float));
      std::memcpy(
          agg_buf_.global_feat.data() +
              static_cast<std::size_t>(dst) * static_cast<std::size_t>(kGlobalFeatureDim),
          buf.global_feat.data(),
          static_cast<std::size_t>(n) *
              static_cast<std::size_t>(kGlobalFeatureDim) * sizeof(float));
      std::memcpy(
          agg_buf_.legal_mask.data() +
              static_cast<std::size_t>(dst) * static_cast<std::size_t>(kActionCount),
          buf.legal_mask.data(),
          static_cast<std::size_t>(n) *
              static_cast<std::size_t>(kActionCount) * sizeof(uint8_t));
      std::memcpy(agg_buf_.node_ids.data() + static_cast<std::size_t>(dst),
                  buf.node_ids.data(),
                  static_cast<std::size_t>(n) * sizeof(int32_t));
      std::memcpy(agg_buf_.tree_ids.data() + static_cast<std::size_t>(dst),
                  buf.tree_ids.data(),
                  static_cast<std::size_t>(n) * sizeof(int32_t));

      agg_buf_.count += n;

      // 重置被彙整的 buffer，歸還給 worker
      buf.count = 0;
      buf.is_ready.store(false, std::memory_order_release);
    }
  }

  {
    std::lock_guard<std::mutex> lock(swap_reason_mtx_);
    last_swap_reason_ = "aggregate:" +
        std::to_string(agg_buf_.count) + "leaves";
  }
}

// ══════════════════════════════════════════════════════════════════════
// 結果取得
// ══════════════════════════════════════════════════════════════════════

bool SearchManager::is_complete() const {
  if (completed_count_.load(std::memory_order_acquire) < tree_count_) {
    return false;
  }
  std::lock_guard<std::mutex> results_lock(results_mtx_);
  return pending_results_.empty();
}

int SearchManager::completed_tree_count() const {
  return completed_count_.load(std::memory_order_relaxed);
}

std::string SearchManager::last_swap_reason() const {
  std::lock_guard<std::mutex> lock(swap_reason_mtx_);
  return last_swap_reason_;
}

int SearchManager::total_remaining_simulations() const {
  int remaining = 0;
  for (const auto& tree_ptr : trees_) {
    if (!tree_ptr->completed) {
      remaining += (tree_ptr->session->simulations_requested() -
                    tree_ptr->session->simulations_processed());
    }
  }
  return std::max(0, remaining);
}

std::vector<SearchResult> SearchManager::finish_all() {
  std::vector<SearchResult> results;
  results.reserve(trees_.size());

  for (auto& tree_ptr : trees_) {
    results.push_back(tree_ptr->session->finish());
  }

  return results;
}

// ══════════════════════════════════════════════════════════════════════
// 輔助函式
// ══════════════════════════════════════════════════════════════════════

int64_t SearchManager::now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch()
         ).count();
}

}  // namespace tzaar
