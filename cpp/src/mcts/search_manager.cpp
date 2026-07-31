// search_manager.cpp
// 實作 SearchManager：多樹 MCTS 搜尋的管理器。
// 使用固定數量 worker thread + 樹 ID 佇列 + 雙 buffer 架構。
//
// 本檔案對應 multi_thread_logic.md 的設計：
//   - Worker 從 ConcurrentQueue 取出樹 ID，無樹時 wait
//   - 一棵樹由 ConcurrentQueue 確保唯一持有，無需 per-tree mutex
//   - 模擬結果用 write_index/active_writers 原子變數寫入 shared buffer
//   - try_swap_buffer 用 atomic exchange lock 確保互斥
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
// ════════════════ㄐㄧ══════════════════════════════════════════════════════

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

        // 建立常駐 threads（佇列空的，workers 直接在 tree_queue_.wait_and_pop 中 wait）
  stop_ = false;
  result_handler_stop_ = false;

  result_handler_ = std::thread(&SearchManager::result_handler_loop, this);
  pool_.reserve(static_cast<std::size_t>(num_threads_));
  for (int i = 0; i < num_threads_; ++i) {
    pool_.emplace_back(&SearchManager::worker_loop, this, i);
  }
}

void SearchManager::init_buffers() {
  // buffer capacity = 樹的總數 * leaf_batch_size（符合 multi_thread_logic.md）
  const int buf_capacity = tree_count_ * config_.leaf_batch_size;
  for (int b = 0; b < 2; ++b) {
    EvalBuffer& buf = buffers_[b];
    buf.capacity = buf_capacity;
    buf.board_flat.resize(static_cast<std::size_t>(buf_capacity) *
                          static_cast<std::size_t>(kBoardFlatSize), 0.0f);
    buf.global_feat.resize(static_cast<std::size_t>(buf_capacity) *
                           static_cast<std::size_t>(kGlobalFeatureDim), 0.0f);
    buf.legal_mask.resize(static_cast<std::size_t>(buf_capacity) *
                          static_cast<std::size_t>(kActionCount), 0);
    buf.node_ids.resize(static_cast<std::size_t>(buf_capacity), 0);
    buf.tree_ids.resize(static_cast<std::size_t>(buf_capacity), 0);
    buf.write_index = 0;
    buf.active_writers = 0;
    buf.pending_count = 0;
    buf.eval_done = true;
  }
    fillable_[0].store(true, std::memory_order_relaxed);
  fillable_[1].store(false, std::memory_order_relaxed);
}

// ══════════════════════════════════════════════════════════════════════
// 生命週期
// ══════════════════════════════════════════════════════════════════════

void SearchManager::enqueue_all_trees() {
  for (int i = 0; i < tree_count_; ++i) {
    tree_queue_.push(i);
  }
}

void SearchManager::reset(const std::vector<PhaseGameState>& root_states,
                           SearchConfig config) {
  // 不用暫停 workers，直接重置
  // workers 此時可能在 acquire_tree_from_queue 中 wait（佇列空的）
  // 或正在處理上一次的樹（但我們即將清空佇列並放新樹）
  config_ = std::move(config);
  tree_count_ = static_cast<int>(root_states.size());

  if (tree_count_ <= 0)
    throw std::invalid_argument("root_states must not be empty");

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

  // 重置 buffers
  init_buffers();

  // 重置所有計數器和狀態
  completed_count_ = 0;
  stop_ = false;
  result_handler_stop_ = false;
  active_buffer_ = 0;
  swapping_ = false;
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

// Worker 的節奏（使用 ConcurrentQueue 取出即獨佔，無需 per-tree mutex）：
// 1. 從 ConcurrentQueue 取得一棵樹 ID（佇列空時 wait）
// 2. 檢查樹狀態（pending leaves / completed），必要時放回或跳過
// 3. 多次呼叫 simulate_into_buffers，直到 local buffer 滿 leaf_batch_size、
//    樹已完成、或樹 stalled（有 pending leaves 在等 GPU）
// 4. 將 local buffer flush 到 shared buffer（用 write_index/active_writers 原子變數）
// 5. 嘗試 swap_buffer
// 6. 找下一棵樹（重新入隊由 result_handler 負責）
//
// 注意：重新入隊的責任歸 result_handler（依 multi_thread_logic.md），
//       worker 只管模擬→flush→找下一棵樹。
void SearchManager::worker_loop(int thread_id) {
  ThreadLocalBuffer local;
  const int local_cap = config_.leaf_batch_size;
  local.capacity = local_cap;
  local.count = 0;
  local.board_flat.resize(static_cast<std::size_t>(local_cap) *
                          static_cast<std::size_t>(kBoardFlatSize), 0.0f);
  local.global_feat.resize(static_cast<std::size_t>(local_cap) *
                           static_cast<std::size_t>(kGlobalFeatureDim), 0.0f);
  local.legal_mask.resize(static_cast<std::size_t>(local_cap) *
                          static_cast<std::size_t>(kActionCount), 0);
  local.node_ids.resize(static_cast<std::size_t>(local_cap), 0);
  local.tree_ids.resize(static_cast<std::size_t>(local_cap), 0);

  while (!stop_) {
    // 1. 從 ConcurrentQueue 取得一棵樹（佇列空則 wait）
    int tree_id;
    if (!tree_queue_.wait_and_pop(tree_id)) {
      // 收到停止訊號
      break;
    }

    // ConcurrentQueue 取出即獨佔：此 tree_id 只有本 worker 持有
    SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
    bool simulated = false;

    // 跳過有 pending leaves 的樹（還在等 GPU 結果）
    if (tree.session->has_pending_leaves()) {
      // 放回佇列讓其他 worker 有機會
      tree_queue_.push(tree_id);
      continue;
    }

    // 檢查樹是否已經完成
    if (tree.session->is_complete()) {
      complete_tree(tree);
      if (is_complete()) break;
      continue;
    }

    // 2. 重設 flushed_to_buffer，開始新一輪模擬
    tree.flushed_to_buffer = false;

    simulate_tree_into_local(tree_id, local);

    if (local.count > 0) {
      simulated = true;
    }

    // 檢查樹是否已完成
    if (tree.session->is_complete()) {
      complete_tree(tree);
      if (is_complete()) {
        break;
      }
    }

    // ═══ 不需解鎖（無 per-tree mutex）═══

    // 3. 將 local buffer flush 到 shared buffer
    //    使用 write_index/active_writers 原子變數，不用 mutex
    if (simulated) {
      flush_local_to_shared(thread_id, local, tree_id);
    }

    // 4. 嘗試 swap_buffer（重新入隊由 result_handler 負責）
    swap_caller_.store("worker", std::memory_order_relaxed);
    try_swap_buffer();
  }

    if (local.count > 0) {
    flush_local_to_shared(thread_id, local, -1);
  }

  nvtxRangePop();  // worker_N
}

// 若樹未完成且無 pending leaves，重新放入佇列
// 注意：此函式僅由 result_handler 呼叫（依 multi_thread_logic.md），
//       worker 不負責重新入隊（問題 2 修復）。
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

void SearchManager::simulate_tree_into_local(int tree_id,
                                              ThreadLocalBuffer& local) {
  SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
  auto& session = tree.session;

  local.count = 0;

  // 依 multi_thread_logic.md：
  // 「除非模擬次數消耗完，否則會持續蒐集到 leaf_batch_size 個葉節點」
  // 在同一棵樹上持續呼叫 simulate_into_buffers，直到：
  //   - local buffer 滿了（count >= leaf_batch_size）
  //   - 樹的模擬次數已耗盡（is_complete）
  //   - 樹 stalled（simulate_into_buffers 回傳 0 且有 pending leaves）
  const int local_cap = local.capacity;
  while (local.count < local_cap) {
    // 檢查是否已完成或正在等 GPU
    if (session->is_complete()) break;
    if (session->has_pending_leaves()) break;

    const int remaining_cap = local_cap - local.count;
    const int chunk = std::min(config_.leaf_batch_size, remaining_cap);
    if (chunk <= 0) break;

    // 計算要寫入 local buffer 的位置偏移
    float* board_ptr = local.board_flat.data() +
        static_cast<std::size_t>(local.count) * static_cast<std::size_t>(kBoardFlatSize);
    float* global_ptr = local.global_feat.data() +
        static_cast<std::size_t>(local.count) * static_cast<std::size_t>(kGlobalFeatureDim);
    uint8_t* mask_ptr = local.legal_mask.data() +
        static_cast<std::size_t>(local.count) * static_cast<std::size_t>(kActionCount);
    int32_t* node_ptr = local.node_ids.data() + static_cast<std::size_t>(local.count);
    int32_t* tree_ptr = local.tree_ids.data() + static_cast<std::size_t>(local.count);

        int n = session->simulate_into_buffers(
        chunk,
        board_ptr, global_ptr, mask_ptr,
        node_ptr, tree_ptr,
        tree_id);

        if (n > 0) {
      local.count += n;
    }

    // simulate_into_buffers 回傳 0 表示 stalled（所有路徑都被 pending leaf 阻塞）
    // 或樹已 complete。跳出 while 迴圈，不要 busy loop。
        if (n == 0) {
      break;
    }
  }
  nvtxRangePop();  // simulate_tree_into_local
}

// 將 local buffer 的內容 flush 到 shared buffer。
// 使用 write_index / active_writers 原子變數，不用 mutex。
//
// 流程（依 multi_thread_logic.md）：
// 1. 等待 buffer 處於 fillable 狀態
// 2. Increment active_writers (memory_order_relaxed)
// 3. fetch_add write_index 取得寫入位置 (memory_order_relaxed)
// 4. 寫入資料
// 5. 更新 pending_count (memory_order_relaxed)
// 6. Decrement active_writers (memory_order_release)
// 7. 若 local 清空，設 tree.flushed_to_buffer = true
void SearchManager::flush_local_to_shared(int thread_id,
                                           ThreadLocalBuffer& local,
                                           int tree_id) {
  if (local.count <= 0) return;

  const int to_copy = local.count;
  int active = active_buffer_.load();

    // 1. 等待 buffer fillable（使用 spin + yield 而非 mutex cv）
  //    因為我們希望寫入是 lock-free 的
  int spin_count = 0;
  while (!fillable_[active].load(std::memory_order_acquire)) {
    spin_count++;
    if (spin_count > 1000) {
      // 長時間不 fillable，可能是 buffer 正被 GPU 使用，等 swap
      std::this_thread::yield();
      spin_count = 0;
    }
    active = active_buffer_.load(std::memory_order_relaxed);  // 重新檢查
  }

  EvalBuffer& buf = buffers_[active];

  // 2. Increment active_writers (relaxed)
  buf.active_writers.fetch_add(1, std::memory_order_relaxed);

  // 3. 取得寫入位置 (relaxed)
  const int slot = buf.write_index.fetch_add(to_copy, std::memory_order_relaxed);

  // 檢查是否超過 capacity（需要 swap）
  if (slot + to_copy > buf.capacity) {
    // 空間不足，回退 write_index
    buf.write_index.fetch_sub(to_copy, std::memory_order_relaxed);
    buf.active_writers.fetch_sub(1, std::memory_order_release);

    // 嘗試 swap
    try_swap_buffer();
    // 重試 flush（遞迴呼叫）
    flush_local_to_shared(thread_id, local, tree_id);
    return;
  }

    // 4. 寫入資料（從連續 local buffer 直接整段 memcpy）
  {
    const std::size_t dst_offset = static_cast<std::size_t>(slot);

    std::memcpy(
        buf.board_flat.data() + dst_offset * static_cast<std::size_t>(kBoardFlatSize),
        local.board_flat.data(),
        static_cast<std::size_t>(to_copy) *
            static_cast<std::size_t>(kBoardFlatSize) * sizeof(float));

    std::memcpy(
        buf.global_feat.data() + dst_offset * static_cast<std::size_t>(kGlobalFeatureDim),
        local.global_feat.data(),
        static_cast<std::size_t>(to_copy) *
            static_cast<std::size_t>(kGlobalFeatureDim) * sizeof(float));

    std::memcpy(
        buf.legal_mask.data() + dst_offset * static_cast<std::size_t>(kActionCount),
        local.legal_mask.data(),
        static_cast<std::size_t>(to_copy) *
            static_cast<std::size_t>(kActionCount) * sizeof(uint8_t));

    std::memcpy(
        buf.node_ids.data() + dst_offset,
        local.node_ids.data(),
        static_cast<std::size_t>(to_copy) * sizeof(int32_t));

    std::memcpy(
        buf.tree_ids.data() + dst_offset,
        local.tree_ids.data(),
        static_cast<std::size_t>(to_copy) * sizeof(int32_t));
  }

    // 5. 更新 pending_count (relaxed)
  buf.pending_count.fetch_add(to_copy, std::memory_order_relaxed);

  // 6. Decrement active_writers (release) — 確保寫入對 swapper 可見
  buf.active_writers.fetch_sub(1, std::memory_order_release);

  // 7. 標記樹已 flush
  if (tree_id >= 0) {
    trees_[static_cast<std::size_t>(tree_id)]->flushed_to_buffer = true;
  }

    // 清空 local
  local.count = 0;

  nvtxRangePop();  // flush_to_shared

  // 檢查 swap 條件：
  // - pending_count >= swap_threshold 且
  // - 另一個 buffer 已完成 GPU 推論（eval_done = true）
  const int swap_threshold = std::min(
      static_cast<int>(config_.min_batch_for_swap > 0 ? config_.min_batch_for_swap : max_batch_),
      max_batch_);
    if (buf.pending_count.load(std::memory_order_relaxed) >= swap_threshold) {
    swap_caller_.store("flush", std::memory_order_relaxed);
    try_swap_buffer();
  }
}

// 嘗試交換雙 buffer（原子交換鎖保護）。
// 流程（依 multi_thread_logic.md）：
// 1. atomic exchange 鎖定 swapping_ (acquire)
// 2. 檢查 precondition：另一個 buffer 已完成 GPU 推論
// 3. 檢查條件（至少一項滿足）：
//    A. pending_count >= swap_threshold
//    B. 所有樹都已 stalled
// 4. Swap fillable flag：舊 buffer=不可填入，新 buffer=可填入
// 5. 等待舊 buffer 的 active_writers == 0 (acquire)
// 6. 切換 active_buffer
// 7. 將舊 buffer 送 GPU（設 eval_done=false，通知 Python）
// 8. 釋放 swapping_


bool SearchManager::try_swap_buffer() {
  nvtxRangePushA("swap_buffer");
  // 1. 嘗試取得交換鎖
  bool expected = false;
  if (!swapping_.compare_exchange_strong(expected, true,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed)) {
    return false;  // 另一個 thread 正在 swap
  }

  const int active = active_buffer_.load();
  const int other = 1 - active;

  EvalBuffer& active_buf = buffers_[active];
  EvalBuffer& other_buf = buffers_[other];

    // 2. Precondition：另一個 buffer 必須已完成 GPU 推論
    if (!other_buf.eval_done) {
      swapping_.store(false, std::memory_order_release);
      return false;
    }

    // 3. 檢查條件
    const int pending = active_buf.pending_count.load(std::memory_order_acquire);
    const int swap_threshold = std::min(
        static_cast<int>(config_.min_batch_for_swap > 0 ? config_.min_batch_for_swap : max_batch_),
        max_batch_);
    bool condition_a = (pending >= swap_threshold);
    bool condition_b = trees_all_stalled() && pending > 0;

    if (!condition_a && !condition_b) {
      swapping_.store(false, std::memory_order_release);
      return false;
    }

    // 4. Swap fillable flags（使用 atomic store，問題 9 修復）
  fillable_[active].store(false, std::memory_order_release);
  fillable_[other].store(true, std::memory_order_release);

  // 5. 等待 active buffer 上所有 writer 完成
  while (active_buf.active_writers.load(std::memory_order_acquire) > 0) {
    std::this_thread::yield();
  }

  // 6. 切換 active_buffer
  active_buffer_.store(other);
  last_swap_time_ms_.store(now_ms());

    // 記錄 swap 原因（包含呼叫來源）
    {
      const char* caller = swap_caller_.load(std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(swap_reason_mtx_);
      std::string cond;
      if (condition_a && condition_b) {
        cond = "threshold+stalled";
      } else if (condition_a) {
        cond = "threshold";
      } else {
        cond = "stalled";
      }
      last_swap_reason_ = std::string(caller) + ":" + cond;
    }

  // 7. 將舊 buffer 送 GPU
  active_buf.eval_done = false;

    {
    std::lock_guard<std::mutex> lock(cv_mtx_);
    batch_ready_ = true;
  }
  batch_ready_cv_.notify_one();

  // 8. 釋放交換鎖
  swapping_.store(false, std::memory_order_release);

  nvtxRangePop();  // swap_buffer
  return true;
}

// ══════════════════════════════════════════════════════════════════════
// Python 端專用介面
// ══════════════════════════════════════════════════════════════════════

bool SearchManager::has_ready_batch() const {
  // 檢查是否有 buffer 處於「待評估」狀態
  for (int b = 0; b < 2; ++b) {
    if (!buffers_[b].eval_done && buffers_[b].pending_count > 0) {
      return true;
    }
  }
  return false;
}

SearchManager::PackedBatch SearchManager::get_ready_batch() {
  {
    std::unique_lock<std::mutex> lock(cv_mtx_);
        batch_ready_cv_.wait(lock, [this]() {
      return batch_ready_;
    });
    batch_ready_ = false;
  }

  PackedBatch result;

  // 找出哪個 buffer 需要評估
  for (int b = 0; b < 2; ++b) {
    EvalBuffer& buf = buffers_[b];
    if (!buf.eval_done && buf.pending_count > 0) {
            const int count = buf.pending_count.load();
      result.buffer_id = b;
      result.batch_size = count;
      result.node_ids = buf.node_ids.data();
      result.board_state_flat = buf.board_flat.data();
      result.global_features = buf.global_feat.data();
      result.legal_masks = buf.legal_mask.data();
      result.tree_ids = buf.tree_ids.data();
      return result;
    }
  }

    // 沒有 ready batch
  result.buffer_id = -1;
  result.batch_size = 0;
  return result;
}

void SearchManager::submit_eval_batch(int buffer_id,
                                       const int32_t* /*node_ids*/,
                                       const float* priors,
                                       const float* values,
                                       int batch_size) {
  
  if (buffer_id < 0 || buffer_id > 1)
    throw std::invalid_argument("buffer_id must be 0 or 1");
  if (batch_size <= 0)
    throw std::invalid_argument("batch_size must be >= 1");

    EvalBuffer& buf = buffers_[buffer_id];

  // 將 GPU 結果放入 pending_results_
  {
    std::lock_guard<std::mutex> lock(results_mtx_);

    PendingEvalResult res;
    res.buffer_id = buffer_id;
    res.batch_size = batch_size;
    res.node_ids.assign(buf.node_ids.begin(),
                        buf.node_ids.begin() + static_cast<std::size_t>(batch_size));
    res.tree_ids.assign(buf.tree_ids.begin(),
                        buf.tree_ids.begin() + static_cast<std::size_t>(batch_size));
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

    // 標記 buffer 為可用
    buf.eval_done = true;
  buf.write_index = 0;
  buf.pending_count = 0;
  buf.active_writers = 0;

    // 嘗試 swap — 另一個 buffer 可能已有資料
    swap_caller_.store("submit_eval", std::memory_order_relaxed);
    try_swap_buffer();
}

// ══════════════════════════════════════════════════════════════════════
// 專用 GPU 結果處理執行緒
// ══════════════════════════════════════════════════════════════════════

// 專用的 GPU 結果處理執行緒主迴圈。
// 職責：
// 1. 等待 GPU 回傳結果（由 Python 端透過 submit_eval_batch 放入）
// 2. 將 priors/value 回填到對應的樹
// 3. 若樹尚未完成且無 pending leaves，重新入隊
// 4. 嘗試 swap_buffer（另一個 buffer 可能已有資料）
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
      // 依 md：先鎖住資料對應的樹，在鎖內復原 virtual loss + expand + backup
      // 然後檢查是否需要重新入隊
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
        // 注意：不需 per-tree lock，因為 ConcurrentQueue 確保 worker
        // 和 result_handler 不會同時操作同一棵樹的衝突資料。
        // Worker 操作的是尚未送出的 pending leaves，
        // result_handler 操作的是已送回來的 GPU 結果，兩者不重疊。
        tree.session->submit_single_eval(node_id, priors_row, value);

        // 若此樹尚有剩餘模擬次數且無 pending leaves，則重新入隊
        // 注意：不用 tree.completed 判斷，因為 worker 模擬完若還有 pending leaves
        // 就不會設 completed=true，要等 GPU 結果回來 process_pending_evals 後才知道
        if (!tree.session->is_complete() && !tree.session->has_pending_leaves()) {
          reenqueue_tree_if_needed(tree_id);
        } else if (tree.session->is_complete()) {
          complete_tree(tree);
        }
      }
  }

    // 記憶體屏障：確保寫入對 worker 可見
    std::atomic_thread_fence(std::memory_order_release);

    nvtxRangePop();  // result_handler_process_batch

        // 嘗試 swap_buffer
    swap_caller_.store("result_handler", std::memory_order_relaxed);
    try_swap_buffer();
  }
  nvtxRangePop();  // result_handler_loop
}

// ══════════════════════════════════════════════════════════════════════
// 處理 GPU 回傳的結果（保留給 wait_for_completion 的最終清理用）
// ══════════════════════════════════════════════════════════════════════

// 將 GPU 已算完的 priors/value 回填到各棵 SearchSession。
// 這一步會真正解除先前 virtual loss 造成的暫時偏移。
//
// 注意：正常運行時由 result_handler_loop() 負責呼叫 submit_single_eval。
// 這個函式只作為 wait_for_completion 中的「最終清理」備用路徑，
// 處理 result handler 停止後來不及處理的殘留結果。
void SearchManager::process_pending_results() {
  // 從 pending_results_ 取出所有結果並分配給對應的樹
  std::vector<PendingEvalResult> results_to_process;

  {
    std::lock_guard<std::mutex> lock(results_mtx_);
        results_to_process.swap(pending_results_);
  }

  if (results_to_process.empty()) {
    return;
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

      // 不需 per-tree lock（理由同 result_handler_loop）
      tree.session->submit_single_eval(node_id, priors_row, value);
    }
  }

  // 記憶體屏障：確保寫入對後續 worker 可見
  std::atomic_thread_fence(std::memory_order_release);
}

// ══════════════════════════════════════════════════════════════════════
// 結果取得
// ══════════════════════════════════════════════════════════════════════

bool SearchManager::is_complete() const {
  // 使用原子 completed_count_ 快速檢查（問題 3 修復）
  // 只需要檢查計數器是否等於總樹數，不需遍歷所有樹
  if (completed_count_.load(std::memory_order_acquire) < tree_count_) {
    return false;
  }
  // pending_results_ 可能被 result_handler 或 submit_eval_batch 修改，
  // 但 is_complete() 本身就是一個「快速檢查」，不保證絕對精確，
  // 因為就算現在 empty，下一秒可能又有新結果進來。
  // 這裡只確保 tree 都 completed，pending_results_ 大致清空即可。
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
  // 收集所有樹的搜尋結果
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

// 檢查是否「沒有新的樹可以被模擬了」。
// 依 multi_thread_logic.md 的寬鬆定義：
// 「所有的樹都已經被模擬過或是正在被模擬或是模擬次數結束」
// 當所有未 completed 的樹都有 pending_leaves（已在等 GPU）時回傳 true。
// 不需要檢查 flushed_to_buffer — 只要樹已經送出 leaf 在等 GPU，
// 就代表沒有進度可以再做，應觸發 swap 讓 GPU 消化這些 pending leaves。
bool SearchManager::trees_all_stalled() const {
  for (const auto& tree_ptr : trees_) {
    if (tree_ptr->completed) continue;
    // 只要有一棵未 completed 的樹沒有 pending leaves（還能繼續模擬），就非 stalled
    if (!tree_ptr->session->has_pending_leaves()) return false;
  }
  return true;  // 所有未 completed 的樹都已送出 leaf 在等 GPU
}



}  // namespace tzaar
