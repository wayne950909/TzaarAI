// search_manager.cpp
// 實作 SearchManager：多樹 MCTS 搜尋的管理器。
// 使用固定數量 worker thread + 樹 ID 佇列 + 雙 buffer 架構。
//
// 本檔案對應 multi_thread_logic.md 的設計：
//   - Worker 從佇列取出樹 ID，無樹時 wait（condition_variable）
//   - 一棵樹由 per-tree mutex 保護，一次只由一個 worker 模擬
//   - 模擬結果用 write_index/active_writers 原子變數寫入 shared buffer
//   - try_swap_buffer 用 atomic exchange lock 確保互斥
//   - result_handler 處理 GPU 結果並將未完成樹重新入隊

#include "mcts/search_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdarg>
#include <stdexcept>
#include <fstream>
#include <mutex>

// Thread-safe debug log
static std::mutex g_sm_log_mtx;
static void sm_log(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    std::lock_guard<std::mutex> lk(g_sm_log_mtx);
    std::ofstream log("C:/temp/sm_log.txt", std::ios::app);
    log << buf << std::endl;
}


namespace tzaar {

// ══════════════════════════════════════════════════════════════════════
// 建構子
// ══════════════════════════════════════════════════════════════════════

SearchManager::SearchManager(const std::vector<PhaseGameState>& root_states,
                             SearchConfig config,
                             int num_threads,
                             int max_batch)
    : config_(std::move(config)),
      num_threads_(num_threads),
      max_batch_(max_batch),
      tree_count_(static_cast<int>(root_states.size())) {

  if (tree_count_ <= 0)
    throw std::invalid_argument("root_states must not be empty");
  if (num_threads_ <= 0)
    throw std::invalid_argument("num_threads must be >= 1");
  if (max_batch_ <= 0)
    throw std::invalid_argument("max_batch must be >= 1");

  trees_.reserve(static_cast<std::size_t>(tree_count_));
  for (int i = 0; i < tree_count_; ++i) {
    auto tree = std::make_unique<SearchTree>();
    tree->root_state = root_states[static_cast<std::size_t>(i)].clone();
    tree->session = std::make_unique<SearchSession>(tree->root_state, config_);
    tree->tree_id = i;
    trees_.push_back(std::move(tree));
  }

  init_buffers();
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
  fillable_[0] = true;
  fillable_[1] = false;
}

// ══════════════════════════════════════════════════════════════════════
// 生命週期
// ══════════════════════════════════════════════════════════════════════

void SearchManager::enqueue_all_trees() {
  std::lock_guard<std::mutex> lock(queue_mtx_);
  for (int i = 0; i < tree_count_; ++i) {
    tree_queue_.push(i);
  }
  queue_cv_.notify_all();
}

void SearchManager::start_workers() {
  sm_log("start_workers: %d workers + 1 result handler", num_threads_);

  // 把所有樹 ID 放入佇列
  enqueue_all_trees();

  result_handler_stop_ = false;
  result_handler_ = std::thread(&SearchManager::result_handler_loop, this);

  pool_.reserve(static_cast<std::size_t>(num_threads_));
  for (int i = 0; i < num_threads_; ++i) {
    pool_.emplace_back(&SearchManager::worker_loop, this, i);
  }
}

void SearchManager::wait_for_completion() {
  sm_log("wait_for_completion: start");

  // 先設停止旗標，確保 threads 可以跳出 wait
  stop_ = true;
  result_handler_stop_ = true;
  {
    std::lock_guard<std::mutex> lock(queue_mtx_);
    queue_cv_.notify_all();
  }
  batch_ready_cv_.notify_all();
  results_cv_.notify_all();

  // 1. 等待所有 CPU worker threads 完成
  for (auto& t : pool_) {
    if (t.joinable()) {
      t.join();
    }
  }
  pool_.clear();
  sm_log("wait_for_completion: all workers joined");

  // 2. 通知 result handler 停止（它可能在等 results_cv_）
  {
    std::lock_guard<std::mutex> lock(results_mtx_);
    result_handler_stop_ = true;
  }
  results_cv_.notify_all();

  // 3. 等待 result handler 完成
  if (result_handler_.joinable()) {
    result_handler_.join();
  }
  sm_log("wait_for_completion: result handler joined");

  // 4. 確保任何殘留結果被處理（如果 result handler 在停止時來不及處理的遺留）
  process_pending_results();
  sm_log("wait_for_completion: done");
}

void SearchManager::run() {
  sm_log("run: start");
  start_workers();
  wait_for_completion();
}

void SearchManager::shutdown() {
  stop_ = true;
  result_handler_stop_ = true;
  batch_ready_cv_.notify_all();
  results_cv_.notify_all();
  {
    std::lock_guard<std::mutex> lock(queue_mtx_);
    queue_cv_.notify_all();
  }
}

// ══════════════════════════════════════════════════════════════════════
// Worker Thread 主迴圈
// ══════════════════════════════════════════════════════════════════════

// Worker 的節奏（依 multi_thread_logic.md）：
// 1. 從佇列取得一棵樹 ID（佇列空時 wait）
// 2. 鎖住該樹的 mutex
// 3. 模擬 leaf_batch_size 次，填入 local buffer
// 4. 解鎖樹
// 5. 用 write_index/active_writers 原子寫入 shared buffer（已不持有樹鎖）
// 6. 檢查 swap 條件，必要時 try_swap_buffer
// 7. 若樹未完成且無 pending leaves，重新入隊
void SearchManager::worker_loop(int thread_id) {
  sm_log("worker %d: started cap=%d", thread_id, config_.leaf_batch_size);
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

  int loop_count = 0;
  while (!stop_) {
    loop_count++;

    // 1. 從佇列取得一棵樹（佇列空則 wait）
    int tree_id = acquire_tree_from_queue();
    if (tree_id < 0) {
      // 收到停止訊號
      break;
    }

                // 2. 嘗試鎖住該樹的 mutex（依 md：鎖住失敗就跳到步驟 1，找下一棵樹）
    SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
    bool simulated = false;
    {
      std::unique_lock<std::mutex> lock(tree.mtx, std::try_to_lock);
            if (!lock.owns_lock()) {
        // 鎖不住：直接回步驟 1，讓其他 worker 有機會鎖這棵樹
        // 自己不重新入隊（樹本來就在佇列中或已被其他 worker 取出）
        continue;
      }

      // 鎖定後再次檢查：可能已被其他 worker 處理完成
      if (tree.completed) {
        continue;
      }
      if (tree.session->is_complete()) {
        tree.completed = true;
        continue;
      }
      // 跳過有 pending leaves 的樹（還在等 GPU 結果）
      if (tree.session->has_pending_leaves()) {
        continue;
      }

      // 3. 重設 flushed_to_buffer，開始新一輪模擬
      tree.flushed_to_buffer = false;

      sm_log("worker %d: loop=%d tree=%d simulate", thread_id, loop_count, tree_id);
      simulate_tree_into_local(tree_id, local);

      if (local.count > 0) {
        simulated = true;
      }

      // 檢查樹是否已完成（在鎖內更新 completed，解鎖後不會重新入隊）
      if (tree.session->is_complete()) {
        tree.completed = true;
      }

      // 4. 解鎖（lock 離開 scope 時自動釋放）
      //    依 md：flush 到 shared buffer 不應在持有樹鎖時進行
    }
    // ═══ mutex 已解鎖 ═══

    // 5. 解鎖後，將 local buffer flush 到 shared buffer
    //    使用 write_index/active_writers 原子變數，不用 mutex
    if (simulated) {
      sm_log("worker %d: loop=%d tree=%d flushing %d leaves",
             thread_id, loop_count, tree_id, local.count);
      flush_local_to_shared(thread_id, local, tree_id);
    }

    // 6. 若樹未完成且無 pending leaves，重新入隊
    if (!tree.completed && !tree.session->has_pending_leaves()) {
      reenqueue_tree_if_needed(tree_id);
    }

    // 7. 每次模擬後嘗試 swap_buffer
    swap_caller_.store("worker", std::memory_order_relaxed);
    try_swap_buffer();

        // 8. 若所有樹都完成，結束
    if (is_complete()) {
      sm_log("worker %d: loop=%d complete, exit", thread_id, loop_count);
      // 通知 Python get_ready_batch 醒來
      {
        std::lock_guard<std::mutex> lock(cv_mtx_);
        batch_ready_ = true;
      }
      batch_ready_cv_.notify_one();
      break;
    }

        // 9. timeout swap 檢查
        if (loop_count % 10 == 0) {
          const int timeout_ms = config_.flush_timeout_ms;
          if (timeout_ms > 0) {
            const int64_t now = now_ms();
            const int64_t last = last_swap_time_ms_.load();
            if (now - last > static_cast<int64_t>(timeout_ms)) {
              swap_caller_.store("timeout", std::memory_order_relaxed);
              try_swap_buffer();
            }
          }
        }
  }

  if (local.count > 0) {
    sm_log("worker %d: final flush local=%d", thread_id, local.count);
    flush_local_to_shared(thread_id, local, -1);
  }
  sm_log("worker %d: exit (total loops=%d)", thread_id, loop_count);
}

// 從佇列取得一棵樹 ID。
// 佇列為空時在 condition_variable 上 wait。
// 回傳 -1 表示收到停止訊號或所有樹已完成。
int SearchManager::acquire_tree_from_queue() {
    std::unique_lock<std::mutex> lock(queue_mtx_);
  sm_log("worker: waiting on queue (queue_size=%zu)", tree_queue_.size());
  queue_cv_.wait(lock, [this]() {
    return !tree_queue_.empty() || stop_ || is_complete();
  });

  if (stop_) {
    sm_log("worker: awake (stop signal)");
    return -1;
  }
  if (tree_queue_.empty()) {
    sm_log("worker: awake (all trees complete, queue empty)");
    return -1;
  }

  int tree_id = tree_queue_.front();
  tree_queue_.pop();
  sm_log("worker: awake (got tree=%d, queue_size=%zu)", tree_id, tree_queue_.size());
  return tree_id;
}

// 若樹未完成且無 pending leaves，重新放入佇列
void SearchManager::reenqueue_tree_if_needed(int tree_id) {
  SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
  if (tree.completed) return;
  if (tree.session->has_pending_leaves()) return;  // 還在等 GPU

    // 檢查是否真的還需要模擬
  if (tree.session->is_complete()) {
    tree.completed = true;
    // 通知 queue_cv_，讓 worker 有機會檢查 SearchManager::is_complete()
    queue_cv_.notify_all();
    return;
  }

  {
    std::lock_guard<std::mutex> lock(queue_mtx_);
    tree_queue_.push(tree_id);
  }
  queue_cv_.notify_one();
}

void SearchManager::simulate_tree_into_local(int tree_id,
                                              ThreadLocalBuffer& local) {
  SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
  auto& session = tree.session;

  const int chunk = config_.leaf_batch_size;
  local.count = 0;

  // 依 md：一次模擬 leaf_batch_size 次，累積 16 個 leaf
  // 直接傳 chunk=leaf_batch_size 給 simulate_into_buffers，
  // 讓它在內部一次完成 16 次 selection → expand
  // 寫入 local 的連續 buffer（board_flat/global_feat/legal_mask/node_ids/tree_ids）
  const int actual_chunk = std::min(chunk, local.capacity);
  if (actual_chunk <= 0) return;

    int count = session->simulate_into_buffers(
      actual_chunk,
      local.board_flat.data(),
      local.global_feat.data(),
      local.legal_mask.data(),
      local.node_ids.data(),
      local.tree_ids.data(),
      tree_id);

  local.count = count;
  int remaining = session->simulations_requested() - session->simulations_processed();
  sm_log("    simulate tree=%d: simulated=%d (remaining=%d)", tree_id, count, remaining);

  if (session->is_complete()) {
    tree.completed = true;
    sm_log("    simulate tree=%d: COMPLETED", tree_id);
  }
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
  while (!fillable_[active]) {
    spin_count++;
    if (spin_count > 1000) {
      // 長時間不 fillable，可能是 buffer 正被 GPU 使用，等 swap
      std::this_thread::yield();
      spin_count = 0;
    }
    active = active_buffer_.load();  // 重新檢查
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

    sm_log("  flush %d: wrote %d leaves at slot=%d, pending=%d, tree=%d",
         thread_id, to_copy, slot, buf.pending_count.load(std::memory_order_relaxed), tree_id);

  // 清空 local
  local.count = 0;

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

  // 4. Swap fillable flags
  fillable_[active] = false;
  fillable_[other] = true;

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

  sm_log("  try_swap: buf%d->buf%d (pending=%d)", active, other, pending);

  // 7. 將舊 buffer 送 GPU
  active_buf.eval_done = false;

  {
    std::lock_guard<std::mutex> lock(cv_mtx_);
    batch_ready_ = true;
  }
  batch_ready_cv_.notify_one();

  // 8. 釋放交換鎖
  swapping_.store(false, std::memory_order_release);

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
  sm_log("  get_ready_batch: waiting on batch_ready_cv_");
  {
    std::unique_lock<std::mutex> lock(cv_mtx_);
    batch_ready_cv_.wait(lock, [this]() {
      if (batch_ready_) {
        sm_log("  get_ready_batch: batch_ready_=true, wake up");
        return true;
      }
      if (is_complete()) {
        sm_log("  get_ready_batch: is_complete, wake up");
        return true;
      }
      return false;
    });
    batch_ready_ = false;
  }

  PackedBatch result;

  // 找出哪個 buffer 需要評估
  for (int b = 0; b < 2; ++b) {
    EvalBuffer& buf = buffers_[b];
    if (!buf.eval_done && buf.pending_count > 0) {
      const int count = buf.pending_count.load();
      sm_log("  get_ready_batch: returning buf%d size=%d", b, count);
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
  sm_log("  get_ready_batch: no ready buffer found");
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
                                        
  // DEBUG: 記錄進來的資料樣本
  {
    // 取前 min(3, batch_size) 筆來看 value 跟 tree_id/node_id
    int samples = batch_size < 3 ? batch_size : 3;
    std::string sample_str;
    for (int i = 0; i < samples; ++i) {
      int t = buf.tree_ids[i];
      int n = buf.node_ids[i];
      float v = values[i];
      float p0 = priors[i * kActionCount];
      char tmp[128];
      std::snprintf(tmp, sizeof(tmp), "[%d]tree=%d node=%d val=%.4f p0=%.4f", i, t, n, v, p0);
      if (i > 0) sample_str += " | ";
      sample_str += tmp;
    }
    sm_log("  submit_EVAL_BATCH: buf%d size=%d %s", buffer_id, batch_size, sample_str.c_str());
  }

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
    sm_log("  submit_EVAL_BATCH: pending_results_ size now=%zu", pending_results_.size());
  }
  results_cv_.notify_all();

    // 標記 buffer 為可用
  buf.eval_done = true;
  buf.write_index = 0;
  buf.pending_count = 0;
  buf.active_writers = 0;
  sm_log("  submit_EVAL_BATCH: buf%d eval_done=true", buffer_id);

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
  sm_log("result_handler: started");

  while (!result_handler_stop_) {
    std::unique_lock<std::mutex> lock(results_mtx_);

    results_cv_.wait(lock, [this]() {
      return !pending_results_.empty() || result_handler_stop_;
    });

    sm_log("result_handler: woke up (stop=%d pending=%zu)",
           (int)result_handler_stop_, pending_results_.size());

    if (result_handler_stop_ && pending_results_.empty()) {
      sm_log("result_handler: stop signal, no pending data, exit");
      break;
    }

    auto results_to_process = std::move(pending_results_);
    pending_results_.clear();
    lock.unlock();

    sm_log("result_handler: processing %zu result batches", results_to_process.size());

        for (auto& res : results_to_process) {
      sm_log("result_handler: processing batch buf%d (size=%d)",
             res.buffer_id, res.batch_size);

      // 依 md：先鎖住資料對應的樹，在鎖內復原 virtual loss + expand + backup
      // 然後檢查是否需要重新入隊
      for (int i = 0; i < res.batch_size; ++i) {
        const int tree_id = res.tree_ids[static_cast<std::size_t>(i)];
        const int node_id = res.node_ids[static_cast<std::size_t>(i)];

        if (tree_id < 0 || tree_id >= tree_count_) {
          sm_log("result_handler:   SKIP tree_id=%d out of range", tree_id);
          continue;
        }
        SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
        if (tree.completed) {
          sm_log("result_handler:   SKIP tree=%d already completed", tree_id);
          continue;
        }

        const float* priors_row =
            res.priors.data() + (static_cast<std::size_t>(i) *
                                 static_cast<std::size_t>(kActionCount));
        const float value = res.values[static_cast<std::size_t>(i)];

                // 鎖住樹，在鎖內復原 virtual loss + expand + backup
        // 依 md：result_handler 先鎖住資料對應的樹，將 virtual loss 復原，
        // 然後將推論完的先驗機率跟 value 加到樹裡
                {
                    std::lock_guard<std::mutex> tree_lock(tree.mtx);

          // 將 NN 評估結果寫入樹（復原 virtual loss + expand + backup）
          tree.session->submit_single_eval(node_id, priors_row, value);

          // 若此樹的模擬次數尚未達到目標次數，則將樹的 id 放到佇列
          if (!tree.completed && !tree.session->has_pending_leaves()) {
            sm_log("result_handler:   -> reenqueue tree=%d", tree_id);
            reenqueue_tree_if_needed(tree_id);
          }
        }
        // 解鎖 — 依 md
      }
    }

        // 記憶體屏障：確保寫入對 worker 可見
        std::atomic_thread_fence(std::memory_order_release);

        // 嘗試 swap_buffer
        swap_caller_.store("result_handler", std::memory_order_relaxed);
        try_swap_buffer();

        // 若所有樹都完成，通知 batch_ready_cv_ 讓 get_ready_batch 跳出
        if (is_complete()) {
          sm_log("result_handler: all trees complete, notify get_ready_batch");
          {
            std::lock_guard<std::mutex> lock(cv_mtx_);
            batch_ready_ = true;
          }
          batch_ready_cv_.notify_one();
        }
  }

  sm_log("result_handler: EXIT");
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
    sm_log("  process_pending_results: took %zu batches from pending_results_", results_to_process.size());
  }

  if (results_to_process.empty()) {
    sm_log("  process_pending_results: nothing to process");
    return;
  }

    for (auto& res : results_to_process) {
    sm_log("  process_pending_results: processing batch buf%d (size=%d)", res.buffer_id, res.batch_size);
    for (int i = 0; i < res.batch_size; ++i) {
      const int tree_id = res.tree_ids[static_cast<std::size_t>(i)];
      const int node_id = res.node_ids[static_cast<std::size_t>(i)];

      if (tree_id < 0 || tree_id >= tree_count_) {
        sm_log("  process_pending_results:   SKIP tree_id=%d OOB", tree_id);
        continue;
      }
      SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
      if (tree.completed) {
        sm_log("  process_pending_results:   SKIP tree=%d completed", tree_id);
        continue;
      }

      const float* priors_row =
          res.priors.data() + (static_cast<std::size_t>(i) *
                               static_cast<std::size_t>(kActionCount));
      const float value = res.values[static_cast<std::size_t>(i)];

      sm_log("  process_pending_results: submit_single_eval(tree=%d, node=%d, val=%.4f)",
             tree_id, node_id, value);

      // 在鎖內復原 virtual loss + expand + backup（與 result_handler 一致）
      {
        std::lock_guard<std::mutex> lock(tree.mtx);
        tree.session->submit_single_eval(node_id, priors_row, value);
      }
    }
  }

  // 記憶體屏障：確保寫入對後續 worker 可見
  std::atomic_thread_fence(std::memory_order_release);
}

// ══════════════════════════════════════════════════════════════════════
// 結果取得
// ══════════════════════════════════════════════════════════════════════

bool SearchManager::is_complete() const {
  for (const auto& tree_ptr : trees_) {
    if (!tree_ptr->completed) return false;
  }
  // pending_results_ 可能被 result_handler 或 submit_eval_batch 修改，
  // 但 is_complete() 本身就是一個「快速檢查」，不保證絕對精確，
  // 因為就算現在 empty，下一秒可能又有新結果進來。
  // 這裡只確保 tree 都 completed，pending_results_ 大致清空即可。
  std::lock_guard<std::mutex> lock(results_mtx_);
  return pending_results_.empty();
}

int SearchManager::completed_tree_count() const {
  int count = 0;
  for (const auto& tree_ptr : trees_) {
    if (tree_ptr->completed) ++count;
  }
  return count;
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
  sm_log("finish_all: start");

  // 先設定停止旗標，讓 worker threads 和 result handler 跳出等待
  stop_ = true;
  result_handler_stop_ = true;
  {
    std::lock_guard<std::mutex> lock(queue_mtx_);
    queue_cv_.notify_all();
  }
  batch_ready_cv_.notify_all();
  results_cv_.notify_all();

  // 確保所有 worker threads 完成
  for (auto& t : pool_) {
    if (t.joinable()) {
      t.join();
    }
  }
  pool_.clear();

  // 等待 result handler 完成
  if (result_handler_.joinable()) {
    result_handler_.join();
  }

  // 處理殘留結果（如果有）
  process_pending_results();

  // 收集所有樹的搜尋結果
  std::vector<SearchResult> results;
  results.reserve(trees_.size());

  for (auto& tree_ptr : trees_) {
    results.push_back(tree_ptr->session->finish());
  }

  sm_log("finish_all: done, %zu results", results.size());
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
