#include "mcts/search_manager.h"

#include <algorithm>
#include <cstring>
#include <cstdarg>
#include <stdexcept>
#include <fstream>
#include <mutex>

// Thread-safe debug log
// 目前保留這個簡單 logger，是因為 SearchManager 還在整合期，
// 多執行緒行為出問題時比起一般 print 更容易追查時序。
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

// 建構時一次建立所有搜尋樹與雙 buffer。
// 之後 worker thread 只反覆操作這些既有物件，不再重建它們。
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

  // 建立所有搜尋樹（每個 clone 一份 root state）。
  // 每棵樹完全獨立，避免不同對局互相污染狀態。
  trees_.reserve(static_cast<std::size_t>(tree_count_));
  for (int i = 0; i < tree_count_; ++i) {
    auto tree = std::make_unique<SearchTree>();
    tree->root_state = root_states[static_cast<std::size_t>(i)].clone();
    tree->session = std::make_unique<SearchSession>(tree->root_state, config_);
    tree->tree_id = i;
    trees_.push_back(std::move(tree));
  }

  // 初始化雙 buffer
  init_buffers();
}

void SearchManager::init_buffers() {
  for (int b = 0; b < 2; ++b) {
    EvalBuffer& buf = buffers_[b];
    buf.capacity = max_batch_;
    buf.board_flat.resize(static_cast<std::size_t>(max_batch_) *
                          static_cast<std::size_t>(kBoardFlatSize), 0.0f);
    buf.global_feat.resize(static_cast<std::size_t>(max_batch_) *
                           static_cast<std::size_t>(kGlobalFeatureDim), 0.0f);
    buf.legal_mask.resize(static_cast<std::size_t>(max_batch_) *
                          static_cast<std::size_t>(kActionCount), 0);
    buf.node_ids.resize(static_cast<std::size_t>(max_batch_), 0);
    buf.tree_ids.resize(static_cast<std::size_t>(max_batch_), 0);
    buf.pending_count = 0;
    buf.eval_done = true;
  }
}

// ══════════════════════════════════════════════════════════════════════
// 生命週期
// ══════════════════════════════════════════════════════════════════════

// 啟動固定數量 worker threads。
// 這裡的 thread pool 是 SearchManager 相對於 Python async_worker 的核心差異。
void SearchManager::start_workers() {
  sm_log("start_workers: %d threads", num_threads_);
  pool_.reserve(static_cast<std::size_t>(num_threads_));
  for (int i = 0; i < num_threads_; ++i) {
    pool_.emplace_back(&SearchManager::worker_loop, this, i);
  }
}

void SearchManager::wait_for_completion() {
  sm_log("wait_for_completion: start");
  for (auto& t : pool_) {
    if (t.joinable()) {
      t.join();
    }
  }
  pool_.clear();
  process_pending_results();
}

void SearchManager::run() {
  sm_log("run: start");
  start_workers();
  wait_for_completion();
}

void SearchManager::shutdown() {
  stop_ = true;
  batch_ready_cv_.notify_all();
  eval_done_cv_.notify_all();
  results_cv_.notify_all();
}

// ══════════════════════════════════════════════════════════════════════
// Worker Thread 主迴圈
// ══════════════════════════════════════════════════════════════════════

// Worker 的節奏：
// 1. 先處理 GPU 已回傳但尚未分發的結果
// 2. 再嘗試鎖定一棵樹去模擬
// 3. local buffer 滿了就 flush 到 shared buffer
// 4. 無樹可模擬時，必要就強制 swap buffer 給 GPU
void SearchManager::worker_loop(int thread_id) {
  sm_log("worker %d: started cap=%d", thread_id, config_.leaf_batch_size);
  ThreadLocalBuffer local;
  const int local_capacity = config_.leaf_batch_size;
  local.slots.resize(static_cast<std::size_t>(local_capacity));
  local.count = 0;

  int loop_count = 0;
  while (!stop_) {
    loop_count++;
    // 1. 先檢查有沒有 GPU 結果要處理
    {
      std::unique_lock<std::mutex> lock(results_mtx_);
      if (!pending_results_.empty()) {
        auto res = std::move(pending_results_.back());
        pending_results_.pop_back();
        lock.unlock();

        sm_log("worker %d: loop=%d processing pending result", thread_id, loop_count);
        process_pending_results();
        continue;
      }
    }

    // 2. 沒有 GPU 結果要處理，嘗試鎖定一棵樹來模擬
    int tree_id = try_lock_tree();
    if (tree_id >= 0) {
      sm_log("worker %d: loop=%d tree=%d simulate (local=%d)", thread_id, loop_count, tree_id, local.count);
      simulate_tree_into_local(tree_id, local);
      trees_[static_cast<std::size_t>(tree_id)]->locked = false;
      sm_log("worker %d: loop=%d tree=%d unlocked (local=%d now)", thread_id, loop_count, tree_id, local.count);

      if (local.count >= local_capacity) {
        sm_log("worker %d: loop=%d tree=%d local full, flush", thread_id, loop_count, tree_id);
        flush_local_to_shared(thread_id, local);
      }
    } else {
      if (local.count > 0) {
        sm_log("worker %d: loop=%d no tree, flushing local=%d", thread_id, loop_count, local.count);
        flush_local_to_shared(thread_id, local);
      }

      if (is_complete()) {
        sm_log("worker %d: loop=%d complete, exit", thread_id, loop_count);
        break;
      }

      // 沒有樹可鎖 → buffer 有資料就強制送給 Python（即使未滿 max_batch）
      // 但只送「還沒送 GPU 的 buffer」（eval_done==true 表示還沒送）
      if (buffers_[0].pending_count > 0 && buffers_[0].eval_done) {
        sm_log("worker %d: loop=%d no tree, forcing swap buf0 (pending=%d)", thread_id, loop_count, buffers_[0].pending_count.load());
        swap_buffer(0);
      } else if (buffers_[1].pending_count > 0 && buffers_[1].eval_done) {
        sm_log("worker %d: loop=%d no tree, forcing swap buf1 (pending=%d)", thread_id, loop_count, buffers_[1].pending_count.load());
        swap_buffer(1);
      }

      if (loop_count % 100 == 0) {
        sm_log("worker %d: loop=%d yielding (no tree, local=%d)", thread_id, loop_count, local.count);
      }
      std::this_thread::yield();
    }
  }

  if (local.count > 0) {
    sm_log("worker %d: final flush local=%d", thread_id, local.count);
    flush_local_to_shared(thread_id, local);
  }
  sm_log("worker %d: exit (total loops=%d)", thread_id, loop_count);
}

int SearchManager::try_lock_tree() {
  // 用 round-robin 方式嘗試鎖定樹，避免所有 thread 搶同一棵
  const int start = next_tree_index_.fetch_add(1) % tree_count_;

  for (int i = 0; i < tree_count_; ++i) {
    const int idx = (start + i) % tree_count_;
    SearchTree& tree = *trees_[static_cast<std::size_t>(idx)];

    if (tree.completed) continue;

    // CAS：如果 locked 是 false，設為 true
    bool expected = false;
    if (tree.locked.compare_exchange_weak(expected, true,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
      // 鎖定成功後再檢查一次是否已完成（避免 race）
      if (tree.completed) {
        tree.locked = false;
        continue;
      }
      // 檢查這棵樹是否在等待 GPU 結果（有 pending leaves）
      // 如果有，跳過它讓它先處理完
      if (tree.session->has_pending_leaves()) {
        tree.locked = false;
        continue;
      }
      // 檢查是否已完成
      if (tree.session->is_complete()) {
        tree.completed = true;
        tree.locked = false;
        continue;
      }
      sm_log("  lock_tree: claimed tree %d", idx);
      return idx;
    }
  }

  sm_log("  lock_tree: none available");
  return -1;  // 沒有可鎖定的樹
}

void SearchManager::simulate_tree_into_local(int tree_id,
                                              ThreadLocalBuffer& local) {
    SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
  auto& session = tree.session;

  const int chunk = config_.leaf_batch_size;

  // 計算 local buffer 中還有多少空位
  const int local_capacity = static_cast<int>(local.slots.size());
  const int space = local_capacity - local.count;
  if (space <= 0) return;

  const int slot_start = local.count;
  const int to_simulate = std::min(chunk, space);

  // 逐個 slot 填入
  int simulated = 0;
  for (int i = 0; i < to_simulate; ++i) {
    if (session->is_complete()) break;

    const int idx = slot_start + i;
    auto& slot = local.slots[static_cast<std::size_t>(idx)];

    // 使用 simulate_into_buffers 但一次只模擬 1 個
    int count = session->simulate_into_buffers(
        1,
        slot.board_flat,
        slot.global_feat,
        slot.legal_mask,
        &slot.node_id,
        &slot.tree_id,
        tree_id);

    if (count > 0) {
      simulated++;
    } else {
      break;  // 無法再模擬（已完成）
    }
  }

  local.count += simulated;
  sm_log("    simulate tree=%d: simulated=%d local=%d", tree_id, simulated, local.count);

  // 檢查樹是否已完成
  if (session->is_complete()) {
    tree.completed = true;
    sm_log("    simulate tree=%d: COMPLETED", tree_id);
  }
}

// 把 thread-local 收集到的 leaves 批次搬到 shared buffer。
// 這一步是 CPU worker 與 Python/GPU 邊界的真正交會點。
void SearchManager::flush_local_to_shared(int thread_id,
                                           ThreadLocalBuffer& local) {
  if (local.count <= 0) return;

  while (local.count > 0) {
    const int active = active_buffer_.load();
    EvalBuffer& buf = buffers_[active];
    sm_log("  flush %d: active_buf=%d pending=%d eval_done=%d local=%d",
           thread_id, active, buf.pending_count.load(), (int)buf.eval_done, local.count);

    // 等待 buffer 的 eval 完成（GPU 可能還在處理）
    {
      std::unique_lock<std::mutex> lock(cv_mtx_);
      sm_log("  flush %d: waiting on eval_done_cv_", thread_id);
      eval_done_cv_.wait(lock, [&buf]() {
        return buf.eval_done;
      });
      sm_log("  flush %d: eval_done_cv_ returned (eval_done=%d)", thread_id, (int)buf.eval_done);
    }

    // 寫入 shared buffer
    const int slot = buf.pending_count.load();
    const int space = buf.capacity - slot;

    if (space <= 0) {
      // buffer 滿了，需要 swap
      int old_buf = active_buffer_.load();
      swap_buffer(old_buf);
      continue;  // 重新嘗試
    }

        const int to_copy = std::min(local.count, space);

    sm_log("  flush %d: copying %d slots (pending %d->%d)", thread_id, to_copy, slot, slot + to_copy);

  // 將 local 的資料連續複製到 shared buffer
  for (int i = 0; i < to_copy; ++i) {
    const auto& src = local.slots[static_cast<std::size_t>(i)];
    const std::size_t dst_slot = static_cast<std::size_t>(slot + i);

    std::memcpy(
        buf.board_flat.data() + dst_slot * static_cast<std::size_t>(kBoardFlatSize),
        src.board_flat,
        static_cast<std::size_t>(kBoardFlatSize) * sizeof(float));

    std::memcpy(
        buf.global_feat.data() + dst_slot * static_cast<std::size_t>(kGlobalFeatureDim),
        src.global_feat,
        static_cast<std::size_t>(kGlobalFeatureDim) * sizeof(float));

    std::memcpy(
        buf.legal_mask.data() + dst_slot * static_cast<std::size_t>(kActionCount),
        src.legal_mask,
        static_cast<std::size_t>(kActionCount) * sizeof(uint8_t));

    buf.node_ids[dst_slot] = src.node_id;
    buf.tree_ids[dst_slot] = src.tree_id;
  }

  buf.pending_count = slot + to_copy;

  // 更新 local count：移除已 flush 的部分，將剩餘的移到前方
  const int remaining = local.count - to_copy;
  if (remaining > 0) {
    for (int i = 0; i < remaining; ++i) {
      local.slots[static_cast<std::size_t>(i)] =
          local.slots[static_cast<std::size_t>(to_copy + i)];
    }
  }
  local.count = remaining;

    // 檢查是否要 swap buffer
  if (buf.pending_count >= max_batch_) {
    sm_log("  flush %d: pending=%d >= max_batch=%d, swapping", thread_id, buf.pending_count.load(), max_batch_);
    swap_buffer(active_buffer_.load());
  }

    sm_log("  flush %d: done, remaining local=%d", thread_id, local.count);
    break;  // 跳出 while 迴圈（已成功 flush）
  }  // end while
}

// 交換雙 buffer：
// - old buffer 標記成待 GPU 評估
// - new buffer 變成 CPU 持續填入的目標
void SearchManager::swap_buffer(int old_buffer_id) {
  int new_buf = 1 - old_buffer_id;
  active_buffer_.store(new_buf);

  // 把舊 buffer 標記為「等待 GPU 評估」
  EvalBuffer& old_buf = buffers_[old_buffer_id];
  old_buf.eval_done = false;

  sm_log("  swap: buf%d->buf%d active (old_buf pending=%d)", old_buffer_id, new_buf, old_buf.pending_count.load());

  // 通知 Python 端
  {
    std::lock_guard<std::mutex> lock(cv_mtx_);
    batch_ready_ = true;
    sm_log("  swap: batch_ready_=true, notifying");
  }
  batch_ready_cv_.notify_one();
  sm_log("  swap: notified");
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
                                       const int32_t* node_ids,
                                       const float* priors,
                                       const float* values,
                                       int batch_size) {
  if (buffer_id < 0 || buffer_id > 1)
    throw std::invalid_argument("buffer_id must be 0 or 1");
  if (batch_size <= 0)
    throw std::invalid_argument("batch_size must be >= 1");

  EvalBuffer& buf = buffers_[buffer_id];
  sm_log("  submit: buf%d size=%d", buffer_id, batch_size);

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

  // 標記 buffer 為可用 — 在 cv_mtx_ 保護下設定並 notify
  {
    std::lock_guard<std::mutex> lock(cv_mtx_);
    buf.eval_done = true;
    buf.pending_count = 0;
    sm_log("  submit: buf%d eval_done=true, notifying eval_done_cv_", buffer_id);
  }
  eval_done_cv_.notify_all();
  sm_log("  submit: notified eval_done_cv_");
}

// ══════════════════════════════════════════════════════════════════════
// 處理 GPU 回傳的結果
// ══════════════════════════════════════════════════════════════════════

// 將 GPU 已算完的 priors/value 回填到各棵 SearchSession。
// 這一步會真正解除先前 virtual loss 造成的暫時偏移。
void SearchManager::process_pending_results() {
  // 從 pending_results_ 取出所有結果並分配給對應的樹
  std::vector<PendingEvalResult> results_to_process;

  {
    std::lock_guard<std::mutex> lock(results_mtx_);
    results_to_process.swap(pending_results_);
  }

    for (auto& res : results_to_process) {
    sm_log("  process: processing batch %d (size=%d)", res.buffer_id, res.batch_size);
    for (int i = 0; i < res.batch_size; ++i) {
      const int tree_id = res.tree_ids[static_cast<std::size_t>(i)];
      const int node_id = res.node_ids[static_cast<std::size_t>(i)];

      if (tree_id < 0 || tree_id >= tree_count_) continue;
      SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
      if (tree.completed) continue;

      const float* priors_row =
          res.priors.data() + (static_cast<std::size_t>(i) *
                               static_cast<std::size_t>(kActionCount));
      const float value = res.values[static_cast<std::size_t>(i)];

      tree.session->submit_single_eval(node_id, priors_row, value);
    }
  }
}

// ══════════════════════════════════════════════════════════════════════
// 結果取得
// ══════════════════════════════════════════════════════════════════════

bool SearchManager::is_complete() const {
  for (const auto& tree_ptr : trees_) {
    if (!tree_ptr->completed) return false;
  }
  return pending_results_.empty();
}

std::vector<SearchResult> SearchManager::finish_all() {
  // 確保所有執行緒已完成
  shutdown();

  std::vector<SearchResult> results;
  results.reserve(trees_.size());

  for (auto& tree_ptr : trees_) {
    results.push_back(tree_ptr->session->finish());
  }

  return results;
}

}  // namespace tzaar
