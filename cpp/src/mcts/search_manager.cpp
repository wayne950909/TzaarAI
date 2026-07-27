// search_manager.cpp (v3 — 新架構)
// 實作 SearchManager：多樹 MCTS 搜尋的管理器。
//
// 對應 multi_thread_logic.md 與 migration_plan_v3.md 的新規劃：
//   - 固定分配樹給每個 CPU worker，內部活躍清單管理
//   - 16 次模擬為一個單位，指標傳遞進 ConcurrentQueue
//   - 1:1 配對的 Result Handler
//   - 防空轉 condition_variable 睡眠/喚醒機制

#include "mcts/search_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <mutex>

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

  stop_ = false;
  result_handler_stop_ = false;

  // 初始化每個 worker 的同步原語
  worker_cv_mtx_.reserve(static_cast<std::size_t>(num_threads_));
  worker_cv_.reserve(static_cast<std::size_t>(num_threads_));
  for (int i = 0; i < num_threads_; ++i) {
    worker_cv_mtx_.push_back(std::make_unique<std::mutex>());
    worker_cv_.push_back(std::make_unique<std::condition_variable>());
  }
  assigned_trees_.resize(static_cast<std::size_t>(num_threads_));

  // 初始化每個 Result Handler 的專屬 pending queue
  per_worker_pending_mtx_.reserve(static_cast<std::size_t>(num_threads_));
  per_worker_pending_results_.resize(static_cast<std::size_t>(num_threads_));
  per_worker_pending_cv_.reserve(static_cast<std::size_t>(num_threads_));
  for (int i = 0; i < num_threads_; ++i) {
    per_worker_pending_mtx_.push_back(std::make_unique<std::mutex>());
    per_worker_pending_cv_.push_back(std::make_unique<std::condition_variable>());
  }

  // 建立常駐 CPU workers
  pool_.reserve(static_cast<std::size_t>(num_threads_));
  for (int i = 0; i < num_threads_; ++i) {
    pool_.emplace_back(&SearchManager::worker_loop, this, i);
  }

  // 建立 1:1 配對的 Result Handlers
  result_handlers_.reserve(static_cast<std::size_t>(num_threads_));
  for (int i = 0; i < num_threads_; ++i) {
    result_handlers_.emplace_back(&SearchManager::result_handler_loop, this, i);
  }
}

// ══════════════════════════════════════════════════════════════════════
// 生命週期
// ══════════════════════════════════════════════════════════════════════

void SearchManager::reset(const std::vector<PhaseGameState>& root_states,
                           SearchConfig config) {
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

  // === 固定分配樹給每個 Worker ===
  int trees_per_worker = tree_count_ / num_threads_;
  int remainder = tree_count_ % num_threads_;
  int idx = 0;
  for (int w = 0; w < num_threads_; ++w) {
    assigned_trees_[static_cast<std::size_t>(w)].clear();
    int count = trees_per_worker + (w < remainder ? 1 : 0);
    for (int j = 0; j < count; ++j) {
      assigned_trees_[static_cast<std::size_t>(w)].push_back(idx++);
    }
  }

  // 重置所有計數器和狀態
  completed_count_ = 0;
  stop_ = false;
  result_handler_stop_ = false;
  search_done_ = false;
  accumulated_leaves_ = 0;

  // 清空 queue
  {
    EvalBatch dummy;
    while (eval_queue_.try_dequeue(dummy)) {
      for (LeafData* leaf : dummy.leaves) {
        delete leaf;
      }
    }
  }

  // 清空每個 worker 的 pending queue
  for (int w = 0; w < num_threads_; ++w) {
    std::lock_guard<std::mutex> lk(*per_worker_pending_mtx_[static_cast<std::size_t>(w)]);
    per_worker_pending_results_[static_cast<std::size_t>(w)].clear();
  }

  // 喚醒所有 workers
  for (int w = 0; w < num_threads_; ++w) {
    std::lock_guard<std::mutex> lock(*worker_cv_mtx_[static_cast<std::size_t>(w)]);
    worker_cv_[static_cast<std::size_t>(w)]->notify_one();
  }
}

void SearchManager::shutdown() {
  stop_ = true;
  result_handler_stop_ = true;

  for (int w = 0; w < num_threads_; ++w) {
    std::lock_guard<std::mutex> lock(*worker_cv_mtx_[static_cast<std::size_t>(w)]);
    worker_cv_[static_cast<std::size_t>(w)]->notify_one();
  }

  // 喚醒所有 Result Handler
  for (int w = 0; w < num_threads_; ++w) {
    std::lock_guard<std::mutex> lk(*per_worker_pending_mtx_[static_cast<std::size_t>(w)]);
    per_worker_pending_cv_[static_cast<std::size_t>(w)]->notify_one();
  }

  {
    std::lock_guard<std::mutex> lock(search_done_mtx_);
    search_done_ = true;
    search_done_cv_.notify_all();
  }
}

void SearchManager::join_workers() {
  {
    EvalBatch dummy;
    while (eval_queue_.try_dequeue(dummy)) {
      for (LeafData* leaf : dummy.leaves) {
        delete leaf;
      }
    }
  }

  shutdown();

  for (auto& t : pool_) {
    if (t.joinable()) t.join();
  }
  pool_.clear();

  for (auto& t : result_handlers_) {
    if (t.joinable()) t.join();
  }
  result_handlers_.clear();
}

// ══════════════════════════════════════════════════════════════════════
// CPU Worker 主迴圈
// ══════════════════════════════════════════════════════════════════════

void SearchManager::worker_loop(int worker_id) {
  std::vector<int>& my_trees = assigned_trees_[static_cast<std::size_t>(worker_id)];
  const int batch_increment = (config_.batch_increment > 0) ? config_.batch_increment : kBatchIncrement;

  while (!stop_) {
    bool did_work = false;

    for (auto it = my_trees.begin(); it != my_trees.end(); ) {
      int tree_id = *it;
      if (tree_id < 0 || tree_id >= tree_count_) {
        it = my_trees.erase(it);
        continue;
      }

      SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];

      if (tree.session->is_complete()) {
        if (!tree.completed) complete_tree(tree);
        it = my_trees.erase(it);
        continue;
      }

      if (tree.session->has_pending_leaves()) {
        ++it;
        continue;
      }

      int simulated_count = 0;
      std::vector<LeafData*> batch_leaves;
      batch_leaves.reserve(static_cast<std::size_t>(batch_increment));

      while (simulated_count < batch_increment &&
             !tree.session->is_complete() &&
             !tree.session->has_pending_leaves()) {

        LeafData* leaf = simulate_single_step(tree_id);
        if (leaf) batch_leaves.push_back(leaf);
        simulated_count++;
      }

      if (!batch_leaves.empty()) {
        did_work = true;
        EvalBatch batch;
        batch.worker_id = worker_id;
        batch.leaves = std::move(batch_leaves);
        eval_queue_.enqueue(std::move(batch));
        accumulated_leaves_.fetch_add(simulated_count, std::memory_order_relaxed);
      }

      if (tree.session->is_complete()) {
        if (!tree.completed) complete_tree(tree);
        it = my_trees.erase(it);
        continue;
      }
      ++it;
    }

    if (!did_work) {
      std::unique_lock<std::mutex> lk(*worker_cv_mtx_[static_cast<std::size_t>(worker_id)]);
      if (HasAnyReadyTree(my_trees)) continue;
      worker_cv_[static_cast<std::size_t>(worker_id)]->wait(lk, [this, &my_trees]() {
        return stop_ || HasAnyReadyTree(my_trees);
      });
    }

    if (completed_count_.load(std::memory_order_acquire) >= tree_count_) {
      // 所有樹完成，進入睡眠等待下一次 reset
      std::unique_lock<std::mutex> lk(*worker_cv_mtx_[static_cast<std::size_t>(worker_id)]);
      worker_cv_[static_cast<std::size_t>(worker_id)]->wait(lk, [&]() { return stop_.load(); });
      if (stop_) break;
    }
  }

}

// ══════════════════════════════════════════════════════════════════════
// 1:1 Result Handler 主迴圈
// ══════════════════════════════════════════════════════════════════════
//
// 不再從 eval_queue_ 拿資料，只等自己的 per_worker_pending_queue。
// 當 Python 呼叫 submit_leaf_evals(worker_id, ...) 時，資料被直接
// 放進對應 worker_id 的 queue，然後喚醒對應的 Result Handler。
// Result Handler 用 tree_id 找到樹，呼叫 submit_single_eval 寫回。

void SearchManager::result_handler_loop(int worker_id) {
  const std::size_t idx = static_cast<std::size_t>(worker_id);

  while (!result_handler_stop_) {
    // 只等自己的 queue
    std::unique_lock<std::mutex> lk(*per_worker_pending_mtx_[idx]);
    per_worker_pending_cv_[idx]->wait(lk, [this, idx]() {
      return result_handler_stop_ || !per_worker_pending_results_[idx].empty();
    });

    if (result_handler_stop_) break;

    // 取出自己 queue 裡的全部結果
    std::vector<PendingEvalResult> results;
    results.swap(per_worker_pending_results_[idx]);
    lk.unlock();

    // 處理結果 — 這些 tree_id 一定是我負責的
    for (auto& res : results) {
      for (int i = 0; i < res.batch_size; ++i) {
        const int node_id = res.node_ids[static_cast<std::size_t>(i)];
        const int tree_id = res.tree_ids[static_cast<std::size_t>(i)];
        const float* priors_row =
            res.priors.data() + (static_cast<std::size_t>(i) *
                                 static_cast<std::size_t>(kActionCount));
        const float value = res.values[static_cast<std::size_t>(i)];

        process_pending_eval_result(tree_id, node_id, priors_row, value);
      }
    }
  }

}

// ══════════════════════════════════════════════════════════════════════
// 處理 GPU 推論結果（寫回樹、解除 pending、喚醒 worker）
// ══════════════════════════════════════════════════════════════════════

void SearchManager::process_pending_eval_result(int tree_id, int node_id,
                                                 const float* priors,
                                                 float value) {
  if (tree_id < 0 || tree_id >= tree_count_) return;

  SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
  if (tree.completed) return;

  bool became_ready = false;

  tree.session->submit_single_eval(node_id, priors, value);

  if (!tree.session->is_complete() && !tree.session->has_pending_leaves()) {
    became_ready = true;
  } else if (tree.session->is_complete()) {
    if (!tree.completed) complete_tree(tree);
  }

  // 如果樹解除 pending 且未完成，喚醒對應的 CPU Worker
  if (became_ready) {
    for (int w = 0; w < num_threads_; ++w) {
      const auto& assigned = assigned_trees_[static_cast<std::size_t>(w)];
      if (std::find(assigned.begin(), assigned.end(), tree_id) != assigned.end()) {
        std::lock_guard<std::mutex> lk(*worker_cv_mtx_[static_cast<std::size_t>(w)]);
        worker_cv_[static_cast<std::size_t>(w)]->notify_one();
        break;
      }
    }
  }
}

// ══════════════════════════════════════════════════════════════════════
// Python 端專用介面
// ══════════════════════════════════════════════════════════════════════

SearchManager::PackedBatch SearchManager::dequeue_batch(int max_batch,
                                                        int timeout_ms) {
  PackedBatch result;

  // 先把 queue 中能拿的都拿出來，累積到一個 vector 中
  std::vector<LeafData*> all_leaves;
  std::vector<int> all_worker_ids;
  int total = 0;

  // ── 等待模式（timeout_ms > 0） ─────────────────
  if (timeout_ms > 0) {
    int spin = 0;
    auto start = std::chrono::steady_clock::now();
    while (true) {
      EvalBatch batch;
      if (eval_queue_.try_dequeue(batch)) {
        const std::size_t n = batch.leaves.size();
        for (std::size_t i = 0; i < n; ++i) {
          all_leaves.push_back(batch.leaves[i]);
          all_worker_ids.push_back(batch.worker_id);
        }
        total += static_cast<int>(n);
        if (total >= max_batch) break;
        continue;
      }
      // queue 空了
      if (total > 0) break;  // 已有資料，直接回傳
      if (stop_ || all_trees_completed()) break;
      spin++;
      if (spin > 10000) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= timeout_ms) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        spin = 0;
      }
    }
  } else {
    // ── non-blocking ──────────────────────────────
    EvalBatch batch;
    while (eval_queue_.try_dequeue(batch)) {
      const std::size_t n = batch.leaves.size();
      for (std::size_t i = 0; i < n; ++i) {
        all_leaves.push_back(batch.leaves[i]);
        all_worker_ids.push_back(batch.worker_id);
      }
      total += static_cast<int>(n);
      if (total >= max_batch) break;
    }
  }

  if (total == 0) {
    result.batch_size = 0;
    return result;
  }

  const std::size_t bsz = static_cast<std::size_t>(total);

  dequeued_node_ids_.resize(bsz);
  dequeued_tree_ids_.resize(bsz);
  dequeued_worker_ids_.resize(bsz);
  dequeued_board_flat_.resize(bsz * static_cast<std::size_t>(kBoardFlatSize));
  dequeued_global_feat_.resize(bsz * static_cast<std::size_t>(kGlobalFeatureDim));
  dequeued_legal_mask_.resize(bsz * static_cast<std::size_t>(kActionCount));

  for (std::size_t i = 0; i < bsz; ++i) {
    LeafData* leaf = all_leaves[i];
    dequeued_node_ids_[i] = leaf->node_id;
    dequeued_tree_ids_[i] = leaf->tree_id;
    dequeued_worker_ids_[i] = all_worker_ids[i];

    std::memcpy(dequeued_board_flat_.data() + i * static_cast<std::size_t>(kBoardFlatSize),
                leaf->board_flat,
                sizeof(float) * static_cast<std::size_t>(kBoardFlatSize));
    std::memcpy(dequeued_global_feat_.data() + i * static_cast<std::size_t>(kGlobalFeatureDim),
                leaf->global_feat,
                sizeof(float) * static_cast<std::size_t>(kGlobalFeatureDim));
    std::memcpy(dequeued_legal_mask_.data() + i * static_cast<std::size_t>(kActionCount),
                leaf->legal_mask,
                sizeof(uint8_t) * static_cast<std::size_t>(kActionCount));
  }

    for (LeafData* leaf : all_leaves) delete leaf;

  result.batch_size = total;
  result.node_ids          = dequeued_node_ids_.data();
  result.tree_ids          = dequeued_tree_ids_.data();
  result.worker_ids        = dequeued_worker_ids_.data();
  result.board_state_flat  = dequeued_board_flat_.data();
  result.global_features   = dequeued_global_feat_.data();
  result.legal_masks       = dequeued_legal_mask_.data();

  return result;
}

void SearchManager::submit_leaf_evals(
    const int32_t* worker_ids,
    const int32_t* node_ids,
    const int32_t* tree_ids,
    const float* priors,
    const float* values,
    int batch_size) {
  if (batch_size <= 0)
    throw std::invalid_argument("batch_size must be >= 1");

  // 先按 worker_id 分組
  // 為避免每次重新分配記憶體，先用 map 結構收集
  // worker_id -> (node_ids, tree_ids, priors_row, values)
  // 因為 worker 數固定且不多，使用固定大小陣列

  // 先掃一遍，統計每個 worker 的數量
  std::vector<int> per_worker_count(static_cast<std::size_t>(num_threads_), 0);
  for (int i = 0; i < batch_size; ++i) {
    int w = worker_ids[i];
    if (w >= 0 && w < num_threads_) {
      per_worker_count[static_cast<std::size_t>(w)]++;
    }
  }

  // 為每個有資料的 worker 建立 PendingEvalResult
  for (int w = 0; w < num_threads_; ++w) {
    int cnt = per_worker_count[static_cast<std::size_t>(w)];
    if (cnt == 0) continue;

    PendingEvalResult res;
    res.batch_size = cnt;
    res.node_ids.reserve(static_cast<std::size_t>(cnt));
    res.tree_ids.reserve(static_cast<std::size_t>(cnt));
    res.priors.resize(static_cast<std::size_t>(cnt) *
                      static_cast<std::size_t>(kActionCount));
    res.values.reserve(static_cast<std::size_t>(cnt));

    int dst_idx = 0;
    for (int i = 0; i < batch_size; ++i) {
      if (worker_ids[i] != w) continue;
      res.node_ids.push_back(node_ids[i]);
      res.tree_ids.push_back(tree_ids[i]);
      std::memcpy(
          res.priors.data() + static_cast<std::size_t>(dst_idx) *
                                  static_cast<std::size_t>(kActionCount),
          priors + static_cast<std::size_t>(i) *
                       static_cast<std::size_t>(kActionCount),
          static_cast<std::size_t>(kActionCount) * sizeof(float));
      res.values.push_back(values[i]);
      dst_idx++;
    }

    // 放進對應 worker 的 queue，喚醒
    const std::size_t idx = static_cast<std::size_t>(w);
    {
      std::lock_guard<std::mutex> lock(*per_worker_pending_mtx_[idx]);
      per_worker_pending_results_[idx].push_back(std::move(res));
    }
    per_worker_pending_cv_[idx]->notify_one();
  }
}

// ══════════════════════════════════════════════════════════════════════
// 輔助方法
// ══════════════════════════════════════════════════════════════════════

SearchManager::LeafData* SearchManager::simulate_single_step(int tree_id) {
  SearchTree& tree = *trees_[static_cast<std::size_t>(tree_id)];
  auto& session = tree.session;
  if (session->is_complete() || session->has_pending_leaves()) return nullptr;

  auto leaf = std::make_unique<LeafData>();
  leaf->tree_id = tree_id;

  int n = session->simulate_into_buffers(
      1, leaf->board_flat, leaf->global_feat, leaf->legal_mask,
      &leaf->node_id, &leaf->tree_id, tree_id);

  return (n > 0) ? leaf.release() : nullptr;
}

bool SearchManager::HasAnyReadyTree(const std::vector<int>& tree_ids) const {
  for (int tid : tree_ids) {
    if (tid < 0 || tid >= tree_count_) continue;
    const auto& tree = *trees_[static_cast<std::size_t>(tid)];
    if (!tree.session->is_complete() && !tree.session->has_pending_leaves())
      return true;
  }
  return false;
}

void SearchManager::complete_tree(SearchTree& tree) {
  if (tree.completed) return;
  tree.completed = true;
  completed_count_.fetch_add(1, std::memory_order_release);

  if (all_trees_completed()) {
    std::lock_guard<std::mutex> lock(search_done_mtx_);
    search_done_ = true;
    search_done_cv_.notify_all();
  }
}

bool SearchManager::all_trees_completed() const {
  return completed_count_.load(std::memory_order_acquire) >= tree_count_;
}

// ══════════════════════════════════════════════════════════════════════
// 結果取得
// ══════════════════════════════════════════════════════════════════════

bool SearchManager::is_complete() const {
  return all_trees_completed();
}

int SearchManager::completed_tree_count() const {
  return completed_count_.load(std::memory_order_relaxed);
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

}  // namespace tzaar
