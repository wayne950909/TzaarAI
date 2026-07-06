#include "mcts/bench.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace tzaar {

// ─── Thread-local RNG（每個 thread 獨立，減少鎖競爭）────
static std::mt19937& get_tls_rng() {
  thread_local std::mt19937 rng(static_cast<unsigned>(std::random_device{}()));
  return rng;
}

// ══════════════════════════════════════════════════════════════════════
// 公開 API
// ══════════════════════════════════════════════════════════════════════

CpuBenchResult CpuBench::run(int num_trees, int num_threads, int simulations,
                              int leaf_batch_size) {
  if (num_trees <= 0) throw std::invalid_argument("num_trees must be >= 1");
  if (num_threads <= 0) throw std::invalid_argument("num_threads must be >= 1");
  if (simulations <= 0) throw std::invalid_argument("simulations must be >= 1");
  if (leaf_batch_size <= 0) throw std::invalid_argument("leaf_batch_size must be >= 1");

  // ─── 建立搜尋配置 ───────────────────────────────────
  SearchConfig config;
  config.simulations = simulations;
  config.leaf_batch_size = leaf_batch_size;
  config.puct_c = 1.5f;
  config.add_root_dirichlet_noise = false;

  // ─── 建立樹與同步物件 ─────────────────────────────
  std::vector<std::unique_ptr<SearchSession>> sessions;
  std::vector<std::atomic<bool>> tree_locks(num_trees);
  std::vector<std::atomic<bool>> tree_completed(num_trees);
  sessions.reserve(static_cast<std::size_t>(num_trees));

  for (int i = 0; i < num_trees; ++i) {
    PhaseGameState state;
    sessions.push_back(std::make_unique<SearchSession>(state, config));
    tree_locks[static_cast<std::size_t>(i)] = false;
    tree_completed[static_cast<std::size_t>(i)] = false;
  }

  // ─── 啟動 worker threads 並計時 ─────────────────
  std::atomic<int> next_tree_index{0};

  auto t_start = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(num_threads));
  for (int i = 0; i < num_threads; ++i) {
    pool.emplace_back(worker_loop, i, num_trees,
                      std::ref(sessions), std::ref(tree_locks),
                      std::ref(tree_completed), simulations,
                      leaf_batch_size, std::ref(next_tree_index));
  }

  // 等待所有 worker 完成
  for (auto& t : pool) {
    if (t.joinable()) t.join();
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  // ─── 統計結果 ────────────────────────────────────────
  CpuBenchResult result;
  result.elapsed_seconds = elapsed;
  result.num_trees = num_trees;
  result.num_threads = num_threads;
  result.simulations_per_tree = simulations;
  result.total_simulations = num_trees * simulations;

  int total_sims_done = 0;
  result.tree_simulations_done.resize(static_cast<std::size_t>(num_trees));
  result.tree_node_counts.resize(static_cast<std::size_t>(num_trees));

  for (int i = 0; i < num_trees; ++i) {
    SearchResult sr = sessions[static_cast<std::size_t>(i)]->finish();
    int sims_done = sr.simulations_processed;
    result.tree_simulations_done[static_cast<std::size_t>(i)] = sims_done;
    total_sims_done += sims_done;

    // 完成次數即為產生的節點數近似值
    result.tree_node_counts[static_cast<std::size_t>(i)] = sims_done;
  }

  result.total_simulations_done = total_sims_done;
  result.total_nodes_created = total_sims_done;  // 近似
  result.sims_per_second = elapsed > 0.0
      ? static_cast<double>(total_sims_done) / elapsed
      : 0.0;

  return result;
}

// ══════════════════════════════════════════════════════════════════════
// 內部輔助：產生隨機 priors（正規化為機率分佈）
// ══════════════════════════════════════════════════════════════════════

static void gen_random_priors(float* priors, int num_actions) {
  auto& rng = get_tls_rng();
  float sum = 0.0f;
  for (int a = 0; a < num_actions; ++a) {
    const float v = static_cast<float>(rng() % 10000) / 10000.0f;
    priors[a] = v;
    sum += v;
  }
  if (sum > 0.0f) {
    const float inv_sum = 1.0f / sum;
    for (int a = 0; a < num_actions; ++a) {
      priors[a] *= inv_sum;
    }
  } else {
    const float uniform = 1.0f / static_cast<float>(num_actions);
    for (int a = 0; a < num_actions; ++a) {
      priors[a] = uniform;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════
// Worker Thread 主迴圈
// ══════════════════════════════════════════════════════════════════════

void CpuBench::worker_loop(
    int thread_id,
    int num_trees,
    std::vector<std::unique_ptr<SearchSession>>& sessions,
    std::vector<std::atomic<bool>>& tree_locks,
    std::vector<std::atomic<bool>>& tree_completed,
    int simulations,
    int leaf_batch_size,
    std::atomic<int>& next_tree_index) {

  (void)thread_id;
  (void)simulations;
  (void)leaf_batch_size;

  // 每個 thread 預先分配 buffer
  std::vector<float> board_buf(static_cast<std::size_t>(kBoardFlatSize));
  std::vector<float> global_buf(static_cast<std::size_t>(kGlobalFeatureDim));
  std::vector<uint8_t> mask_buf(static_cast<std::size_t>(kActionCount));
  int32_t node_id_out;
  int32_t tree_id_out;

  std::vector<float> priors_buf(static_cast<std::size_t>(kActionCount));

  while (true) {
    // ─── 用 round-robin CAS 找一棵未完成的樹 ────────
    int tree_id = -1;
    const int start = next_tree_index.fetch_add(1) % num_trees;

    for (int i = 0; i < num_trees; ++i) {
      const int idx = (start + i) % num_trees;
      if (tree_completed[static_cast<std::size_t>(idx)].load(
              std::memory_order_acquire))
        continue;

      bool expected = false;
      if (tree_locks[static_cast<std::size_t>(idx)].compare_exchange_weak(
              expected, true, std::memory_order_acquire,
              std::memory_order_relaxed)) {
        if (tree_completed[static_cast<std::size_t>(idx)].load(
                std::memory_order_acquire)) {
          tree_locks[static_cast<std::size_t>(idx)].store(
              false, std::memory_order_release);
          continue;
        }
        tree_id = idx;
        break;
      }
    }

    // 沒有可鎖定的樹 → 全部完成
    if (tree_id < 0) break;

    auto& session = *sessions[static_cast<std::size_t>(tree_id)];

    // ─── 對這棵樹一直做模擬直到完成 ────────────────
    while (!session.is_complete()) {
      const int count = session.simulate_into_buffers(
          1,
          board_buf.data(),
          global_buf.data(),
          mask_buf.data(),
          &node_id_out,
          &tree_id_out,
          tree_id);

      if (count > 0) {
        // 成功模擬出一個葉節點 → mock NN 評估
        gen_random_priors(priors_buf.data(), kActionCount);
        session.submit_single_eval(node_id_out, priors_buf.data(), 0.0f);
      } else {
        // 模擬 count == 0 → 可能有 pending leaves 未處理
        if (session.has_pending_leaves()) {
          // 用 mock priors 補齊 pending eval
          auto packed = session.collect_pending_leaves_packed(
              kActionCount);  // 取全部
          if (packed.batch_size > 0) {
            std::vector<float> big_priors(
                static_cast<std::size_t>(packed.batch_size) *
                static_cast<std::size_t>(kActionCount));
            std::vector<float> big_values(
                static_cast<std::size_t>(packed.batch_size), 0.0f);
            for (int j = 0; j < packed.batch_size; ++j) {
              gen_random_priors(
                  big_priors.data() +
                      static_cast<std::size_t>(j) * kActionCount,
                  kActionCount);
            }
            session.submit_leaf_eval_batch(
                packed.node_ids, big_priors.data(), big_values.data(),
                packed.batch_size);
          }
        } else {
          break;  // 真的完成了
        }
      }
    }

    // 標記樹為完成
    tree_completed[static_cast<std::size_t>(tree_id)].store(
        true, std::memory_order_release);
    tree_locks[static_cast<std::size_t>(tree_id)].store(
        false, std::memory_order_release);
  }
}

}  // namespace tzaar
