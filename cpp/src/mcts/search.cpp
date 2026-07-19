#include "mcts/search.h"
#include "core/action.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tzaar {

// ══════════════════════════════════════════════════════════════════════
// 建構子
// ══════════════════════════════════════════════════════════════════════

// SearchSession 代表「單棵搜尋樹」。
// 它本身不碰多執行緒調度，只專注在：
// - 節點選擇
// - 葉節點狀態重建
// - pending eval 管理
// - backup / expand / root noise
SearchSession::SearchSession(const PhaseGameState& root_state, SearchConfig config)
    : config_(std::move(config)), rng_(std::random_device{}()) {

  if (config_.simulations < 0) throw std::invalid_argument("simulations must be >= 0");
  if (config_.leaf_batch_size <= 0) throw std::invalid_argument("leaf_batch_size must be >= 1");
  if (config_.puct_c <= 0.0f) throw std::invalid_argument("puct_c must be > 0");

  root_state_ = root_state.clone();

  MctsNode root;
  root.prior = 1.0f;
  root.parent_idx = -1;
  root.action_from_parent = -1;
  root.to_play = root_state_.current_player();
  root.is_terminal = root_state_.is_done();
  root.winner = root_state_.winner();
  root.has_player = !root.is_terminal;
  if (root.is_terminal) {
    root.legal_mask_ready = true;
    root.cached_legal_mask.assign(static_cast<std::size_t>(kActionCount), static_cast<std::uint8_t>(0));
  }

  nodes_.push_back(std::move(root));

  // 根節點 CNN 特徵快取。
  // 之後所有葉節點 state 都可從這裡出發，沿動作序列重建。
  root_board_flat_.resize(static_cast<std::size_t>(kBoardFlatSize), 0.0f);
  root_global_feat_.resize(static_cast<std::size_t>(kGlobalFeatureDim), 0.0f);
  if (!root_state_.is_done()) {
    const auto& board = root_state_.game().board();
    const auto counts = root_state_.game().piece_counts();
    build_cnn_features_into(
        board, root_state_.current_player(), root_state_.turn_number(),
        root_state_.phase(), counts,
        root_board_flat_.data(), root_global_feat_.data());
  }
}

// ══════════════════════════════════════════════════════════════════════
// 公開 API
// ══════════════════════════════════════════════════════════════════════

bool SearchSession::is_complete() const {
  return simulations_processed_ >= config_.simulations && !has_pending_leaves();
}

std::vector<LeafSnapshot> SearchSession::collect_pending_leaves(int max_batch) {
  bool has_pending = prepare_pending_leaves(max_batch);
  if (has_pending) {
    return pending_leaf_snapshots();
  }
  return {};
}

SearchSession::PackedLeaves SearchSession::collect_pending_leaves_packed(int max_batch) {
  bool has_pending = prepare_pending_leaves(max_batch);

  PackedLeaves result;
  if (!has_pending) {
    result.batch_size = 0;
    return result;
  }

  result.batch_size = static_cast<int>(pending_node_order_.size());
  result.node_ids = batch_node_ids_.data();
  result.legal_masks = batch_legal_mask_.data();
  result.board_state_flat = batch_board_flat_.data();
  result.global_features = batch_global_feat_.data();
  return result;
}

void SearchSession::submit_leaf_eval(int node_id, const std::vector<float>& priors, float value) {
  const int node_idx = node_id - 1;
  if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size()))
    throw std::invalid_argument("unknown node_id for current search session");
  if (!has_pending_leaves())
    throw std::invalid_argument("no pending leaves to evaluate");
  if (pending_paths_.find(node_idx) == pending_paths_.end())
    throw std::invalid_argument("node_id is not in pending leaves");
  if (pending_eval_map_.find(node_idx) != pending_eval_map_.end())
    throw std::invalid_argument("leaf evaluation already submitted for this node");
  if (priors.size() != static_cast<std::size_t>(kActionCount))
    throw std::invalid_argument("priors must have length N_ACTIONS");

  pending_eval_map_.emplace(node_idx, PendingEval{priors, value});

  if (pending_eval_map_.size() == pending_node_order_.size()) {
    process_pending_evals();
  }
}

void SearchSession::submit_leaf_eval_batch(
    const int32_t* node_ids,
    const float* priors,
    const float* values,
    int batch_size) {

  if (!has_pending_leaves())
    throw std::invalid_argument("no pending leaves to evaluate");

  std::vector<float> prior_row(static_cast<std::size_t>(kActionCount), 0.0f);

  for (int i = 0; i < batch_size; ++i) {
    const int node_id = static_cast<int>(node_ids[i]);
    const int node_idx = node_id - 1;

    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size()))
      throw std::invalid_argument("unknown node_id for current search session");
    if (pending_paths_.find(node_idx) == pending_paths_.end())
      throw std::invalid_argument("node_id is not in pending leaves");
    if (pending_eval_map_.find(node_idx) != pending_eval_map_.end())
      throw std::invalid_argument("leaf evaluation already submitted for this node");

    std::memcpy(prior_row.data(),
                priors + (i * static_cast<std::int64_t>(kActionCount)),
                static_cast<std::size_t>(kActionCount) * sizeof(float));
    pending_eval_map_.emplace(node_idx, PendingEval{prior_row, values[i]});
  }

  if (pending_eval_map_.size() == pending_node_order_.size()) {
    process_pending_evals();
  }
}

SearchResult SearchSession::finish() {
  SearchResult result;
  result.root_node_id = root_node_id_;
  result.root_player = nodes_[root_node_index_].to_play;
  result.winner = nodes_[root_node_index_].winner;
  result.is_done = nodes_[root_node_index_].is_terminal;
  result.is_complete = is_complete();
  result.needs_root_eval = has_pending_leaves();
  result.simulations_requested = config_.simulations;
  result.simulations_processed = simulations_processed_;
  result.pending_leaf_count = static_cast<int>(pending_node_order_.size());
  result.root_value = nodes_[root_node_index_].mean_value();

  MctsNode& root_node = nodes_[root_node_index_];
  if (!root_node.legal_mask_ready) {
    PhaseGameState tmp = root_state_.clone();
    get_legal_mask_for_node(root_node_index_, tmp);
  }

  result.legal_mask.assign(root_node.cached_legal_mask.begin(),
                           root_node.cached_legal_mask.end());

  result.root_policy.assign(static_cast<std::size_t>(kActionCount), 0.0f);
  result.root_visits.assign(static_cast<std::size_t>(kActionCount), 0.0f);

  const MctsNode& root = nodes_[root_node_index_];
  for (const auto& pair : root.children) {
    const int action = pair.first;
    const int child_idx = pair.second;
    if (action < 0 || action >= kActionCount) continue;
    const MctsNode& child = nodes_[child_idx];
    result.root_policy[static_cast<std::size_t>(action)] = child.prior;
    result.root_visits[static_cast<std::size_t>(action)] = static_cast<float>(child.visit_count);
  }

  return result;
}

LeafSnapshot SearchSession::root_snapshot() {
  LeafSnapshot snapshot;
  root_state_.fill_snapshot_metadata(snapshot, root_node_id_);
  snapshot.board_state_flat = root_board_flat_;
  snapshot.global_features = root_global_feat_;
  return snapshot;
}

// ══════════════════════════════════════════════════════════════════════
// 供 SearchManager 使用的方法
// ══════════════════════════════════════════════════════════════════════

// 這是給 SearchManager 使用的「零配置批次模擬」介面。
// 它會把待推論葉節點直接寫進外部 buffer，而不是先建立 Python 物件。
int SearchSession::simulate_into_buffers(int chunk,
                                         float* board_out,
                                         float* global_out,
                                         uint8_t* mask_out,
                                         int32_t* node_ids_out,
                                         int32_t* tree_ids_out,
                                         int tree_id) {
    if (chunk <= 0) return 0;

  int simulated_count = 0;
  FILE* dbg = fopen("C:\\temp\\sim_debug.txt", "a");
  for (int sim = 0; sim < chunk; ++sim) {
    if (is_complete()) break;

    // ── Selection：從 root 開始，避開已有 pending leaf 的子樹 ──
    int node_idx = root_node_index_;
    std::vector<int> path;
    path.reserve(128);
    path.push_back(node_idx);

    while (true) {
      MctsNode& node = nodes_[node_idx];

      // 情況 A：終端節點 → backup，繼續下一次模擬
      if (node.is_terminal) {
        fprintf(dbg, "  sim=%d TERMINAL node_idx=%d\n", sim, node_idx);
        const float leaf_value = terminal_value_for_current_player(node.winner, node.to_play);
        backup(path, leaf_value, node.to_play);
        simulations_processed_ += 1;
                goto next_simulation;
      }

      // 情況 B：已展開節點 → selection 繼續往下
      if (node.expanded) {
        fprintf(dbg, "  sim=%d EXPANDED node_idx=%d children=%zu\n", sim, node_idx, node.children.size());
        if (node.children.empty()) {
          // 已展開但無合法子節點（可能因為遊戲結束判斷不同步）
          backup(path, 0.0f, node.to_play);
          simulations_processed_ += 1;
          goto next_simulation;
        }
                const int action = select_child_action(node_idx);
        fprintf(dbg, "    select_child_action -> action=%d\n", action);
        // select_child_action 回傳 -1 代表所有子節點都有 pending leaf
        if (action < 0) {
          // 整棵樹已 stalled → 無法再產生新 leaf，直接返回
          goto finished;
        }
        node_idx = node.children.at(action);
        path.push_back(node_idx);
        continue;
      }

            // 情況 C：未展開葉節點
            fprintf(dbg, "  sim=%d UNEXPANDED_LEAF node_idx=%d (simulated_count=%d)\n", sim, node_idx, simulated_count);

            // 首次訪問：重建狀態
      PhaseGameState state = reconstruct_state_for_node(node_idx);
      node.to_play = state.current_player();
      node.is_terminal = state.is_done();
      node.winner = state.winner();
      node.has_player = !node.is_terminal;

      if (node.is_terminal) {
        node.expanded = true;
        node.legal_mask_ready = true;
        node.cached_legal_mask.assign(static_cast<std::size_t>(kActionCount), static_cast<std::uint8_t>(0));
        const float leaf_value = terminal_value_for_current_player(node.winner, node.to_play);
        backup(path, leaf_value, node.to_play);
        simulations_processed_ += 1;
        goto next_simulation;
      }

      // ─── 將 CNN 特徵寫入外部 buffer ─────────────────
      {
        const auto counts = state.game().piece_counts();
        const std::size_t offset = static_cast<std::size_t>(simulated_count);
        build_cnn_features_into(
            state.game().board(), state.current_player(), state.turn_number(),
            state.phase(), counts,
            board_out + (offset * static_cast<std::size_t>(kBoardFlatSize)),
            global_out + (offset * static_cast<std::size_t>(kGlobalFeatureDim)));
      }

      // ─── 合法遮罩寫入外部 buffer ────────────────────
      const std::vector<bool> legal = state.legal_mask();
      node.cached_legal_mask.resize(static_cast<std::size_t>(kActionCount), 0);
      {
        const std::size_t offset = static_cast<std::size_t>(simulated_count) * static_cast<std::size_t>(kActionCount);
        for (std::size_t j = 0; j < legal.size() && j < static_cast<std::size_t>(kActionCount); ++j) {
          const uint8_t v = legal[j] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
          node.cached_legal_mask[j] = v;
          mask_out[offset + j] = v;
        }
      }
      node.legal_mask_ready = true;

            // ─── 建立子節點樁 ──────────────────────────────
            // 先保留空間，避免 push_back 時 vector reallocation 使 reference 失效
            nodes_.reserve(nodes_.size() + static_cast<std::size_t>(kActionCount));
            for (int action = 0; action < kActionCount; ++action) {
              if (!legal[static_cast<std::size_t>(action)]) continue;
              const int new_idx = static_cast<int>(nodes_.size());
              MctsNode child;
              child.prior = 0.0f;
              child.parent_idx = node_idx;
              child.action_from_parent = action;
              nodes_.push_back(std::move(child));
              nodes_[node_idx].children[action] = new_idx;
            }

                        // ─── 標記節點已展開（關鍵！否則下次又會走到相同節點）──
            // 用直接索引存取，避免 reference 可能失效的風險
            nodes_[node_idx].expanded = true;

      // ─── Virtual loss ──────────────────────────────
      for (const int idx : path) {
        nodes_[idx].visit_count += 1;
        nodes_[idx].value_sum -= 1.0f;
      }

      // ─── 記錄 pending ──────────────────────────────
      pending_node_order_.push_back(node_idx);
      node_ids_out[simulated_count] = static_cast<int32_t>(node_idx + 1);
      if (tree_ids_out) {
        tree_ids_out[simulated_count] = static_cast<int32_t>(tree_id);
      }
      pending_paths_[node_idx].push_back(path);
      simulated_count++;
      simulations_processed_ += 1;
      goto next_simulation;
    }

  next_simulation:
    continue;
  }

finished:
  fprintf(dbg, "  => return simulated_count=%d\n", simulated_count);
  fclose(dbg);
  return simulated_count;
}

void SearchSession::submit_single_eval(int node_id,
                                       const float* priors,
                                       float value) {
  const int node_idx = node_id - 1;
  if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size()))
    throw std::invalid_argument("unknown node_id for current search session");
  if (pending_paths_.find(node_idx) == pending_paths_.end())
    throw std::invalid_argument("node_id is not in pending leaves");
  if (pending_eval_map_.find(node_idx) != pending_eval_map_.end())
    throw std::invalid_argument("leaf evaluation already submitted for this node");

  std::vector<float> prior_row(static_cast<std::size_t>(kActionCount), 0.0f);
  std::memcpy(prior_row.data(), priors,
              static_cast<std::size_t>(kActionCount) * sizeof(float));

  pending_eval_map_.emplace(node_idx, PendingEval{std::move(prior_row), value});

  // 當所有 pending 都到齊時自動處理
  if (pending_eval_map_.size() == pending_node_order_.size()) {
    process_pending_evals();
  }
}

// ══════════════════════════════════════════════════════════════════════
// 內部方法
// ══════════════════════════════════════════════════════════════════════

// 只記錄 parent/action，而不是在每個節點存完整 state。
// 這樣可大量降低樹的記憶體成本。
std::vector<int> SearchSession::collect_action_path_to_node(int node_idx) const {
  std::vector<int> reversed;
  int current = node_idx;
  while (current != root_node_index_) {
    const MctsNode& node = nodes_[current];
    reversed.push_back(node.action_from_parent);
    current = node.parent_idx;
  }
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

// 由 root_state_ + action path 重建任意節點局面。
// 這正是 mctsLogic.md 中描述的「根狀態快取 + 動作序列重建」實作。
PhaseGameState SearchSession::reconstruct_state_for_node(int node_idx) const {
  PhaseGameState state = root_state_.clone();
  const std::vector<int> actions = collect_action_path_to_node(node_idx);
  for (const int action : actions) {
    state.apply_action_trusted(action);
  }
  return state;
}

void SearchSession::update_node_metadata_from_state(int node_idx, const PhaseGameState& state) {
  MctsNode& node = nodes_[node_idx];
  node.to_play = state.current_player();
  node.is_terminal = state.is_done();
  node.winner = state.winner();
  node.has_player = !node.is_terminal;
  if (node.is_terminal) {
    node.legal_mask_ready = true;
    node.cached_legal_mask.assign(static_cast<std::size_t>(kActionCount), static_cast<std::uint8_t>(0));
  }
}

std::vector<bool> SearchSession::get_legal_mask_for_node(int node_idx, PhaseGameState& state) {
  MctsNode& node = nodes_[node_idx];
  if (node.legal_mask_ready) {
    std::vector<bool> cached(static_cast<std::size_t>(kActionCount), false);
    const std::size_t n = std::min(cached.size(), node.cached_legal_mask.size());
    for (std::size_t i = 0; i < n; ++i) {
      cached[i] = node.cached_legal_mask[i] != 0;
    }
    return cached;
  }

  std::vector<bool> mask = state.legal_mask();
  node.cached_legal_mask.assign(mask.size(), static_cast<std::uint8_t>(0));
  for (std::size_t i = 0; i < mask.size(); ++i) {
    node.cached_legal_mask[i] = mask[i] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
  }
  node.legal_mask_ready = true;
  return mask;
}

bool SearchSession::prepare_pending_leaves(int max_batch) {
  if (max_batch <= 0) throw std::invalid_argument("max_batch must be >= 1");

  if (is_complete()) return false;

  if (has_pending_leaves()) {
    if (static_cast<int>(pending_node_order_.size()) > max_batch)
      throw std::invalid_argument("max_batch is smaller than current pending leaf count");
    return true;
  }

  while (!is_complete() && !has_pending_leaves()) {
    const int remaining = config_.simulations - simulations_processed_;
    if (remaining <= 0) break;
    const int chunk = std::min(config_.leaf_batch_size, remaining);
    simulate_chunk(chunk);
  }

  if (!has_pending_leaves()) return false;
  if (static_cast<int>(pending_node_order_.size()) > max_batch)
    throw std::invalid_argument("max_batch is smaller than current pending leaf count");
  return true;
}

std::vector<LeafSnapshot> SearchSession::pending_leaf_snapshots() {
  std::vector<LeafSnapshot> leaves;
  leaves.reserve(pending_node_order_.size());
  for (const int node_idx : pending_node_order_) {
    leaves.push_back(build_leaf_snapshot(node_idx));
  }
  return leaves;
}

LeafSnapshot SearchSession::build_leaf_snapshot(int node_idx) {
  MctsNode& node = nodes_[node_idx];
  PhaseGameState state = reconstruct_state_for_node(node_idx);
  update_node_metadata_from_state(node_idx, state);

  LeafSnapshot snapshot;
  state.fill_snapshot_metadata(snapshot, node_idx + 1);
  if (!node.is_terminal) {
    const auto counts = state.game().piece_counts();
    build_cnn_features(
        state.game().board(), state.current_player(), state.turn_number(),
        state.phase(), counts,
        snapshot.board_state_flat, snapshot.global_features);
  }
  return snapshot;
}

void SearchSession::simulate_chunk(int chunk) {
  pending_node_order_.clear();
  pending_paths_.clear();
  pending_eval_map_.clear();
  batch_board_flat_.clear();
  batch_global_feat_.clear();
  batch_legal_mask_.clear();
  batch_node_ids_.clear();

  for (int sim = 0; sim < chunk; ++sim) {
    simulations_processed_ += 1;

    int node_idx = root_node_index_;
    std::vector<int> path;
    path.reserve(128);
    path.push_back(node_idx);

    while (true) {
      MctsNode& node = nodes_[node_idx];

      if (node.is_terminal) {
        const float leaf_value = terminal_value_for_current_player(node.winner, node.to_play);
        backup(path, leaf_value, node.to_play);
        break;
      }

      if (node.expanded) {
        if (node.children.empty()) {
          backup(path, 0.0f, node.to_play);
          break;
        }
        const int action = select_child_action(node_idx);
        node_idx = node.children.at(action);
        path.push_back(node_idx);
        continue;
      }

      // 未展開葉節點
      if (pending_paths_.count(node_idx) > 0) {
        // 已在佇列中：應用 virtual loss
        for (const int idx : path) {
          nodes_[idx].visit_count += 1;
          nodes_[idx].value_sum -= 1.0f;
        }
        pending_paths_[node_idx].push_back(path);
        break;
      }

      // 首次訪問：重建狀態一次
      PhaseGameState state = reconstruct_state_for_node(node_idx);
      node.to_play = state.current_player();
      node.is_terminal = state.is_done();
      node.winner = state.winner();
      node.has_player = !node.is_terminal;

      if (node.is_terminal) {
        node.expanded = true;
        node.legal_mask_ready = true;
        node.cached_legal_mask.assign(static_cast<std::size_t>(kActionCount), static_cast<std::uint8_t>(0));
        const float leaf_value = terminal_value_for_current_player(node.winner, node.to_play);
        backup(path, leaf_value, node.to_play);
        break;
      }

      // 建構 CNN 特徵到連續批次緩衝區
      {
        const auto counts = state.game().piece_counts();
        const std::size_t offset_board = batch_board_flat_.size();
        const std::size_t offset_global = batch_global_feat_.size();
        batch_board_flat_.resize(offset_board + static_cast<std::size_t>(kBoardFlatSize));
        batch_global_feat_.resize(offset_global + static_cast<std::size_t>(kGlobalFeatureDim));
        build_cnn_features_into(
            state.game().board(), state.current_player(), state.turn_number(),
            state.phase(), counts,
            batch_board_flat_.data() + offset_board,
            batch_global_feat_.data() + offset_global);
      }

      // 快取合法遮罩並填入批次緩衝區
      const std::vector<bool> legal = state.legal_mask();
      node.cached_legal_mask.resize(static_cast<std::size_t>(kActionCount), 0);
      const std::size_t offset_mask = batch_legal_mask_.size();
      batch_legal_mask_.resize(offset_mask + static_cast<std::size_t>(kActionCount), 0);
      for (std::size_t j = 0; j < legal.size() && j < static_cast<std::size_t>(kActionCount); ++j) {
        const uint8_t v = legal[j] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
        node.cached_legal_mask[j] = v;
        batch_legal_mask_[offset_mask + j] = v;
      }
      node.legal_mask_ready = true;

            // 建立子節點樁（僅記錄 parent+action，不 clone 狀態）
      nodes_.reserve(nodes_.size() + static_cast<std::size_t>(kActionCount));
      for (int action = 0; action < kActionCount; ++action) {
        if (!legal[static_cast<std::size_t>(action)]) continue;
        const int new_idx = static_cast<int>(nodes_.size());
        MctsNode child;
        child.prior = 0.0f;  // expand 時由 NN 輸出設定
        child.parent_idx = node_idx;
        child.action_from_parent = action;
        nodes_.push_back(std::move(child));
        nodes_[node_idx].children[action] = new_idx;
      }

      // 加入 NN 批次佇列 — 應用 virtual loss
      for (const int idx : path) {
        nodes_[idx].visit_count += 1;
        nodes_[idx].value_sum -= 1.0f;
      }
      pending_node_order_.push_back(node_idx);
      batch_node_ids_.push_back(static_cast<int32_t>(node_idx + 1));
      pending_paths_[node_idx].push_back(path);
      break;
    }
  }
}

int SearchSession::select_child_action(int node_idx) {
  MctsNode& node = nodes_[node_idx];
  if (node.children.empty())
    throw std::runtime_error("select_child_action called on node without children");

  int total_visits = 0;
  for (const auto& pair : node.children) {
    total_visits += nodes_[pair.second].visit_count;
  }

  const float sqrt_total = std::sqrt(static_cast<float>(total_visits + 1));
  float best_score = -std::numeric_limits<float>::infinity();
  std::vector<int> best_actions;

  for (const auto& pair : node.children) {
    const int child_idx = pair.second;
    const int action = pair.first;

    // ── 跳過底下已有 pending leaf 的子節點 ──────────
    // 目的是讓同一棵樹在單次 simulate_into_buffers 呼叫中，
    // 每次 selection 都走向不同的未展開節點，累積 leaf_batch_size 個 leaf
    if (has_pending_descendant(child_idx)) continue;

    const MctsNode& child = nodes_[child_idx];
    const float q_child = child.mean_value();

    float q = q_child;
    if (node.has_player && child.has_player && child.to_play != node.to_play) {
      q = -q_child;
    }

    const float u = config_.puct_c * child.prior * sqrt_total / static_cast<float>(1 + child.visit_count);
    const float score = q + u;

    if (score > best_score + 1e-12f) {
      best_score = score;
      best_actions.clear();
      best_actions.push_back(action);
    } else if (std::fabs(score - best_score) <= 1e-12f) {
      best_actions.push_back(action);
    }
  }

  // ── 所有子節點都有 pending leaf → 無法 selection ──
  if (best_actions.empty()) {
    return -1;
  }

  std::uniform_int_distribution<int> pick(0, static_cast<int>(best_actions.size()) - 1);
  return best_actions[static_cast<std::size_t>(pick(rng_))];
}

// 反向傳播時，若節點玩家與 leaf_to_play 不同，就翻號。
// 這對應文件中「玩家不同 value 要加負號」的規則。
void SearchSession::backup(const std::vector<int>& path, float leaf_value, int leaf_to_play) {
  for (const int idx : path) {
    MctsNode& node = nodes_[idx];
    const int node_player = node.has_player ? node.to_play : leaf_to_play;
    const float value_for_node = (node_player == leaf_to_play) ? leaf_value : -leaf_value;
    node.value_sum += value_for_node;
    node.visit_count += 1;
  }
}

// 當一批 pending eval 都到齊後：
// 1. 先 expand 對應葉節點
// 2. 對所有等待同一節點結果的 path 還原 virtual loss
// 3. 再做真正 backup
void SearchSession::process_pending_evals() {
  for (const int node_idx : pending_node_order_) {
    auto eval_it = pending_eval_map_.find(node_idx);
    if (eval_it == pending_eval_map_.end())
      throw std::runtime_error("missing pending eval while processing leaf batch");

    expand_node(node_idx, eval_it->second.priors);

    const int leaf_to_play = nodes_[node_idx].to_play;
    const float leaf_value = eval_it->second.value;

    auto path_it = pending_paths_.find(node_idx);
    for (const auto& path : path_it->second) {
      // 還原 virtual loss
      for (const int idx : path) {
        nodes_[idx].visit_count -= 1;
        nodes_[idx].value_sum += 1.0f;
      }
      backup(path, leaf_value, leaf_to_play);
    }
  }

  pending_node_order_.clear();
  pending_paths_.clear();
  pending_eval_map_.clear();
}

void SearchSession::expand_node(int node_idx, const std::vector<float>& priors) {
  if (nodes_[node_idx].expanded) return;

  MctsNode& parent = nodes_[node_idx];
  if (parent.children.empty()) {
    parent.expanded = true;
    return;
  }

  float sum_legal = 0.0f;
  for (const auto& kv : parent.children) {
    const int action = kv.first;
    const float p = (action >= 0 && action < static_cast<int>(priors.size()))
                        ? priors[static_cast<std::size_t>(action)]
                        : 0.0f;
    const float safe = std::isfinite(p) && p > 0.0f ? p : 0.0f;
    nodes_[kv.second].prior = safe;
    sum_legal += safe;
  }

  if (!(sum_legal > 0.0f)) {
    const float uniform = 1.0f / static_cast<float>(parent.children.size());
    for (const auto& kv : parent.children) {
      nodes_[kv.second].prior = uniform;
    }
  } else {
    for (const auto& kv : parent.children) {
      nodes_[kv.second].prior /= sum_legal;
    }
  }

  parent.expanded = true;

  if (node_idx == root_node_index_ && config_.add_root_dirichlet_noise && !root_noise_applied_) {
    apply_root_dirichlet_noise();
    root_noise_applied_ = true;
  }
}

// 遞迴檢查 node_idx 底下是否有 pending leaf。
// 用於 simulate_into_buffers 中的 selection：避開已有 pending leaf 的子樹，
// 讓每次 selection 都走向新的未展開節點。
bool SearchSession::has_pending_descendant(int node_idx) const {
  // 如果這個節點本身就在 pending 中（未展開的葉節點）
  if (pending_paths_.find(node_idx) != pending_paths_.end()) return true;

  const MctsNode& node = nodes_[node_idx];
  if (!node.expanded) {
    // 未展開且不在 pending 中 → 沒有 pending descendant
    return false;
  }

  // 已展開：遞迴檢查所有子節點
  for (const auto& pair : node.children) {
    if (has_pending_descendant(pair.second)) return true;
  }
  return false;
}

void SearchSession::apply_root_dirichlet_noise() {
  MctsNode& root = nodes_[root_node_index_];
  if (root.children.empty()) return;

  const float alpha = config_.root_dirichlet_alpha;
  if (!(alpha > 0.0f)) return;

  std::vector<int> actions;
  actions.reserve(root.children.size());
  for (const auto& pair : root.children) {
    actions.push_back(pair.first);
  }

  std::gamma_distribution<float> gamma(alpha, 1.0f);
  std::vector<float> noise(actions.size(), 0.0f);
  float noise_sum = 0.0f;
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const float sample = gamma(rng_);
    noise[i] = sample;
    noise_sum += sample;
  }
  if (!(noise_sum > 0.0f)) return;

  const float eps = std::clamp(config_.root_dirichlet_eps, 0.0f, 1.0f);
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const int child_idx = root.children[actions[i]];
    MctsNode& child = nodes_[child_idx];
    const float dir = noise[i] / noise_sum;
    child.prior = (1.0f - eps) * child.prior + eps * dir;
  }
}

}  // namespace tzaar
