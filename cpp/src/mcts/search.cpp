#include "mcts/search.h"
#include "core/action.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

// ??? NVTX Profiling ????????????????????????????????????
#ifdef TZAAR_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#else
// No-op stubs when NVTX not available
#define nvtxRangePushA(x)
#define nvtxRangePop()
#endif

namespace tzaar {
namespace {

// RAII 頛嚗脣?憛? push NVTX range嚗??憛???early-exit/return嚗??芸? pop??
// ?冽 simulate_into_buffers ???挾?葫嚗Ⅱ靽?push/pop 銝摰?撠?
struct NvtxRangeGuard {
  explicit NvtxRangeGuard(const char* name) {
#ifdef TZAAR_USE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~NvtxRangeGuard() {
#ifdef TZAAR_USE_NVTX
    nvtxRangePop();
#endif
  }
  NvtxRangeGuard(const NvtxRangeGuard&) = delete;
  NvtxRangeGuard& operator=(const NvtxRangeGuard&) = delete;
};

}  // namespace

// ??????????????????????????????????????????????????????????????????????
// 撱箸?摮?
// ??????????????????????????????????????????????????????????????????????

// SearchSession 隞?”?璉菜?撠邦??
// 摰頨思?蝣啣??瑁?蝺矽摨佗??芸?瘜典嚗?
// - 蝭暺??
// - ??暺???撱?
// - pending eval 蝞∠?
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

  // ?寧?暺?CNN ?孵噩敹怠???
  // 銋????蝭暺?state ?賢敺ㄐ?箇嚗窒??摨??遣??
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

// ??????????????????????????????????????????????????????????????????????
// ?祇? API
// ??????????????????????????????????????????????????????????????????????

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
  result.node_count = static_cast<int>(nodes_.size());  // ??蝯?敺璉菜邦??暺嚗?寧?暺?
  result.root_value = nodes_[root_node_index_].mean_value();

  MctsNode& root_node = nodes_[root_node_index_];
  if (!root_node.legal_mask_ready) {
    PhaseGameState tmp = root_state_.clone();
    get_legal_mask_for_node(root_node_index_, tmp);
  }

  result.legal_mask.assign(root_node.cached_legal_mask.begin(),
                           root_node.cached_legal_mask.end());

  // Compute the maximum legal-move count over all nodes in this tree.
  {
    int max_legal_moves = 0;
    for (const MctsNode& nd : nodes_) {
      if (!nd.legal_mask_ready) continue;
      int cnt = 0;
      for (std::uint8_t v : nd.cached_legal_mask) {
        if (v != 0) ++cnt;
      }
      if (cnt > max_legal_moves) max_legal_moves = cnt;
    }
    result.max_node_legal_moves = max_legal_moves;
  }


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

// ??????????????????????????????????????????????????????????????????????
// 靘?SearchManager 雿輻?瘜?
// ??????????????????????????????????????????????????????????????????????

// ?蝯?SearchManager 雿輻??蔭?寞活璅⊥???Ｕ?
// 摰????刻???暺?亙神?脣???buffer嚗??臬?撱箇? Python ?拐辣??
// 瘜冽?嚗??喳澆?賢???chunk嚗??綽?
//   - 蝯垢蝭暺??芋?祆活?訾?銝??leaf嚗ackup 撌脣??嚗?
//   - ??楝敺鋡?pending leaf ?餃???stall嚗?????
int SearchSession::simulate_into_buffers(int chunk,
                                                                                  float* board_out,
                                         float* global_out,
                                         uint8_t* mask_out,
                                         int32_t* node_ids_out,
                                         int32_t* tree_ids_out,
                                         int tree_id) {
  if (chunk <= 0) return 0;

  // ?游撘惜蝝? NVTX range嚗AII嚗??return嚗tall嚗??望?瑽??pop??
  NvtxRangeGuard sim_guard("simulate_into_buffers");

  int simulated_count = 0;
  const int max_attempts = chunk + 256;  // ??蝯衣?蝡舐?暺??岫蝛粹?

  for (int attempts = 0; attempts < max_attempts; ++attempts) {
    if (simulated_count >= chunk) break;
    if (is_complete()) break;

    // ?? 畾?嚗邦?韏啗赤???walk + 蝪輯?嚗??項??翮隞????
    // RAII guard 蝣箔??喃蝙 terminal / stall / early-exit 銋??? pop??
    NvtxRangeGuard stage1("sib_stage1_selection");

    int node_idx = root_node_index_;
    std::vector<int> path;
    path.reserve(128);
        path.push_back(node_idx);

    while (true) {
      MctsNode& node = nodes_[node_idx];

      // ?? A嚗?蝡舐?暺???backup嚗???leaf嚗歲??while 霈?撅日?閰?
      if (node.is_terminal) {
        const float leaf_value = terminal_value_for_current_player(node.winner, node.to_play);
        backup(path, leaf_value, node.to_play);
        simulations_processed_ += 1;
        break;
      }

      // ?? B嚗歇撅?蝭暺???selection 蝜潛?敺銝?
      if (node.expanded) {
        if (node.children.empty()) {
          // 撌脣????∪?瘜?蝭暺??航??蝯??斗銝?甇伐?
          backup(path, 0.0f, node.to_play);
          simulations_processed_ += 1;
          break;
        }
        const int action = select_child_action(node_idx);
        // select_child_action ? -1 隞?”???蝭暺??pending leaf
        if (action < 0) {
          // ?湔ㄤ璅孵歇 stalled ???⊥??? leaf嚗??喟?歇蝝舐???
          return simulated_count;  // sib_stage1_selection ??guard ????pop
        }
        node_idx = node.children.at(action);
        path.push_back(node_idx);
        continue;  // 隞韏啗赤嚗挾1??閮?
      }

                              // ?? C嚗撅???暺?
      // 韏啗赤摰?嚗挾1?唳迨嚗??典?撅?sib_stage1_selection 蝭??改?

      // ?? 畾?嚗??脩???鋆質???憟 ??????????????
      PhaseGameState state;
      {
        NvtxRangeGuard stage3("sib_stage3_state_clone");
        state = reconstruct_state_for_node(node_idx);
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
          break;  // 頝喳 while嚗?撅?sib_stage1_selection ?冽迨餈凋誨蝯???雿?pop
        }
      }  // sib_stage3_state_clone pop

      // ?? 畾?嚗敺萄????神?亦楨銵?嚗uild_cnn_features_into嚗??
      {
        NvtxRangeGuard stage4("sib_stage4_feature_write");
        const auto counts = state.game().piece_counts();
        const std::size_t offset = static_cast<std::size_t>(simulated_count);
        build_cnn_features_into(
            state.game().board(), state.current_player(), state.turn_number(),
            state.phase(), counts,
            board_out + (offset * static_cast<std::size_t>(kBoardFlatSize)),
            global_out + (offset * static_cast<std::size_t>(kGlobalFeatureDim)));
      }  // sib_stage4_feature_write pop

      // ?? 畾?嚗?瘜?雿蝵拍???撖怠嚗tate.legal_mask()嚗??
      const std::vector<bool> legal = state.legal_mask();
      {
        NvtxRangeGuard stage5("sib_stage5_legal_mask");
        node.cached_legal_mask.resize(static_cast<std::size_t>(kActionCount), 0);
        const std::size_t mask_offset = static_cast<std::size_t>(simulated_count) * static_cast<std::size_t>(kActionCount);
        for (std::size_t j = 0; j < legal.size() && j < static_cast<std::size_t>(kActionCount); ++j) {
          const uint8_t v = legal[j] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
          node.cached_legal_mask[j] = v;
          mask_out[mask_offset + j] = v;
        }
        node.legal_mask_ready = true;
      }  // sib_stage5_legal_mask pop

      // ?? 畾?嚗?暺??園??蔭?隞嗅遣瑽???????????????
      // ???征???踹? push_back ??vector reallocation 雿?reference 憭望?
      {
        NvtxRangeGuard stage2("sib_stage2_node_alloc");
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
        nodes_[node_idx].expanded = true;
      }  // sib_stage2_node_alloc pop

      // ?? Virtual loss + 閮? pending嚗蔥?交挾1 sib_stage1_selection嚗??
      for (const int idx : path) {
        nodes_[idx].visit_count += 1;
        nodes_[idx].value_sum -= 1.0f;
      }
      pending_node_order_.push_back(node_idx);
      node_ids_out[simulated_count] = static_cast<int32_t>(node_idx + 1);
            if (tree_ids_out) {
        tree_ids_out[simulated_count] = static_cast<int32_t>(tree_id);
      }
      pending_paths_[node_idx].push_back(path);

      simulated_count++;
      simulations_processed_ += 1;
      break;
    }  // 餈凋誨蝯?嚗ib_stage1_selection ??guard ????pop
  }

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

  // ?嗆???pending ?賢朣??芸???
  if (pending_eval_map_.size() == pending_node_order_.size()) {
    process_pending_evals();
  }
}

// ??????????????????????????????????????????????????????????????????????
// ?折?寞?
// ??????????????????????????????????????????????????????????????????????

// ?芾???parent/action嚗??臬瘥?暺?摰 state??
// ?見?臬之??雿邦???園????
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

// ??root_state_ + action path ?遣隞餅?蝭暺??Ｕ?
// ?迤??mctsLogic.md 銝剜?餈啁????翰??+ ??摨??遣?祕雿?
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

      // ?芸???蝭暺?
      if (pending_paths_.count(node_idx) > 0) {
        // 撌脣雿?銝哨?? virtual loss
        for (const int idx : path) {
          nodes_[idx].visit_count += 1;
          nodes_[idx].value_sum -= 1.0f;
        }
        pending_paths_[node_idx].push_back(path);
        break;
      }

      // 擐活閮芸?嚗?撱箇???甈?
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

      // 撱箸? CNN ?孵噩?圈???寞活蝺抵??
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

      // 敹怠????桃蔗銝血‵?交甈∠楨銵?
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

      // 撱箇?摮?暺?嚗?閮? parent+action嚗? clone ???
      nodes_.reserve(nodes_.size() + static_cast<std::size_t>(kActionCount));
      for (int action = 0; action < kActionCount; ++action) {
        if (!legal[static_cast<std::size_t>(action)]) continue;
        const int new_idx = static_cast<int>(nodes_.size());
        MctsNode child;
        child.prior = 0.0f;  // expand ? NN 頛詨閮剖?
        child.parent_idx = node_idx;
        child.action_from_parent = action;
        nodes_.push_back(std::move(child));
        nodes_[node_idx].children[action] = new_idx;
      }

      // ? NN ?寞活雿? ??? virtual loss
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

    // ?? 頝喲?摨?撌脫? pending leaf ??蝭暺???????????
    // ?桃??航???璉菜邦?典甈?simulate_into_buffers ?澆銝哨?
    // 瘥活 selection ?質粥?????芸???暺?蝝舐? leaf_batch_size ??leaf
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

  // ?? ???蝭暺??pending leaf ???⊥? selection ??
  if (best_actions.empty()) {
    return -1;
  }

  std::uniform_int_distribution<int> pick(0, static_cast<int>(best_actions.size()) - 1);
  return best_actions[static_cast<std::size_t>(pick(rng_))];
}

// ???單???亦?暺摰嗉? leaf_to_play 銝?嚗停蝧餉???
// ????隞嗡葉?摰嗡???value 閬?鞎???閬???
void SearchSession::backup(const std::vector<int>& path, float leaf_value, int leaf_to_play) {
  for (const int idx : path) {
    MctsNode& node = nodes_[idx];
    const int node_player = node.has_player ? node.to_play : leaf_to_play;
    const float value_for_node = (node_player == leaf_to_play) ? leaf_value : -leaf_value;
    node.value_sum += value_for_node;
    node.visit_count += 1;
  }
}

// ?嗡???pending eval ?賢朣?嚗?
// 1. ??expand 撠???暺?
// 2. 撠???敺?銝蝭暺??? path ?? virtual loss
// 3. ???迤 backup
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
      // ?? virtual loss
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

// ?艘瑼Ｘ node_idx 摨??臬??pending leaf??
// ?冽 simulate_into_buffers 銝剔? selection嚗?歇??pending leaf ??璅對?
// 霈?甈?selection ?質粥??撅?蝭暺?
bool SearchSession::has_pending_descendant(int node_idx) const {
  // 憒???暺頨怠停??pending 銝哨??芸?????暺?
  if (pending_paths_.find(node_idx) != pending_paths_.end()) return true;

  const MctsNode& node = nodes_[node_idx];
  if (!node.expanded) {
    // ?芸???銝 pending 銝???瘝? pending descendant
    return false;
  }

  // 撌脣????艘瑼Ｘ???蝭暺?
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
