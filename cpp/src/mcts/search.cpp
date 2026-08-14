#include "mcts/search.h"
#include "core/action.h"
#include "mcts/debug_log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

// ─── NVTX Profiling ────────────────────────────────────
#ifdef TZAAR_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#else
// No-op stubs when NVTX not available
#define nvtxRangePushA(x)
#define nvtxRangePop()
#endif

// ─── C++ Debug Log ─────────────────────────────────────
// 具備「關閉即零成本」的可開關 logger（debug_log.h 提供）。
// 關閉時呼叫端只做一次 relaxed atomic load（likely_disabled）即跳過，
// 不格式化、不鎖、不寫檔，因此完全不打擾效能。
#define TZAAR_DEBUG_LOG(...)                                     \
  do {                                                           \
    if (!DebugLogger::instance().likely_disabled())              \
      DebugLogger::instance().log(__VA_ARGS__);                  \
  } while (0)

namespace tzaar {
namespace {

// RAII 輔助：進入區塊時 push NVTX range，離開區塊（含 early-exit/return）時自動 pop。
// 用於 simulate_into_buffers 的五階段量測，確保 push/pop 一定成對。
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

  // 零動態分配 Flat Node Pool：一次 reserve 到位，之後不再 reallocation。
  // reset() 只重設內容與狀態，不再觸發 reserve/resize。
  if (next_free_node_idx_ <= root_node_index_) next_free_node_idx_ = 1;
  nodes_.reserve(static_cast<std::size_t>(kMaxNodesPerTree));
  nodes_.resize(static_cast<std::size_t>(kMaxNodesPerTree));

  reset(root_state, config_);
}

// ══════════════════════════════════════════════════════════════════════
// 重用（重置）
// ══════════════════════════════════════════════════════════════════════

// 重用既有 session 進行新一輪搜尋，保留節點池記憶體（nodes_ 不釋放/不重建）。
// 安全性保證：
//   - root（slot 0）在此手動完整重設所有欄位。
//   - root 以外的槽位：每次被當作子節點 allocate 時，經由 reset_node() 覆蓋舊值。
//   - 未在下一次搜尋中 allocate 的槽位（殘留資料）不會被任何搜尋途徑觸及。
// rng_ 刻意保留，跨搜尋延續隨機序列（少一次 random_device 開銷）。
void SearchSession::reset(const PhaseGameState& root_state, SearchConfig config) {
  if (config.simulations < 0) throw std::invalid_argument("simulations must be >= 0");
  if (config.leaf_batch_size <= 0) throw std::invalid_argument("leaf_batch_size must be >= 1");
  if (config.puct_c <= 0.0f) throw std::invalid_argument("puct_c must be > 0");

  config_ = std::move(config);
  root_state_ = root_state.clone();

  // 節點池：只把分配指標指回 root 之後，不 memset 整池。
  // 由 reset_node() 在每次 allocate 時覆蓋舊值。
  next_free_node_idx_ = 1;

  // 根節點 slot 0：手動完整重設。
  MctsNode& root = nodes_[root_node_index_];
  reset_node(root);
  root.prior = 1.0f;
  root.parent_idx = -1;
  root.action_from_parent = -1;
  root.to_play = root_state_.current_player();
  root.is_terminal = root_state_.is_done();
  root.winner = root_state_.winner();

  // 根節點 CNN 特徵快取（尺寸固定，assign 更新內容）。
  root_board_flat_.assign(static_cast<std::size_t>(kBoardFlatSize), 0.0f);
  root_global_feat_.assign(static_cast<std::size_t>(kGlobalFeatureDim), 0.0f);
  if (!root_state_.is_done()) {
    const auto& board = root_state_.game().board();
    const auto counts = root_state_.game().piece_counts();
    build_cnn_features_into(
        board, root_state_.current_player(), root_state_.turn_number(),
        root_state_.phase(), counts,
        root_board_flat_.data(), root_global_feat_.data());
  }

  // 清空搜尋期間的暫態狀態。
  simulations_processed_ = 0;
  root_noise_applied_ = false;
  pending_node_order_.clear();
  pending_paths_.clear();
  pending_eval_map_.clear();
  batch_board_flat_.clear();
  batch_global_feat_.clear();
  batch_legal_mask_.clear();
  batch_node_ids_.clear();
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
  if (node_idx < 0 || node_idx >= next_free_node_idx_)
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

        if (node_idx < 0 || node_idx >= next_free_node_idx_)
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

    // 根節點 legal mask：臨時產生，不長期佔用節點空間。
  {
    PhaseGameState tmp = root_state_.clone();
    const std::vector<bool> m = tmp.legal_mask();
    result.legal_mask.assign(m.begin(), m.end());
  }

  result.root_policy.assign(static_cast<std::size_t>(kActionCount), 0.0f);
  result.root_visits.assign(static_cast<std::size_t>(kActionCount), 0.0f);

  const MctsNode& root = nodes_[root_node_index_];
  const int base = root.children_base_idx;
  const int cnt = root.children_count;
  for (int i = 0; i < cnt; ++i) {
    const MctsNode& child = nodes_[base + i];
    const int action = child.action_from_parent;
    if (action < 0 || action >= kActionCount) continue;
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
// 注意：回傳值可能少於 chunk，因為：
//   - 終端節點消耗模擬次數但不產出 leaf（backup 已內處理）
//   - 所有路徑都被 pending leaf 阻塞時（stall）提前結束
int SearchSession::simulate_into_buffers(int chunk,
                                                                                  float* board_out,
                                         float* global_out,
                                         uint8_t* mask_out,
                                         int32_t* node_ids_out,
                                         int32_t* tree_ids_out,
                                         int tree_id) {
  if (chunk <= 0) return 0;

  // 整個函式層級的 NVTX range（RAII）。早期 return（stall）時由析構自動 pop。
  NvtxRangeGuard sim_guard("simulate_into_buffers");

  int simulated_count = 0;
  const int max_attempts = chunk + 400;  // 預留給終端節點的重試空間

  for (int attempts = 0; attempts < max_attempts; ++attempts) {
    if (simulated_count >= chunk) break;
    if (simulations_processed_ >= config_.simulations) break;

    // ── 段1：樹狀走訪與選擇（walk + 簿記，範圍涵蓋整個迭代）──
    // RAII guard 確保即使 terminal / stall / early-exit 也會成對 pop。
    NvtxRangeGuard stage1("sib_stage1_selection");

    int node_idx = root_node_index_;
    std::vector<int> path;
    path.reserve(128);
        path.push_back(node_idx);

    while (true) {
      MctsNode& node = nodes_[node_idx];

      // 情況 A：終端節點 → backup，不產 leaf，跳出 while 讓外層重試
      if (node.is_terminal) {
        const float leaf_value = terminal_value_for_current_player(node.winner, node.to_play);
        backup(path, leaf_value, node.to_play);
        simulations_processed_ += 1;
        break;
      }

            // 情況 B：已展開節點 → selection 繼續往下
      if (node.expanded) {
        if (node.children_count == 0) {
          // 已展開但無合法子節點（可能因為遊戲結束判斷不同步）
          backup(path, 0.0f, node.to_play);
          simulations_processed_ += 1;
          break;
        }
        const int child_idx = select_child_action(node_idx);
        // select_child_action 回傳 -1 代表所有子節點都有 pending leaf
        if (child_idx < 0) {
          // 整棵樹已 stalled → 無法再產生新 leaf，回傳目前已累積的
          return simulated_count;  // sib_stage1_selection 由 guard 析構時 pop
        }
        node_idx = child_idx;
        path.push_back(node_idx);
        continue;  // 仍在走訪，段1持續計時
      }

                              // 情況 C：未展開葉節點
      // 走訪完成，段1到此（仍在外層 sib_stage1_selection 範圍內）

      // ── 段3：遊戲狀態複製與動作套用 ──────────────
      PhaseGameState state;
      {
        NvtxRangeGuard stage3("sib_stage3_state_clone");
        state = reconstruct_state_for_node(node_idx);
        node.to_play = state.current_player();
        node.is_terminal = state.is_done();
        node.winner = state.winner();

        if (node.is_terminal) {
          node.expanded = true;
          const float leaf_value = terminal_value_for_current_player(node.winner, node.to_play);
          backup(path, leaf_value, node.to_play);
          simulations_processed_ += 1;
          break;  // 跳出 while，外層 sib_stage1_selection 在此迭代結束時一併 pop
        }
      }  // sib_stage3_state_clone pop

            // ── 生成本節點合法動作遮罩（計數用；真正的寫入在段5）──
      // 診斷：若 node.is_terminal 為 false（搜尋認為局面未結束），
      // 但合法動作數量為 0（局面實際上無步可下），代表「終局判斷漏掉」
      // 或「重建出的局面與實際遊戲不一致」——這正是可能造成整棵樹
      // 卡住（無法模擬、剩餘模擬次數不減）的關鍵原因。
      {
        // 在呼叫 legal_mask()「之前」先記錄原始 phase/player，避免
        // legal_mask() 內部把局勢判定為終局（並把 stage 改為 Done）干擾診斷資訊。
        const std::string dbg_phase_before = state.phase();
        const int dbg_player = state.current_player();
        const int dbg_turn = state.turn_number();
        const int dbg_winner = state.winner();
        const int dbg_done_node = node.is_terminal ? 1 : 0;

        const std::vector<bool> dbg_legal = state.legal_mask();
        int dbg_legal_count = 0;
        for (std::size_t j = 0; j < dbg_legal.size(); ++j) {
          if (dbg_legal[j]) dbg_legal_count++;
        }
        if (dbg_legal_count == 0) {
          TZAAR_DEBUG_LOG(
              "[mcts/simulate] TREE %d | node_idx=%d | phase_before=%s | "
              "current_player=%d | turn=%d | node.is_terminal=%d | "
              "winner=%d | NO-LEGAL-MOVES=%d !!! "
              "(搜尋判定未終局，但重建局面無合法步子 -> 疑似終局判斷漏判或重建不一致)",
              tree_id, node_idx, dbg_phase_before.c_str(),
              dbg_player, dbg_turn, dbg_done_node, dbg_winner, 1);
        }
      }

      // ── 段4：特徵序列化與寫入緩衝區（build_cnn_features_into）──
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

            // ── 段5：合法動作遮罩生成與寫入（state.legal_mask()）──
      // 遮罩僅寫入輸出 buffer（SearchManager 跨執行緒輸出），不存入節點。
      const std::vector<bool> legal = state.legal_mask();
      {
        NvtxRangeGuard stage5("sib_stage5_legal_mask");
        const std::size_t mask_offset = static_cast<std::size_t>(simulated_count) * static_cast<std::size_t>(kActionCount);
        for (std::size_t j = 0; j < legal.size() && j < static_cast<std::size_t>(kActionCount); ++j) {
          const uint8_t v = legal[j] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
          mask_out[mask_offset + j] = v;
        }
      }  // sib_stage5_legal_mask pop

      // ── 段2：節點記憶體配置（Flat Node Pool 連續區塊）──
      // 在已 reserve 的 nodes_ 陣列上直接分配，不做 push_back / reallocation。
      {
        NvtxRangeGuard stage2("sib_stage2_node_alloc");
                MctsNode& parent = nodes_[node_idx];
        if (next_free_node_idx_ + kActionCount >= kMaxNodesPerTree)
          throw std::overflow_error("MCTS node pool exhausted (kMaxNodesPerTree)");
                parent.children_base_idx = next_free_node_idx_;
        parent.children_count = 0;
        for (int action = 0; action < kActionCount; ++action) {
          if (!legal[static_cast<std::size_t>(action)]) continue;
          MctsNode& child = nodes_[next_free_node_idx_++];
          reset_node(child);      // 手動整格覆蓋舊 value，avoid MctsNode{} 臨時物件
          child.prior = 0.0f;     // expand 時由 NN 輸出設定
          child.parent_idx = node_idx;
          child.action_from_parent = action;
          parent.children_count++;
        }
        parent.expanded = true;
      }  // sib_stage2_node_alloc pop

      // ── Virtual loss + 記錄 pending（併入段1 sib_stage1_selection）──
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
    }  // 迭代結束，sib_stage1_selection 由 guard 析構時 pop
  }

  // 診斷：這次呼叫完全沒有產生任何 leaf。這可能是本次 tree 被「丟掉」的徵兆。
  //   - 若本樹其實還沒 complete，卻回傳 0 → 需要追查是 is_complete() 提前成立，
  //     還是 selection 全部被 pending/無合法子節點卡住。
  if (simulated_count == 0) {
    TZAAR_DEBUG_LOG(
        "[mcts/simulate] TREE %d | simulate_into_buffers ch=%d RETURNED 0 "
        "(proc=%d/%d | pending=%d) -> 此樹可能被丟掉/卡住",
        tree_id, chunk, simulations_processed_, config_.simulations,
        static_cast<int>(pending_node_order_.size()));
  }

  return simulated_count;
}

bool SearchSession::submit_single_eval(int node_id,
                                       const float* priors,
                                       float value) {
  const int node_idx = node_id - 1;
  if (node_idx < 0 || node_idx >= next_free_node_idx_)
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
    return true;
  }
  return false;
}

// ══════════════════════════════════════════════════════════════════════
// 內部方法
// ══════════════════════════════════════════════════════════════════════

// 只記錄 parent/action，而不是在每個節點存完整 state。
// 這樣可大量降低樹的記憶體成本。

// 將節點池中單一 slot 完整重設為初始空值。
// 手動逐欄賦值（不使用 MctsNode{} 臨時物件），避免不必要的建構/拷貝成本。
// 每個欄位都必須明確重設，否則會殘留上一次 search 的舊值
// （尤其 children_base_idx / children_count / expanded / visit_count /
//  value_sum / is_terminal / to_play 等會影響路徑決策的欄位）。
void SearchSession::reset_node(MctsNode& node) {
  node.prior = 0.0f;
  node.parent_idx = -1;
  node.action_from_parent = -1;
  node.children_base_idx = -1;
  node.children_count = 0;
  node.visit_count = 0;
  node.value_sum = 0.0f;
  node.expanded = false;
  node.to_play = 0;
  node.winner = 0;
  node.is_terminal = false;
}

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
}

std::vector<bool> SearchSession::get_legal_mask_for_node(int node_idx, PhaseGameState& state) {
  (void)node_idx;
  return state.legal_mask();
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
        if (node.children_count == 0) {
          backup(path, 0.0f, node.to_play);
          break;
        }
        const int child_idx = select_child_action(node_idx);
        node_idx = child_idx;
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

      if (node.is_terminal) {
        node.expanded = true;
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

            // 填寫合法遮罩到批次緩衝區（不存入節點）
      const std::vector<bool> legal = state.legal_mask();
      const std::size_t offset_mask = batch_legal_mask_.size();
      batch_legal_mask_.resize(offset_mask + static_cast<std::size_t>(kActionCount), 0);
      for (std::size_t j = 0; j < legal.size() && j < static_cast<std::size_t>(kActionCount); ++j) {
        const uint8_t v = legal[j] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
        batch_legal_mask_[offset_mask + j] = v;
      }

      // 建立子節點樁（Flat Node Pool 連續區塊，僅記錄 parent+action，不 clone 狀態）
      if (next_free_node_idx_ + kActionCount >= kMaxNodesPerTree)
        throw std::overflow_error("MCTS node pool exhausted (kMaxNodesPerTree)");
            node.children_base_idx = next_free_node_idx_;
      node.children_count = 0;
      for (int action = 0; action < kActionCount; ++action) {
        if (!legal[static_cast<std::size_t>(action)]) continue;
        MctsNode& child = nodes_[next_free_node_idx_++];
        reset_node(child);      // 手動整格覆蓋舊 value，avoid MctsNode{} 臨時物件
        child.prior = 0.0f;     // expand 時由 NN 輸出設定
        child.parent_idx = node_idx;
        child.action_from_parent = action;
        node.children_count++;
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
  if (node.children_count == 0)
    throw std::runtime_error("select_child_action called on node without children");

  const int base = node.children_base_idx;
  const int cnt = node.children_count;

  int total_visits = 0;
  for (int i = 0; i < cnt; ++i) {
    total_visits += nodes_[base + i].visit_count;
  }

  const float sqrt_total = std::sqrt(static_cast<float>(total_visits + 1));
  float best_score = -std::numeric_limits<float>::infinity();
  // 用 on-stack 固定陣列保存最佳候選的 slot 偏移，避免 heap 配置。
  int best_slots[64];          // 上限 64 個候選（kActionCount 遠小於此）
  int best_count = 0;

  for (int i = 0; i < cnt; ++i) {
    const int child_idx = base + i;

    // ── 跳過底下已有 pending leaf 的子節點 ──────────
    if (has_pending_descendant(child_idx)) continue;

    const MctsNode& child = nodes_[child_idx];
    const float q_child = child.mean_value();

    // 玩家翻號：selection 中 node 與 child 皆為非終端（有 to_play）
    const float q = (child.to_play != node.to_play) ? -q_child : q_child;

    const float u = config_.puct_c * child.prior * sqrt_total / static_cast<float>(1 + child.visit_count);
    const float score = q + u;

    if (score > best_score + 1e-12f) {
      best_score = score;
      best_count = 1;
      best_slots[0] = i;
    } else if (std::fabs(score - best_score) <= 1e-12f) {
      if (best_count < 64) best_slots[best_count++] = i;
    }
  }

  // ── 所有子節點都有 pending leaf → 無法 selection ──
  if (best_count == 0) {
    return -1;
  }

  std::uniform_int_distribution<int> pick(0, best_count - 1);
  const int slot = best_slots[pick(rng_)];
  return base + slot;
}

// 反向傳播時，若節點玩家與 leaf_to_play 不同，就翻號。
// 這對應文件中「玩家不同 value 要加負號」的規則。
void SearchSession::backup(const std::vector<int>& path, float leaf_value, int leaf_to_play) {
  for (const int idx : path) {
    MctsNode& node = nodes_[idx];
    // 終端節點無 to_play，視為與 leaf 同玩家；否則比對玩家。
    const int node_player = node.is_terminal ? leaf_to_play : node.to_play;
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
  if (parent.children_count == 0) {
    parent.expanded = true;
    return;
  }

  const int base = parent.children_base_idx;
  const int cnt = parent.children_count;
  const int priors_size = static_cast<int>(priors.size());

  float sum_legal = 0.0f;
  for (int i = 0; i < cnt; ++i) {
    MctsNode& child = nodes_[base + i];
    const int action = child.action_from_parent;
    const float p = (action >= 0 && action < priors_size)
                        ? priors[static_cast<std::size_t>(action)]
                        : 0.0f;
    const float safe = std::isfinite(p) && p > 0.0f ? p : 0.0f;
    child.prior = safe;
    sum_legal += safe;
  }

  if (!(sum_legal > 0.0f)) {
    const float uniform = 1.0f / static_cast<float>(cnt);
    for (int i = 0; i < cnt; ++i) {
      nodes_[base + i].prior = uniform;
    }
  } else {
    for (int i = 0; i < cnt; ++i) {
      nodes_[base + i].prior /= sum_legal;
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

    // 已展開：遞迴檢查所有子節點（連續區塊）
  const int base = node.children_base_idx;
  const int cnt = node.children_count;
  for (int i = 0; i < cnt; ++i) {
    if (has_pending_descendant(base + i)) return true;
  }
  return false;
}

void SearchSession::apply_root_dirichlet_noise() {
  MctsNode& root = nodes_[root_node_index_];
  const int root_base = root.children_base_idx;
  const int root_cnt = root.children_count;
  if (root_cnt == 0) return;

  const float alpha = config_.root_dirichlet_alpha;
  if (!(alpha > 0.0f)) return;

    std::gamma_distribution<float> gamma(alpha, 1.0f);
  // on-stack 陣列存 Dirichlet noise，避免 heap 配置。
  // 大小 >= kActionCount，涵蓋任何合法子節點數量上限。
  float noise[kActionCount];
  float noise_sum = 0.0f;
  for (int i = 0; i < root_cnt; ++i) {
    const float sample = gamma(rng_);
    noise[i] = sample;
    noise_sum += sample;
  }
  if (!(noise_sum > 0.0f)) return;

  const float eps = std::clamp(config_.root_dirichlet_eps, 0.0f, 1.0f);
  for (int i = 0; i < root_cnt; ++i) {
    MctsNode& child = nodes_[root_base + i];
    const float dir = noise[i] / noise_sum;
    child.prior = (1.0f - eps) * child.prior + eps * dir;
  }
}

}  // namespace tzaar
