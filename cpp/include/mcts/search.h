#ifndef TZAAR_MCTS_SEARCH_H_
#define TZAAR_MCTS_SEARCH_H_

#include "core/constants.h"
#include "state/phase_state.h"
#include "mcts/config.h"
#include "features/cnn_builder.h"

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace tzaar {

class SearchSession {
 public:
  SearchSession(const PhaseGameState& root_state, SearchConfig config);

  // 重用既有 session（保留節點池記憶體，僅重設狀態）。
  // 供 SearchManager 在每批搜尋間重複利用同一棵樹的 session，
  // 避免每次重新 reserve/resize 節點池（kMaxNodesPerTree）。
  void reset(const PhaseGameState& root_state, SearchConfig config);

  // ─── 唯讀查詢 ─────────────────────────────────────────
  SearchConfig config() const { return config_; }
  bool has_pending_leaves() const { return !pending_node_order_.empty(); }
  bool is_complete() const;
  int simulations_processed() const { return simulations_processed_; }
  int simulations_requested() const { return config_.simulations; }

  // ─── 葉節點收集（舊 API） ─────────────────────────────
  std::vector<LeafSnapshot> collect_pending_leaves(int max_batch);

  // ─── 葉節點收集（packed API：零拷貝批次緩衝區） ──────
  // 回傳指針到內部緩衝區，呼叫端應在下次 simulate_chunk 前複製
  struct PackedLeaves {
    int batch_size = 0;
    const int32_t* node_ids = nullptr;
    const uint8_t* legal_masks = nullptr;   // [N x kActionCount]
    const float* board_state_flat = nullptr; // [N x kBoardFlatSize]
    const float* global_features = nullptr;  // [N x kGlobalFeatureDim]
  };
  PackedLeaves collect_pending_leaves_packed(int max_batch);

  // ─── 提交 NN 評估結果 ───────────────────────────────
  void submit_leaf_eval(int node_id, const std::vector<float>& priors, float value);
  void submit_leaf_eval_batch(
      const int32_t* node_ids,
      const float* priors,
      const float* values,
      int batch_size);

  // ─── 完成搜尋 ─────────────────────────────────────────
  SearchResult finish();

  // ─── 根節點快照 ───────────────────────────────────────
  LeafSnapshot root_snapshot();

  // ─── 供 SearchManager 使用的內部批次模擬 ─────────────
  // 模擬 chunk 次，將結果寫入指定的外部 buffer
  // board_out / global_out / mask_out / node_ids_out 是外部預分配的連續記憶體
  // tree_id 會寫入 tree_ids_out（如果非 null）
  // 回傳實際模擬的葉節點數量
    // path_buffer：呼叫端（SearchManager 的執行緒）提供的重用容器。
  // 執行緒可預先 reserve 容量並跨多次呼叫/多次 run_search 重用，
  // 避免 simulate_into_buffers 每次迭代重新配置 path（heap 熱點）。
  int simulate_into_buffers(int chunk,
                            float* board_out,
                            float* global_out,
                            uint8_t* mask_out,
                            int32_t* node_ids_out,
                            int32_t* tree_ids_out,
                            int tree_id,
                            std::vector<int>& path_buffer);

  // 提交單一節點的評估結果（供 SearchManager 使用）
  // 與 submit_leaf_eval 不同，不會觸發 process_pending_evals
  // 而是將結果暫存，等所有 pending 都到齊後自動處理
  bool submit_single_eval(int node_id,
                          const float* priors,
                          float value);

 private:
  struct PendingEval {
    std::vector<float> priors;
    float value = 0.0f;
  };

  struct MctsNode {
    float prior = 0.0f;
        int parent_idx = -1;
    int action_from_parent = -1;
    // ─── 子節點連續區塊（Flat Node Pool 的隱含合法性） ──
    // 合法子節點落在 [children_base_idx, children_base_idx + children_count)
    int children_base_idx = -1;
    int children_count = 0;

    int visit_count = 0;
    float value_sum = 0.0f;
    bool expanded = false;

    // ─── 標量狀態欄位（保留：熱路徑終端檢查與玩家翻號 O(1)） ──
    int to_play = 0;
    int winner = 0;
    bool is_terminal = false;

    float mean_value() const {
      if (visit_count <= 0) return 0.0f;
      return value_sum / static_cast<float>(visit_count);
    }
  };

  static constexpr int root_node_index_ = 0;
  static constexpr int kMaxNodesPerTree = 120000;   // 節點池容量上限
  SearchConfig config_;
  int root_node_id_ = 1;
  int simulations_processed_ = 0;
  bool root_noise_applied_ = false;

  PhaseGameState root_state_;
  // ─── 零動態分配 Flat Node Pool ─────────────────────
  std::vector<MctsNode> nodes_;       // reserve(kMaxNodesPerTree) 後不再 reallocation
  int next_free_node_idx_ = 1;        // 節點池分配指標（0 = root）

  // ─── Pending leaves 狀態 ──────────────────────────
  std::vector<int> pending_node_order_;
  std::unordered_map<int, std::vector<std::vector<int>>> pending_paths_;
  std::unordered_map<int, PendingEval> pending_eval_map_;

  // ─── 批次緩衝區（供 collect_pending_leaves_packed 使用） ─
  std::vector<float>    batch_board_flat_;
  std::vector<float>    batch_global_feat_;
  std::vector<uint8_t>  batch_legal_mask_;
  std::vector<int32_t>  batch_node_ids_;

  // ─── 根節點 CNN 特徵快取 ──────────────────────────
  std::vector<float> root_board_flat_;
  std::vector<float> root_global_feat_;

  std::mt19937 rng_;

  // ─── 內部方法 ─────────────────────────────────────────
  // 將節點池中單一 slot 完整重設為初始空值（手動逐欄賦值，避免創建臨時物件）。
  // 用於「重用節點池而不整個 memset」：root 在 reset 時手動重設，
  // 其餘槽位在每次被當作子節點 allocate 時也經由此函式覆蓋舊 search 殘留值。
  void reset_node(MctsNode& node);
  std::vector<int> collect_action_path_to_node(int node_idx) const;
  PhaseGameState reconstruct_state_for_node(int node_idx) const;
  void update_node_metadata_from_state(int node_idx, const PhaseGameState& state);
  std::vector<bool> get_legal_mask_for_node(int node_idx, PhaseGameState& state);
  bool prepare_pending_leaves(int max_batch);
  std::vector<LeafSnapshot> pending_leaf_snapshots();
  LeafSnapshot build_leaf_snapshot(int node_idx);
  void simulate_chunk(int chunk);
  int select_child_action(int node_idx);
  void backup(const std::vector<int>& path, float leaf_value, int leaf_to_play);
  void process_pending_evals();
  void expand_node(int node_idx, const std::vector<float>& priors);
  void apply_root_dirichlet_noise();
  
  // ─── 供 simulate_into_buffers 使用的輔助方法 ──────
  // 檢查子節點底下是否有 pending leaf（遞迴檢查）
  bool has_pending_descendant(int node_idx) const;
};

}  // namespace tzaar

#endif  // TZAAR_MCTS_SEARCH_H_

