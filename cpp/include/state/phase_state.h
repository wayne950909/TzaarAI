#ifndef TZAAR_STATE_PHASE_STATE_H_
#define TZAAR_STATE_PHASE_STATE_H_

#include "core/constants.h"
#include "core/game.h"
#include "features/cnn_builder.h"

#include <string>
#include <vector>

namespace tzaar {

class PhaseGameState {
 public:
  PhaseGameState();

  // ─── 唯讀查詢 ─────────────────────────────────────────
  bool is_done() const;
  int current_player() const;
  int winner() const;
  std::string phase() const;
  int turn_number() const;
  const TzaarGame& game() const { return game_; }

  // ─── 合法動作遮罩 ─────────────────────────────────────
  std::vector<bool> legal_mask();

  // ─── 動作執行 ─────────────────────────────────────────
  void apply_action(int action_idx);
  void apply_action_trusted(int action_idx);  // internal fast path

  // ─── 複製 ─────────────────────────────────────────────
  PhaseGameState clone() const;

  // ─── Python 綁定用輔助 ───────────────────────────────
  LeafSnapshot leaf_snapshot(int node_id);
  void fill_snapshot_metadata(LeafSnapshot& snapshot, int node_id);

  // ─── Python cpp_adapter 用輔助 ───────────────────────
  // 回傳 TzaarGame 的 piece_counts；pybind11 綁定在 module.cpp 中轉 dict
  const TzaarGame& game_ref() const { return game_; }
 private:
  TzaarGame game_;
  Stage stage_;
  bool cache_valid_ = false;
  std::vector<bool> cached_mask_;

  void invalidate_cache();
  void advance_from_first_src();
  void fill_step1_mask(std::vector<bool>& mask) const;
  void fill_step2_mask(std::vector<bool>& mask) const;
  Move reconstruct_unified_action(int action_idx, bool is_step1) const;
};

}  // namespace tzaar

#endif  // TZAAR_STATE_PHASE_STATE_H_


