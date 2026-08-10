#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "core/constants.h"
#include "core/action.h"
#include "state/phase_state.h"
#include "mcts/config.h"
#include "mcts/search.h"
#include "mcts/search_manager.h"
#include "mcts/bench.h"

namespace py = pybind11;
namespace tz = tzaar;

// ══════════════════════════════════════════════════════════════════════
// pybind11 模組：tzaar_cpp
// ══════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(tzaar_cpp, m) {
  m.doc() = "Tzaar C++ rules engine bridge module";

  // ─── 常數 ───────────────────────────────────────────────
  m.attr("N_ACTIONS")       = py::int_(tz::kActionCount);
  m.attr("PASS_ACTION_IDX") = py::int_(tz::kPassActionIdx);
  m.attr("CAPTURE_OFFSET")  = py::int_(tz::kCaptureOffset);
  m.attr("REINFORCE_OFFSET")= py::int_(tz::kReinforceOffset);
  m.attr("N_EDGE_ACTIONS")  = py::int_(tz::kEdgeActionCount);

  // ─── 動作編解碼 ───────────────────────────────────────
  m.def("encode_action_idx", &tz::encode_action_idx,
        py::arg("kind"), py::arg("edge_idx") = py::none(),
        "Encode (kind, edge_idx) into unified action index.");

  m.def("decode_action_idx",
        [](int action_idx) {
          const tz::DecodedAction decoded = tz::decode_action_idx(action_idx);
          py::dict out;
          out["kind"]         = py::str(tz::move_kind_to_string(decoded.kind));
          out["edge_idx"]     = py::int_(decoded.edge_idx);
          out["src_idx"]      = py::int_(decoded.src_idx);
          out["direction_idx"]= py::int_(decoded.direction_idx);
          return out;
        },
        py::arg("action_idx"),
        "Decode unified action index into kind/edge/src/direction.");

  m.def("action_edge_table",
        []() {
          py::list out;
          for (const auto& edge : tz::action_space().edge_table()) {
            out.append(py::make_tuple(edge.first, edge.second));
          }
          return out;
        },
        "Return generated fixed edge table as (src_idx, direction_idx) pairs.");

  // ─── LeafSnapshot ───────────────────────────────────────
  py::class_<tz::LeafSnapshot>(m, "LeafSnapshot")
    .def(py::init<>())
    .def_readwrite("node_id",         &tz::LeafSnapshot::node_id)
    .def_readwrite("current_player",  &tz::LeafSnapshot::current_player)
    .def_readwrite("turn_number",     &tz::LeafSnapshot::turn_number)
    .def_readwrite("winner",          &tz::LeafSnapshot::winner)
    .def_readwrite("is_done",         &tz::LeafSnapshot::is_done)
    .def_readwrite("phase",           &tz::LeafSnapshot::phase)
    .def_readwrite("white_counts",    &tz::LeafSnapshot::white_counts)
    .def_readwrite("black_counts",    &tz::LeafSnapshot::black_counts)
    .def_readwrite("board_state_flat", &tz::LeafSnapshot::board_state_flat)
    .def_readwrite("global_features", &tz::LeafSnapshot::global_features)
    .def_readwrite("legal_mask",      &tz::LeafSnapshot::legal_mask);

  // ─── SearchConfig ─────────────────────────────────────
  py::class_<tz::SearchConfig>(m, "SearchConfig")
    .def(py::init<>())
    .def_readwrite("simulations",              &tz::SearchConfig::simulations)
    .def_readwrite("leaf_batch_size",           &tz::SearchConfig::leaf_batch_size)
    .def_readwrite("puct_c",                   &tz::SearchConfig::puct_c)
        .def_readwrite("add_root_dirichlet_noise", &tz::SearchConfig::add_root_dirichlet_noise)
    .def_readwrite("root_dirichlet_eps",       &tz::SearchConfig::root_dirichlet_eps)
    .def_readwrite("root_dirichlet_alpha",     &tz::SearchConfig::root_dirichlet_alpha)
    .def_readwrite("min_batch_for_swap",       &tz::SearchConfig::min_batch_for_swap)
    .def_readwrite("flush_timeout_ms",         &tz::SearchConfig::flush_timeout_ms)
    .def_readwrite("buffer_capacity_per_tree", &tz::SearchConfig::buffer_capacity_per_tree)
    .def_readwrite("ready_flush_leaves",       &tz::SearchConfig::ready_flush_leaves)
    .def_readwrite("debug_log_enabled",        &tz::SearchConfig::debug_log_enabled)
    .def_readwrite("debug_log_path",           &tz::SearchConfig::debug_log_path);

  // ─── SearchResult ─────────────────────────────────────
  py::class_<tz::SearchResult>(m, "SearchResult")
    .def(py::init<>())
    .def_readwrite("root_node_id",          &tz::SearchResult::root_node_id)
    .def_readwrite("root_player",           &tz::SearchResult::root_player)
    .def_readwrite("winner",                &tz::SearchResult::winner)
    .def_readwrite("is_done",               &tz::SearchResult::is_done)
    .def_readwrite("is_complete",           &tz::SearchResult::is_complete)
    .def_readwrite("needs_root_eval",       &tz::SearchResult::needs_root_eval)
    .def_readwrite("simulations_requested", &tz::SearchResult::simulations_requested)
    .def_readwrite("simulations_processed", &tz::SearchResult::simulations_processed)
    .def_readwrite("pending_leaf_count",    &tz::SearchResult::pending_leaf_count)
    .def_readwrite("node_count",            &tz::SearchResult::node_count)
    .def_readwrite("max_node_legal_moves",  &tz::SearchResult::max_node_legal_moves)
    .def_readwrite("root_value",            &tz::SearchResult::root_value)
    .def_readwrite("legal_mask",            &tz::SearchResult::legal_mask)
    .def_readwrite("root_policy",           &tz::SearchResult::root_policy)
    .def_readwrite("root_visits",           &tz::SearchResult::root_visits);

  // ─── PhaseGameState ────────────────────────────────────
  py::class_<tz::PhaseGameState>(m, "PhaseGameState")
    .def(py::init<>())
    .def("is_done",         &tz::PhaseGameState::is_done)
    .def("current_player",  &tz::PhaseGameState::current_player)
    .def("turn_number",     &tz::PhaseGameState::turn_number)
    .def("winner",          &tz::PhaseGameState::winner)
    .def("phase",           &tz::PhaseGameState::phase)
    .def("leaf_snapshot",   &tz::PhaseGameState::leaf_snapshot, py::arg("node_id") = 0)
    .def("legal_mask",      &tz::PhaseGameState::legal_mask)
    .def("apply_action",    &tz::PhaseGameState::apply_action, py::arg("action_idx"))
    .def("clone",           &tz::PhaseGameState::clone)
    .def("piece_counts",
         [](const tz::PhaseGameState& self) -> py::dict {
           const auto& game = self.game_ref();
           const auto counts = game.piece_counts();
           py::dict out;
           py::dict white, black;
           white[py::int_(1)] = py::int_(counts[0][1]);
           white[py::int_(2)] = py::int_(counts[0][2]);
           white[py::int_(3)] = py::int_(counts[0][3]);
           black[py::int_(1)] = py::int_(counts[1][1]);
           black[py::int_(2)] = py::int_(counts[1][2]);
           black[py::int_(3)] = py::int_(counts[1][3]);
           out[py::int_(tz::kWhite)] = std::move(white);
           out[py::int_(tz::kBlack)] = std::move(black);
           return out;
         })
    .def("board_cells",
         [](const tz::PhaseGameState& self) -> py::list {
           py::list out;
           const auto& board = self.game_ref().board();
           for (const auto& pos : board.valid_positions()) {
             if (board.is_empty(pos)) continue;
             py::dict cell;
             cell["row"]       = py::int_(pos.row);
             cell["col"]       = py::int_(pos.col);
             cell["top_piece"] = py::int_(board.top_piece(pos));
             cell["height"]    = py::int_(board.height(pos));
             out.append(std::move(cell));
           }
           return out;
         });

  // ─── SearchSession ─────────────────────────────────────
  py::class_<tz::SearchSession>(m, "SearchSession")
    .def(py::init<const tz::PhaseGameState&, tz::SearchConfig>(),
         py::arg("root_state"), py::arg("config"))
    .def("config", &tz::SearchSession::config)
    .def("has_pending_leaves", &tz::SearchSession::has_pending_leaves)

    // 舊 API：傳回 LeafSnapshot 列表
    .def("collect_pending_leaves", &tz::SearchSession::collect_pending_leaves,
         py::arg("max_batch"))

    // Packed API：傳回包含 numpy views 的 dict（零拷貝）
    .def("collect_pending_leaves_packed",
         [](tz::SearchSession& session, int max_batch) {
           auto packed = session.collect_pending_leaves_packed(max_batch);

           py::dict out;
           if (packed.batch_size == 0) {
             out["node_ids"]         = py::array_t<int32_t>({0});
             out["legal_masks"]      = py::array_t<uint8_t>({0, tz::kActionCount});
             out["board_state_flat"] = py::array_t<float>({0, tz::kBoardFlatSize});
             out["global_features"]  = py::array_t<float>({0, tz::kGlobalFeatureDim});
             return out;
           }

           const py::ssize_t bsz = static_cast<py::ssize_t>(packed.batch_size);

           // Capsules with no-op destructors: the SearchSession vectors own the memory
           py::capsule ids_cap  (packed.node_ids,         [](void*) {});
           py::capsule mask_cap (packed.legal_masks,       [](void*) {});
           py::capsule board_cap(packed.board_state_flat,  [](void*) {});
           py::capsule glob_cap (packed.global_features,   [](void*) {});

           out["node_ids"]         = py::array_t<int32_t>({bsz}, packed.node_ids, ids_cap);
           out["legal_masks"]      = py::array_t<uint8_t>(
               {bsz, static_cast<py::ssize_t>(tz::kActionCount)},
               packed.legal_masks, mask_cap);
           out["board_state_flat"] = py::array_t<float>(
               {bsz, static_cast<py::ssize_t>(tz::kBoardFlatSize)},
               packed.board_state_flat, board_cap);
           out["global_features"]  = py::array_t<float>(
               {bsz, static_cast<py::ssize_t>(tz::kGlobalFeatureDim)},
               packed.global_features, glob_cap);
           return out;
         },
         py::arg("max_batch"))

    // 單葉評估提交
    .def("submit_leaf_eval", &tz::SearchSession::submit_leaf_eval,
         py::arg("node_id"), py::arg("priors"), py::arg("value"))

    // 批次評估提交
    .def("submit_leaf_eval_batch",
         [](tz::SearchSession& session,
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> node_ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> priors,
            py::array_t<float, py::array::c_style | py::array::forcecast> values) {

           if (node_ids.ndim() != 1)
             throw std::invalid_argument("node_ids must be a 1D array");
           if (priors.ndim() != 2)
             throw std::invalid_argument("priors must be a 2D array [B, N_ACTIONS]");
           if (values.ndim() != 1)
             throw std::invalid_argument("values must be a 1D array [B]");

           const py::ssize_t bsz = node_ids.shape(0);
           if (priors.shape(0) != bsz || values.shape(0) != bsz)
             throw std::invalid_argument("batch size mismatch among node_ids/priors/values");
           if (priors.shape(1) != static_cast<py::ssize_t>(tz::kActionCount))
             throw std::invalid_argument("priors second dim must equal N_ACTIONS");

           const float* priors_ptr = static_cast<const float*>(priors.request().ptr);

           session.submit_leaf_eval_batch(
               static_cast<const int32_t*>(node_ids.request().ptr),
               priors_ptr,
               static_cast<const float*>(values.request().ptr),
               static_cast<int>(bsz));
         },
         py::arg("node_ids"), py::arg("priors"), py::arg("values"))

    .def("finish", &tz::SearchSession::finish,
         py::call_guard<py::gil_scoped_release>())
    .def("root_snapshot", &tz::SearchSession::root_snapshot);

  // ─── SearchManager ─────────────────────────────────────
  py::class_<tz::SearchManager>(m, "SearchManager")
    .def(py::init<tz::SearchConfig, int, int>(),
         py::arg("config"), py::arg("num_threads") = 8, py::arg("max_batch") = 480)

    .def("reset", &tz::SearchManager::reset,
         py::arg("root_states"), py::arg("config"))
    .def("join_workers", &tz::SearchManager::join_workers)
    .def("has_ready_batch", &tz::SearchManager::has_ready_batch)

        // get_ready_batch: 回傳 dict 包含 numpy views（零拷貝）
    .def("get_ready_batch",
         [](tz::SearchManager& mgr) {
           auto packed = mgr.get_ready_batch();
           py::dict out;

           if (packed.batch_size == 0) {
             out["batch_size"] = py::int_(0);
             return out;
           }

           const py::ssize_t bsz = static_cast<py::ssize_t>(packed.batch_size);

           // Capsules with no-op destructors
           py::capsule ids_cap  (packed.node_ids,         [](void*) {});
           py::capsule mask_cap (packed.legal_masks,       [](void*) {});
           py::capsule board_cap(packed.board_state_flat,  [](void*) {});
           py::capsule glob_cap (packed.global_features,   [](void*) {});
           py::capsule tree_cap (packed.tree_ids,          [](void*) {});

           out["batch_size"] = py::int_(packed.batch_size);
           out["node_ids"] = py::array_t<int32_t>({bsz}, packed.node_ids, ids_cap);
           out["tree_ids"] = py::array_t<int32_t>({bsz}, packed.tree_ids, tree_cap);
           out["legal_masks"] = py::array_t<uint8_t>(
               {bsz, static_cast<py::ssize_t>(tz::kActionCount)},
               packed.legal_masks, mask_cap);
                      out["board_state_flat"] = py::array_t<float>(
               {bsz, static_cast<py::ssize_t>(tz::kBoardFlatSize)},
               packed.board_state_flat, board_cap);
           out["global_features"] = py::array_t<float>(
               {bsz, static_cast<py::ssize_t>(tz::kGlobalFeatureDim)},
               packed.global_features, glob_cap);
           return out;
         })

    // submit_eval_batch
    .def("submit_eval_batch",
         [](tz::SearchManager& mgr,
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> node_ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> priors,
            py::array_t<float, py::array::c_style | py::array::forcecast> values) {

           if (node_ids.ndim() != 1)
             throw std::invalid_argument("node_ids must be a 1D array");
           if (priors.ndim() != 2)
             throw std::invalid_argument("priors must be a 2D array [B, N_ACTIONS]");
           if (values.ndim() != 1)
             throw std::invalid_argument("values must be a 1D array [B]");

           const py::ssize_t bsz = node_ids.shape(0);
           if (priors.shape(0) != bsz || values.shape(0) != bsz)
             throw std::invalid_argument("batch size mismatch among node_ids/priors/values");
           if (priors.shape(1) != static_cast<py::ssize_t>(tz::kActionCount))
             throw std::invalid_argument("priors second dim must equal N_ACTIONS");

           mgr.submit_eval_batch(
               static_cast<const int32_t*>(node_ids.request().ptr),
               static_cast<const float*>(priors.request().ptr),
               static_cast<const float*>(values.request().ptr),
               static_cast<int>(bsz));
         },
         py::arg("node_ids"), py::arg("priors"),
         py::arg("values"))

            .def("is_complete", &tz::SearchManager::is_complete)
    .def("completed_tree_count", &tz::SearchManager::completed_tree_count)
    .def("last_swap_reason", &tz::SearchManager::last_swap_reason)
    .def("finish_all", &tz::SearchManager::finish_all)
    .def("shutdown", &tz::SearchManager::shutdown)
    .def("total_remaining_simulations", &tz::SearchManager::total_remaining_simulations);

  // ─── CpuBench（純 CPU MCTS 效能測試） ────────────────
  py::class_<tz::CpuBenchResult>(m, "CpuBenchResult")
    .def(py::init<>())
    .def_readwrite("elapsed_seconds",      &tz::CpuBenchResult::elapsed_seconds)
    .def_readwrite("num_trees",            &tz::CpuBenchResult::num_trees)
    .def_readwrite("num_threads",          &tz::CpuBenchResult::num_threads)
    .def_readwrite("simulations_per_tree", &tz::CpuBenchResult::simulations_per_tree)
    .def_readwrite("total_simulations",    &tz::CpuBenchResult::total_simulations)
    .def_readwrite("total_simulations_done", &tz::CpuBenchResult::total_simulations_done)
    .def_readwrite("total_nodes_created",  &tz::CpuBenchResult::total_nodes_created)
    .def_readwrite("sims_per_second",      &tz::CpuBenchResult::sims_per_second)
    .def_readwrite("tree_simulations_done", &tz::CpuBenchResult::tree_simulations_done)
    .def_readwrite("tree_node_counts",     &tz::CpuBenchResult::tree_node_counts);

  m.def("run_cpu_bench", &tz::CpuBench::run,
        py::arg("num_trees"), py::arg("num_threads"),
        py::arg("simulations"), py::arg("leaf_batch_size") = 8,
        "Run CPU-only MCTS benchmark. "
        "Creates num_trees trees, uses num_threads workers, "
        "each tree runs simulations times. "
        "Returns CpuBenchResult with timing and stats.");
}
