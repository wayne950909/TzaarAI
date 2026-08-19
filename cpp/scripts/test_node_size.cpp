// test_node_size.cpp
// ============================================================================
// 目的：量測 MCTS 樹節點 (MctsNode) 的記憶體大小，與節點池總記憶體需求。
//
// 為什麼要用「複刻」struct 來量測？
//   MctsNode 是 SearchSession 的 private 巢狀 struct，外部無法直接 sizeof。
//   但它的每個成員型別都是已知的標量（float / int / bool），
//   因此複刻一份完全相同欄位順序的 struct 即可得到 100% 正確的 sizeof。
//   同時此檔也用 offsetof 逐一驗證每個欄位的對齊與偏移，方便分析 padding。
//
// 編譯方式（任何 C++17 編譯器，不需連結專案，純量測，無外部依賴）：
//   Windows MSVC:   cl /std:c++17 /EHsc test_node_size.cpp
//   Linux g++/clang: g++ -std=c++17 test_node_size.cpp -o test_node_size
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>

// 複刻自 include/mcts/search.h 的 MctsNode（欄位與順序完全相同）
struct MctsNode {
    float prior = 0.0f;
    int   parent_idx = -1;
    int   action_from_parent = -1;
    int   children_base_idx = -1;
    int   children_count = 0;
    int   visit_count = 0;
    float value_sum = 0.0f;
    bool  expanded = false;
    int   to_play = 0;
    int   winner = 0;
    bool  is_terminal = false;
    int   pending_slot = -1;
};

int main() {
    // ── 1. 各欄位偏移（分析 padding）────────────────────
    std::printf("=== 欄位偏移 (offsetof) ===\n");
    std::printf("  prior(%)              [float] %zu\n", (size_t)offsetof(MctsNode, prior));
    std::printf("  parent_idx            [int]   %zu\n", (size_t)offsetof(MctsNode, parent_idx));
    std::printf("  action_from_parent    [int]   %zu\n", (size_t)offsetof(MctsNode, action_from_parent));
    std::printf("  children_base_idx     [int]   %zu\n", (size_t)offsetof(MctsNode, children_base_idx));
    std::printf("  children_count        [int]   %zu\n", (size_t)offsetof(MctsNode, children_count));
    std::printf("  visit_count           [int]   %zu\n", (size_t)offsetof(MctsNode, visit_count));
    std::printf("  value_sum             [float] %zu\n", (size_t)offsetof(MctsNode, value_sum));
    std::printf("  expanded              [bool]  %zu\n", (size_t)offsetof(MctsNode, expanded));
    std::printf("  to_play               [int]   %zu\n", (size_t)offsetof(MctsNode, to_play));
    std::printf("  winner                [int]   %zu\n", (size_t)offsetof(MctsNode, winner));
    std::printf("  is_terminal           [bool]  %zu\n", (size_t)offsetof(MctsNode, is_terminal));
    std::printf("  pending_slot          [int]   %zu\n", (size_t)offsetof(MctsNode, pending_slot));
    std::printf("\n");

    // ── 2. 合計大小與各基本型別大小 ──────────────────────
    const size_t sz        = sizeof(MctsNode);
    const size_t align     = alignof(MctsNode);
    std::printf("=== 大小彙總 ===\n");
    std::printf("  sizeof(bool)   = %zu\n", sizeof(bool));
    std::printf("  sizeof(int)    = %zu\n", sizeof(int));
    std::printf("  sizeof(float)  = %zu\n", sizeof(float));
    std::printf("  sizeof(void*)  = %zu\n", sizeof(void*));
    std::printf("  sizeof(MctsNode) = %zu bytes\n", sz);
    std::printf("  alignof(MctsNode) = %zu bytes\n", align);

    // 理論上「欄位原始總和」（不含 padding）
    const size_t raw_sum = sizeof(float) * 2 + sizeof(int) * 7 + sizeof(bool) * 2;
    std::printf("  欄位原始總和 (無 padding) = %zu bytes\n", raw_sum);
    std::printf("  padding 造成 => %zu bytes\n", sz - raw_sum);
    std::printf("\n");

    // ── 3. 節點池總記憶體需求 ───────────────────────────
    constexpr size_t kMaxNodesPerTree = 240000;   // include/mcts/search.h
    std::printf("=== 節點池 (kMaxNodesPerTree = %zu) ===\n", kMaxNodesPerTree);
    std::printf("  單一 SearchSession 節點池 = %zu bytes = %.2f MB\n",
                sz * kMaxNodesPerTree,
                (double)(sz * kMaxNodesPerTree) / (1024.0 * 1024.0));

    // 實際 vector 保留（capacity）後的大小
    std::vector<MctsNode> pool;
    pool.reserve(kMaxNodesPerTree);
    pool.resize(kMaxNodesPerTree);
    std::printf("  實際 vector<uint8 容量> : %zu slots\n", pool.capacity());
    std::printf("  vector 保留後大小        : %zu bytes\n", pool.size() * sizeof(MctsNode));
    std::printf("\n");

    // ── 4. 多執行緒（多棵樹並存）情境 ───────────────────
    std::printf("=== 多棵樹並存 (SearchManager 每執行緒一棵) ===\n");
    for (int tree = 1; tree <= 8; ++tree) {
        const size_t bytes = sz * kMaxNodesPerTree * (size_t)tree;
        std::printf("  %d 棵樹 => %zu bytes = %.2f MB\n",
                    tree, bytes, (double)bytes / (1024.0 * 1024.0));
    }

    return 0;
}
