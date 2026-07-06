"""
bench_cpu_mcts.py — 純 CPU MCTS 多執行緒效能測試

使用方法：
  python bench_cpu_mcts.py                          # 執行預設測試組合
  python bench_cpu_mcts.py --quick                   # 快速測試（少量參數）
  python bench_cpu_mcts.py --custom 8 4 500          # 自訂: trees=8 threads=4 sims=500

參數說明：
  num_trees     : 搜尋樹數量
  num_threads   : Worker 執行緒數量
  simulations   : 每棵樹的模擬次數
  leaf_batch_size : 每批模擬的葉節點數（預設 8）
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cpp", "build", "Release"))
import tzaar_cpp as cpp


def run_single_test(num_trees, num_threads, simulations, leaf_batch_size=8):
    """執行單一測試，回傳 CpuBenchResult"""
    print(f"    trees={num_trees:3d}  threads={num_threads:2d}  "
          f"sims={simulations:5d}  batch={leaf_batch_size:2d}  ...", end=" ", flush=True)

    t0 = time.perf_counter()
    result = cpp.run_cpu_bench(num_trees, num_threads, simulations, leaf_batch_size)
    t1 = time.perf_counter()

    # 加上 Python 端的 overhead（非常小）
    elapsed = result.elapsed_seconds
    py_overhead = (t1 - t0) - elapsed

    print(f"  {elapsed:8.3f}s  "
          f"({result.sims_per_second:8.0f} sims/s)  "
          f"done={result.total_simulations_done}/{result.total_simulations}  "
          f"py_overhead={py_overhead*1000:.1f}ms")

    return result


def print_separator(char="=", width=72):
    print(char * width)


def run_default_suite():
    """執行完整的測試組合"""
    SIMS = 500
    BATCH = 8

    print()
    print_separator()
    print("  CPU MCTS Multi-threading Benchmark (pure CPU, no GPU)")
    print(f"  基本設定: simulations={SIMS}, leaf_batch_size={BATCH}")
    print_separator()

    # ─── Test 1: 固定 tree=4, 改變 thread 數量 ──────────
    print()
    print("─" * 72)
    print("  Test 1: Fixed trees=4, vary threads")
    print("─" * 72)
    print(f"  {'trees':>6} {'threads':>8} {'sims':>8} {'time':>9} {'sims/s':>10}  {'speedup':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*10} {'─'*8}")

    base_time = None
    for threads in [1, 2, 4, 8]:
        result = run_single_test(4, threads, SIMS, BATCH)
        speedup = base_time / result.elapsed_seconds if base_time else 1.0
        print(f"  {4:6d} {threads:8d} {SIMS:8d} {result.elapsed_seconds:9.3f}s "
              f"{result.sims_per_second:10.0f}  {speedup:7.2f}x")
        if base_time is None:
            base_time = result.elapsed_seconds

    # ─── Test 2: 固定 threads=4, 改變 tree 數量 ──────────
    print()
    print("─" * 72)
    print("  Test 2: Fixed threads=4, vary trees")
    print("─" * 72)
    print(f"  {'trees':>6} {'threads':>8} {'sims':>8} {'time':>9} {'sims/s':>10}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*10}")

    for trees in [1, 2, 4, 8, 16, 32, 64]:
        result = run_single_test(trees, 4, SIMS, BATCH)
        print(f"  {trees:6d} {4:8d} {SIMS:8d} {result.elapsed_seconds:9.3f}s "
              f"{result.sims_per_second:10.0f}")

    # ─── Test 3: 改變模擬深度 ─────────────────────────────
    print()
    print("─" * 72)
    print("  Test 3: Fixed trees=4, threads=4, vary simulations")
    print("─" * 72)
    print(f"  {'trees':>6} {'threads':>8} {'sims':>8} {'time':>9} {'sims/s':>10}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*10}")

    for sims in [100, 200, 500, 1000, 2000]:
        result = run_single_test(4, 4, sims, BATCH)
        print(f"  {4:6d} {4:8d} {sims:8d} {result.elapsed_seconds:9.3f}s "
              f"{result.sims_per_second:10.0f}")

    # ─── Test 4: 改變 leaf_batch_size ────────────────────
    print()
    print("─" * 72)
    print("  Test 4: Fixed trees=4, threads=4, sims=500, vary batch_size")
    print("─" * 72)
    print(f"  {'trees':>6} {'threads':>8} {'batch':>8} {'time':>9} {'sims/s':>10}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*10}")

    for batch in [1, 4, 8, 16, 32]:
        result = run_single_test(4, 4, 500, batch)
        print(f"  {4:6d} {4:8d} {batch:8d} {result.elapsed_seconds:9.3f}s "
              f"{result.sims_per_second:10.0f}")

    print()
    print_separator("=")
    print("  Benchmark complete!")
    print_separator("=")
    print()


def run_quick_test():
    """快速測試"""
    print()
    print_separator()
    print("  Quick CPU MCTS Benchmark")
    print_separator()
    print()

    for trees, threads, sims in [(1, 1, 100), (2, 2, 100), (4, 2, 100), (4, 4, 100)]:
        run_single_test(trees, threads, sims)

    print()
    print("  Done!")
    print()


def run_custom_test(num_trees, num_threads, simulations, leaf_batch_size=8):
    """自訂參數測試"""
    print()
    print_separator()
    print(f"  Custom CPU MCTS Benchmark")
    print(f"  trees={num_trees}, threads={num_threads}, sims={simulations}, batch={leaf_batch_size}")
    print_separator()
    print()

    result = run_single_test(num_trees, num_threads, simulations, leaf_batch_size)

    print()
    print("─" * 72)
    print("  Detailed Results:")
    print("─" * 72)
    print(f"    Elapsed:          {result.elapsed_seconds:.3f} sec")
    print(f"    Sims/second:      {result.sims_per_second:.0f}")
    print(f"    Trees:            {result.num_trees}")
    print(f"    Threads:          {result.num_threads}")
    print(f"    Sims/tree:        {result.simulations_per_tree}")
    print(f"    Total sims req:   {result.total_simulations}")
    print(f"    Total sims done:  {result.total_simulations_done}")
    print(f"    Nodes created:    {result.total_nodes_created}")

    # 各樹分配
    if len(result.tree_simulations_done) <= 16:
        print(f"    Per-tree sims:    {result.tree_simulations_done}")
    else:
        print(f"    Per-tree sims:    min={min(result.tree_simulations_done)}, "
              f"max={max(result.tree_simulations_done)}, "
              f"avg={sum(result.tree_simulations_done)/len(result.tree_simulations_done):.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="CPU MCTS Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--custom", nargs=3, type=int, metavar=("TREES", "THREADS", "SIMS"),
                        help="Run custom test: trees threads sims")
    parser.add_argument("--batch", type=int, default=8, help="Leaf batch size (default: 8)")

    args = parser.parse_args()

    if args.custom:
        run_custom_test(args.custom[0], args.custom[1], args.custom[2], args.batch)
    elif args.quick:
        run_quick_test()
    else:
        run_default_suite()


if __name__ == "__main__":
    main()
