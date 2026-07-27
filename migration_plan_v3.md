# Tzaar MCTS 多執行緒架構重構計畫 (v3 最終完整版)

本文件詳細記錄了從舊有的 **「全域動態搶樹佇列 + 雙緩衝區 (Double Buffer) + 單一全域 Result Handler」** 架構，遷移至新計畫規定的 **「固定分配樹給 CPU Worker + 指標型 ConcurrentQueue + 每位 CPU Worker 配對專屬 Result Handler + 防空轉睡眠機制」** 架構的完整重構藍圖。

---

## 一、 核心架構設計與演進

| 設計維度 | 舊計畫架構 (Current) | 新計畫架構 (New / 最終版) |
| :--- | :--- | :--- |
| **樹的分配方式** | 所有樹放入全域 `tree_queue_`，Worker 動態搶奪 (Dynamic Stealing)。 | 每個 CPU Worker **固定分配並掌管一組內部樹 ID 清單**，不需要搶奪。 |
| **完成樹處理** | 標記完成後留在系統中。 | 樹一旦完成 (`is_complete()`)，**立即從 Worker 的內部活躍清單中移除**，不再納入輪詢。 |
| **防空轉機制 (CPU 100%)** | 透過 `std::this_thread::yield()` 或 Spin 迴圈等待。 | 引入專屬的 `std::condition_variable`。當無事可做時 Worker **主動進入睡眠**，由對應的 Result Handler 喚醒。 |
| **資料傳遞結構** | 雙緩衝區（`EvalBuffer`）與平坦陣列 (`std::vector<float>`) 進行 `memcpy`。 | 使用指標型 `ConcurrentQueue`，資料進出均以**指標**傳遞。 |
| **模擬與批次觸發** | Worker 持續模擬直到 local buffer 滿或 stall。 | **每進行 16 次模擬**為一個單位，將收集到的一批葉節點指標送進 `ConcurrentQueue`。 |
| **Result Handler** | 僅有**單一全域**的 `result_handler_` 執行緒。 | **1 對 1 配對**：每個 CPU Worker 各伴隨一個專屬的 Result Handler。 |

---

## 二、 核心模組與重構實作細節

### 1. 樹的固定分配與內部活躍清單 (Thread-Local Active Trees)
- 在 `SearchManager::reset()` 時，將總樹數平均切分，指派給各個 CPU Worker（例如 `assigned_trees_[worker_id]`）。
- 每個 Worker 在自己的執行緒內部維護一份活躍樹 ID 清單。
- **動態過濾完成樹：** 當 Worker 輪詢到某棵樹並發現其 `session->is_complete()` 達成時：
  1. 將全域完成計數原子變數 `completed_count_` +1。
  2. **直接從該 Worker 自己的內部清單中 `erase` 該樹**，之後的迴圈將完全略過它。

### 2. 避免 CPU Busy-Spinning 的睡眠與喚醒機制
當 Worker 輪詢完自己分配的所有樹，發現：
- 要麼全部樹都已經 `is_complete()`。
- 要麼還沒完成的樹**全部都在等 GPU 結果 (`has_pending_leaves() == true`)**。
- 此時若讓 Worker 繼續用 `while(true)` 掃描會導致 CPU 飆高。

**解決方案：**
- 每個 Worker 擁有自己的 `std::mutex` 與 `std::condition_variable`。
- 當 Worker 發現本輪「完全沒有做到任何實際模擬（Did Work = false）」時，調用 `cv.wait()` 進入 **Sleep 狀態**。
- **誰來喚醒？** 當對應的 1:1 **Result Handler** 從 GPU 拿回推論結果、更新完樹（`submit_single_eval`）之後，該樹的 `has_pending_leaves()` 變回 `false`。此時 Result Handler 會立即呼叫 `worker_cv.notify_one()`，精準喚醒對應的 Worker 繼續工作。

### 3. 「16 次模擬為一個批次」的 Worker 輪詢邏輯
- Worker 迴圈開始掃描它未完成的內部樹清單。
- 對選中的樹執行**最多 16 次模擬**（模擬一次最多產生一個待推論葉節點，遇到底部節點可能不產生）。
- 累積滿 16 次模擬（或該樹暫時無法再模擬）後，將這一批葉節點指標**整批送入 `ConcurrentQueue`**。
- 檢查全域累積資料量是否達到閥值，若達到則喚醒 Python 端進行 GPU 推論。

### 4. 1 對 1 配對的 Result Handler 職責
- 每個 CPU Worker 配對一個專屬的 Result Handler。
- 專屬 Result Handler 負責：
  1. 接收對應 Worker 產出並經 GPU 推論完的結果。
  2. 將 Prior 與 Value 寫回樹中（`submit_single_eval`），這會解除樹的 pending 狀態。
  3. **檢查樹是否完成**：若未完成，確保該樹可被 Worker 繼續模擬；若已完成則交由全域計數統計。
  4. 透過 `notify_one()` 喚醒對應的 Worker。

---

## 三、 Worker 迴圈虛擬碼 (Pseudo-code) 參考

```cpp
void WorkerLoop(int worker_id) {
    // 1. 取得該 worker 內部擁有的固定樹 ID 清單
    std::vector<int>& my_trees = assigned_trees_[worker_id];

    while (!stop_) {
        bool did_work = false;

        // 2. 輪詢自己內部的樹清單
        for (auto it = my_trees.begin(); it != my_trees.end(); ) {
            int tree_id = *it;
            SearchTree& tree = *trees_[tree_id];

            // 若樹已完成，直接從內部清單移除，從此不理會
            if (tree.session->is_complete()) {
                it = my_trees.erase(it);
                continue;
            }

            // 若樹正在等 GPU 結果 (pending leaves)，跳過它去看下一棵
            if (tree.session->has_pending_leaves()) {
                ++it;
                continue;
            }

            // 3. 執行核心邏輯：對這棵樹進行最多 16 次模擬
            int simulated_count = 0;
            std::vector<LeafNode*> batch_leaves;
            
            {
                std::lock_guard<std::mutex> lock(tree.mtx);
                while (simulated_count < 16 && 
                       !tree.session->is_complete() && 
                       !tree.session->has_pending_leaves()) {
                    
                    // 單次模擬，產生葉節點指標
                    LeafNode* leaf = tree.session->simulate_single_step(); 
                    if (leaf) {
                        batch_leaves.push_back(leaf);
                    }
                    simulated_count++;
                }

                // 4. 若收集到葉節點，整批送進 ConcurrentQueue
                if (!batch_leaves.empty()) {
                    did_work = true;
                    concurrent_queue_.push_batch(worker_id, batch_leaves);
                    // 檢查總量是否超過閥值以喚醒 Python
                    CheckAndNotifyPythonIfNeeded();
                }

                // 5. 模擬完後檢查這棵樹是否正式完成
                if (tree.session->is_complete()) {
                    CompleteTree(tree); // 全局完成數 +1
                    it = my_trees.erase(it); // 從內部清單拔除
                    continue;
                }
            }
            ++it;
        }

        // 6. 如果本輪完全沒有做到工作（所有樹都完成或都在等 GPU），進入睡眠防空轉
        if (!did_work) {
            std::unique_lock<std::mutex> lk(worker_cv_mtx_[worker_id]);
            if (HasAnyReadyTree(my_trees)) {
                continue; // 雙重檢查若剛好被解鎖就繼續
            }
            // 進入睡眠，直到對應的 1:1 Result Handler 喚醒或 reset/stop
            worker_cv_[worker_id].wait(lk, [this, &my_trees]() {
                return stop_ || HasAnyReadyTree(my_trees) || reset_signaled_;
            });
        }
    }
}
```
