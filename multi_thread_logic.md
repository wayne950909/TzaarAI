# 多執行緒目的
預設建立多個 c++ 的 CPU 執行緒和 1 個 python 的 GPU 執行緒，目的是讓 **CPU 和GPU同時運行**。
**執行緒不會反覆被創立並刪除**，生命週期與 `SearchManager` 綁定。

---

# 一、 樹的固定分配與輪詢機制 (CPU Worker)
- **固定指派**：每次呼叫 `reset_search()` 時，將總樹數平均分配給每個 CPU worker，每個 worker 內部維護一個專屬的「活躍樹 ID 清單」（Active Trees）。
- **輪詢與過濾**：CPU worker 只輪詢自己內部清單中的樹。若某棵樹已完成（`is_complete()`），**立即從該 worker 的內部活躍清單中移除（`erase`）**，之後不再納入輪詢。

---

# 二、 模擬與 16 次批次資料收集
- **每輪 16 次模擬**：CPU worker 輪詢其活躍樹，對選中的樹進行**最多 16 次模擬**（模擬一次最多產生一個待推論葉節點，遇到底部節點可能不產生）。
- **打包送出**：累積滿 16 次模擬（或該樹暫時無法再模擬）後，將收集到的一批待推論葉節點**以指標形式**整批送進 `concurrentQueue`。
- **閥值檢查**：送入後檢查累積資料有沒有超過設定的閥值量，如果有，則喚醒 Python 端去做 GPU 推論。

---

# 三、 防空轉機制 (Worker Sleep / Wakeup)
- **主動睡眠**：若 CPU worker 輪詢完自己所有活躍的樹，發現：
  1. 樹全部都已完成 (`is_complete()`)；或者
  2. 尚未完成的樹**全部都在等 GPU 推論 (`has_pending_leaves() == true`)**。
  - 此時 worker 會透過專屬的 `std::condition_variable` **主動進入睡眠狀態**，避免 CPU 發生 100% Busy-Spinning。
- **精準喚醒**：當對應的 1:1 `result_handler` 處理完 GPU 結果並解鎖該樹後，會發出 `notify_one()` 將睡眠中的 CPU worker 喚醒繼續工作。

---

# 四、 ConcurrentQueue 設計
- 每個 CPU worker 負責將資料指標（或批次指標）丟入佇列，Python 從裡面拿資料去進行神經網路推論。
- 資料的進出全程**使用指標來傳遞**，實現零拷貝高效能。

---

# 五、 一棵樹的模擬完成與狀態更新
- **完成定義**：模擬完成的定義是**消耗完模擬次數且沒有待推論葉節點**（`is_complete()`）。
- **已完成計數**：存在一個已完成樹的數量原子變數，從 0 開始遞增。
- **重新入隊/解鎖**：樹有沒有完成會影響 `result_handler` 是否要解除該樹的 pending 狀態，讓 worker 可以繼續對其進行後續模擬。

---

# 六、 處理推論完的資料 (Result Handler Worker - 1:1 配對)
- **1 對 1 伴隨創建**：每個負責模擬 MCTS 的 CPU worker 都會**伴隨一個專屬的 `result_handler` 執行緒**（模擬執行緒數量等於 `result_handler` 數量）。
- **專屬處理**：事先在每個資料封包標記所屬的 CPU worker。每個 `result_handler` 專責挑選該 CPU worker 負責的 GPU 推論結果。
- **回填與喚醒**：將先驗機率（Prior）與 Value 加到對應的樹上（`submit_single_eval`），解除 pending 狀態，若樹未完成則透過 `notify_one()` 喚醒對應的 CPU worker。

---

# 七、 MCTS 搜尋結束
- 每當已完成的樹的數量增加時，檢查是否等於樹的總數。
- 若等於總樹數，則喚醒 Python 執行緒，Python 端即會察覺搜尋結束並收回搜尋結果。

---

# 八、 Worker 的存活與生命週期
- `SearchManager` 在多場對局中不重建。每次執行 `run_search()` 時，只將樹與 `concurrentQueue` 清空並重新指派。
- CPU worker 和 Result Handler 在結束搜尋時不關閉，而是**進入 `wait` 狀態**，等到下一次呼叫 `run_search()` 時被喚醒全部。
- 只有當整場遊戲結束、呼叫 `shutdown()` 時，才會正式關閉並銷毀所有執行緒。
