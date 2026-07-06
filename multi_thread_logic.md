預設建立10個c++的CPU執行緒和1個python的GPU執行緒，目的是讓**CPU和GPU同時運行**
**執行緒不會反覆被創立並刪除**

# CPU執行緒模擬
- 每個MCTS遇到固定數量的待推論的子節點就會停止模擬，直到推論完才會繼續模擬，這邊預設為16
- 預設一次會模擬200場遊戲，也就是說會同時創建200棵MCTS樹
- 每個CPU執行緒，**會優先處理GPU推論完的資料**，然後**同時去尋找待模擬的搜尋樹並霸佔模擬**，模擬完再重複動作

# 雙buffer
過程如下:
1. CPU模擬並把資料放進暫存的16格資料的區塊，收集到16個後一次丟進buffer
2. GPU接著會把buffer的資料拿去推論，條件為
    - buffer內的資料數量達到一定值
    - 沒有新的樹可以被模擬了(樹的狀態不是模擬過就是已經模擬完次數)
    - 時間到達最大值限制
3. 在GPU推論的同時，CPU會在另外一個buffer繼續模擬(如果可以的話)，這將做為下一次推論的buffer，然後buffer會一直交替使用
4. GPU推論完後，會將推論完資料給回CPU執行緒去處理，以解鎖停止模擬的樹

# 目前程式對應（2026-07）
- 啟動入口統一為 `start_training.py`，它只負責設定預設 `TZAAR_STATE_BACKEND=cpp` 並呼叫 `training.loop.main()`。
- 訓練模式摘要會在啟動時輸出，例如：
    - backend
    - selfplay_mode（`async-batch` 或 `sync-single`）
    - `parallel_games / infer_max_batch / infer_max_wait_ms / response_timeout_s`
- `training/loop.py` 的 self-play 現在會維護 active game pool：
    1. 同時啟動多局（上限 `ASYNC_MCTS_CFG.parallel_games`）
    2. 對 active 局面做一次 `run_mcts_batch`
    3. 逐局採樣動作並更新局面
    4. 終局時回填 value target
- `mcts/async_worker.py` 增加了 timeout 診斷訊息（session/batch/queue），以及 request queue 滿時的有限等待邏輯，降低卡住時無訊息的問題。