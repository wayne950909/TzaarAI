# 訓練架構現況與目標差距（2026-07）

## 目標摘要
目前的目標架構是：
- Python 負責神經網路推論、訓練主流程、樣本收集。
- C++ 負責遊戲狀態、MCTS 搜尋、葉節點打包、多樹搜尋管理。
- 正式訓練時使用常駐的 C++ CPU worker pool 與 Python GPU worker，讓 CPU 模擬與 GPU 推論同時進行。
- 透過雙 buffer 讓 CPU 持續收集下一批葉節點，而 GPU 同時處理前一批資料。

## 目前 Python / C++ 訓練架構

### Python 端
- [start_training.py](C:/Users/user/Desktop/training/start_training.py)
  - 最外層啟動入口。
  - 設定預設 `TZAAR_STATE_BACKEND=cpp`，再呼叫 `training.loop.main()`。
- [training/loop.py](C:/Users/user/Desktop/training/training/loop.py)
  - 訓練主流程 orchestration。
  - 負責：載入模型、後端解析、self-play、train step、gate、checkpoint、log。
- [training/selfplay_engine.py](C:/Users/user/Desktop/training/training/selfplay_engine.py)
  - 目前 self-play 排程核心。
  - 負責：active game pool 管理、批次 MCTS 呼叫、樣本收集、async 失敗 fallback。
- [core/env.py](C:/Users/user/Desktop/training/core/env.py)
  - 統一 C++ / Python 後端環境介面。
  - 正式訓練目前主要透過這層取得 observation、step、winner。
- [mcts/mcts_api.py](C:/Users/user/Desktop/training/mcts/mcts_api.py)
  - Python 側 MCTS 統一入口。
  - 依條件選擇同步 `cpp_session`、非同步 `async_worker`、或 Python fallback。
- [mcts/cpp_session.py](C:/Users/user/Desktop/training/mcts/cpp_session.py)
  - 單棵樹同步 C++ SearchSession 包裝。
- [mcts/async_worker.py](C:/Users/user/Desktop/training/mcts/async_worker.py)
  - 目前正式訓練使用的 async 批次路徑。
  - 但它是「每次 batch search 臨時建立 thread」，不是常駐 worker pool。

### C++ 端
- [cpp/src/mcts/search.cpp](C:/Users/user/Desktop/training/cpp/src/mcts/search.cpp)
  - `SearchSession` 真正的單棵樹 MCTS 核心。
  - 已實作：
    - 根節點特徵快取
    - 動作路徑重建 state
    - pending leaf queue
    - virtual loss
    - terminal node 直接 backup
    - policy/value 回填與反向傳播
- [cpp/include/mcts/search_manager.h](C:/Users/user/Desktop/training/cpp/include/mcts/search_manager.h)
- [cpp/src/mcts/search_manager.cpp](C:/Users/user/Desktop/training/cpp/src/mcts/search_manager.cpp)
  - 多棵樹、多 CPU worker、雙 buffer 的正式底層設計。
  - 已實作：
    - 固定數量 worker threads
    - thread-local leaf buffer
    - shared double buffer
    - 在無樹可模擬時強制 swap buffer
    - GPU 結果回寫後解鎖樹
  - 但目前尚未接進正式訓練主流程。

## 已做到的部分
- `leaf_batch_size=16` 已對應到正式訓練配置。
- Python 訓練主流程已支援批次 self-play。
- async 失敗時可自動 fallback 到同步單局搜尋。
- C++ `SearchSession` 已完成文件中描述的大多數 MCTS 細節：
  - 根特徵快取
  - 路徑重建 state
  - terminal 直接回傳
  - virtual loss
  - backup 時依玩家切換翻號
- C++ `SearchManager` 已有常駐 worker + 雙 buffer 的底層實作。

## 還沒做到的部分

### 1. 正式訓練還沒接上 C++ SearchManager
目前正式訓練的 async 路徑仍然是：
- `training/selfplay_engine.py` -> `mcts.run_mcts_batch()` -> `mcts.async_worker.run_mcts_cpp_batch_async()`

這條路徑的問題：
- 每次批次搜尋都會新建 Python thread。
- 執行完就 `join()` 回收。
- 沒有真正使用常駐的 C++ worker pool。
- 沒有使用 `SearchManager` 的雙 buffer。

### 2. 文件中的「固定 10 個 CPU 執行緒」尚未在正式訓練落地
- `SearchManager` 有 `num_threads=10` 預設值。
- 但正式訓練沒有呼叫 `SearchManager`。
- 現在 Python async 路徑是「每個 root state 一個 thread」，數量與 active roots 相同，不是固定 10。

### 3. 文件中的「一次同時建立 200 棵樹」尚未落地
- 文件寫的是 200 棵樹並行。
- 目前正式設定是 `parallel_games=15`。
- `games_per_update=150` 是每輪總局數，不是同時 150 局。

### 4. GPU batch 觸發條件只部分符合文件描述
現在 Python async worker 只根據：
- request queue 裡已有的請求數
- `infer_max_batch`
- `infer_max_wait_ms`

它不知道：
- 全域上是不是已經沒有樹可以模擬
- 哪個 buffer 是當前 CPU 填寫、哪個 buffer 是 GPU 讀取

這些正是 `SearchManager` 已實作但尚未接線的部分。

## 接下來還需要做什麼

### Phase 1：把正式訓練從 Python async_worker 改接 C++ SearchManager
需要做的事：
1. 在 Python 端新增 `search_manager` 包裝層。
2. 將多局 root states 一次交給 C++ `SearchManager`。
3. Python 端只負責：
   - 取出 ready batch
   - 用 GPU 做 forward
   - 把 priors/value 回填給 `SearchManager`
4. 將目前 `mcts/async_worker.py` 轉為：
   - 備援實作
   - 或單元測試用對照路徑

### Phase 2：把配置對齊文件目標
需要確認與調整：
1. `parallel_games` 是否要提升到 200。
2. `num_threads` 是否固定 10，或改為可配置。
3. `max_batch` 是否明確對應 `parallel_games * leaf_batch_size`。
4. 是否需要把 `SearchManager` 的 batch 行為對齊文件中的 GPU 觸發規則。

### Phase 3：補效能與可觀測性
建議增加：
1. 每輪平均 batch size
2. 每個 buffer 的 flush 次數
3. GPU wait time / CPU idle time
4. 每局平均 decision 數
5. 每秒模擬數（正式訓練，不只 bench）

## 目前建議的核心檔案閱讀順序
1. [start_training.py](C:/Users/user/Desktop/training/start_training.py)
2. [training/loop.py](C:/Users/user/Desktop/training/training/loop.py)
3. [training/selfplay_engine.py](C:/Users/user/Desktop/training/training/selfplay_engine.py)
4. [mcts/mcts_api.py](C:/Users/user/Desktop/training/mcts/mcts_api.py)
5. [mcts/async_worker.py](C:/Users/user/Desktop/training/mcts/async_worker.py)
6. [mcts/cpp_session.py](C:/Users/user/Desktop/training/mcts/cpp_session.py)
7. [core/env.py](C:/Users/user/Desktop/training/core/env.py)
8. [cpp/src/mcts/search.cpp](C:/Users/user/Desktop/training/cpp/src/mcts/search.cpp)
9. [cpp/include/mcts/search_manager.h](C:/Users/user/Desktop/training/cpp/include/mcts/search_manager.h)
10. [cpp/src/mcts/search_manager.cpp](C:/Users/user/Desktop/training/cpp/src/mcts/search_manager.cpp)

## 一句話總結
目前狀態是：
- 單棵樹 MCTS 細節已大致完成。
- 多樹常駐 worker + 雙 buffer 的底層也已完成。
- 但正式訓練主流程還沒接到這個底層，所以離最終目標還差「正式接線」這一步。 
