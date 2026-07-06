# 蒙地卡羅分數公式
- score = Q(s, a) + U(s, a)
- Q(s, a) = child.value_sum / child.visit_count
- U(s, a) = C_CUPT * P(s, a) * sqrt(N_all) / N(s, a)
- P(s, a)為神經網路的先驗機率
- N_all 為所有子節點訪問次數總和加1
- N(s, a) 為自身節點的訪問次數加1

# 遊戲狀態特徵state取得方法:
- 先處理根節點，因為最後要放進replay buffer，所以這邊我選擇**根節點的state全部都做一份複製，做成快取**。
- 每當擴展節點時，會在該節點紀錄所做的遊戲動作。
- 每當在葉節點需要state去做推論時，會先從根節點的state複製一份到batch的記憶空間，根據根節點到葉節點的動作序列將這個state製作成目前的state。
- 要注意的是，最後要根據目前是誰的回合去做state的轉換。

# 模擬過程
會從根節點一路選擇**分數最高**的向下，直到到達葉節點(尚未展開)或是最底部的節點(遊戲結束):
## 到達尚未展開的節點
若是到達尚未展開的葉節點
- 由於這個節點是第一次到達，所以是還沒有記錄動作的，並將此動作新增到動作序列
- 然後複製根節點的state到一塊連續的batch的記憶空間(每次模擬就依序放)，將這個state根據動作序列做成葉節點的state，然後展開子節點
- 將節點的分數加上**虛擬的負分**，讓同一batch的資料不要包含重複節點，在**新的batch再將負分拿除**
- 這個節點將會成為待會要batch到神經網路的節點，state已經設定完畢。
## 到達底部的節點
- 若是到達底部的節點，直接從勝負+-1得出value，向上傳播，此節點**不用做神經網路推論(不用放到batch)**，因此不需要製作state，也不用去紀錄動作。
- 經過N次的模擬後，得到一個節點的batch，這些節點的state已經被處理成連續記憶體的資料，所以直接用這個資料區塊的指標，做成tensor，時間複雜度O(1)，丟進神經網路做推論
- 得到每個節點的value還有policy的機率分布，policy的機率分布值會儲存到展開的每個子節點，當作代進分數公式的機率，value則會做反向傳播。
- 這邊說明value反向傳播的過程，傳播就是下來的路徑傳回去，如果此節點的與上面節點的**玩家不一樣，value傳遞時就要加上負號**，一樣的玩家則不用，
- 假如每個節點的訪問次數為N，所以當value傳遞到節點時，就是Q=(Q*N+value)/(N+1)，然後N=N+1。

# 模型訓練
## policy network
在所有根節點，用子節點的訪問次數分布去訓練
## value network
在所有的根節點，如果是獲勝者，分數用+1，輸家則用-1

# 目前程式呼叫路徑（2026-07）
- 訓練入口：`start_training.py` -> `training.loop.main()` -> `training.loop.run(TITLE)`
- Self-play 由 `training.loop._collect_selfplay_samples()` 調度：
	- 若條件成立（cpp backend + async 開啟 + parallel_games > 1），走 `mcts.run_mcts_batch()`
	- 否則回退到單局 `mcts.run_mcts()`
- `mcts.run_mcts_batch()` 在可用時會進入 `mcts.async_worker.run_mcts_cpp_batch_async()`：
	- 多個 CPU worker 負責 SearchSession 收集葉節點
	- 單一 GPU worker 進行批次網路推論並回傳 priors/value
- Python / C++ 邊界：
	- Python：訓練主循環、資料收集、網路前向與反向
	- C++：PhaseGameState、SearchSession、葉節點打包與樹內模擬





