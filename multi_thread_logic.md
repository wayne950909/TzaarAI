# 多執行緒目的
預設建立多個c++的CPU執行緒和1個python的GPU執行緒，目的是讓**CPU和GPU同時運行**
**執行緒不會反覆被創立並刪除**


# 蒐集資料
cpu worker負責收集mcts待推論的葉節點，一棵樹只會由一個執行緒來模擬，一次**最多收集16的葉節點**
定義模擬次數，模擬一次最多產生一個待推論葉節點，也有可能不會產生，像是遇到底部節點
存在一個佇列(queue)，裡面放著尚需模擬的樹id

## CPU worker
1. 若佇列內有id，負責模擬mcts的CPU worker從佇列拿一個取出一個樹id；若佇列內沒有id，worker進入wait狀態，直到佇列內有id
2. 嘗試鎖住樹(mutex)，鎖住失敗就跳到步驟1.，成功鎖住的話則開始跑mcts
3. 模擬的過程，worker會把資料放進local暫存的地方(大小設為16，資料不會和其他樹混合)
4. 除非模擬次數消耗完，否則會持續蒐集到16個葉節點
5. 如果將模擬次數消耗完，沒有待推論的葉節點，則**已完成的樹數量增加1**
6. 完成模擬後，直接將所有節點資料丟進可以被填入的buffer
7. 填入資料後，檢查是否可以swap_buffer，可以的話就swap_buffer
8. 解鎖
9. 重複以上動作，找到下一棵樹來模擬

# buffer的運作
存在兩個buffer，每個的空間大小限制為樹的總數*16，也就是說**不用判定模擬完的資料放不放得下**，可以直接放
會設定一個原子變數，代表**資料要填到buffer 0或1**

## buffer寫入
cpu worker將資料從local寫到buffer會用到
1. 每個buffer存在兩個原子變數write_index和acitve_writers，分別記錄下一個寫入的位置跟正在寫入的執行緒數量
2. 避免位於同一個 Cache Line，以免false sharing
3. 當cpu worker要寫入資料時，增加write_index和active_writers(都用std::memory_order_relaxed，write_index根據要寫入的資料數量增加，active_writers增加1)，並根據write_index找到資料該寫入buffer的位置
4. 當cpu worker寫完資料後，再將active_writers(std::memory_order_release)減少1
這樣的原子變數確保active_writers的減少在write_index和active_writers的增加後面，這麼一來，active_writers可以藉由std::memory_order_acquire取得正確的active_writers數量

## swap_buffer
swap_buffer就是先讓資料轉為填入另一個buffer，然後將蒐集到資料的buffer拿去做gpu推論

### swap_buffer的檢查時機
- 每個cpu worker往buffer填入資料後
- gpu推論完將結果傳回result_handler時
### swap_buffer條件
---
swap_buffer的前提:
- 目前非填入資料的buffer已經gpu推論完成，這是cpu worker填入資料後要先檢查的
- 沒有其他執行緒在swap_buffer
---
若前提都成立，則以下條件至少符合一項就swap_buffer:
- buffer內的資料數量達到一定值
- 沒有新的樹可以被模擬了(所有的樹都已經被模擬過或是正在被模擬或是模擬次數結束)
如果沒有符合條件，也就代表目前蒐集的資料不夠
### swap_buffer流程
1. 若cpu worker決定要swap_buffer，必須先用原子交換鎖來避免多個worker去swap_buffer，其他worker若發現正在swap，略過就好
2. 假設現在填入的為buffer 0，另一個是buffer 1:
    1. 先將兩個buffer的能否被填入的布林值對調，代表接下來填入資料都變成填到buffer 1
    2. 接著等待buffer 0的active_writers變為0(std::memory_order_acquire)，沒有正在寫入的執行緒才能送進gpu
    3. 接著將buffer 0的資料送進gpu推論

# 一棵樹的模擬完成
模擬完成的定義是**消耗完模擬次數且沒有待推論葉節點**
樹有沒有完成會影響result_handler是否要再把樹放回佇列
存在**一個已完成樹的數量原子變數，從0開始**


# 處理推論完的資料 - result_handler worker
跟著負責模擬mcts的cpu worker一起被創建，result_handler負責處理gpu推論完的資料
1. 當result_handler worker收到buffer推論完的資料，會嘗試swap_buffer(不用檢查是否gpu推論完)
2. result_handler會根據資料，先鎖住資料對應的樹，將樹裡的virtual loss復原，然後將推論完的先驗機率跟value加到樹裡
3. 若此樹的模擬次數尚未達到目標次數，則將樹的id放到尚需模擬的佇列；若到達模擬次數，則不用放回佇列，並將已完成的樹數量增加1
4. 解鎖
5. 如果有將樹的id放回佇列，喚醒一個worker

# mcts搜尋結束
每當已完成的樹的數量增加時，檢查是否等於樹的總數，是的話則喚醒cpu worker執行緒並結束cpu worker和result_handler執行緒，最後喚醒python執行緒


