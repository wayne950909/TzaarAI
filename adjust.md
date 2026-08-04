我會把buffer的最大容量調到樹數量*32(填不滿)

# 每個執行緒的local buffer的is_ready變為true的條件
前提是兩個buffer都要為false，然後每次會對正在寫入的buffer做以下的條件檢查，符合一項ready就會變true
- 到達一定的資料量，如果沒有其他ready的buffer(這個量不是buffer的最大容量)
- 所有的樹都stalled了，不用特地去判斷，可以做成該執行緒無法從佇列獲得樹的時候，將自己正在寫入的buffer設為ready

註: 當buffer的ready從false轉變成true，執行緒會轉為寫入另一個buffer


# buffer的 ready 的狀況列舉
若buffer_id指向正在填入的buffer，以下分為兩種狀況:
1. 其中一個buffer的ready為true，另一個buffer的ready為false
    偵測本來是ready的buffer，如果為true，執行緒就持續將資料填入ready為false的buffer，如果本來是ready的buffer從true變成false，則會跳到第二種狀況
2. 兩個buffer都為false
    在第一種情況的ready的buffer被主執行緒偵測到後，複製資料然後送出，然後buffer會被清空，接著true會被轉變成false，此時正在填入的buffer獲取變成true的資格，條件到了就可以變為true

註: ready的原子變數改變(true=>false or false=>true)的時候都是release，獲取的時候都是acquire

# 其他狀況
當執行緒沒有樹可以模擬的時候，進入睡眠狀態，直到result_handler處理完一批資料後再去喚醒
當然，會藉由調整ready的資料量閥值，盡量讓執行緒不要進入睡眠
