# 新的多執行緒計畫的原因
在multi_thread_logic.md的規劃中，由於使用多個原子變數和鎖，導致效率不佳，必須得找一個更平行化的方法
# 舊方法的大約的時間數據
8 games, 8 threads, 256 simulations
  CPU_sim         | count=4494 | total=217598341 us | avg= 48419 us | min=   712 us | max=202981 us
  flush           | count=4494 | total= 1952500 us | avg=   434 us | min=    89 us | max=  3315 us
  GPU_infer       | count=3404 | total=17047026 us | avg=  5007 us | min=  1524 us | max=1787510 us
  GPU_backfill    | count=3404 | total= 2750838 us | avg=   808 us | min=    95 us | max=  7503 us


8 games, 8 threads, 128 simulations
  CPU_sim         | count=2509 | total=46271850 us | avg= 18442 us | min=  1273 us | max= 89334 us
  flush           | count=2509 | total= 1403085 us | avg=   559 us | min=    91 us | max=  3694 us
  GPU_infer       | count=1364 | total= 6380695 us | avg=  4677 us | min=  1367 us | max=1796895 us
  GPU_backfill    | count=1364 | total= 1873216 us | avg=  1373 us | min=    95 us | max= 15880 us

舊的方法大部分時間還是在CPU模擬，GPU模擬時間比較固定

# 概念

1. 改成每個執行緒會有固定掌管的多棵樹(也就是將樹平均分配給每個執行緒)，而且每個負責模擬的執行緒就會伴隨一個result_handler(也就是說模擬的執行緒數量等於result_handler的執行緒數量)，這麼一來，執行緒不用鎖住樹，只需在模擬前判斷是否可以模擬
2. 使用moodycamel::ConcurrentQueue取代雙buffer，worker模擬完16次後，將蒐集到的資料一次enqueue，然後喚醒python端，python端會去檢查資料量有沒有超過預先設定的閥值，如果有則取出資料開始推論。

20 games 128 simulations 35s

