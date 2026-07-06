"""新版訓練啟動腳本 — 直接執行這個檔案即可開始訓練。

用法：
    python start_training.py          # 一般訓練
    set DEBUG=1 && python start_training.py   # 偵錯模式

這個檔案刻意保持很薄：
- 只做 workspace path 注入
- 只設定預設狀態後端為 cpp
- 真正的訓練 orchestration 都集中在 training.loop.main()

這樣做的目的，是讓之後手動維護訓練流程時，入口與核心邏輯分離。
通常閱讀順序會是：
    start_training.py -> training/loop.py -> training/selfplay_engine.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TZAAR_STATE_BACKEND", "cpp")

from training.loop import main

if __name__ == "__main__":
    main()
