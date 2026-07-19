# C++ pybind11 模組編譯指南

## 前置條件

- Visual Studio 2022 Build Tools（或完整版 VS 2022）
- Python 3.14（目前使用的版本）
- pybind11（已安裝在 Python site-packages 中）

## 編譯步驟

```powershell
# 1. 從專案根目錄執行
cd C:\Users\user\Desktop\training

# 2. 清理並重建 build 目錄
Remove-Item -Recurse -Force cpp/build -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path cpp/build

# 3. 設定 Visual Studio 環境變數 + CMake 配置
cmd.exe /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" && cd /d cpp/build && cmake .. -G `"Visual Studio 17 2022`" -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=C:\Users\user\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\pybind11\share\cmake\pybind11"

# 4. 編譯
cmd.exe /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" && cd /d cpp/build && cmake --build . --config Release --target tzaar_cpp"

# 5. 複製 .pyd 到專案根目錄
Copy-Item cpp/build/Release/tzaar_cpp.cp314-win_amd64.pyd tzaar_cpp.cp314-win_amd64.pyd -Force
```

## 快速編譯（一鍵執行）

從專案根目錄（`C:\Users\user\Desktop\training`）執行：

```powershell
cmd.exe /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" && cd /d cpp && if exist build rmdir /s /q build && mkdir build && cd build && cmake .. -G `"Visual Studio 17 2022`" -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=C:\Users\user\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\pybind11\share\cmake\pybind11 && cmake --build . --config Release --target tzaar_cpp && copy /y Release\tzaar_cpp.cp314-win_amd64.pyd ..\..\tzaar_cpp.cp314-win_amd64.pyd"
```

## 關鍵路徑

| 項目 | 路徑 |
|------|------|
| CMakeLists.txt | `cpp/CMakeLists.txt` |
| 原始碼 | `cpp/src/` |
| 標頭檔 | `cpp/include/` |
| 編譯輸出 | `cpp/build/Release/tzaar_cpp.cp314-win_amd64.pyd` |
| 目標位置 | `tzaar_cpp.cp314-win_amd64.pyd`（專案根目錄） |
| pybind11 cmake 設定 | `C:\Users\user\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\pybind11\share\cmake\pybind11` |
| VS BuildTools | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |

## 注意事項

- 使用 Visual Studio 17 2022 generator（即使 BuildTools 版本是 2022）
- pybind11_DIR 必須指定，否則 CMake 找不到 pybind11
- 每次修改 `search_manager.cpp`、`search.cpp` 或任何 `.h` 檔後都需要重新編譯
- .pyd 必須複製到專案根目錄（Python 會在那裡 import）
