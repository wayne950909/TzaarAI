# C++ pybind11 模組編譯指南

## 前置條件

- Visual Studio 2022 Build Tools（或完整版 VS 2022）
- Python 3.12（目前實際使用的版本，位於 pytorch_env）
- pybind11（已安裝在 Python site-packages 中）

> 注意：目前實際使用環境與本文件早期版本（曾標示 Python 3.14）不同。
> 本文件以「實際存在並可編譯成功」的現況為準（Python 3.12 / CPython 312 後綴）。

## 專案根目錄

本專案實際根目錄為：

```
C:\Users\user\Desktop\training2\TzaarAI
```

所有相對路徑（`cpp/`、`tzaar_cpp.cp312-win_amd64.pyd` 等）均以此為基準。

## 編譯步驟

```powershell
# 1. 從專案根目錄執行
cd C:\Users\user\Desktop\training2\TzaarAI

# 2. 清理並重建 build 目錄
Remove-Item -Recurse -Force cpp/build -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path cpp/build

# 3. 設定 Visual Studio 環境變數 + CMake 配置
cmd.exe /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" && cd /d cpp/build && cmake .. -G `"Visual Studio 17 2022`" -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=C:\Users\user\Desktop\pytorch_env\Lib\site-packages\pybind11\share\cmake\pybind11"

# 4. 編譯
cmd.exe /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" && cd /d cpp/build && cmake --build . --config Release --target tzaar_cpp"

# 5. 複製 .pyd 到專案根目錄
Copy-Item cpp/build/Release/tzaar_cpp.cp312-win_amd64.pyd tzaar_cpp.cp312-win_amd64.pyd -Force
```

## 快速編譯（一鍵執行）

從專案根目錄（`C:\Users\user\Desktop\training2\TzaarAI`）執行：

```powershell
cmd.exe /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" && cd /d cpp && if exist build rmdir /s /q build && mkdir build && cd build && cmake .. -G `"Visual Studio 17 2022`" -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=C:\Users\user\Desktop\pytorch_env\Lib\site-packages\pybind11\share\cmake\pybind11 && cmake --build . --config Release --target tzaar_cpp && copy /y Release\tzaar_cpp.cp312-win_amd64.pyd ..\..\tzaar_cpp.cp312-win_amd64.pyd"
```

## 使用 build_cpp_module.ps1（自動偵測路徑）

`cpp/build_cpp_module.ps1` 會用 vswhere 自動偵測 VS 安裝路徑，且不需手動指定
`pybind11_DIR`（只要執行時使用的 Python 已裝 pybind11，CMake 可自動找到）。

```powershell
cd C:\Users\user\Desktop\training2\TzaarAI\cpp
powershell -ExecutionPolicy Bypass -File build_cpp_module.ps1
```

模組產出位置：`cpp/build/Release/tzaar_cpp.cp312-win_amd64.pyd`

## 關鍵路徑

| 項目 | 路徑 |
|------|------|
| CMakeLists.txt | `cpp/CMakeLists.txt` |
| 原始碼 | `cpp/src/` |
| 標頭檔 | `cpp/include/` |
| 編譯輸出 | `cpp/build/Release/tzaar_cpp.cp312-win_amd64.pyd` |
| 目標位置 | `tzaar_cpp.cp312-win_amd64.pyd`（專案根目錄） |
| Python 執行環境 | `C:\Users\user\Desktop\pytorch_env\Scripts\python.exe` |
| pybind11 cmake 設定 | `C:\Users\user\Desktop\pytorch_env\Lib\site-packages\pybind11\share\cmake\pybind11` |
| VS BuildTools | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |

## 注意事項

- 使用 Visual Studio 17 2022 generator（即使 BuildTools 版本是 2022）
- 必須以「C++ bridge import 的 Python」執行 cmake，以便 `find_package` 找到正確的 Python 3.12 與 pybind11。
  若 CMake 找不到 pybind11，請指定 `-Dpybind11_DIR=<site-packages>/pybind11/share/cmake/pybind11`
- 每次修改 `search_manager.cpp`、`search.cpp` 或任何 `.h` 檔後都需要重新編譯
- .pyd 必須複製到專案根目錄（Python 會在那裡 import），或已由 `cpp_adapter.py` 加入 `cpp/build/Release` 到 `sys.path`
- Python import 之後綴需與執行環境 CPython 版本一致：
  - CPython 3.12 → `tzaar_cpp.cp312-win_amd64.pyd`
  - 若更換 Python 版本，請同步更新上述檔名與 pybind11/Include 路徑
