# build_cpp_module.ps1
# 用 Visual Studio 2022 建置 C++ pybind11 模組
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File build_cpp_module.ps1

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# 找到 VS 安裝路徑
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -property installationPath
} else {
    # 備用：檢查常見路徑
    $candidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        "C:\Program Files\Microsoft Visual Studio\18\Community"
    )
    $vsPath = $null
    foreach ($c in $candidates) {
        if (Test-Path "$c\Common7\Tools\VsDevCmd.bat") {
            $vsPath = $c
            break
        }
    }
}

if (-not $vsPath) {
    Write-Error "找不到 Visual Studio 2022 安裝路徑"
    exit 1
}

Write-Host "找到 Visual Studio: $vsPath"

# 載入 VS 開發環境
$devCmd = "$vsPath\Common7\Tools\VsDevCmd.bat"
Write-Host "載入開發環境: $devCmd"

# 清空並重建 build 目錄
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
New-Item -ItemType Directory -Force -Path "build" | Out-Null

# 執行 CMake
Write-Host "執行 CMake 配置..."
& cmd.exe /c "`"$devCmd`" && cd /d `"$scriptDir\build`" && cmake .. -G `"Visual Studio 17 2022`" -DCMAKE_BUILD_TYPE=Release"
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake 配置失敗"
    exit 1
}

# 編譯
Write-Host "編譯中..."
& cmd.exe /c "`"$devCmd`" && cd /d `"$scriptDir\build`" && cmake --build . --config Release --target tzaar_cpp"
if ($LASTEXITCODE -ne 0) {
    Write-Error "編譯失敗"
    exit 1
}

Write-Host ""
Write-Host "✅ 編譯成功！模組位置：build/Release/tzaar_cpp.pyd"
Write-Host ""
Write-Host "執行測試："
Write-Host "  cd $scriptDir\.."
Write-Host "  python -c `"from state.cpp_adapter import try_load_cpp_backend; m = try_load_cpp_backend(); print('OK:', dir(m))`""
