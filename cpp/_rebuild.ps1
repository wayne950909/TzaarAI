$ErrorActionPreference = "Continue"
Set-Location "C:\Users\user\Desktop\training2\TzaarAI"

$vsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
$pybind = "C:\Users\user\Desktop\pytorch_env\Lib\site-packages\pybind11\share\cmake\pybind11"

# Remove existing build dir
if (Test-Path "cpp\build") { Remove-Item -Recurse -Force "cpp\build" }
New-Item -ItemType Directory -Force -Path "cpp\build" | Out-Null

# CMake configure
Write-Host "=== CMake configure ==="
cmd.exe /c "`"$vsDevCmd`" >nul 2>nul && cd /d C:\Users\user\Desktop\training2\TzaarAI\cpp\build && cmake .. -G `"Visual Studio 17 2022`" -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=`"$pybind`""
Write-Host "CONFIGURE EXIT: $LASTEXITCODE"

# Build
Write-Host "=== Build tzaar_cpp ==="
cmd.exe /c "`"$vsDevCmd`" >nul 2>nul && cd /d C:\Users\user\Desktop\training2\TzaarAI\cpp\build && cmake --build . --config Release --target tzaar_cpp"
Write-Host "BUILD EXIT: $LASTEXITCODE"

# Copy
Write-Host "=== Copy pyd ==="
Copy-Item "cpp\build\Release\tzaar_cpp.cp312-win_amd64.pyd" "tzaar_cpp.cp312-win_amd64.pyd" -Force
Write-Host "DONE"
