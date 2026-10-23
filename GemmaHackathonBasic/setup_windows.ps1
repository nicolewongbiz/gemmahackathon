# One-shot setup for Windows. Run in PowerShell from this folder:
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== 1. Python venv (3.11 recommended) =="
$py = "py"; $ver = "-3.11"
try { & $py $ver --version } catch { $ver = "-3" }
& $py $ver -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`n== 2. Kinova Kortex API wheel =="
$whl = Get-ChildItem -Filter "kortex_api*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($whl) {
  & .\.venv\Scripts\python.exe -m pip install $whl.FullName
  & .\.venv\Scripts\python.exe -m pip install "protobuf==3.20.3"
} else {
  Write-Host "  No kortex_api*.whl in this folder. Download it from Kinova's Artifactory (link in the"
  Write-Host "  github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L README), drop it here, re-run this script."
}

Write-Host "`n== 3. Gemma model (LiteRT-LM) =="
if (-not (Test-Path models\gemma.litertlm)) { & .\.venv\Scripts\python.exe scripts\download_model.py }
else { Write-Host "  models\gemma.litertlm already present" }

Write-Host "`n== 4. Self-test (no robot, no model) =="
& .\.venv\Scripts\python.exe scripts\selftest.py

Write-Host "`nDone. Next:"
Write-Host "  run_mock.bat        chat with Gemma, fake arm"
Write-Host "  scripts\set_robot_ip_windows.ps1 (as admin), then run_real.bat"
