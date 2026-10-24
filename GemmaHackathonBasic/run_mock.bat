@echo off
REM Chat with Gemma, fake arm. Add --backend rules to skip the model entirely.
cd /d %~dp0
REM Open the browser only once the server is listening (model load takes a few seconds).
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "while (-not (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue)) { Start-Sleep 1 }; Start-Process http://127.0.0.1:7860"
.venv\Scripts\python.exe app.py --mock %*
pause
