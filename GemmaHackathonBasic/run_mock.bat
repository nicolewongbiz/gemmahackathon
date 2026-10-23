@echo off
REM Chat with Gemma, fake arm. Add --backend rules to skip the model entirely.
cd /d %~dp0
start "" http://127.0.0.1:7860
.venv\Scripts\python.exe app.py --mock %*
