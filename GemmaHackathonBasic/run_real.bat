@echo off
REM Chat with Gemma and drive the REAL Kinova Gen3. Keep the E-stop in reach.
cd /d %~dp0
start "" http://127.0.0.1:7860
.venv\Scripts\python.exe app.py --real %*
