@echo off
REM One camera frame to Gemma; prints what it sees and the latency.
cd /d %~dp0
.venv\Scripts\python.exe -m scripts.test_gemma %*
pause
