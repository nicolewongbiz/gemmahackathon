@echo off
REM Laptop server for the Pi voice agent (and curl): http://<laptop-ip>:8000. Ctrl+C stops it.
cd /d %~dp0
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 %*
pause
