@echo off
REM Save a frame from each camera index to data\cam*.jpg.
cd /d %~dp0
.venv\Scripts\python.exe -m scripts.find_camera %*
pause
