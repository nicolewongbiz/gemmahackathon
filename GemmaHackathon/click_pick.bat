@echo off
REM Click an object in the view and the arm picks it into bin A. Esc quits. Keep the E-stop in reach.
cd /d %~dp0
.venv\Scripts\python.exe -m scripts.test_click_pick %*
pause
