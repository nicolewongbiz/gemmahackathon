@echo off
REM Live camera view + CALIBRATED check. Moves the arm to LOOK_POSE unless USE_MOCK_ARM = True.
cd /d %~dp0
.venv\Scripts\python.exe -m perception.calibrate %*
pause
