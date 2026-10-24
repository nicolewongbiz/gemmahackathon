@echo off
REM Measure TABLE_TAGS by jogging the gripper onto each tag in the Web App. Reads the pose only; never moves the arm.
cd /d %~dp0
.venv\Scripts\python.exe -m scripts.measure_tags %*
pause
