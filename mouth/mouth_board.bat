@echo off
rem The mouth net running ON THE ESP32-CAM: bars = values from the board, picture = its 80x60 input.
rem L LED, Q quit. Optional: COM port, --smooth 0.5
set "PY=%~dp0..\.venv\Scripts\python.exe"
"%PY%" "%~dp0tools\mouth_live.py" --board %*
if errorlevel 1 pause
