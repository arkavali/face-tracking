@echo off
rem Live demo of the trained mouth model: camera picture + value bars. L LED, Q quits.
rem Optional: COM port, --babble (adds Babble's values as thin orange bars), --smooth 0.5
set "PY=%~dp0..\.venv\Scripts\python.exe"
"%PY%" "%~dp0tools\mouth_live.py" %*
if errorlevel 1 pause
