@echo off
rem Mouth camera live view: R record (out\cam\rec_*), S snapshot, L flash LED, Q quit.
rem Optional argument: COM port (the CH340 port is found automatically otherwise).
set "PY=%~dp0..\brow_tracker\.venv\Scripts\python.exe"
"%PY%" "%~dp0tools\mouth_cam.py" %*
if errorlevel 1 pause
