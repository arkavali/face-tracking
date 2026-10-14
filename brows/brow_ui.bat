@echo off
rem Brow camera UI (opens in the browser): live view, ROI drawing, calibration,
rem brow plots, snapshots and recordings (out\cam). Close this window to stop it.
rem Optional argument: COM port (found automatically otherwise).
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (echo Python venv not found: %PY% & pause & exit /b 1)
"%PY%" "%~dp0tools\ui\browui.py" %*
if errorlevel 1 pause
