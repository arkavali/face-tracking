@echo off
rem Simulated mask-camera view for one camera placement (contact sheet, opens when done).
rem Arguments go to tools\preview_sim.py, e.g.:
rem   sim_preview.bat --cam-y 57 --cam-z 39 --actor 4 --glasses
set "PY=%~dp0..\.venv\Scripts\python.exe"
pushd "%~dp0tools"
"%PY%" preview_sim.py %* --open
set "ERR=%errorlevel%"
popd
if not "%ERR%"=="0" pause
