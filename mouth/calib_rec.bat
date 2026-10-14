@echo off
rem Prompted calibration recording (~3 min per pass): the window shows an expression, copy it.
rem Space start/pause, arrows (paused) redo, N skip, L LED, R plain video, Q stop. Saved to out\calib\cal_*. Optional: COM port, --reps 2.
set "PY=%~dp0..\.venv\Scripts\python.exe"
"%PY%" "%~dp0tools\calib_rec.py" %*
if errorlevel 1 pause
