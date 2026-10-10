@echo off
rem Live view of the eye camera with the tracker overlay. R = record to out\cam, Q = quit.
rem Usage: eye_live.bat COM9   (the board's "USB" port; see Device Manager)
if "%~1"=="" (echo Usage: eye_live.bat COMx & pause & exit /b 1)
"%~dp0..\brow_tracker\.venv\Scripts\python.exe" "%~dp0tools\eye_cam.py" %1 live %2 %3
if errorlevel 1 pause
