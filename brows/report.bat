@echo off
rem Rebuild out\report.html (simulator sweep + real-footage results) and open it.
set "PY=%~dp0..\.venv\Scripts\python.exe"
"%PY%" "%~dp0tools\report.py" %* || (pause & exit /b 1)
start "" "%~dp0out\report.html"
