@echo off
rem Track the brows offline in a video, an image folder or a brow UI recording (out\cam\rec_*).
rem Drag the file or folder onto this .bat. A window opens on the first frame: draw one
rem rectangle per brow (Enter after each, Esc when done). The start of the clip must be a
rem neutral face. Results (plot + overlay sheet) open when done; files in out\video.
set "PY=%~dp0.venv\Scripts\python.exe"
if "%~1"=="" (echo Drag a video file or a recording folder onto this file. & pause & exit /b 1)
"%PY%" "%~dp0tools\track_video.py" %* --open
if errorlevel 1 pause
