@echo off
rem Build firmware\ and flash it to the ESP32-CAM. Optional arguments go to pio, e.g.:
rem   flash_firmware.bat --upload-port COM3
set "PIO=pio"
where pio >nul 2>nul || set "PIO=%APPDATA%\Python\Python314\Scripts\pio.exe"
pushd "%~dp0firmware"
"%PIO%" run -t upload %*
set "ERR=%errorlevel%"
popd
if not "%ERR%"=="0" (echo. & echo FLASHING FAILED & pause & exit /b %ERR%)
echo Flashed OK.
timeout /t 5 >nul 2>nul
