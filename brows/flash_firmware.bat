@echo off
rem Build firmware\ and flash it to the ESP32-CAM. If the brow UI is running, it lets go of
rem the serial port for the upload and takes it back afterwards.
rem Optional arguments go to pio, e.g.:  flash_firmware.bat --upload-port COM5
set "PIO=pio"
where pio >nul 2>nul || set "PIO=%APPDATA%\Python\Python314\Scripts\pio.exe"
curl -s -m 2 -d close http://127.0.0.1:8766/port >nul 2>nul
pushd "%~dp0firmware"
"%PIO%" run -t upload %*
set "ERR=%errorlevel%"
popd
curl -s -m 2 -d open http://127.0.0.1:8766/port >nul 2>nul
if not "%ERR%"=="0" (echo. & echo FLASHING FAILED & pause & exit /b %ERR%)
echo.
echo Flashed OK.
timeout /t 5 >nul 2>nul
exit /b 0
