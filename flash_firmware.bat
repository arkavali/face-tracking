@echo off
rem Build firmware\ and flash it to the ESP32-S3 CAM over its "USB" port.
rem Optional arguments go to pio, e.g.:  flash_firmware.bat --upload-port COM9
set "PIO=pio"
where pio >/dev/null 2>/dev/null || set "PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe"
pushd "%~dp0firmware"
"%PIO%" run -t upload %*
set "ERR=%errorlevel%"
popd
if not "%ERR%"=="0" (echo. & echo FLASHING FAILED & pause & exit /b %ERR%)
echo.
echo Flashed OK.
timeout /t 5 >/dev/null 2>nul
exit /b 0
