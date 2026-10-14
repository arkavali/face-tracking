@echo off
rem Plain text serial console of the eye camera (type commands, e.g. help, status).
rem Close eye_live first: only one program can hold the port. Ctrl+C to quit.
set "PIO=pio"
where pio >/dev/null 2>/dev/null || set "PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe"
pushd "%~dp0firmware"
"%PIO%" device monitor --echo --eol LF --rts 0 --dtr 0 %*
popd
