@echo off
rem Plain text serial console of the brow camera (type commands, e.g. help, status).
rem Close brow_ui first: only one program can hold the port. Ctrl+C to quit.
set "PIO=pio"
where pio >nul 2>nul || set "PIO=%APPDATA%\Python\Python314\Scripts\pio.exe"
pushd "%~dp0firmware"
"%PIO%" device monitor --echo --eol LF --rts 0 --dtr 0 %*
popd
