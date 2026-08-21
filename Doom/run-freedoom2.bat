@echo off
set "ROOT=%~dp0"
start "" "%ROOT%build-local\src\Release\chocolate-doom.exe" -window -iwad "%ROOT%iwads\freedoom-0.13.0\freedoom-0.13.0\freedoom2.wad"
