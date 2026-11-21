@echo off
rem Serves runs\ so runs\live.html (training dashboard) and runs\viewer.html open at http://localhost:8770/
cd /d "%~dp0.."
python -m http.server 8770 --bind 127.0.0.1 --directory runs
